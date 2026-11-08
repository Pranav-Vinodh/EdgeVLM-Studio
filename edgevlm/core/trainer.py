"""
EdgeVLM Trainer.

Engineers a memory-efficient training loop supporting mixed-precision (FP16/BF16),
gradient accumulation, visual token pruning, multi-task losses, and step-level VRAM profiling.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgevlm.config import EdgeVLMConfig, load_config
from edgevlm.core.loss_functions import MultiTaskVLMLoss
from edgevlm.core.visual_pruner import SpatialVisualTokenPruner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EdgeVLMTrainer:
    """
    Modular training loop optimized for single-GPU consumer & edge environments.
    """

    def __init__(
        self,
        config: EdgeVLMConfig,
        model: nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Loss function
        loss_cfg = self.config.training.losses
        self.loss_fn = MultiTaskVLMLoss(
            lambda_ce=loss_cfg.lambda_ce,
            lambda_l1=loss_cfg.lambda_l1,
            lambda_giou=loss_cfg.lambda_giou,
        )

        # Visual token pruner
        if self.config.pruner.enabled:
            self.pruner = SpatialVisualTokenPruner(
                keep_ratio=self.config.pruner.keep_ratio,
                min_keep_patches=self.config.pruner.min_keep_patches,
                pooling_strategy=self.config.pruner.pooling_strategy,
            ).to(self.device)
        else:
            self.pruner = None

        # Optimizer
        if optimizer is None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # Mixed precision setup
        amp_mode = self.config.training.mixed_precision.lower()
        self.use_amp = (self.device.type == "cuda") and (amp_mode in ["fp16", "bf16"])
        self.amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
        # Scaler setup with PyTorch modern amp API
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and amp_mode == "fp16"))
        except (TypeError, AttributeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and amp_mode == "fp16"))

        os.makedirs(self.config.training.output_dir, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single forward + backward training step on a batch."""
        self.model.train()

        input_ids = batch.get("input_ids")
        labels = batch.get("labels", input_ids)
        pixel_values = batch.get("pixel_values")
        pred_boxes = batch.get("pred_boxes")
        target_boxes = batch.get("target_boxes")

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if pred_boxes is not None:
            pred_boxes = pred_boxes.to(self.device)
        if target_boxes is not None:
            target_boxes = target_boxes.to(self.device)

        # Autocast context
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Model forward call
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                return_dict=True,
            )

            logits = getattr(outputs, "logits", outputs)

            # Compute multi-task loss
            total_loss, loss_dict = self.loss_fn(
                lm_logits=logits,
                labels=labels,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
            )

            # Scale loss for gradient accumulation
            accum_loss = total_loss / self.config.training.gradient_accumulation_steps

        # Backward pass
        if self.scaler.is_enabled():
            self.scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        return loss_dict

    def train_epoch(self, epoch_idx: int) -> Dict[str, float]:
        """Trains for one full epoch across train_dataloader."""
        if self.train_dataloader is None:
            logger.warning("No train_dataloader provided; skipping epoch.")
            return {}

        epoch_loss = 0.0
        step_count = 0
        start_time = time.time()

        for step, batch in enumerate(self.train_dataloader):
            loss_dict = self.train_step(batch)
            epoch_loss += loss_dict.get("loss_total", 0.0)
            step_count += 1

            # Gradient accumulation step
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()

            if (step + 1) % self.config.training.logging_steps == 0:
                vram_info = ""
                if self.device.type == "cuda":
                    alloc = torch.cuda.memory_allocated() / (1024**2)
                    vram_info = f" | VRAM: {alloc:.1f}MB"
                logger.info(
                    f"Epoch {epoch_idx} | Step {step+1}/{len(self.train_dataloader)} | "
                    f"Loss: {loss_dict.get('loss_total', 0.0):.4f}{vram_info}"
                )

        duration = time.time() - start_time
        avg_loss = epoch_loss / max(1, step_count)
        logger.info(
            f"Epoch {epoch_idx} Complete | Avg Loss: {avg_loss:.4f} | Time: {duration:.2f}s"
        )
        return {"avg_loss": avg_loss, "epoch_time_seconds": duration}

    def save_checkpoint(self, path: str) -> None:
        """Saves adapter weights or full state dictionary."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
            logger.info(f"Saved PEFT adapter checkpoint to {path}")
        else:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved model state dictionary to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EdgeVLM Training Entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config} for model {cfg.model.name_or_path}")
    print(f"EdgeVLM ready to train {cfg.model.family} using rank {cfg.peft.language_rank}")


if __name__ == "__main__":
    main()
