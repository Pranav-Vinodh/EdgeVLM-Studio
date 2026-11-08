"""
Decoupled Multi-Modal PEFT Engine.

Orchestrates asymmetric parameter-efficient fine-tuning (LoRA / DoRA)
across the Vision Transformer, Multi-Modal Projector, and Language Decoder.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DecoupledVLMAdapter:
    """
    Orchestrates asymmetric parameter-efficient fine-tuning across
    Vision Transformer, Multi-Modal Projector, and Language Decoder.
    Supports Weight-Decomposed LoRA (DoRA) and 4-bit NF4 quantization.
    """

    MODEL_TARGET_MAP = {
        "smolvlm": {
            "vision": [
                "vision_model.encoder.layers.*.self_attn.q_proj",
                "vision_model.encoder.layers.*.self_attn.v_proj",
            ],
            "projector": [
                "multi_modal_projector.linear_1",
                "multi_modal_projector.linear_2",
            ],
            "language": [
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
                "model.layers.*.self_attn.o_proj",
            ],
        },
        "paligemma": {
            "vision": [
                "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
                "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
            ],
            "projector": [
                "multi_modal_projector.linear",
            ],
            "language": [
                "language_model.model.layers.*.self_attn.q_proj",
                "language_model.model.layers.*.self_attn.v_proj",
            ],
        },
        "qwen2_vl": {
            "vision": [
                "visual.blocks.*.attn.qkv",
                "visual.blocks.*.attn.proj",
            ],
            "projector": [
                "visual.merger.mlp.*",
            ],
            "language": [
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
                "model.layers.*.self_attn.o_proj",
            ],
        },
        "default": {
            "vision": ["q_proj", "v_proj"],
            "projector": ["linear_1", "linear_2"],
            "language": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        model_family: str = "smolvlm",
        vision_rank: int = 8,
        vision_alpha: int = 16,
        language_rank: int = 16,
        language_alpha: int = 32,
        projector_rank: int = 32,
        projector_alpha: int = 64,
        use_dora: bool = True,
        target_vision: bool = True,
        target_projector: bool = True,
        target_language: bool = True,
        lora_dropout: float = 0.05,
    ):
        self.base_model = base_model
        self.model_family = model_family.lower()
        self.vision_rank = vision_rank
        self.vision_alpha = vision_alpha
        self.language_rank = language_rank
        self.language_alpha = language_alpha
        self.projector_rank = projector_rank
        self.projector_alpha = projector_alpha
        self.use_dora = use_dora
        self.target_vision = target_vision
        self.target_projector = target_projector
        self.target_language = target_language
        self.lora_dropout = lora_dropout

    def build_target_modules(self) -> List[str]:
        """Collect target module patterns depending on model family and adapter flags."""
        patterns = self.MODEL_TARGET_MAP.get(self.model_family, self.MODEL_TARGET_MAP["default"])
        target_modules: List[str] = []

        if self.target_vision and "vision" in patterns:
            target_modules.extend(patterns["vision"])
        if self.target_projector and "projector" in patterns:
            target_modules.extend(patterns["projector"])
        if self.target_language and "language" in patterns:
            target_modules.extend(patterns["language"])

        # Deduplicate while preserving order
        seen = set()
        unique_targets = [x for x in target_modules if not (x in seen or seen.add(x))]
        return unique_targets

    @staticmethod
    def get_quantization_config(
        quant_type: str = "nf4",
        double_quant: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> Optional[Any]:
        """Generate BitsAndBytesConfig for 4-bit or 8-bit quantized fine-tuning."""
        if not BNB_AVAILABLE:
            logger.warning("bitsandbytes is not installed; returning None for quantization config.")
            return None

        if quant_type == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quant_type == "fp4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quant_type == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def configure_peft(self, model: Optional[nn.Module] = None) -> nn.Module:
        """
        Injects decoupled PEFT adapters into the given or internal base model.
        Enables gradient checkpointing and prints parameter allocation stats.
        """
        if not PEFT_AVAILABLE:
            raise ImportError("peft package is required for DecoupledVLMAdapter.configure_peft()")

        target_model = model if model is not None else self.base_model
        if target_model is None:
            raise ValueError("No model provided for PEFT configuration.")

        # Enable gradient checkpointing if supported
        if hasattr(target_model, "gradient_checkpointing_enable"):
            try:
                target_model.gradient_checkpointing_enable()
            except Exception as e:
                logger.debug(f"Gradient checkpointing not enabled: {e}")

        # Prepare for k-bit training if quantized
        if hasattr(target_model, "is_loaded_in_4bit") or hasattr(target_model, "is_loaded_in_8bit"):
            target_model = prepare_model_for_kbit_training(target_model)

        target_modules = self.build_target_modules()
        task_type = "CAUSAL_LM" if hasattr(target_model, "prepare_inputs_for_generation") else None
        
        lora_config = LoraConfig(
            r=self.language_rank,
            lora_alpha=self.language_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=task_type,
            use_dora=self.use_dora,
        )

        peft_model = get_peft_model(target_model, lora_config)
        self.print_trainable_parameters(peft_model)
        return peft_model

    @staticmethod
    def print_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
        """Computes and logs the ratio of trainable parameters to total parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = (100.0 * trainable / total) if total > 0 else 0.0

        stats = {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio_percent": ratio,
        }
        logger.info(
            f"Trainable Params: {trainable:,} | Total Params: {total:,} | Ratio: {ratio:.2f}%"
        )
        return stats
