import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytest

from edgevlm.config import EdgeVLMConfig
from edgevlm.core.trainer import EdgeVLMTrainer


class DummyDataset(Dataset):
    def __init__(self, size=8, seq_len=16, vocab_size=100):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "pred_boxes": torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            "target_boxes": torch.tensor([[0.22, 0.18, 0.58, 0.62]]),
        }


class DummyVLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        logits = self.fc(h)
        return logits


def test_edgevlm_trainer_lifecycle():
    temp_dir = tempfile.mkdtemp()
    try:
        config = EdgeVLMConfig()
        config.training.output_dir = temp_dir
        config.training.gradient_accumulation_steps = 2
        config.training.logging_steps = 2
        config.training.mixed_precision = "no"
        config.pruner.enabled = False

        model = DummyVLM()
        dataset = DummyDataset(size=4)
        dataloader = DataLoader(dataset, batch_size=2)

        trainer = EdgeVLMTrainer(
            config=config,
            model=model,
            train_dataloader=dataloader,
            device=torch.device("cpu"),
        )

        # Test single train step
        batch = next(iter(dataloader))
        loss_dict = trainer.train_step(batch)
        assert "loss_total" in loss_dict
        assert loss_dict["loss_total"] > 0

        # Test full epoch
        epoch_stats = trainer.train_epoch(epoch_idx=1)
        assert "avg_loss" in epoch_stats
        assert epoch_stats["avg_loss"] > 0

        # Test saving checkpoint
        ckpt_path = os.path.join(temp_dir, "model_checkpoint.pt")
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
