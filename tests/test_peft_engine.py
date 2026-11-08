import torch
import torch.nn as nn
import pytest
from edgevlm.core.peft_engine import DecoupledVLMAdapter


class MockVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


class MockProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(64, 64)
        self.linear_2 = nn.Linear(64, 64)

    def forward(self, x):
        return self.linear_2(self.linear_1(x))


class MockLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)

    def forward(self, x):
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))


class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = MockVisionTower()
        self.multi_modal_projector = MockProjector()
        self.language_model = MockLanguageModel()

    def forward(self, x):
        v = self.vision_tower(x)
        p = self.multi_modal_projector(v)
        return self.language_model(p)


def test_target_modules_discovery():
    adapter = DecoupledVLMAdapter(model_family="smolvlm")
    targets = adapter.build_target_modules()
    assert len(targets) > 0
    assert any("self_attn.q_proj" in t for t in targets)
    assert any("multi_modal_projector" in t for t in targets)


def test_trainable_parameters_stats():
    model = MockVLM()
    stats = DecoupledVLMAdapter.print_trainable_parameters(model)
    assert stats["trainable_params"] > 0
    assert stats["total_params"] > 0
    assert stats["trainable_ratio_percent"] == 100.0


def test_configure_peft_injection():
    model = MockVLM()
    adapter = DecoupledVLMAdapter(
        base_model=model,
        model_family="default",
        language_rank=8,
        use_dora=False,
    )
    peft_model = adapter.configure_peft()
    assert peft_model is not None
    # Frozen base parameters vs trainable LoRA adapters
    stats = DecoupledVLMAdapter.print_trainable_parameters(peft_model)
    assert stats["trainable_params"] < stats["total_params"]
    assert stats["trainable_ratio_percent"] < 100.0
