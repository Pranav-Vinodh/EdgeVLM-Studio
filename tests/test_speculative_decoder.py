import torch
import torch.nn as nn
import pytest
from edgevlm.serving.speculative_decoder import CrossModalSpeculativeDecoder


class MockModel(nn.Module):
    """Generates predictable next-token logits based on simple linear mapping."""

    def __init__(self, vocab_size: int = 50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 32)
        self.head = nn.Linear(32, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        h = self.embedding(input_ids)
        logits = self.head(h)
        return logits


def test_speculative_decoder_generation():
    vocab_size = 50
    torch.manual_seed(42)

    draft_model = MockModel(vocab_size)
    target_model = MockModel(vocab_size)

    decoder = CrossModalSpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        lookahead_k=3,
    )

    batch_size = 1
    prompt_len = 4
    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len))

    output_ids = decoder.generate(input_ids, max_new_tokens=6)

    assert output_ids.shape[0] == batch_size
    assert output_ids.shape[1] > prompt_len

    stats = decoder.get_telemetry()
    assert stats["total_drafted"] > 0
    assert "acceptance_rate_percent" in stats
