import torch
import pytest
from edgevlm.core.visual_pruner import SpatialVisualTokenPruner


def test_visual_pruner_shape_reduction():
    batch_size = 2
    num_patches = 100
    embed_dim = 128
    keep_ratio = 0.40

    pruner = SpatialVisualTokenPruner(keep_ratio=keep_ratio, min_keep_patches=10)
    visual_embeds = torch.randn(batch_size, num_patches, embed_dim)

    # Synthetic saliency weights
    saliency = torch.rand(batch_size, num_patches)

    pruned_embeds, indices, returned_saliency = pruner(
        visual_embeds, saliency_weights=saliency
    )

    expected_keep = 40
    assert pruned_embeds.shape == (batch_size, expected_keep, embed_dim)
    assert indices.shape == (batch_size, expected_keep)
    # Check that indices are strictly monotonically increasing (preserving 2D raster order)
    for b in range(batch_size):
        diffs = indices[b, 1:] - indices[b, :-1]
        assert (diffs > 0).all()


def test_visual_pruner_cross_attention_pooling():
    pruner = SpatialVisualTokenPruner(keep_ratio=0.50)
    b, h, nt, nv = 2, 4, 16, 64
    d = 32

    cross_attn = torch.rand(b, h, nt, nv)
    visual_embeds = torch.randn(b, nv, d)

    pruned, indices, saliency = pruner(visual_embeds, cross_attn_map=cross_attn)
    assert pruned.shape == (b, 32, d)
    assert saliency.shape == (b, nv)


def test_compression_stats():
    stats = SpatialVisualTokenPruner.get_compression_stats(1000, 450)
    assert stats["tokens_eliminated"] == 550
    assert pytest.approx(stats["compression_ratio_percent"], 0.1) == 55.0
    assert pytest.approx(stats["theoretical_speedup_factor"], 0.05) == 2.22
