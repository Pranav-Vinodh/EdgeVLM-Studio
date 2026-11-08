import torch
import pytest
from edgevlm.serving.kv_cache import VisualPrefixKVCacheManager


def test_kv_cache_registration_and_retrieval():
    manager = VisualPrefixKVCacheManager(device=torch.device("cpu"))
    session_id = "test_user_session_1"

    # [Layers, Batch, Heads, Vis_Len, Head_Dim]
    k_states = torch.randn(4, 1, 8, 128, 64)
    v_states = torch.randn(4, 1, 8, 128, 64)

    manager.register_visual_prefix(session_id, k_states, v_states)
    assert len(manager) == 1

    retrieved = manager.retrieve_prefix(session_id)
    assert retrieved is not None
    assert torch.equal(retrieved[0], k_states)
    assert torch.equal(retrieved[1], v_states)


def test_kv_cache_concatenation():
    manager = VisualPrefixKVCacheManager(device=torch.device("cpu"))
    session_id = "session_concat"

    vis_k = torch.randn(4, 1, 8, 100, 64)
    vis_v = torch.randn(4, 1, 8, 100, 64)
    manager.register_visual_prefix(session_id, vis_k, vis_v)

    text_k = torch.randn(4, 1, 8, 20, 64)
    text_v = torch.randn(4, 1, 8, 20, 64)

    full_k, full_v = manager.concatenate_prefix(session_id, text_k, text_v)
    assert full_k.shape == (4, 1, 8, 120, 64)
    assert full_v.shape == (4, 1, 8, 120, 64)


def test_kv_cache_purge_and_memory():
    manager = VisualPrefixKVCacheManager(device=torch.device("cpu"))
    k = torch.randn(2, 1, 4, 64, 32)
    v = torch.randn(2, 1, 4, 64, 32)

    manager.register_visual_prefix("s1", k, v)
    assert manager.get_cache_memory_mb() > 0

    assert manager.purge_session("s1") is True
    assert len(manager) == 0
    assert manager.get_cache_memory_mb() == 0.0
