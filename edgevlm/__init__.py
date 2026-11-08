"""
EdgeVLM-Studio: Modular Fine-Tuning and Quantized Edge Inference Framework.
"""

from edgevlm.config import EdgeVLMConfig, load_config
from edgevlm.core.peft_engine import DecoupledVLMAdapter
from edgevlm.core.visual_pruner import SpatialVisualTokenPruner
from edgevlm.core.loss_functions import MultiTaskVLMLoss
from edgevlm.core.trainer import EdgeVLMTrainer
from edgevlm.serving.kv_cache import VisualPrefixKVCacheManager
from edgevlm.serving.speculative_decoder import CrossModalSpeculativeDecoder

__version__ = "0.1.0"
__all__ = [
    "EdgeVLMConfig",
    "load_config",
    "DecoupledVLMAdapter",
    "SpatialVisualTokenPruner",
    "MultiTaskVLMLoss",
    "EdgeVLMTrainer",
    "VisualPrefixKVCacheManager",
    "CrossModalSpeculativeDecoder",
]
