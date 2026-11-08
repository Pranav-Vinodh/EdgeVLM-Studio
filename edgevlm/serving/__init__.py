"""
Edge serving and low-latency inference modules.
"""

from edgevlm.serving.kv_cache import VisualPrefixKVCacheManager
from edgevlm.serving.speculative_decoder import CrossModalSpeculativeDecoder
from edgevlm.serving.awq_quantizer import WeightQuantizer

__all__ = [
    "VisualPrefixKVCacheManager",
    "CrossModalSpeculativeDecoder",
    "WeightQuantizer",
]
