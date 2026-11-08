"""
Core components for parameter-efficient fine-tuning, token pruning, and training.
"""

from edgevlm.core.peft_engine import DecoupledVLMAdapter
from edgevlm.core.visual_pruner import SpatialVisualTokenPruner
from edgevlm.core.loss_functions import MultiTaskVLMLoss
from edgevlm.core.trainer import EdgeVLMTrainer

__all__ = [
    "DecoupledVLMAdapter",
    "SpatialVisualTokenPruner",
    "MultiTaskVLMLoss",
    "EdgeVLMTrainer",
]
