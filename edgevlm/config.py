"""
Configuration schemas and loaders for EdgeVLM-Studio.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    name_or_path: str = "HuggingFaceTB/SmolVLM-Instruct"
    family: str = "smolvlm"  # 'smolvlm', 'paligemma', 'qwen2_vl'
    torch_dtype: str = "bfloat16"  # 'float16', 'bfloat16', 'float32'
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class PEFTConfig:
    vision_rank: int = 8
    vision_alpha: int = 16
    language_rank: int = 16
    language_alpha: int = 32
    projector_rank: int = 32
    projector_alpha: int = 64
    use_dora: bool = True
    target_vision: bool = True
    target_projector: bool = True
    target_language: bool = True
    quant_type: str = "nf4"  # 'nf4', 'fp4', 'int8', 'none'
    double_quant: bool = True
    lora_dropout: float = 0.05


@dataclass
class PrunerConfig:
    enabled: bool = True
    keep_ratio: float = 0.45  # Keeps top 45% salient tokens (55% reduction)
    min_keep_patches: int = 64
    pooling_strategy: str = "mean_cross_attn"  # 'mean_cross_attn', 'max_cross_attn', 'cls_similarity'
    layer_idx: int = -1  # layer from which to harvest cross-attention maps


@dataclass
class LossWeightsConfig:
    lambda_ce: float = 1.0
    lambda_l1: float = 1.0
    lambda_giou: float = 2.0


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"  # 'no', 'fp16', 'bf16'
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    losses: LossWeightsConfig = field(default_factory=LossWeightsConfig)


@dataclass
class ServingConfig:
    kv_cache_prefix_enabled: bool = True
    speculative_enabled: bool = True
    draft_model_name: Optional[str] = "HuggingFaceTB/SmolVLM-256M-Instruct"
    lookahead_k: int = 4
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class EdgeVLMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeVLMConfig":
        model_cfg = ModelConfig(**data.get("model", {}))
        peft_cfg = PEFTConfig(**data.get("peft", {}))
        pruner_cfg = PrunerConfig(**data.get("pruner", {}))
        
        train_data = data.get("training", {})
        losses_data = train_data.pop("losses", {})
        loss_cfg = LossWeightsConfig(**losses_data) if losses_data else LossWeightsConfig()
        train_cfg = TrainingConfig(**train_data, losses=loss_cfg)
        
        serving_cfg = ServingConfig(**data.get("serving", {}))
        
        return cls(
            model=model_cfg,
            peft=peft_cfg,
            pruner=pruner_cfg,
            training=train_cfg,
            serving=serving_cfg,
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def load_config(config_path: str) -> EdgeVLMConfig:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return EdgeVLMConfig.from_dict(data)
