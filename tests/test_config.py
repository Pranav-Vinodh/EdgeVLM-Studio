import os
import pytest
from edgevlm.config import EdgeVLMConfig, load_config


def test_default_config():
    config = EdgeVLMConfig()
    assert config.model.family == "smolvlm"
    assert config.peft.use_dora is True
    assert config.peft.quant_type == "nf4"
    assert config.pruner.keep_ratio == 0.45
    assert config.training.losses.lambda_giou == 2.0


def test_load_yaml_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "smolvlm_vqa_lora.yaml"
    )
    cfg = load_config(config_path)
    assert cfg.model.family == "smolvlm"
    assert cfg.peft.vision_rank == 8
    assert cfg.peft.language_rank == 16
    assert cfg.pruner.enabled is True
    assert cfg.pruner.keep_ratio == 0.45


def test_paligemma_grounding_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "paligemma2_grounding.yaml"
    )
    cfg = load_config(config_path)
    assert cfg.model.family == "paligemma"
    assert cfg.training.losses.lambda_l1 == 1.0
    assert cfg.training.losses.lambda_giou == 2.0


def test_missing_config_raises_error():
    with pytest.raises(FileNotFoundError):
        load_config("/non/existent/path/config.yaml")
