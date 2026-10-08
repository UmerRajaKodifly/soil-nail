"""Phase 0: config loader smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soilnail.config import ProjectConfig, load_project_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_project_yaml_loads() -> None:
    cfg = load_project_config(ROOT / "configs" / "project.yaml")
    assert cfg.project_name == "soilnail-smarteye"
    assert cfg.schema_version == "0"
    assert cfg.seeds.global_seed == 42
    assert "decision" in cfg.seeds.source or "reproducibility" in cfg.seeds.source


def test_load_yaml_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(ROOT / "configs" / "does_not_exist.yaml")


def test_project_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {
                "project_name": "x",
                "schema_version": "0",
                "seeds": {"global_seed": 1, "source": "t"},
                "unexpected": True,
            }
        )
