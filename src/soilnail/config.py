"""Load and validate YAML configs with pydantic.

Every numeric constant used by the production pipeline must enter through here
(or CLI overrides of config fields). Magic numbers in call sites are bugs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SourceNote(BaseModel):
    """Provenance for a config value (spec: every constant has a source note)."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description="upstream default | measured by script X | my decision | …",
    )


class SeedsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_seed: int = Field(description="Master RNG seed for sampling and bootstraps.")
    source: str


class ProjectConfig(BaseModel):
    """Minimal root config validated in Phase 0; expanded in later phases."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    schema_version: str
    seeds: SeedsConfig


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_project_config(path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml(path))
