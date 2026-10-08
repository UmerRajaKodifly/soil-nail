"""Typed result schemas (Detection, Failure, RunManifest)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RLE(BaseModel):
    """COCO-compatible run-length encoding (uncompressed counts)."""

    model_config = ConfigDict(extra="forbid")

    counts: list[int]
    size: list[int]  # [h, w]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    model: str
    config_id: str
    instance_id: int
    mask_rle: RLE
    bbox_xywh: list[float]  # COCO xywh
    area_px: int
    area_fraction: float
    raw_score: float = Field(
        description=(
            "Uncalibrated model score (SAM IoU head or SegGPT proxy). "
            "NOT a calibrated probability — do not present as confidence."
        )
    )
    touches_border: bool = False
    merged_across_tiles: bool = False
    area_flag: bool = False
    source_tile_ids: list[str] = Field(default_factory=list)
    stage_tag: Literal["hole_only", "bar_installed", "grouted", "unclear"] | None = None


class Failure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    stage: str
    exception_type: str
    message: str
    model: str | None = None
    config_id: str | None = None


class ImageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    status: Literal["succeeded", "failed", "skipped"]
    detections: list[Detection] = Field(default_factory=list)
    failure: Failure | None = None
    runtime_s: float | None = None
    peak_vram_mb: float | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    run_id: str
    created_utc: datetime
    git_commit: str | None
    config_hash: str
    config_path: str
    seeds: dict[str, int]
    environment: dict[str, Any]
    input_file_sha256: dict[str, str]
    n_inputs: int
    n_succeeded: int
    n_failed: int
    n_skipped: int

    def reconcile(self) -> None:
        if self.n_inputs != self.n_succeeded + self.n_failed + self.n_skipped:
            raise AssertionError(
                f"manifest reconcile failed: {self.n_inputs} != "
                f"{self.n_succeeded}+{self.n_failed}+{self.n_skipped}"
            )
