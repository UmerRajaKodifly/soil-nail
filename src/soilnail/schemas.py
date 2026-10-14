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


class SphericalGeom(BaseModel):
    """Geometry on the unit sphere (radians unless noted)."""

    model_config = ConfigDict(extra="forbid")

    lon_rad: float
    lat_rad: float
    angular_width_deg: float
    angular_height_deg: float
    solid_angle_sr: float


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    model: str
    config_id: str
    instance_id: int
    mask_rle: RLE
    bbox_xywh: list[float]  # COCO xywh in face (or equirect for legacy) coords
    area_px: int
    area_fraction: float = Field(description="Fraction of the face pixel area (mask_px / face_px).")
    area_fraction_solid_angle: float | None = Field(
        default=None,
        description="solid_angle_sr / (4π); None for non-360 / legacy runs.",
    )
    raw_score: float = Field(
        description=(
            "Uncalibrated model score (SAM IoU head or SegGPT proxy). "
            "NOT a calibrated probability — do not present as confidence."
        )
    )
    touches_border: bool = False
    merged_across_tiles: bool = False
    merged_across_faces: bool = False
    area_flag: bool = False
    source_tile_ids: list[str] = Field(default_factory=list)
    face_id: str | None = Field(
        default=None, description="Cubemap face id F|R|B|L|U|D; None for flat/legacy."
    )
    spherical: SphericalGeom | None = None
    equirect_polygon_xy: list[list[float]] | None = Field(
        default=None,
        description="Largest equirect ring [[x,y],…] (convenience); prefer equirect_polygons_xy.",
    )
    equirect_polygons_xy: list[list[list[float]]] | None = Field(
        default=None,
        description=(
            "One or more closed rings in equirect pixel coords. "
            "Split into multiple rings when the contour crosses the ±180° wrap."
        ),
    )
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
    # Top-level model/config so empty-detection rows still key correctly in the store.
    model: str | None = None
    config_id: str | None = None


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
