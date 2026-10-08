"""Pydantic models for ingest config and audit records."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: Path
    extract_root: Path
    manifest_dir: Path
    audit_md: Path
    source: str


class ArchiveSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    zip_filename: str
    csv_filename: str
    source: str


class ZipLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_uncompressed_bytes: int
    max_compression_ratio: float
    max_members: int
    forbid_symlinks: bool
    forbid_absolute_paths: bool
    source: str


class ImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_extensions: list[str]
    max_pixels: int
    apply_exif_transpose: bool
    source: str


class StampProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_size: int
    corner_frac_w: float
    corner_frac_h: float
    sat_delta_min: float
    sat_abs_min: float
    likely_fraction_threshold: float
    measured_cover_frac_w: float
    measured_cover_frac_h: float
    crop_enabled_default: bool
    source: str


class SeedsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_seed: int
    source: str


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    paths: PathsConfig
    archives: list[ArchiveSpec]
    zip_limits: ZipLimits
    image: ImageConfig
    stamp_probe: StampProbeConfig
    seeds: SeedsConfig


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    archive: str
    relative_path: str
    filename: str
    sha256: str
    bytes: int
    status: Literal["ok", "failed", "skipped"]
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    orig_width: int | None = None
    orig_height: int | None = None
    effective_width: int | None = None
    effective_height: int | None = None
    mode: str | None = None
    format: str | None = None
    exif_orientation: int | None = None
    transposed: bool = False
    csv_matched: bool = False


class CsvProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archive: str
    path: str
    encoding: str
    delimiter: str
    has_bom: bool
    n_rows: int
    n_cols: int
    columns: list[str]
    null_rates: dict[str, float]
    duplicate_row_count: int
    coordinate_kind: Literal[
        "wgs84_latlon",
        "local_metres",
        "hk1980_grid_candidate",
        "unknown",
        "none",
    ]
    coordinate_rationale: str
    join_key: str
    rows_matched_to_images: int
    rows_unmatched: int
    images_unmatched: int
    unmatched_row_keys: list[str] = Field(default_factory=list)
    unmatched_image_names: list[str] = Field(default_factory=list)
