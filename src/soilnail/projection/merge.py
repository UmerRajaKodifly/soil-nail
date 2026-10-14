"""Cross-face detection merge (overlap-primary, centroid secondary)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from soilnail.projection.sphere_ops import angular_distance_rad


@dataclass(frozen=True)
class MergeConfig:
    overlap_jaccard_min: float = 0.05
    secondary_centroid_deg: float = 2.0
    coarse_width: int = 360  # 1° bins — faster
    coarse_height: int = 180
    source: str = (
        "overlap_jaccard_min=0.05 on 1° equirect bins; "
        "secondary_centroid_deg=2.0 (=4× min_instance_angular_width_deg 0.5°); "
        "only cross-face pairs where ≥1 touches_border are overlap-tested; "
        "same-face merges require pixel overlap only. "
        "Sensitivity: outputs/pilot_cubemap/merge_sensitivity.json"
    )


@dataclass
class MergeItem:
    lon: float
    lat: float
    face_id: str
    touches_border: bool
    mask: NDArray[np.bool_]
    payload: Any
    _canvas: NDArray[np.uint8] | None = field(default=None, repr=False)


def _ensure_canvas(
    item: MergeItem,
    *,
    face_size: int,
    fov_deg: float,
    margin_deg: float,
    yaw_offset_deg: float,
    cfg: MergeConfig,
) -> NDArray[np.uint8]:
    if item._canvas is not None:
        return item._canvas
    from soilnail.projection.cubemap import face_pixel_to_direction
    from soilnail.projection.geometry import lonlat_to_equirect_xy, xyz_to_lonlat

    h, w = cfg.coarse_height, cfg.coarse_width
    canvas = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(item.mask)
    if len(xs) == 0:
        item._canvas = canvas
        return canvas
    step = max(1, len(xs) // 800)
    for u, v in zip(xs[::step], ys[::step], strict=True):
        d = face_pixel_to_direction(
            item.face_id,
            float(u),
            float(v),
            face_size=face_size,
            fov_deg=fov_deg,
            margin_deg=margin_deg,
            yaw_offset_deg=yaw_offset_deg,
        )
        lon, lat = xyz_to_lonlat(d.reshape(1, 3))
        x, y = lonlat_to_equirect_xy(float(lon[0]), float(lat[0]), w, h)
        xi = int(np.mod(np.rint(x), w))
        yi = int(np.clip(np.rint(y), 0, h - 1))
        canvas[yi, xi] = 1
    item._canvas = canvas
    return canvas


def should_merge_pair(
    a: MergeItem,
    b: MergeItem,
    *,
    face_size: int,
    fov_deg: float,
    margin_deg: float,
    yaw_offset_deg: float,
    cfg: MergeConfig,
) -> bool:
    if a.face_id == b.face_id:
        inter = int(np.logical_and(a.mask, b.mask).sum())
        if inter == 0:
            return False
        union = int(np.logical_or(a.mask, b.mask).sum())
        return (inter / max(union, 1)) >= cfg.overlap_jaccard_min

    # Cross-face: require at least one border touch (seam candidate)
    if not (a.touches_border or b.touches_border):
        return False

    ca = _ensure_canvas(
        a,
        face_size=face_size,
        fov_deg=fov_deg,
        margin_deg=margin_deg,
        yaw_offset_deg=yaw_offset_deg,
        cfg=cfg,
    )
    cb = _ensure_canvas(
        b,
        face_size=face_size,
        fov_deg=fov_deg,
        margin_deg=margin_deg,
        yaw_offset_deg=yaw_offset_deg,
        cfg=cfg,
    )
    inter = int(np.logical_and(ca, cb).sum())
    union = int(np.logical_or(ca, cb).sum())
    jacc = inter / max(union, 1)
    if jacc >= cfg.overlap_jaccard_min:
        return True
    if a.touches_border and b.touches_border:
        d = math.degrees(angular_distance_rad(a.lon, a.lat, b.lon, b.lat))
        if d < cfg.secondary_centroid_deg:
            return True
    return False


def merge_detections(
    items: list[MergeItem],
    *,
    face_size: int,
    fov_deg: float,
    margin_deg: float,
    yaw_offset_deg: float = 0.0,
    cfg: MergeConfig | None = None,
) -> list[Any]:
    if cfg is None:
        cfg = MergeConfig()
    kept: list[MergeItem] = []
    for it in items:
        if any(
            should_merge_pair(
                it,
                k,
                face_size=face_size,
                fov_deg=fov_deg,
                margin_deg=margin_deg,
                yaw_offset_deg=yaw_offset_deg,
                cfg=cfg,
            )
            for k in kept
        ):
            continue
        kept.append(it)
    return [k.payload for k in kept]


def merge_detections_angular(
    items: list[tuple[float, float, object]],
    *,
    merge_deg: float,
) -> list[object]:
    kept: list[tuple[float, float, object]] = []
    for lon, lat, payload in items:
        if any(
            angular_distance_rad(lon, lat, klon, klat) < math.radians(merge_deg)
            for klon, klat, _ in kept
        ):
            continue
        kept.append((lon, lat, payload))
    return [p for _, _, p in kept]
