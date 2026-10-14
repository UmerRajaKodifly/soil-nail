"""Spherical geometry helpers for detections (angular size, merge)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from soilnail.projection.geometry import lonlat_to_xyz, xyz_to_lonlat


def angular_distance_rad(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    a = lonlat_to_xyz(lon1, lat1)
    b = lonlat_to_xyz(lon2, lat2)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def mask_centroid_lonlat(
    mask: NDArray[np.bool_],
    face_id: str,
    face_size: int,
    *,
    fov_deg: float = 90.0,
    margin_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> tuple[float, float]:
    from soilnail.projection.cubemap import face_pixel_to_direction

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    u, v = float(xs.mean()), float(ys.mean())
    d = face_pixel_to_direction(
        face_id,
        u,
        v,
        face_size=face_size,
        fov_deg=fov_deg,
        margin_deg=margin_deg,
        yaw_offset_deg=yaw_offset_deg,
    )
    lon, lat = xyz_to_lonlat(d.reshape(1, 3))
    return float(lon[0]), float(lat[0])


def approx_angular_bbox_deg(
    mask: NDArray[np.bool_],
    face_id: str,
    face_size: int,
    *,
    fov_deg: float = 90.0,
    margin_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Return (angular_width_deg, angular_height_deg, solid_angle_approx_sr)."""
    from soilnail.projection.cubemap import face_pixel_to_direction

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    # Sample bbox corners in face pixels → directions → angular spans
    corners = [
        (float(xs.min()), float(ys.min())),
        (float(xs.max()), float(ys.min())),
        (float(xs.min()), float(ys.max())),
        (float(xs.max()), float(ys.max())),
    ]
    dirs = [
        face_pixel_to_direction(
            face_id,
            u,
            v,
            face_size=face_size,
            fov_deg=fov_deg,
            margin_deg=margin_deg,
            yaw_offset_deg=yaw_offset_deg,
        )
        for u, v in corners
    ]

    # Pairwise max angular distance ≈ diagonal; width/height from edge pairs
    def ang(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))

    width = max(ang(dirs[0], dirs[1]), ang(dirs[2], dirs[3]))
    height = max(ang(dirs[0], dirs[2]), ang(dirs[1], dirs[3]))
    # Solid angle ≈ face pixel fraction * face solid angle
    # Face FOV solid angle for square tan-FOV ≈ 4 * asin(sin²(fov/2)) for 90° = π/2?
    # Simpler: steradian ≈ (width_rad * height_rad) for small patches
    solid = math.radians(width) * math.radians(height)
    return width, height, solid


def merge_detections_angular(
    items: Sequence[tuple[float, float, object]],
    *,
    merge_deg: float,
) -> list[object]:
    """Greedy keep-first merge by angular distance. items: (lon, lat, payload)."""
    kept: list[tuple[float, float, object]] = []
    for lon, lat, payload in items:
        if any(
            angular_distance_rad(lon, lat, klon, klat) < math.radians(merge_deg)
            for klon, klat, _ in kept
        ):
            continue
        kept.append((lon, lat, payload))
    return [p for _, _, p in kept]
