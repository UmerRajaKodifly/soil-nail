"""Reproject a face-space mask contour into equirect polygons (wrap-aware)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from soilnail.projection.cubemap import face_pixel_to_direction
from soilnail.projection.geometry import lonlat_to_equirect_xy, xyz_to_lonlat


def _contour_points(mask: NDArray[np.bool_], *, max_points: int = 64) -> NDArray[np.float64]:
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    # Subsample uniformly
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts  # columns = (u, v) = (x, y)


def face_mask_to_equirect_polygons(
    mask: NDArray[np.bool_],
    face_id: str,
    *,
    face_size: int,
    equirect_width: int,
    equirect_height: int,
    fov_deg: float = 90.0,
    margin_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> list[list[list[float]]]:
    """Return list of closed rings in equirect pixel coords.

    If the contour crosses the ±180° longitude seam, it is split into
    separate polygons (one per contiguous lon unwrap segment).
    """
    pts = _contour_points(mask)
    if len(pts) < 3:
        return []

    lons: list[float] = []
    lats: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for u, v in pts:
        d = face_pixel_to_direction(
            face_id,
            float(u),
            float(v),
            face_size=face_size,
            fov_deg=fov_deg,
            margin_deg=margin_deg,
            yaw_offset_deg=yaw_offset_deg,
        )
        lon, lat = xyz_to_lonlat(d.reshape(1, 3))
        lon_f, lat_f = float(lon[0]), float(lat[0])
        x, y = lonlat_to_equirect_xy(lon_f, lat_f, equirect_width, equirect_height)
        # Wrap x into [−0.5, W−0.5]
        x = float(np.mod(x + 0.5, equirect_width) - 0.5)
        lons.append(lon_f)
        lats.append(lat_f)
        xs.append(float(x))
        ys.append(float(np.clip(y, -0.5, equirect_height - 0.5)))

    # Detect wrap crossing: adjacent vertices with |Δx| > W/2
    w = float(equirect_width)
    rings: list[list[list[float]]] = []
    current: list[list[float]] = [[xs[0], ys[0]]]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if abs(dx) > w / 2.0:
            # Close current ring and start a new one
            if len(current) >= 3:
                if current[0] != current[-1]:
                    current.append(current[0])
                rings.append(current)
            current = [[xs[i], ys[i]]]
        else:
            current.append([xs[i], ys[i]])
    if len(current) >= 3:
        if current[0] != current[-1]:
            current.append(current[0])
        rings.append(current)

    # Also handle case where first/last edge wraps
    if len(rings) == 1 and abs(xs[0] - xs[-1]) > w / 2.0 and len(xs) >= 3:
        # Split at max lon discontinuity index already handled; if still one ring
        # with wrap between last→first, leave as-is only if area is sensible.
        pass

    return rings
