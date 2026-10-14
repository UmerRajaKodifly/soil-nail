"""Spherical ↔ Cartesian ↔ equirect pixel maths (docs/PROJECTION.md)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]


def equirect_xy_to_lonlat(
    x: ArrayF | float,
    y: ArrayF | float,
    width: int,
    height: int,
) -> tuple[ArrayF | float, ArrayF | float]:
    """Pixel-centre convention: lon in [-π,π], lat in [-π/2,π/2]."""
    lon = 2.0 * math.pi * (np.asarray(x, dtype=np.float64) + 0.5) / width - math.pi
    lat = math.pi / 2.0 - math.pi * (np.asarray(y, dtype=np.float64) + 0.5) / height
    if np.isscalar(x) and np.isscalar(y):
        return float(lon), float(lat)
    return lon, lat


def lonlat_to_equirect_xy(
    lon: ArrayF | float,
    lat: ArrayF | float,
    width: int,
    height: int,
) -> tuple[ArrayF | float, ArrayF | float]:
    """Inverse of equirect_xy_to_lonlat (lon may be outside [-π,π]; wrapped later)."""
    x = (np.asarray(lon, dtype=np.float64) / (2.0 * math.pi) + 0.5) * width - 0.5
    y = (-np.asarray(lat, dtype=np.float64) / math.pi + 0.5) * height - 0.5
    if np.isscalar(lon) and np.isscalar(lat):
        return float(x), float(y)
    return x, y


def lonlat_to_xyz(lon: ArrayF | float, lat: ArrayF | float) -> NDArray[np.float64]:
    """Unit direction; +X right, +Y up, +Z forward."""
    lon_a = np.asarray(lon, dtype=np.float64)
    lat_a = np.asarray(lat, dtype=np.float64)
    cl = np.cos(lat_a)
    xyz = np.stack(
        (cl * np.sin(lon_a), np.sin(lat_a), cl * np.cos(lon_a)),
        axis=-1,
    )
    return xyz


def xyz_to_lonlat(xyz: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of lonlat_to_xyz (matches py360convert xyz2uv)."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.hypot(x, z))
    return lon.astype(np.float64), lat.astype(np.float64)


def wrap_x(x: ArrayF, width: int) -> NDArray[np.float64]:
    """Wrap horizontal sample coords into (-0.5, width-0.5] then for map_coordinates."""
    return np.asarray(np.mod(x + 0.5, width) - 0.5, dtype=np.float64)


def clamp_y(y: ArrayF, height: int) -> NDArray[np.float64]:
    return np.asarray(np.clip(y, -0.5, height - 0.5), dtype=np.float64)


def apply_yaw(xyz: NDArray[np.floating], yaw_deg: float) -> NDArray[np.float64]:
    """Rotate directions about +Y by yaw_deg (degrees, CCW when viewed from above)."""
    if abs(yaw_deg) < 1e-12:
        return np.asarray(xyz, dtype=np.float64)
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return np.stack((c * x + s * z, y, -s * x + c * z), axis=-1)
