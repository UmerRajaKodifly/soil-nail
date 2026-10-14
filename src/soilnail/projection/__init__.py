"""Equirectangular ↔ cubemap projection (see docs/PROJECTION.md)."""

from soilnail.projection.cubemap import (
    FACE_IDS,
    CubemapConfig,
    cubemap_to_equirect,
    direction_to_equirect_xy,
    equirect_to_cubemap,
    face_pixel_to_direction,
    make_cross_layout,
)
from soilnail.projection.errors import NotFullEquirectError, PartialPanoError
from soilnail.projection.geometry import (
    equirect_xy_to_lonlat,
    lonlat_to_equirect_xy,
    lonlat_to_xyz,
    xyz_to_lonlat,
)

__all__ = [
    "FACE_IDS",
    "CubemapConfig",
    "NotFullEquirectError",
    "PartialPanoError",
    "equirect_to_cubemap",
    "cubemap_to_equirect",
    "face_pixel_to_direction",
    "direction_to_equirect_xy",
    "make_cross_layout",
    "equirect_xy_to_lonlat",
    "lonlat_to_equirect_xy",
    "lonlat_to_xyz",
    "xyz_to_lonlat",
]
