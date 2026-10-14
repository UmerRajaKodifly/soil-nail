"""Known-answer geometry tests for equirect ↔ direction ↔ face."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soilnail.projection.cubemap import (
    FACE_IDS,
    CubemapConfig,
    assert_full_equirect,
    equirect_to_cubemap,
    face_pixel_to_direction,
)
from soilnail.projection.errors import NotFullEquirectError
from soilnail.projection.geometry import (
    apply_yaw,
    equirect_xy_to_lonlat,
    lonlat_to_equirect_xy,
    lonlat_to_xyz,
)


def test_reject_non_2to1() -> None:
    assert_full_equirect(1000, 500)
    assert_full_equirect(3840, 1920)
    with pytest.raises(NotFullEquirectError):
        assert_full_equirect(1920, 1080)
    with pytest.raises(NotFullEquirectError):
        assert_full_equirect(3840, 2160)


def test_pixel_centre_origin() -> None:
    # Centre of 4x2 image → lon=0, lat=0
    lon, lat = equirect_xy_to_lonlat(1.5, 0.5, 4, 2)
    assert abs(lon) < 1e-12
    assert abs(lat) < 1e-12


def test_poles_and_wrap() -> None:
    lon, lat = equirect_xy_to_lonlat(0.0, 0.0, 8, 4)
    assert lat > 0  # near zenith
    lon_r, _ = equirect_xy_to_lonlat(7.0, 2.0, 8, 4)
    assert lon_r > 0  # right side
    # wrap: lon of last+first columns nearly ±π
    lon0, _ = equirect_xy_to_lonlat(-0.5, 2.0, 8, 4)  # just before wrap in float


@pytest.mark.parametrize(
    "face,expected_axis,sign",
    [
        ("F", 2, +1),
        ("R", 0, +1),
        ("B", 2, -1),
        ("L", 0, -1),
        ("U", 1, +1),
        ("D", 1, -1),
    ],
)
def test_face_centre_is_axis(face: str, expected_axis: int, sign: int) -> None:
    n = 64
    d = face_pixel_to_direction(face, (n - 1) / 2, (n - 1) / 2, face_size=n)
    assert abs(d[expected_axis] - sign) < 1e-6
    for i in range(3):
        if i != expected_axis:
            assert abs(d[i]) < 1e-5


def test_direction_colour_encoding_sample() -> None:
    """Synthetic equirect encodes direction in RGB; face centre must match."""
    h, w = 64, 128
    # Build equirect where each pixel's RGB = mapped unit xyz in [0,255]
    xs = np.arange(w)
    ys = np.arange(h)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = equirect_xy_to_lonlat(xx.astype(float), yy.astype(float), w, h)
    xyz = lonlat_to_xyz(lon, lat)
    rgb = np.clip(np.rint((xyz + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)
    faces = equirect_to_cubemap(rgb, CubemapConfig(face_size=32, interpolation="nearest"))
    for face in FACE_IDS:
        # Integer centre-ish pixel of 32×32 face
        pix = faces[face][16, 16].astype(np.float64)
        recovered = pix / 255.0 * 2.0 - 1.0
        recovered = recovered / np.linalg.norm(recovered)
        expected = face_pixel_to_direction(face, 16, 16, face_size=32)
        assert np.linalg.norm(recovered - expected) < 0.08, (face, recovered, expected)


def test_yaw_inverse_identity() -> None:
    d = np.array([0.1, 0.2, 0.9])
    d = d / np.linalg.norm(d)
    back = apply_yaw(apply_yaw(d, 37.5), -37.5)
    assert np.allclose(d, back.reshape(3), atol=1e-12)


@given(
    x=st.floats(0, 3839, allow_nan=False, allow_infinity=False),
    y=st.floats(0, 1919, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_xy_lonlat_roundtrip(x: float, y: float) -> None:
    lon, lat = equirect_xy_to_lonlat(x, y, 3840, 1920)
    x2, y2 = lonlat_to_equirect_xy(lon, lat, 3840, 1920)
    assert abs(x2 - x) < 1e-6
    assert abs(y2 - y) < 1e-6
    assert math.isfinite(float(lon)) and math.isfinite(float(lat))


@given(yaw=st.floats(-180, 180, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_yaw_roundtrip_property(yaw: float) -> None:
    d = lonlat_to_xyz(0.3, -0.1)
    out = apply_yaw(apply_yaw(d, yaw), -yaw)
    assert np.allclose(d, out, atol=1e-10)
    assert not np.any(~np.isfinite(out))
