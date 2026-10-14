"""Seam-straddle: synthetic marker across face edge / wrap counted once with stub detector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from soilnail.projection.cubemap import CubemapConfig, equirect_to_cubemap
from soilnail.projection.geometry import lonlat_to_xyz, xyz_to_lonlat


@dataclass
class StubDet:
    face_id: str
    lon: float
    lat: float
    score: float = 1.0


def _paint_disk_equirect(img: np.ndarray, lon: float, lat: float, radius_deg: float) -> None:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    lon_g = 2.0 * math.pi * (xx + 0.5) / w - math.pi
    lat_g = math.pi / 2.0 - math.pi * (yy + 0.5) / h
    # Angular distance via dot product
    t = lonlat_to_xyz(lon, lat).reshape(1, 1, 3)
    g = lonlat_to_xyz(lon_g, lat_g)
    cosang = np.clip((g * t).sum(-1), -1, 1)
    ang = np.arccos(cosang)
    mask = ang < math.radians(radius_deg)
    img[mask] = (255, 255, 0)


def stub_detect_faces(
    faces: dict[str, np.ndarray],
    *,
    face_size: int,
    thresh: int = 200,
) -> list[StubDet]:
    """Deterministic stub: centroid of yellow pixels → spherical direction."""
    dets: list[StubDet] = []
    for fid, face in faces.items():
        yellow = (face[..., 0] > thresh) & (face[..., 1] > thresh) & (face[..., 2] < 80)
        if not yellow.any():
            continue
        ys, xs = np.where(yellow)
        u, v = float(xs.mean()), float(ys.mean())
        # direction via face centre mapping
        from soilnail.projection.cubemap import face_pixel_to_direction

        d = face_pixel_to_direction(fid, u, v, face_size=face_size)
        lon, lat = xyz_to_lonlat(d.reshape(1, 3))
        dets.append(StubDet(fid, float(lon[0]), float(lat[0])))
    return dets


def merge_on_sphere(dets: list[StubDet], merge_deg: float = 5.0) -> list[StubDet]:
    """Greedy angular merge — keep first, drop neighbours within merge_deg."""
    kept: list[StubDet] = []
    for d in dets:
        d_xyz = lonlat_to_xyz(d.lon, d.lat)
        if any(
            float(np.degrees(np.arccos(np.clip(np.dot(d_xyz, lonlat_to_xyz(k.lon, k.lat)), -1, 1))))
            < merge_deg
            for k in kept
        ):
            continue
        kept.append(d)
    return kept


def test_seam_straddle_face_edge_counted_once() -> None:
    """Marker centred on F/R edge (lon=+45°) must merge to one detection."""
    h, w = 128, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_disk_equirect(img, lon=math.pi / 4, lat=0.0, radius_deg=8.0)
    cfg = CubemapConfig(face_size=64, margin_deg=0.0)
    faces = equirect_to_cubemap(img, cfg)
    raw = stub_detect_faces(faces, face_size=64)
    # Without margin, marker may appear on F and/or R
    assert len(raw) >= 1
    merged = merge_on_sphere(raw, merge_deg=10.0)
    assert len(merged) == 1, [(d.face_id, d.lon, d.lat) for d in raw]
    assert abs(merged[0].lon - math.pi / 4) < math.radians(8)
    assert abs(merged[0].lat) < math.radians(8)


def test_seam_straddle_wrap_counted_once() -> None:
    """Marker straddling ±180° wrap."""
    h, w = 128, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_disk_equirect(img, lon=math.pi, lat=0.0, radius_deg=10.0)
    cfg = CubemapConfig(face_size=64)
    faces = equirect_to_cubemap(img, cfg)
    raw = stub_detect_faces(faces, face_size=64)
    merged = merge_on_sphere(raw, merge_deg=12.0)
    assert len(merged) == 1, [(d.face_id, math.degrees(d.lon), math.degrees(d.lat)) for d in raw]
    # lon near ±π
    assert abs(abs(merged[0].lon) - math.pi) < math.radians(15)


def test_corner_spanning_three_faces_merges() -> None:
    """Near F/R/U corner."""
    h, w = 128, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # direction toward upper-right of front
    lon, lat = math.pi / 4, math.pi / 5
    _paint_disk_equirect(img, lon=lon, lat=lat, radius_deg=12.0)
    faces = equirect_to_cubemap(img, CubemapConfig(face_size=64, margin_deg=5.0))
    raw = stub_detect_faces(faces, face_size=64)
    merged = merge_on_sphere(raw, merge_deg=15.0)
    assert len(merged) == 1, [(d.face_id, d.lon, d.lat) for d in raw]
