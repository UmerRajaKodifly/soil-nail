"""End-to-end seam-straddle via production segment_cubemap_image + stub detector."""

from __future__ import annotations

import math

import numpy as np

from soilnail.models.base import PreparedRefs, SegmentOutput
from soilnail.projection.geometry import lonlat_to_xyz
from soilnail.runner import segment_cubemap_image
from soilnail.schemas import Detection


class YellowBlobStub:
    """Deterministic stub: yellow pixels → one mask per face (uint8)."""

    def prepare(self, refs: PreparedRefs) -> None:
        return None

    def segment(self, image_rgb: np.ndarray) -> SegmentOutput:
        yellow = (
            (image_rgb[..., 0] > 200)
            & (image_rgb[..., 1] > 200)
            & (image_rgb[..., 2] < 80)
        )
        if not yellow.any():
            return SegmentOutput(masks=[], scores=[], meta={"stub": True})
        m = (yellow.astype(np.uint8) * 255)
        return SegmentOutput(masks=[m], scores=[1.0], meta={"stub": True})

    def close(self) -> None:
        return None


def _paint_disk(img: np.ndarray, lon: float, lat: float, radius_deg: float) -> None:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    lon_g = 2.0 * math.pi * (xx + 0.5) / w - math.pi
    lat_g = math.pi / 2.0 - math.pi * (yy + 0.5) / h
    t = lonlat_to_xyz(lon, lat).reshape(1, 1, 3)
    g = lonlat_to_xyz(lon_g, lat_g)
    ang = np.arccos(np.clip((g * t).sum(-1), -1, 1))
    img[ang < math.radians(radius_deg)] = (255, 255, 0)


def _run(
    lon: float,
    lat: float,
    *,
    radius_deg: float,
    margin_deg: float,
    merge_deg: float,
) -> list[Detection]:
    h, w = 256, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_disk(img, lon, lat, radius_deg)
    res = segment_cubemap_image(
        YellowBlobStub(),
        img,
        image_id="synth",
        model="stub",
        config_id="e2e_seam",
        face_size=64,
        fov_deg=90.0,
        margin_deg=margin_deg,
        yaw_offset_deg=0.0,
        instance_mode=True,
        min_cc_area=1,
        nms_iou=0.5,
        area_warn_solid_angle_sr=1.0,
        merge_deg=merge_deg,
    )
    assert res.status == "succeeded", res.failure
    return res.detections


def test_e2e_one_face_edge() -> None:
    dets = _run(math.pi / 4, 0.0, radius_deg=8.0, margin_deg=5.0, merge_deg=10.0)
    assert len(dets) == 1
    assert dets[0].face_id is not None
    assert dets[0].spherical is not None
    assert dets[0].equirect_polygons_xy is not None
    assert abs(dets[0].spherical.lon_rad - math.pi / 4) < math.radians(10)


def test_e2e_wrap_seam() -> None:
    dets = _run(math.pi, 0.0, radius_deg=10.0, margin_deg=5.0, merge_deg=12.0)
    assert len(dets) == 1
    assert abs(abs(dets[0].spherical.lon_rad) - math.pi) < math.radians(20)  # type: ignore[union-attr]
    # Wrap-split may yield >1 ring
    assert dets[0].equirect_polygons_xy is not None
    assert len(dets[0].equirect_polygons_xy) >= 1


def test_e2e_corner_two_edges() -> None:
    dets = _run(math.pi / 4, math.pi / 5, radius_deg=12.0, margin_deg=5.0, merge_deg=15.0)
    assert len(dets) == 1
    assert dets[0].spherical is not None
