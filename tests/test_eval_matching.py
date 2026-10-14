"""Matching rule on hand-built equirect boxes."""

from __future__ import annotations

from soilnail.eval.matching import GtNail, box_iou, match_detections
from soilnail.schemas import Detection, RLE, SphericalGeom


def _det(poly: list[list[float]], instance_id: int = 0) -> Detection:
    return Detection(
        image_id="x",
        model="stub",
        config_id="c",
        instance_id=instance_id,
        mask_rle=RLE(counts=[0, 1], size=[1, 1]),
        bbox_xywh=[0, 0, 1, 1],
        area_px=1,
        area_fraction=1.0,
        raw_score=0.5,
        equirect_polygons_xy=[poly],
        spherical=SphericalGeom(
            lon_rad=0.0,
            lat_rad=0.0,
            angular_width_deg=1.0,
            angular_height_deg=1.0,
            solid_angle_sr=0.001,
        ),
    )


def test_box_iou_identical() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_match_overlap() -> None:
    det = _det([[0, 0], [10, 0], [10, 10], [0, 10]])
    gt = [GtNail(bbox_xywh=[1, 1, 8, 8])]
    pairs, ud, ug = match_detections([det], gt)
    assert pairs == [(0, 0)]
    assert ud == [] and ug == []


def test_no_match_far() -> None:
    det = _det([[0, 0], [5, 0], [5, 5], [0, 5]])
    gt = [GtNail(bbox_xywh=[100, 100, 10, 10])]
    pairs, ud, ug = match_detections([det], gt)
    assert pairs == []
    assert ud == [0] and ug == [0]
