"""Merge rule: overlap-primary; distinct nails stay distinct."""

from __future__ import annotations

import math

import numpy as np

from soilnail.projection.merge import MergeConfig, MergeItem, merge_detections, should_merge_pair


def _disk(n: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def test_same_face_no_overlap_no_merge() -> None:
    n = 64
    a = MergeItem(0.0, 0.0, "F", False, _disk(n, 20, 20, 5), "a")
    b = MergeItem(0.05, 0.0, "F", False, _disk(n, 50, 50, 5), "b")
    cfg = MergeConfig()
    assert not should_merge_pair(
        a, b, face_size=n, fov_deg=90, margin_deg=0, yaw_offset_deg=0, cfg=cfg
    )
    out = merge_detections(
        [a, b], face_size=n, fov_deg=90, margin_deg=0, cfg=cfg
    )
    assert out == ["a", "b"]


def test_same_face_overlap_merges() -> None:
    n = 64
    a = MergeItem(0.0, 0.0, "F", False, _disk(n, 32, 32, 10), "a")
    b = MergeItem(0.01, 0.0, "F", False, _disk(n, 34, 34, 10), "b")
    out = merge_detections([a, b], face_size=n, fov_deg=90, margin_deg=0)
    assert out == ["a"]


def test_two_objects_few_degrees_apart_stay_two() -> None:
    """Centroids ~3° apart, no mask overlap, not both border → stay 2."""
    n = 128
    # Place two disks far apart in face pixels so no overlap
    a = MergeItem(0.0, 0.0, "F", False, _disk(n, 30, 30, 4), "a")
    # ~3° on sphere near equator ≈ small; use different faces without overlap
    b = MergeItem(math.radians(3.0), 0.0, "F", False, _disk(n, 100, 100, 4), "b")
    out = merge_detections([a, b], face_size=n, fov_deg=90, margin_deg=0)
    assert len(out) == 2


def test_centroid_secondary_requires_border() -> None:
    """Close centroids but interior (not border) must NOT merge without overlap."""
    n = 64
    a = MergeItem(0.0, 0.0, "F", False, _disk(n, 20, 20, 3), "a")
    b = MergeItem(math.radians(1.0), 0.0, "R", False, _disk(n, 20, 20, 3), "b")
    cfg = MergeConfig(overlap_jaccard_min=0.99, secondary_centroid_deg=5.0)
    assert not should_merge_pair(
        a, b, face_size=n, fov_deg=90, margin_deg=5, yaw_offset_deg=0, cfg=cfg
    )


def test_edge_spanning_fragments_merge_via_overlap() -> None:
    """One object cut across F|R: shared equirect paint in margin region → one."""
    n = 64
    a = MergeItem(
        math.radians(40.0), 0.0, "F", True, _disk(n, 32, n - 4, 6), "a"
    )
    b = MergeItem(
        math.radians(50.0), 0.0, "R", True, _disk(n, 32, 4, 6), "b"
    )
    # Shared margin-region paint (as after reprojection into 1° bins).
    shared = np.zeros((180, 360), dtype=np.uint8)
    shared[90, 40:55] = 1
    a._canvas = shared
    b._canvas = shared.copy()
    cfg = MergeConfig(overlap_jaccard_min=0.05, secondary_centroid_deg=0.5)
    out = merge_detections(
        [a, b], face_size=n, fov_deg=90, margin_deg=15, cfg=cfg
    )
    assert out == ["a"]
    # Without shared paint and with centroids > secondary thr → stay two.
    a2 = MergeItem(
        math.radians(40.0), 0.0, "F", True, _disk(n, 32, n - 4, 6), "a"
    )
    b2 = MergeItem(
        math.radians(50.0), 0.0, "R", True, _disk(n, 32, 4, 6), "b"
    )
    empty = np.zeros((180, 360), dtype=np.uint8)
    empty[90, 40] = 1
    empty2 = np.zeros((180, 360), dtype=np.uint8)
    empty2[90, 55] = 1
    a2._canvas = empty
    b2._canvas = empty2
    out2 = merge_detections(
        [a2, b2], face_size=n, fov_deg=90, margin_deg=15, cfg=cfg
    )
    assert out2 == ["a", "b"]


def test_same_face_never_merge_without_overlap() -> None:
    """Two objects in the same face with no pixel overlap stay two even if close."""
    n = 64
    a = MergeItem(0.0, 0.0, "F", True, _disk(n, 20, 20, 4), "a")
    b = MergeItem(math.radians(1.0), 0.0, "F", True, _disk(n, 20, 40, 4), "b")
    cfg = MergeConfig(secondary_centroid_deg=90.0)
    out = merge_detections([a, b], face_size=n, fov_deg=90, margin_deg=15, cfg=cfg)
    assert out == ["a", "b"]
