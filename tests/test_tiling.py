"""Property-style tiling reconstruction tests."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from soilnail.tiling import make_tiles, merge_masks_union, nms_masks


@given(
    h=st.integers(64, 300),
    w=st.integers(64, 300),
    tile=st.integers(32, 128),
    offset=st.integers(0, 40),
)
@settings(max_examples=30, deadline=None)
def test_tiled_union_reconstructs_objects(h: int, w: int, tile: int, offset: int) -> None:
    img = np.zeros((h, w), dtype=np.uint8)
    # place two blobs
    img[10:20, 10:25] = 255
    cy, cx = min(h - 15, 40 + offset), min(w - 15, 50 + (offset % 17))
    img[cy : cy + 12, cx : cx + 12] = 255

    tiles = make_tiles(h, w, tile=tile, overlap=0.25)
    tile_masks = []
    for t in tiles:
        patch = img[t.y0 : t.y1, t.x0 : t.x1]
        tile_masks.append((t, patch))
    merged, _ = merge_masks_union((h, w), tile_masks)
    assert np.array_equal(merged > 0, img > 0)


def test_nms_removes_overlap() -> None:
    a = np.zeros((20, 20), dtype=bool)
    b = np.zeros((20, 20), dtype=bool)
    a[2:10, 2:10] = True
    b[4:12, 4:12] = True
    keep = nms_masks([a, b], [0.9, 0.5], iou_thresh=0.1)
    assert keep == [0]
