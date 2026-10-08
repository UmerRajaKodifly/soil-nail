"""Sliding-window tiling and mask merge (EXTENSION — not in PerSAM/SegGPT papers)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Tile:
    tile_id: str
    y0: int
    x0: int
    y1: int
    x1: int


def make_tiles(h: int, w: int, tile: int, overlap: float) -> list[Tile]:
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if h <= tile and w <= tile:
        return [Tile("0_0", 0, 0, h, w)]
    stride = max(1, int(tile * (1.0 - overlap)))
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    tiles: list[Tile] = []
    for y0 in ys:
        for x0 in xs:
            tiles.append(
                Tile(
                    tile_id=f"{y0}_{x0}",
                    y0=y0,
                    x0=x0,
                    y1=min(h, y0 + tile),
                    x1=min(w, x0 + tile),
                )
            )
    return tiles


def merge_masks_union(
    shape_hw: tuple[int, int],
    tile_masks: list[tuple[Tile, NDArray[np.uint8]]],
) -> tuple[NDArray[np.uint8], bool]:
    """Union merge. Returns (mask, merged_across_tiles flag)."""
    out = np.zeros(shape_hw, dtype=np.uint8)
    coverage = np.zeros(shape_hw, dtype=np.uint16)
    for tile, m in tile_masks:
        mh, mw = m.shape[:2]
        region = out[tile.y0 : tile.y0 + mh, tile.x0 : tile.x0 + mw]
        cov = coverage[tile.y0 : tile.y0 + mh, tile.x0 : tile.x0 + mw]
        bin_m = (m > 0).astype(np.uint8)
        region[:] = np.maximum(region, bin_m * 255)
        cov[:] = cov + bin_m
    merged = bool((coverage > 1).any() and (out > 0).any())
    return out, merged


def mask_iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def nms_masks(
    masks: list[NDArray[np.bool_]],
    scores: list[float],
    iou_thresh: float,
) -> list[int]:
    """Greedy mask-IoU NMS; returns keep indices (our multi-instance extension)."""
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            if mask_iou(masks[i], masks[j]) >= iou_thresh:
                suppressed.add(j)
    return keep
