from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int


def make_tiles(h: int, w: int, tile: int = 1024, overlap: float = 0.25) -> list[Tile]:
    """EXTENSION: sliding-window tiles. Default OFF at call sites."""
    if tile <= 0 or (h <= tile and w <= tile):
        return [Tile(0, 0, h, w)]
    stride = max(1, int(tile * (1.0 - overlap)))
    tiles: list[Tile] = []
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y0 in ys:
        for x0 in xs:
            tiles.append(Tile(y0, x0, min(h, y0 + tile), min(w, x0 + tile)))
    return tiles


def merge_masks_union(shape_hw: tuple[int, int], tile_masks: list[tuple[Tile, np.ndarray]]) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    for tile, m in tile_masks:
        mh, mw = m.shape[:2]
        out[tile.y0 : tile.y0 + mh, tile.x0 : tile.x0 + mw] = np.maximum(
            out[tile.y0 : tile.y0 + mh, tile.x0 : tile.x0 + mw],
            (m > 0).astype(np.uint8) * 255,
        )
    return out
