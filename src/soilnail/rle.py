"""COCO-style uncompressed RLE encode/decode (row-major, Fortran order like COCO)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from soilnail.schemas import RLE


def encode_rle(mask: NDArray[np.bool_]) -> RLE:
    """Encode binary mask to uncompressed RLE (COCO column-major / Fortran order)."""
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    h, w = mask.shape
    pixels = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    counts: list[int] = []
    if pixels.size == 0:
        return RLE(counts=[], size=[h, w])
    prev = pixels[0]
    # COCO RLE starts with a run of zeros even if first pixel is 1
    if prev == 1:
        counts.append(0)
    run = 1
    for val in pixels[1:]:
        if val == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = val
    counts.append(run)
    return RLE(counts=counts, size=[int(h), int(w)])


def decode_rle(rle: RLE) -> NDArray[np.bool_]:
    h, w = rle.size
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for c in rle.counts:
        if c:
            flat[idx : idx + c] = val
            idx += c
        val = 1 - val
    if idx != total:
        raise ValueError(f"RLE length {idx} != h*w {total}")
    return flat.reshape((h, w), order="F").astype(bool)


def mask_bbox_xywh(mask: NDArray[np.bool_]) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = float(xs.min()), float(xs.max()) + 1.0
    y0, y1 = float(ys.min()), float(ys.max()) + 1.0
    return [x0, y0, x1 - x0, y1 - y0]
