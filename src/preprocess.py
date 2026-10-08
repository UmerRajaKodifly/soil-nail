from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def crop_bottom_right(
    image: np.ndarray,
    frac_w: float = 0.22,
    frac_h: float = 0.08,
    fill: int | tuple = 0,
) -> np.ndarray:
    """Blank out the bottom-right date-stamp region (opt-in extension)."""
    out = image.copy()
    h, w = out.shape[:2]
    x0 = int(w * (1.0 - frac_w))
    y0 = int(h * (1.0 - frac_h))
    out[y0:h, x0:w] = fill
    return out


def mask_bbox(mask: np.ndarray, margin: float = 0.15) -> CropBox | None:
    """Tight axis-aligned box around foreground, expanded by `margin` of box size."""
    m = mask > 0
    if m.ndim == 3:
        m = m.any(axis=-1)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    h, w = m.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = int(round(bw * margin))
    pad_y = int(round(bh * margin))
    return CropBox(
        x0=max(0, x0 - pad_x),
        y0=max(0, y0 - pad_y),
        x1=min(w, x1 + pad_x + 1),
        y1=min(h, y1 + pad_y + 1),
    )


def apply_crop(image: np.ndarray, box: CropBox) -> np.ndarray:
    return image[box.y0 : box.y1, box.x0 : box.x1].copy()


def crop_prompt_pair(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, CropBox | None]:
    """
    EXTENSION (not in original PerSAM/SegGPT): crop prompt image+mask tightly
    around the reference mask with margin.
    """
    box = mask_bbox(mask, margin=margin)
    if box is None:
        return image_rgb, mask, None
    return apply_crop(image_rgb, box), apply_crop(mask, box), box


def load_binary_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8) * 255


def mask_area_pct(mask: np.ndarray) -> float:
    m = mask > 0
    if m.ndim == 3:
        m = m.any(axis=-1)
    return float(100.0 * m.mean())


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    out = image_rgb.copy()
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.any(axis=-1)
    tint = np.zeros_like(out)
    tint[m] = color
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    return out


def to_persam_rgb_mask(mask_gray: np.ndarray) -> np.ndarray:
    """Convert 0/255 mask to PerSAM-style RGB (R=128 on FG after BGR2RGB convention)."""
    rgb = np.zeros((*mask_gray.shape[:2], 3), dtype=np.uint8)
    rgb[mask_gray > 0] = (128, 0, 0)
    return rgb
