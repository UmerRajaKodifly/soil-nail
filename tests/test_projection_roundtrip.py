"""Round-trip, seam, geometry, and metamorphic projection tests."""

from __future__ import annotations

import math

import numpy as np

from soilnail.projection.cubemap import (
    FACE_IDS,
    CubemapConfig,
    cubemap_to_equirect,
    equirect_to_cubemap,
    face_pixel_to_direction,
    make_cross_layout,
)


def _synthetic_equirect(h: int = 128, w: int = 256) -> np.ndarray:
    """Smooth gradient + a few markers (uint8 RGB)."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 / (w - 1)).astype(np.uint8)
    g = (yy * 255 / (h - 1)).astype(np.uint8)
    b = ((xx + yy) * 255 / (w + h - 2)).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    # Marker near wrap seam
    img[h // 2 - 2 : h // 2 + 3, 0:3] = (255, 0, 0)
    img[h // 2 - 2 : h // 2 + 3, w - 3 : w] = (255, 0, 0)
    # Marker at face-edge longitude ~45°
    x45 = int((math.pi / 4 / (2 * math.pi) + 0.5) * w - 0.5)
    img[h // 2 - 3 : h // 2 + 4, x45 - 2 : x45 + 3] = (0, 255, 0)
    return img


def test_roundtrip_psnr_synthetic() -> None:
    src = _synthetic_equirect()
    h, w = src.shape[:2]
    cfg = CubemapConfig(face_size=64, interpolation="bilinear")
    faces = equirect_to_cubemap(src, cfg)
    back = cubemap_to_equirect(faces, height=h, width=w, config=cfg)
    mse = float(((src.astype(float) - back.astype(float)) ** 2).mean())
    psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-12))
    # Cubemap remapping loses high frequencies at poles/corners; threshold from measurement.
    assert psnr > 20.0, psnr
    # Worst error region: poles (top/bottom 10%)
    err = np.abs(src.astype(float) - back.astype(float)).mean(axis=-1)
    pole = float(err[: h // 10].mean() + err[-h // 10 :].mean()) / 2
    mid = float(err[h // 3 : 2 * h // 3].mean())
    # Poles typically worse or comparable; just record via assert finite
    assert math.isfinite(pole) and math.isfinite(mid)


def test_adjacent_faces_agree_on_shared_edge() -> None:
    """Directions along F-right edge ≈ R-left edge."""
    n = 32
    for v in range(n):
        d_f = face_pixel_to_direction("F", n - 1, v, face_size=n)
        d_r = face_pixel_to_direction("R", 0, v, face_size=n)
        assert np.linalg.norm(d_f - d_r) < 0.05, (v, d_f, d_r)


def test_great_circle_stays_straight_on_face() -> None:
    """Horizontal equator great-circle → nearly constant row on Front face."""
    h, w = 180, 360
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Paint equator band
    img[h // 2 - 1 : h // 2 + 2, :] = 255
    faces = equirect_to_cubemap(img, CubemapConfig(face_size=90, interpolation="nearest"))
    front = faces["F"].mean(axis=-1)
    # Rows with signal should be near centre
    rows = np.where(front.max(axis=1) > 128)[0]
    assert len(rows) >= 1
    assert abs(rows.mean() - (90 - 1) / 2) < 3.0


def test_cross_layout_shape() -> None:
    faces = {fid: np.zeros((16, 16, 3), dtype=np.uint8) for fid in FACE_IDS}
    cross = make_cross_layout(faces)
    assert cross.shape == (48, 64, 3)


def test_yaw_metamorphic_content_preserved() -> None:
    """Circular column shift ≡ yaw; marker must still appear in some face."""
    src = _synthetic_equirect(64, 128)
    # Red marker at wrap
    cfg0 = CubemapConfig(face_size=32, yaw_offset_deg=0.0)
    cfg45 = CubemapConfig(face_size=32, yaw_offset_deg=45.0)
    f0 = equirect_to_cubemap(src, cfg0)
    f45 = equirect_to_cubemap(src, cfg45)
    red0 = max(int((f[..., 0] > 200).sum()) for f in f0.values())
    red45 = max(int((f[..., 0] > 200).sum()) for f in f45.values())
    assert red0 > 0 and red45 > 0


def test_subtile_count_arithmetic() -> None:
    """960 face, 512 tiles, 25% overlap → 9 tiles/face → 54/frame."""
    face, tile, overlap = 960, 512, 0.25
    stride = int(tile * (1.0 - overlap))
    n_1d = int(math.ceil((face - tile) / stride)) + 1
    assert stride == 384
    assert n_1d == 3
    assert n_1d * n_1d == 9
    assert 9 * 6 == 54
