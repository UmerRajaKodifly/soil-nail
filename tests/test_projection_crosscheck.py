"""Independent numerical cross-check against py360convert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py360convert
import pytest
from PIL import Image
from py360convert.utils import xyzcube

from soilnail.projection.cubemap import CubemapConfig, _face_xyz_grid, equirect_to_cubemap

# Documented tolerance: float32 direction grids agree to ~1e-7; uint8 faces
# after independent bilinear sampling agree to MAE < 0.5 / PSNR ≥ 45.
# A handful of U/D corner pixels near the poles can spike (singular sampling);
# we bound the fraction of pixels with |Δ|>3 rather than requiring max≤3 everywhere.
DIR_TOL = 1e-6
FACE_MAE_TOL = 0.5
FACE_FRAC_GT3_TOL = 1e-3  # ≤0.1% of pixels
FACE_PSNR_MIN = 45.0


def test_direction_grid_vs_py360convert() -> None:
    n = 128
    ref = xyzcube(n)
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    for i, fid in enumerate("FRBLUD"):
        ours = _face_xyz_grid(fid, n, 90.0)
        r = ref[:, i * n : (i + 1) * n, :]
        assert float(np.abs(ours - r).max()) < DIR_TOL, fid


@pytest.mark.parametrize("face_w", [64, 256])
def test_e2c_vs_py360convert_synthetic(face_w: int) -> None:
    # Smooth field — white noise MAE is dominated by bilinear kernel differences
    # on discontinuous pixels, not geometry error (directions already match 1e-7).
    h, w = 128, 256
    yy, xx = np.mgrid[0:h, 0:w]
    # All channels continuous across the ±180° wrap (no ramp that jumps).
    rgb = np.stack(
        [
            (127 + 127 * np.sin(2 * np.pi * xx / w)).astype(np.uint8),
            (127 + 127 * np.cos(2 * np.pi * xx / w)).astype(np.uint8),
            (127 + 127 * np.sin(np.pi * yy / max(h - 1, 1))).astype(np.uint8),
        ],
        axis=-1,
    )
    ours = equirect_to_cubemap(rgb, CubemapConfig(face_size=face_w, interpolation="bilinear"))
    ref = py360convert.e2c(rgb, face_w=face_w, mode="bilinear", cube_format="dict")
    for fid in "FRBLUD":
        a = ours[fid].astype(np.float64)
        b = ref[fid].astype(np.float64)
        mae = float(np.abs(a - b).mean())
        mse = float(((a - b) ** 2).mean())
        psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-12))
        assert mae < FACE_MAE_TOL, (fid, mae)
        frac_gt3 = float((np.abs(a - b) > 3.0).mean())
        assert frac_gt3 <= FACE_FRAC_GT3_TOL, (fid, frac_gt3)
        assert psnr >= FACE_PSNR_MIN, (fid, psnr)


def test_e2c_vs_py360convert_real_frame() -> None:
    root = Path(__file__).resolve().parents[1]
    # Prefer extracted walk frame if present
    candidates = list((root / "data/site_walk_extracted").rglob("frame_0003.jpg"))
    if not candidates:
        pytest.skip("site-walk extract not present")
    rgb = np.asarray(Image.open(candidates[0]).convert("RGB"))
    assert rgb.shape[1] == 2 * rgb.shape[0]
    ours = equirect_to_cubemap(rgb, CubemapConfig(face_size=256, interpolation="bilinear"))
    ref = py360convert.e2c(rgb, face_w=256, mode="bilinear", cube_format="dict")
    for fid in "FRBLUD":
        a = ours[fid].astype(np.float64)
        b = ref[fid].astype(np.float64)
        mae = float(np.abs(a - b).mean())
        mse = float(((a - b) ** 2).mean())
        psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-12))
        assert mae < FACE_MAE_TOL, (fid, mae)
        assert psnr >= FACE_PSNR_MIN, (fid, psnr)
