"""Probe for burned-in date stamps in bottom-right corner (measured, not assumed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class StampProbeResult:
    sampled: int
    likely_stamp_count: int
    mean_corner_saturation: float
    mean_full_saturation: float
    recommendation: str
    evidence: str


def _saturation(rgb: NDArray[np.uint8]) -> float:
    """Mean HSV-S of an RGB uint8 array."""
    f = rgb.astype(np.float32) / 255.0
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(sat.mean())


def probe_date_stamp(
    image_paths: list[Path],
    *,
    sample_size: int,
    corner_frac_w: float,
    corner_frac_h: float,
    sat_delta_min: float,
    sat_abs_min: float,
    likely_fraction_threshold: float,
    seed: int,
) -> StampProbeResult:
    """Heuristic: reference photos had a saturated orange stamp bottom-right.

    We compare corner saturation vs full-image saturation. This is a *probe*,
    not a crop recommendation unless evidence is strong.
    """
    if not image_paths:
        return StampProbeResult(
            sampled=0,
            likely_stamp_count=0,
            mean_corner_saturation=0.0,
            mean_full_saturation=0.0,
            recommendation="no_crop",
            evidence="no images to probe",
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(image_paths))
    sample = [image_paths[int(i)] for i in order[: min(sample_size, len(image_paths))]]

    corner_sats: list[float] = []
    full_sats: list[float] = []
    likely = 0
    for p in sample:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        x0 = int(w * (1.0 - corner_frac_w))
        y0 = int(h * (1.0 - corner_frac_h))
        corner = rgb[y0:h, x0:w]
        cs = _saturation(corner)
        fs = _saturation(rgb)
        corner_sats.append(cs)
        full_sats.append(fs)
        # Orange stamp tends to be much more saturated than the scene mean
        if cs > fs + sat_delta_min and cs > sat_abs_min:
            likely += 1

    mean_c = float(np.mean(corner_sats)) if corner_sats else 0.0
    mean_f = float(np.mean(full_sats)) if full_sats else 0.0
    frac = likely / len(sample)
    if frac >= likely_fraction_threshold:
        rec = "measure_box_before_crop"
        evidence = (
            f"{likely}/{len(sample)} samples show elevated bottom-right saturation "
            f"(mean corner S={mean_c:.3f} vs full S={mean_f:.3f}). "
            "Do not apply reference-photo crop fractions without measuring the box."
        )
    else:
        rec = "no_crop"
        evidence = (
            f"{likely}/{len(sample)} samples look stamp-like "
            f"(mean corner S={mean_c:.3f} vs full S={mean_f:.3f}). "
            "No burned-in date stamp crop recommended for site-walk."
        )
    return StampProbeResult(
        sampled=len(sample),
        likely_stamp_count=likely,
        mean_corner_saturation=mean_c,
        mean_full_saturation=mean_f,
        recommendation=rec,
        evidence=evidence,
    )
