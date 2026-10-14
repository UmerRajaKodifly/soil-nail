"""Independent cubemap reference (different code path from soilnail.projection).

Used only in tests for cross-checks. Implements e2c via explicit ray casting
and numpy broadcasting — no shared helpers with the production module.
"""

from __future__ import annotations

import math

import numpy as np


def _sample_equirect(img: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = (lon / (2 * math.pi) + 0.5) * w - 0.5
    y = (-lat / math.pi + 0.5) * h - 0.5
    x = np.mod(x, w)
    y = np.clip(y, 0, h - 1)
    # nearest for independence from scipy map_coordinates
    xi = np.clip(np.rint(x).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
    return img[yi, xi]


def independent_e2c(img: np.ndarray, face_w: int) -> dict[str, np.ndarray]:
    """FRBLUD faces via cube rays; nearest sampling (intentionally different)."""
    # Cube coords in [-1,1] then normalize — different from production's [-0.5,0.5] plane
    a = np.linspace(-1, 1, face_w)
    u, v = np.meshgrid(a, -a)  # v: top = +1 world-up for side faces
    faces: dict[str, np.ndarray] = {}
    specs = {
        "F": (u, v, np.ones_like(u)),
        "R": (np.ones_like(u), v, -u),
        "B": (-u, v, -np.ones_like(u)),
        "L": (-np.ones_like(u), v, u),
        "U": (u, np.ones_like(u), -v),
        "D": (u, -np.ones_like(u), v),
    }
    for name, (x, y, z) in specs.items():
        nrm = np.sqrt(x * x + y * y + z * z)
        x, y, z = x / nrm, y / nrm, z / nrm
        lon = np.arctan2(x, z)
        lat = np.arctan2(y, np.hypot(x, z))
        faces[name] = _sample_equirect(img, lon, lat)
    return faces
