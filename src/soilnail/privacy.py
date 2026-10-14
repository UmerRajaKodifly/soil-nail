"""Display-time privacy helpers for report/UI exports.

Never applied to model inputs. Nadir (D) faces and equirect bottom band may show
the operator — blur for human-facing outputs only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def blur_rgb(rgb: NDArray[np.uint8], *, ksize: int = 51) -> NDArray[np.uint8]:
    import cv2

    k = ksize if ksize % 2 == 1 else ksize + 1
    return np.asarray(cv2.GaussianBlur(rgb, (k, k), 0), dtype=np.uint8)


def redact_nadir_face(face_rgb: NDArray[np.uint8], face_id: str) -> NDArray[np.uint8]:
    """Blur the Down face for display; other faces unchanged."""
    if face_id != "D":
        return face_rgb
    return blur_rgb(face_rgb)


def redact_equirect_for_display(
    equirect: NDArray[np.uint8],
    *,
    nadir_fraction: float = 0.18,
) -> NDArray[np.uint8]:
    """Blur the bottom nadir band of an equirect for galleries/PDF only."""
    out = equirect.copy()
    h = out.shape[0]
    y0 = int(h * (1.0 - nadir_fraction))
    out[y0:] = blur_rgb(out[y0:])
    return out


PRIVACY_NOTE = (
    "Privacy: site-walk 360° nadir faces often show the camera operator. "
    "Report figures, UI galleries and PDF exports apply display-time Gaussian "
    "blur to the Down (D) cubemap face and/or the bottom ~18% of the equirect. "
    "Model inputs and stored detections are never altered."
)
