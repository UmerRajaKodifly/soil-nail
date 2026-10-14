"""Camera roll estimate from cubemap side-face verticals (display diagnostics)."""
from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
import cv2

def estimate_face_roll_deg(face_rgb: NDArray[np.uint8]) -> float | None:
    """Roll deg in cv2.getRotationMatrix2D sense; None if too few vertical lines."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=max(20, face_rgb.shape[0] // 5), maxLineGap=20,
    )
    if lines is None:
        return None
    angs: list[float] = []
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in ln)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = abs(ang)
        if a > 90:
            a = 180 - a
        if 60 <= a <= 90:
            dev = ang - 90.0 if ang >= 0 else ang + 90.0
            angs.append(-dev)
    if len(angs) < 3:
        return None
    return float(np.median(angs))
