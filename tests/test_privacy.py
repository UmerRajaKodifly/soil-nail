"""Privacy redaction is display-only and deterministic."""

from __future__ import annotations

import numpy as np

from soilnail.privacy import PRIVACY_NOTE, redact_equirect_for_display, redact_nadir_face


def test_nadir_blur_only_d() -> None:
    face = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(redact_nadir_face(face, "F"), face)
    out = redact_nadir_face(face, "D")
    assert out.shape == face.shape
    assert not np.array_equal(out, face)


def test_equirect_bottom_blur() -> None:
    img = np.random.default_rng(1).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    out = redact_equirect_for_display(img, nadir_fraction=0.2)
    assert np.array_equal(out[:80], img[:80])
    assert not np.array_equal(out[80:], img[80:])


def test_privacy_note_mentions_display_only() -> None:
    assert "never altered" in PRIVACY_NOTE.lower() or "Never" in PRIVACY_NOTE
