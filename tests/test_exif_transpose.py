"""EXIF orientation must change effective size when transpose applies."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from soilnail.ingest.images import audit_image


def test_exif_orientation_6_swaps_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    # Wide image tagged as orientation 6 (rotate 90 CW) → effective tall
    img = Image.new("RGB", (40, 20), color=(10, 20, 30))
    exif = img.getexif()
    exif[274] = 6
    img.save(path, format="JPEG", exif=exif)

    aud = audit_image(path, max_pixels=1_000_000, apply_exif_transpose=True)
    assert aud.status == "ok"
    assert aud.orig_width == 40
    assert aud.orig_height == 20
    assert aud.effective_width == 20
    assert aud.effective_height == 40
    assert aud.exif_orientation == 6
    assert aud.transposed is True


def test_zero_byte_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    aud = audit_image(path, max_pixels=1_000_000, apply_exif_transpose=True)
    assert aud.status == "failed"
    assert aud.failure_type == "ZeroByteImage"
