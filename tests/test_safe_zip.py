"""Zip-slip and zip-bomb guards."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from soilnail.ingest.models import ZipLimits
from soilnail.ingest.safe_zip import UnsafeZipError, safe_extract, validate_zip

LIMITS = ZipLimits(
    max_uncompressed_bytes=10_000_000,
    max_compression_ratio=200.0,
    max_members=100,
    forbid_symlinks=True,
    forbid_absolute_paths=True,
    source="test",
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_zip_slip_rejected(tmp_path: Path) -> None:
    zpath = tmp_path / "evil.zip"
    # Craft a member with .. via ZipInfo
    with zipfile.ZipFile(zpath, "w") as zf:
        info = zipfile.ZipInfo("../escape.txt")
        zf.writestr(info, b"nope")
    with pytest.raises(UnsafeZipError, match="traversal"):
        validate_zip(zpath, LIMITS)


def test_absolute_path_rejected(tmp_path: Path) -> None:
    zpath = tmp_path / "abs.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        info = zipfile.ZipInfo("/tmp/abs.txt")
        zf.writestr(info, b"nope")
    with pytest.raises(UnsafeZipError, match="absolute"):
        validate_zip(zpath, LIMITS)


def test_safe_extract_ok(tmp_path: Path) -> None:
    zpath = tmp_path / "ok.zip"
    _write_zip(zpath, {"a/b.jpg": b"\xff\xd8\xfffake"})
    dest = tmp_path / "out"
    members = safe_extract(zpath, dest, LIMITS)
    assert len(members) == 1
    assert members[0].dest_path.is_file()
    assert members[0].sha256


def test_bomb_ratio_rejected(tmp_path: Path) -> None:
    zpath = tmp_path / "bomb.zip"
    # Highly compressible zeros
    payload = b"\x00" * 500_000
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", payload)
    tight = LIMITS.model_copy(update={"max_compression_ratio": 5.0})
    with pytest.raises(UnsafeZipError, match="ratio"):
        validate_zip(zpath, tight)
