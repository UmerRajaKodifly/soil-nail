"""Zip extraction with zip-slip / bomb / symlink guards."""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

from soilnail.hashutil import sha256_file
from soilnail.ingest.models import ZipLimits

log = logging.getLogger(__name__)


class UnsafeZipError(ValueError):
    """Raised when an archive violates configured safety limits."""


@dataclass(frozen=True)
class ExtractedMember:
    archive_path: str
    dest_path: Path
    file_size: int
    compress_size: int
    sha256: str


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    # Unix symlink: external_attr high bytes create mode includes 0o120000
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def validate_zip(path: Path, limits: ZipLimits) -> list[zipfile.ZipInfo]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > limits.max_members:
            raise UnsafeZipError(
                f"{path.name}: member count {len(infos)} > max_members {limits.max_members}"
            )
        total_uncompressed = 0
        total_compressed = 0
        for info in infos:
            name = info.filename
            if limits.forbid_absolute_paths and (
                name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":")
            ):
                raise UnsafeZipError(f"{path.name}: absolute path member {name!r}")
            # zip-slip: normalize and reject ..
            parts = Path(name).parts
            if ".." in parts:
                raise UnsafeZipError(f"{path.name}: path traversal member {name!r}")
            if limits.forbid_symlinks and _is_symlink_member(info):
                raise UnsafeZipError(f"{path.name}: symlink member {name!r}")
            if info.is_dir():
                continue
            total_uncompressed += int(info.file_size)
            total_compressed += max(1, int(info.compress_size))
        if total_uncompressed > limits.max_uncompressed_bytes:
            raise UnsafeZipError(
                f"{path.name}: uncompressed {total_uncompressed} > "
                f"max_uncompressed_bytes {limits.max_uncompressed_bytes}"
            )
        ratio = total_uncompressed / max(1, total_compressed)
        if ratio > limits.max_compression_ratio:
            raise UnsafeZipError(
                f"{path.name}: compression ratio {ratio:.1f} > "
                f"max_compression_ratio {limits.max_compression_ratio}"
            )
        return infos


def safe_extract(
    zip_path: Path,
    dest_dir: Path,
    limits: ZipLimits,
) -> list[ExtractedMember]:
    """Extract into a fresh directory. Caller must ensure dest_dir is empty/new."""
    infos = validate_zip(zip_path, limits)
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=False)

    members: list[ExtractedMember] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in infos:
            if info.is_dir():
                target = (dest_dir / info.filename).resolve()
                if not str(target).startswith(str(dest_dir) + "/") and target != dest_dir:
                    raise UnsafeZipError(f"zip-slip directory {info.filename!r}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            target = (dest_dir / info.filename).resolve()
            if not str(target).startswith(str(dest_dir) + "/"):
                raise UnsafeZipError(f"zip-slip file {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as out:
                remaining = int(info.file_size)
                while remaining > 0:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
                if remaining != 0:
                    raise UnsafeZipError(f"{zip_path.name}: truncated member {info.filename!r}")
            digest = sha256_file(target)
            members.append(
                ExtractedMember(
                    archive_path=info.filename,
                    dest_path=target,
                    file_size=int(info.file_size),
                    compress_size=int(info.compress_size),
                    sha256=digest,
                )
            )
            log.debug("extracted %s sha256=%s", info.filename, digest[:12])
    return members
