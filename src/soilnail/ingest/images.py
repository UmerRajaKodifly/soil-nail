"""Image decoding audit with EXIF orientation handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

ImageStatus = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True)
class ImageAudit:
    status: ImageStatus
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    orig_width: int | None = None
    orig_height: int | None = None
    effective_width: int | None = None
    effective_height: int | None = None
    mode: str | None = None
    format: str | None = None
    exif_orientation: int | None = None
    transposed: bool = False


def _exif_orientation(img: Image.Image) -> int | None:
    exif = img.getexif()
    if not exif:
        return None
    # Orientation tag 274
    val = exif.get(274)
    return int(val) if val is not None else None


def audit_image(path: Path, *, max_pixels: int, apply_exif_transpose: bool) -> ImageAudit:
    if path.stat().st_size == 0:
        return ImageAudit(
            status="failed",
            failure_stage="stat",
            failure_type="ZeroByteImage",
            failure_message="file size is 0",
        )
    try:
        with Image.open(path) as img:
            img.verify()
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        return ImageAudit(
            status="failed",
            failure_stage="verify",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
    try:
        with Image.open(path) as img:
            fmt = img.format
            mode = img.mode
            ow, oh = img.size
            if ow * oh > max_pixels:
                return ImageAudit(
                    status="failed",
                    failure_stage="decode",
                    failure_type="TooManyPixels",
                    failure_message=f"{ow}x{oh} exceeds max_pixels={max_pixels}",
                    orig_width=ow,
                    orig_height=oh,
                    mode=mode,
                    format=fmt,
                )
            orient = _exif_orientation(img)
            transposed = False
            if apply_exif_transpose:
                img2 = ImageOps.exif_transpose(img)
                transposed = img2.size != (ow, oh) or (orient is not None and orient not in (0, 1))
                ew, eh = img2.size
                mode = img2.mode
            else:
                ew, eh = ow, oh
            return ImageAudit(
                status="ok",
                orig_width=ow,
                orig_height=oh,
                effective_width=ew,
                effective_height=eh,
                mode=mode,
                format=fmt,
                exif_orientation=orient,
                transposed=bool(transposed),
            )
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        return ImageAudit(
            status="failed",
            failure_stage="decode",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
