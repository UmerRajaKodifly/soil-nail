"""Per-image inference with optional tiling, OOM retry, and Detection packing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from soilnail.models.base import PreparedRefs, SegmenterProtocol
from soilnail.rle import encode_rle, mask_bbox_xywh
from soilnail.schemas import Detection, ImageResult
from soilnail.store import failure_record
from soilnail.tiling import make_tiles, merge_masks_union, nms_masks

log = logging.getLogger(__name__)


def load_rgb_exif(path: Path) -> NDArray[np.uint8]:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
    return rgb


def load_ref_pair(
    images_dir: Path, masks_dir: Path, stem: str
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    img_path = images_dir / f"{stem}.jpg"
    msk_path = masks_dir / f"{stem}.png"
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    m = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(msk_path)
    rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    mask = np.asarray((m > 127).astype(np.uint8) * 255, dtype=np.uint8)
    return rgb, mask


def build_prepared_refs(
    images_dir: Path,
    masks_dir: Path,
    stems: list[str],
) -> PreparedRefs:
    imgs: list[NDArray[np.uint8]] = []
    msks: list[NDArray[np.uint8]] = []
    for s in stems:
        im, mk = load_ref_pair(images_dir, masks_dir, s)
        imgs.append(im)
        msks.append(mk)
    return PreparedRefs(names=tuple(stems), images_rgb=tuple(imgs), masks=tuple(msks))


def _cc_instances(mask: NDArray[np.uint8], min_area: int) -> list[NDArray[np.bool_]]:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: list[NDArray[np.bool_]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out.append(labels == i)
    if not out and m.any():
        out.append(m.astype(bool))
    return out


def segment_image(
    segmenter: SegmenterProtocol,
    image_rgb: NDArray[np.uint8],
    *,
    image_id: str,
    model: str,
    config_id: str,
    tiling: bool,
    tile_px: int | None,
    overlap: float,
    instance_mode: bool,
    min_cc_area: int,
    nms_iou: float,
    area_warn_pct: float,
    oom_max_retries: int,
    oom_fallback_tile: int,
) -> ImageResult:
    t0 = time.perf_counter()
    h, w = image_rgb.shape[:2]
    try:
        masks, scores, merged_flag, tile_ids = _segment_with_oom(
            segmenter,
            image_rgb,
            tiling=tiling,
            tile_px=tile_px,
            overlap=overlap,
            oom_max_retries=oom_max_retries,
            oom_fallback_tile=oom_fallback_tile,
        )
        masks_bool: list[NDArray[np.bool_]]
        if instance_mode:
            # Prefer per-instance masks from model; else CC on union
            if len(masks) <= 1:
                union_u8: NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8)
                for m in masks:
                    union_u8 = np.maximum(
                        union_u8, (m > 0).astype(np.uint8) * 255
                    ).astype(np.uint8)
                masks_bool = _cc_instances(union_u8, min_cc_area)
                scores = [float(m.sum()) for m in masks_bool]
            else:
                masks_bool = [m.astype(bool) for m in masks]
                keep = nms_masks(masks_bool, scores, nms_iou)
                masks_bool = [masks_bool[i] for i in keep]
                scores = [scores[i] for i in keep]
        else:
            union_b = np.zeros((h, w), dtype=bool)
            for m in masks:
                union_b |= m.astype(bool)
            masks_bool = [union_b] if union_b.any() else []
            scores = [float(union_b.sum())] if masks_bool else []

        dets: list[Detection] = []
        for i, (mb, sc) in enumerate(zip(masks_bool, scores, strict=True)):
            area = int(mb.sum())
            frac = float(area / (h * w))
            touches = bool(mb[0, :].any() or mb[-1, :].any() or mb[:, 0].any() or mb[:, -1].any())
            dets.append(
                Detection(
                    image_id=image_id,
                    model=model,
                    config_id=config_id,
                    instance_id=i,
                    mask_rle=encode_rle(mb),
                    bbox_xywh=mask_bbox_xywh(mb),
                    area_px=area,
                    area_fraction=frac,
                    raw_score=float(sc),
                    touches_border=touches,
                    merged_across_tiles=merged_flag,
                    area_flag=frac * 100.0 > area_warn_pct,
                    source_tile_ids=tile_ids,
                )
            )
        return ImageResult(
            image_id=image_id,
            status="succeeded",
            detections=dets,
            runtime_s=time.perf_counter() - t0,
            peak_vram_mb=_peak_vram(),
        )
    except Exception as exc:
        log.exception("segment failed image_id=%s", image_id)
        return ImageResult(
            image_id=image_id,
            status="failed",
            failure=failure_record(image_id, "segment", exc, model=model, config_id=config_id),
            runtime_s=time.perf_counter() - t0,
            peak_vram_mb=_peak_vram(),
        )


def _peak_vram() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def _segment_with_oom(
    segmenter: SegmenterProtocol,
    image_rgb: NDArray[np.uint8],
    *,
    tiling: bool,
    tile_px: int | None,
    overlap: float,
    oom_max_retries: int,
    oom_fallback_tile: int,
) -> tuple[list[NDArray[np.uint8]], list[float], bool, list[str]]:
    import torch

    attempt_tile = tile_px
    use_tiling = tiling
    last_exc: BaseException | None = None
    for attempt in range(oom_max_retries + 1):
        try:
            return _segment_once(
                segmenter,
                image_rgb,
                tiling=use_tiling,
                tile_px=attempt_tile,
                overlap=overlap,
            )
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            log.warning(
                "CUDA OOM attempt=%d tiling=%s tile=%s — retry with smaller tile",
                attempt,
                use_tiling,
                attempt_tile,
            )
            torch.cuda.empty_cache()
            use_tiling = True
            attempt_tile = (
                oom_fallback_tile if attempt_tile is None else max(256, int(attempt_tile * 0.75))
            )
    assert last_exc is not None
    raise last_exc


def _segment_once(
    segmenter: SegmenterProtocol,
    image_rgb: NDArray[np.uint8],
    *,
    tiling: bool,
    tile_px: int | None,
    overlap: float,
) -> tuple[list[NDArray[np.uint8]], list[float], bool, list[str]]:
    h, w = image_rgb.shape[:2]
    if not tiling or tile_px is None:
        out = segmenter.segment(image_rgb)
        return out.masks, out.scores, False, []

    tiles = make_tiles(h, w, tile=int(tile_px), overlap=overlap)
    tile_masks: list[Any] = []
    all_scores: list[float] = []
    for t in tiles:
        patch = image_rgb[t.y0 : t.y1, t.x0 : t.x1]
        out = segmenter.segment(patch)
        # merge instance masks on tile into one for union stitch, keep scores
        union = np.zeros(patch.shape[:2], dtype=np.uint8)
        for m in out.masks:
            union = np.maximum(union, (m > 0).astype(np.uint8) * 255)
        tile_masks.append((t, union))
        all_scores.extend(out.scores)
    merged, merged_flag = merge_masks_union((h, w), tile_masks)
    score = float(np.mean(all_scores)) if all_scores else 0.0
    return [merged], [score], merged_flag, [t.tile_id for t in tiles]
