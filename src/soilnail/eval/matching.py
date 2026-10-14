"""Detection ↔ label matching.

Cubemap detection masks are face-local; matching uses equirect polygons /
bbox when present (not face RLE vs equirect GT).
"""

from __future__ import annotations

from dataclasses import dataclass

from soilnail.schemas import Detection


@dataclass(frozen=True)
class MatchConfig:
    """Match if IoU(detection equirect bbox, GT box) ≥ min_iou."""

    min_iou: float = 0.3
    source: str = (
        "my decision — provisional until labelling; IoU≥0.3 between GT equirect "
        "box and detection equirect bbox (from equirect_polygons_xy). Face RLE "
        "is not compared to equirect GT."
    )


@dataclass(frozen=True)
class GtNail:
    bbox_xywh: list[float]  # equirect pixels
    stage_tag: str | None = None


def _det_equirect_bbox_xywh(det: Detection) -> list[float] | None:
    polys = det.equirect_polygons_xy or (
        [det.equirect_polygon_xy] if det.equirect_polygon_xy else None
    )
    if not polys:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for ring in polys:
        for pt in ring:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def match_detections(
    detections: list[Detection],
    nails: list[GtNail],
    *,
    cfg: MatchConfig | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match. Returns (pairs det_i,gt_j), unmatched_dets, unmatched_gts."""
    cfg = cfg or MatchConfig()
    pairs: list[tuple[int, int]] = []
    used_d: set[int] = set()
    used_g: set[int] = set()
    scores: list[tuple[float, int, int]] = []
    for di, d in enumerate(detections):
        db = _det_equirect_bbox_xywh(d)
        if db is None:
            continue
        for gi, g in enumerate(nails):
            iou = box_iou(db, g.bbox_xywh)
            if iou >= cfg.min_iou:
                scores.append((iou, di, gi))
    scores.sort(reverse=True)
    for _, di, gi in scores:
        if di in used_d or gi in used_g:
            continue
        pairs.append((di, gi))
        used_d.add(di)
        used_g.add(gi)
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    unmatched_g = [i for i in range(len(nails)) if i not in used_g]
    return pairs, unmatched_d, unmatched_g
