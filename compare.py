#!/usr/bin/env python3
"""Side-by-side PerSAM / PerSAM-F / SegGPT comparison + summary.csv."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.device import free_cuda, pick_device  # noqa: E402
from src.persam_runner import PerSAMFSegmenter, PerSAMSegmenter  # noqa: E402
from src.preprocess import crop_bottom_right, load_binary_mask, mask_area_pct, overlay_mask  # noqa: E402
from src.segmenter import RefExample  # noqa: E402
from src.seggpt_runner import SegGPTSegmenter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("compare")

MASK_AREA_WARN_PCT = 25.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cls", default="soil_nail")
    p.add_argument("--images", type=str, required=True, help="Test image dir or single file")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--models", nargs="+", default=["persam", "persam_f", "seggpt"])
    p.add_argument("--variant", choices=["A", "B"], default="B", help="B→masks/ (default), A→masks_A/")
    p.add_argument("--ref", nargs="+", default=None, help="Reference image stem(s); default: all with masks")
    p.add_argument("--tight_prompt_crop", action="store_true", help="EXTENSION: crop prompt around mask")
    p.add_argument("--crop_margin", type=float, default=0.15)
    p.add_argument("--tile", action="store_true", help="EXTENSION: sliding-window inference")
    p.add_argument("--multi_instance", action="store_true", help="EXTENSION: PerSAM multi-peak")
    p.add_argument("--crop_bottom_right", action="store_true", help="EXTENSION: blank date stamp")
    p.add_argument("--sam_type", default="vit_h")
    p.add_argument("--mask_area_max_pct", type=float, default=MASK_AREA_WARN_PCT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--persam_f_epochs", type=int, default=1000)
    return p.parse_args()


def variant_mask_dir(cls: str, variant: str) -> Path:
    return ROOT / "data" / "refs" / cls / ("masks" if variant == "B" else "masks_A")


def load_refs(cls: str, variant: str, stems: list[str] | None) -> list[RefExample]:
    img_dir = ROOT / "data" / "refs" / cls / "images"
    mdir = variant_mask_dir(cls, variant)
    if stems is None:
        stems = sorted(p.stem for p in mdir.glob("*.png") if not p.name.endswith("_overlay.png"))
        # exclude overlay jpgs saved as png mistakenly; only pure stems
        stems = [s for s in stems if not s.endswith("_overlay")]
    refs = []
    for stem in stems:
        ip = img_dir / f"{stem}.jpg"
        if not ip.is_file():
            ip = img_dir / f"{stem}.png"
        mp = mdir / f"{stem}.png"
        if not ip.is_file() or not mp.is_file():
            raise FileNotFoundError(f"Missing ref pair for {stem}: {ip} / {mp}")
        rgb = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = load_binary_mask(str(mp))
        area = mask_area_pct(mask)
        flag = area > MASK_AREA_WARN_PCT
        if flag:
            log.warning("REF MASK FLAG %s area=%.2f%% > %.1f%%", stem, area, MASK_AREA_WARN_PCT)
        refs.append(RefExample(image_rgb=rgb, mask=mask, name=stem))
    return refs


def list_test_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 0.0


def build_segmenter(name: str, args, raw_dir: Path):
    common = dict(
        sam_type=args.sam_type,
        device=args.device,
        tight_prompt_crop=args.tight_prompt_crop,
        crop_margin=args.crop_margin,
        multi_instance=args.multi_instance,
    )
    if name == "persam":
        return PerSAMSegmenter(**common)
    if name == "persam_f":
        return PerSAMFSegmenter(**common, train_epoch=args.persam_f_epochs)
    if name == "seggpt":
        return SegGPTSegmenter(
            device=args.device,
            tight_prompt_crop=args.tight_prompt_crop,
            crop_margin=args.crop_margin,
            feature_ensemble=False,
            save_raw_dir=raw_dir,
        )
    raise ValueError(name)


COLORS = {
    "persam": (255, 64, 64),
    "persam_f": (255, 160, 32),
    "seggpt": (32, 200, 64),
}


def make_compare_panel(original, overlays: dict[str, np.ndarray], titles: dict[str, str]) -> np.ndarray:
    """original | model overlays..."""
    panels = [original]
    labels = ["original"]
    for k in overlays:
        panels.append(overlays[k])
        labels.append(titles.get(k, k))
    # resize to same height
    h = min(p.shape[0] for p in panels)
    normed = []
    for p in panels:
        scale = h / p.shape[0]
        w = int(p.shape[1] * scale)
        normed.append(cv2.resize(p, (w, h)))
    # banner with titles
    gap = 8
    total_w = sum(p.shape[1] for p in normed) + gap * (len(normed) - 1)
    banner_h = 36
    canvas = np.ones((h + banner_h, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for lab, p in zip(labels, normed):
        canvas[banner_h : banner_h + h, x : x + p.shape[1]] = p
        cv2.putText(canvas, lab[:48], (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        x += p.shape[1] + gap
    return canvas


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "seggpt_raw"
    device = pick_device(args.device)
    log.info("device=%s variant=%s tight_crop=%s", device, args.variant, args.tight_prompt_crop)

    refs = load_refs(args.cls, args.variant, args.ref)
    if not refs:
        log.error("No reference masks found. Annotate first: python tools/annotate_mask.py --variant B")
        return 1
    tests = list_test_images(args.images)
    if not tests:
        log.error("No test images in %s", args.images)
        return 1

    rows = []
    # Sequential models to fit 8GB
    results_by_image: dict[str, dict] = {p.stem: {} for p in tests}

    for model_name in args.models:
        log.info("=== Model %s ===", model_name)
        free_cuda()
        seg = build_segmenter(model_name, args, raw_dir)
        try:
            seg.prepare(refs)
            for tip in tests:
                rgb = cv2.cvtColor(cv2.imread(str(tip)), cv2.COLOR_BGR2RGB)
                if args.crop_bottom_right:
                    rgb = crop_bottom_right(rgb)
                if args.tile:
                    log.warning("--tile requested but full tiling merge deferred; running full-frame")
                if model_name == "seggpt":
                    result = seg.segment(rgb, stem=tip.stem)
                else:
                    result = seg.segment(rgb)
                union = result.union_mask
                area = mask_area_pct(union)
                area_flag = area > args.mask_area_max_pct
                if area_flag:
                    log.warning("PRED MASK FLAG %s/%s area=%.2f%%", tip.stem, model_name, area)

                mask_path = out / f"{tip.stem}__{model_name}.png"
                ov_path = out / f"{tip.stem}__{model_name}_overlay.jpg"
                json_path = out / f"{tip.stem}__{model_name}.json"
                cv2.imwrite(str(mask_path), union)
                ov = overlay_mask(rgb, union, color=COLORS.get(model_name, (255, 0, 0)))
                cv2.imwrite(str(ov_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                meta = {
                    **result.meta,
                    "scores": result.scores,
                    "mask_area_pct": area,
                    "mask_area_flag": area_flag,
                    "mask_area_max_pct": args.mask_area_max_pct,
                    "ref_stems": [r.name for r in refs],
                    "variant": args.variant,
                }
                # optional GT
                gt_path = ROOT / "data" / "gt" / f"{tip.stem}.png"
                gt_iou = None
                if gt_path.is_file():
                    gt_iou = iou(union, load_binary_mask(str(gt_path)))
                    meta["gt_iou"] = gt_iou
                json_path.write_text(json.dumps(meta, indent=2, default=float))

                results_by_image[tip.stem][model_name] = {
                    "overlay": ov,
                    "original": rgb,
                    "n": result.meta.get("n_instances", len(result.masks)),
                    "runtime": result.meta.get("runtime_s"),
                    "area": area,
                    "flag": area_flag,
                    "diagnosis": result.meta.get("diagnosis"),
                }
                rows.append(
                    {
                        "image": tip.name,
                        "model": model_name,
                        "n_instances": result.meta.get("n_instances", len(result.masks)),
                        "mask_area_pct": round(area, 4),
                        "mask_area_flag": int(area_flag),
                        "runtime_s": result.meta.get("runtime_s"),
                        "peak_vram_mb": result.meta.get("peak_vram_mb"),
                        "device": result.meta.get("device"),
                        "tight_prompt_crop": int(args.tight_prompt_crop),
                        "variant": args.variant,
                        "refs": "|".join(r.name for r in refs),
                        "gt_iou": gt_iou if gt_iou is not None else "",
                        "seggpt_diagnosis": result.meta.get("diagnosis", ""),
                        "seggpt_raw_frac_fg": result.meta.get("raw_frac_closer_to_fg", ""),
                        "seggpt_raw_score_fg_max": result.meta.get("raw_score_fg_max", ""),
                    }
                )
        finally:
            seg.close()
            free_cuda()

    # Compare panels
    for stem, md in results_by_image.items():
        if not md:
            continue
        original = next(iter(md.values()))["original"]
        overlays = {k: v["overlay"] for k, v in md.items()}
        titles = {
            k: f"{k} n={v['n']} t={v['runtime']}s a={v['area']:.1f}%{' FLAG' if v['flag'] else ''}"
            for k, v in md.items()
        }
        panel = make_compare_panel(original, overlays, titles)
        cv2.imwrite(str(out / f"{stem}_compare.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    csv_path = out / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log.info("Wrote %s (%d rows)", csv_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
