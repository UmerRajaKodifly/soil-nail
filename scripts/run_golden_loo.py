#!/usr/bin/env python3
"""Run LOO no-crop golden pairs through soilnail adapters; compare IoU to LOO_REPORT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from soilnail.config import load_yaml  # noqa: E402
from soilnail.models.base import PreparedRefs  # noqa: E402
from soilnail.models.factory import build_segmenter  # noqa: E402
from soilnail.tiling import mask_iou  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8) * 255


def union_mask(masks: list[np.ndarray], hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(hw, dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/golden_loo.yaml")
    ap.add_argument("--out", default="outputs/golden_loo/result.json")
    ap.add_argument("--models", nargs="+", default=["persam", "persam_f", "seggpt"])
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    img_dir = ROOT / cfg["paths"]["images_dir"]
    msk_dir = ROOT / cfg["paths"]["masks_dir"]
    models_cfg = ROOT / cfg["paths"]["models_config"]
    tol = cfg["tolerances"]["abs_iou"]

    rows = []
    failures = []
    for pair in cfg["pairs"]:
        ref = pair["ref"]
        test = pair["test"]
        ref_img = load_rgb(img_dir / f"{ref}.jpg")
        ref_msk = load_mask(msk_dir / f"{ref}.png")
        test_img = load_rgb(img_dir / f"{test}.jpg")
        gt = load_mask(msk_dir / f"{test}.png") > 0
        refs = PreparedRefs(names=(ref,), images_rgb=(ref_img,), masks=(ref_msk,))
        for model in args.models:
            seg = build_segmenter(
                model,
                models_cfg,
                repo_root=ROOT,
                tight_prompt_crop=False,
                multi_instance=False,
            )
            try:
                seg.prepare(refs)
                out = seg.segment(test_img)
                pred = union_mask(out.masks, gt.shape)
                iou = mask_iou(pred, gt)
            finally:
                seg.close()
            exp = float(pair["expected_iou"][model])
            limit = float(tol[model])
            ok = abs(iou - exp) <= limit
            row = {
                "pair": pair["name"],
                "model": model,
                "iou": round(iou, 4),
                "expected": exp,
                "abs_err": round(abs(iou - exp), 4),
                "tolerance": limit,
                "pass": ok,
            }
            rows.append(row)
            print(row)
            if not ok:
                failures.append(row)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tolerance_status": cfg["tolerances"]["status"],
        "rows": rows,
        "n_fail": len(failures),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
