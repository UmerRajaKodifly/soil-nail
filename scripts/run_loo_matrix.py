#!/usr/bin/env python3
"""
Leave-one-out matrix for soil_nail refs:
  02→01, 01→02, 03→02, 02→03
× models persam, persam_f, seggpt
× tight_prompt_crop off/on

Requires variant-B masks in data/refs/soil_nail/masks/<stem>.png
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    ("nail_ref_02_2D16_rebar_install", "nail_ref_01_2D12_rebar_install"),
    ("nail_ref_01_2D12_rebar_install", "nail_ref_02_2D16_rebar_install"),
    ("nail_ref_03_2D17_depth_measure", "nail_ref_02_2D16_rebar_install"),
    ("nail_ref_02_2D16_rebar_install", "nail_ref_03_2D17_depth_measure"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/loo_b")
    ap.add_argument("--models", nargs="+", default=["persam", "persam_f", "seggpt"])
    ap.add_argument("--persam_f_epochs", type=int, default=1000)
    ap.add_argument("--sam_type", default="vit_h")
    args = ap.parse_args()

    mask_dir = ROOT / "data" / "refs" / "soil_nail" / "masks"
    img_dir = ROOT / "data" / "refs" / "soil_nail" / "images"
    needed = set()
    for a, b in PAIRS:
        needed.add(a)
        needed.add(b)
    missing = [s for s in sorted(needed) if not (mask_dir / f"{s}.png").is_file()]
    if missing:
        print("ERROR: Missing variant-B masks:")
        for s in missing:
            print(f"  {mask_dir / (s + '.png')}")
        print("\nAnnotate with:")
        print("  source .venv/bin/activate")
        print("  python tools/annotate_mask.py --variant B")
        print("Variant B = sleeve + bar end + hose roots only (default).")
        return 1

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for crop in (False, True):
        crop_tag = "crop" if crop else "nocrop"
        for ref_stem, test_stem in PAIRS:
            pair_tag = f"{ref_stem.split('_')[2]}to{test_stem.split('_')[2]}"  # 02to01 etc rough
            # clearer tag from nail ids
            def short(s):
                # nail_ref_02_2D16_... → 02
                parts = s.split("_")
                return parts[2] if len(parts) > 2 else s

            tag = f"{short(ref_stem)}_to_{short(test_stem)}_{crop_tag}"
            run_dir = out_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            # write single-image test dir symlink-like copy path via --images file
            test_img = img_dir / f"{test_stem}.jpg"
            cmd = [
                sys.executable,
                str(ROOT / "compare.py"),
                "--cls",
                "soil_nail",
                "--variant",
                "B",
                "--ref",
                ref_stem,
                "--images",
                str(test_img),
                "--out",
                str(run_dir),
                "--models",
                *args.models,
                "--sam_type",
                args.sam_type,
                "--persam_f_epochs",
                str(args.persam_f_epochs),
            ]
            if crop:
                cmd.append("--tight_prompt_crop")
            print("\n>>>", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                print("FAILED", tag, "code", r.returncode)
            csv_path = run_dir / "summary.csv"
            if csv_path.is_file():
                with csv_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["pair"] = tag
                        row["ref"] = ref_stem
                        row["test"] = test_stem
                        row["tight_prompt_crop"] = int(crop)
                        all_rows.append(row)

    master = out_root / "summary.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with master.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nMaster summary: {master} ({len(all_rows)} rows)")
    else:
        print("No rows written — all runs failed?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
