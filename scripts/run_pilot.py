#!/usr/bin/env python3
"""Stratified pilot sample + experiment grid timing → Gate 2 estimate."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from soilnail.config import load_yaml  # noqa: E402
from soilnail.models.factory import build_segmenter  # noqa: E402
from soilnail.runner import build_prepared_refs, load_rgb_exif, segment_image  # noqa: E402
from soilnail.store import ResultsStore  # noqa: E402

log = logging.getLogger("pilot")


def stratified_sample(
    rows: list[tuple[str, str, str]],
    *,
    size: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    """rows: (image_id, archive, relative_path). Stratify by archive."""
    by_arch: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        by_arch[r[1]].append(r)
    rng = np.random.default_rng(seed)
    # proportional allocation
    total = len(rows)
    alloc = {a: max(1, round(size * len(v) / total)) for a, v in by_arch.items()}
    # fix rounding
    while sum(alloc.values()) > size:
        a = max(alloc, key=lambda k: alloc[k])
        if alloc[a] > 1:
            alloc[a] -= 1
        else:
            break
    while sum(alloc.values()) < size:
        a = max(by_arch, key=lambda k: len(by_arch[k]) - alloc.get(k, 0))
        alloc[a] = alloc.get(a, 0) + 1
    chosen: list[tuple[str, str, str]] = []
    for a, n in alloc.items():
        pool = by_arch[a]
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in idx)
    return chosen[:size]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only baseline configs on pilot (faster Gate 2 estimate)",
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)
    models_cfg = ROOT / cfg["paths"]["models_config"]
    models_yaml = load_yaml(models_cfg)

    conn = sqlite3.connect(ROOT / cfg["paths"]["manifest_sqlite"])
    rows = conn.execute(
        "SELECT image_id, archive, relative_path FROM images WHERE status='ok'"
    ).fetchall()
    sample = stratified_sample(
        [(a, b, c) for a, b, c in rows],
        size=int(cfg["pilot"]["size"]),
        seed=int(cfg["pilot"]["seed"]),
    )
    out_dir = ROOT / cfg["paths"]["pilot_out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.json").write_text(
        json.dumps([{"image_id": a, "archive": b, "path": c} for a, b, c in sample], indent=2)
    )

    ref_img = ROOT / cfg["references"]["paths"]["images_dir"]
    ref_msk = ROOT / cfg["references"]["paths"]["masks_dir"]
    extract = ROOT / cfg["paths"]["extract_root"]
    single_refs = list(cfg["references"]["single"])

    # Build config list
    configs: list[dict[str, object]] = []
    if args.quick:
        for model in cfg["baseline_full_run"]["models"]:
            for scale in cfg["baseline_full_run"]["scales"]:
                scale_def = next(
                    s for s in cfg["pilot_grid"]["scales"] if s["name"] == scale
                )
                configs.append(
                    {
                        "config_id": f"baseline_{model}_{scale}_ensemble",
                        "model": model,
                        "refs": single_refs,  # ensemble = all
                        "tiling": scale_def["tiling"],
                        "tile_px": scale_def["tile_px"],
                        "instance_mode": True,
                        "tight_prompt_crop": False,
                        "feature_ensemble": True,
                    }
                )
    else:
        for model in cfg["models"]:
            for scale in cfg["pilot_grid"]["scales"]:
                for refs_mode in cfg["pilot_grid"]["refs_modes"]:
                    if refs_mode == "ensemble":
                        ref_lists = [single_refs]
                        fe = True
                        tag = "ensemble"
                    else:
                        ref_lists = [[r] for r in single_refs]
                        fe = False
                        tag = "single"
                    for refs in ref_lists:
                        rid = refs[0] if len(refs) == 1 else "ensemble"
                        configs.append(
                            {
                                "config_id": f"{model}_{scale['name']}_{tag}_{rid}",
                                "model": model,
                                "refs": refs,
                                "tiling": scale["tiling"],
                                "tile_px": scale["tile_px"],
                                "instance_mode": True,
                                "tight_prompt_crop": False,
                                "feature_ensemble": fe,
                            }
                        )
        if cfg["pilot_grid"]["tight_prompt_crop_ablation"]:
            for model in cfg["models"]:
                configs.append(
                    {
                        "config_id": f"{model}_whole_ensemble_tightcrop",
                        "model": model,
                        "refs": single_refs,
                        "tiling": False,
                        "tile_px": None,
                        "instance_mode": True,
                        "tight_prompt_crop": True,
                        "feature_ensemble": True,
                    }
                )

    store = ResultsStore(out_dir / "pilot.sqlite")
    timings: list[dict[str, object]] = []
    n_images = len(rows)

    for c in configs:
        model = str(c["model"])
        config_id = str(c["config_id"])
        log.info("=== config %s ===", config_id)
        seg = build_segmenter(
            model,
            models_cfg,
            repo_root=ROOT,
            tight_prompt_crop=bool(c["tight_prompt_crop"]),
            multi_instance=bool(c["instance_mode"]) and model.startswith("persam"),
            feature_ensemble=bool(c["feature_ensemble"]),
        )
        refs = build_prepared_refs(ref_img, ref_msk, list(c["refs"]))  # type: ignore[arg-type]
        t_prep = time.perf_counter()
        seg.prepare(refs)
        prep_s = time.perf_counter() - t_prep
        runtimes: list[float] = []
        vrams: list[float] = []
        fails = 0
        for image_id, archive, rel in sample:
            path = extract / archive / rel
            rgb = load_rgb_exif(path)
            res = segment_image(
                seg,
                rgb,
                image_id=image_id,
                model=model,
                config_id=config_id,
                tiling=bool(c["tiling"]),
                tile_px=int(c["tile_px"]) if c["tile_px"] is not None else None,
                overlap=float(models_yaml["tiling"]["overlap"]),
                instance_mode=bool(c["instance_mode"]),
                min_cc_area=int(models_yaml["seggpt_instances"]["min_cc_area_px"]),
                nms_iou=float(models_yaml["multi_instance"]["nms_iou"]),
                area_warn_pct=float(models_yaml["mask_area"]["warn_pct"]),
                oom_max_retries=int(models_yaml["oom"]["max_retries"]),
                oom_fallback_tile=int(models_yaml["oom"]["fallback_tile_px"]),
            )
            store.write_image_result(config_id, res)
            if res.status != "succeeded":
                fails += 1
            if res.runtime_s is not None:
                runtimes.append(res.runtime_s)
            if res.peak_vram_mb is not None:
                vrams.append(res.peak_vram_mb)
        seg.close()
        mean_s = float(np.mean(runtimes)) if runtimes else float("nan")
        p95_s = float(np.percentile(runtimes, 95)) if runtimes else float("nan")
        peak_vram = float(max(vrams)) if vrams else None
        est_full_h = (prep_s + mean_s * n_images) / 3600.0
        timings.append(
            {
                "config_id": config_id,
                "model": model,
                "n_pilot": len(sample),
                "n_failed": fails,
                "prepare_s": round(prep_s, 3),
                "mean_s_per_image": round(mean_s, 3),
                "p95_s_per_image": round(p95_s, 3),
                "peak_vram_mb": peak_vram,
                "est_full_dataset_hours": round(est_full_h, 3),
                "tiling": c["tiling"],
                "tile_px": c["tile_px"],
                "tight_prompt_crop": c["tight_prompt_crop"],
            }
        )
        log.info(
            "%s mean=%.3fs prep=%.1fs est_full=%.2fh fails=%d",
            config_id,
            mean_s,
            prep_s,
            est_full_h,
            fails,
        )

    # Baseline estimate = sum of baseline configs (sequential models)
    baseline_ids = {
        f"baseline_{m}_{s}_ensemble"
        for m in cfg["baseline_full_run"]["models"]
        for s in cfg["baseline_full_run"]["scales"]
    }
    baseline_hours = sum(
        float(t["est_full_dataset_hours"])  # type: ignore[arg-type]
        for t in timings
        if t["config_id"] in baseline_ids or (
            # also match non-quick naming
            str(t["config_id"]).endswith("_ensemble")
            and any(
                str(t["config_id"]).startswith(f"{m}_")
                for m in cfg["baseline_full_run"]["models"]
            )
            and any(s in str(t["config_id"]) for s in cfg["baseline_full_run"]["scales"])
        )
    )
    # cleaner: if quick mode, sum all timings
    if args.quick:
        baseline_hours = sum(float(t["est_full_dataset_hours"]) for t in timings)  # type: ignore[arg-type]

    budget = float(cfg["time_budget_hours"])
    gate = {
        "dataset_size": n_images,
        "pilot_size": len(sample),
        "budget_hours": budget,
        "baseline_estimated_hours": round(baseline_hours, 3),
        "proceed_automatically": baseline_hours <= budget,
        "timings": timings,
    }
    (out_dir / "gate2_estimate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({k: gate[k] for k in gate if k != "timings"}, indent=2))
    print(f"Wrote {out_dir / 'gate2_estimate.json'}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
