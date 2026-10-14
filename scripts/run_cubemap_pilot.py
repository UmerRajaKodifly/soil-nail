#!/usr/bin/env python3
"""Cubemap Gate 2 pilot — detached-friendly, resumable, writes gate2_cubemap_estimate.json.

Do NOT pipe this into head/rg. Launch with nohup/setsid and a log file.
"""

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
from soilnail.models.sam_cache import SamEmbeddingCache  # noqa: E402
from soilnail.runner import (  # noqa: E402
    build_prepared_refs,
    load_rgb_exif,
    segment_cubemap_image,
    segment_image,
)
from soilnail.store import ResultsStore  # noqa: E402

log = logging.getLogger("cubemap_pilot")


def stratified_sample(
    rows: list[tuple[str, str, str]],
    *,
    size: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    by_arch: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        by_arch[r[1]].append(r)
    rng = np.random.default_rng(seed)
    total = len(rows)
    alloc = {a: max(1, round(size * len(v) / total)) for a, v in by_arch.items()}
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument(
        "--micro",
        action="store_true",
        help="Use pilot.micro_size frames for a fast Gate 2 estimate",
    )
    ap.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Only cubemap baseline cells (3 models × ensemble)",
    )
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    proj = load_yaml(ROOT / cfg["paths"]["projection_config"])
    models_cfg = ROOT / cfg["paths"]["models_config"]
    models_yaml = load_yaml(models_cfg)

    conn = sqlite3.connect(ROOT / cfg["paths"]["manifest_sqlite"])
    rows = conn.execute(
        "SELECT image_id, archive, relative_path FROM images WHERE status='ok'"
    ).fetchall()
    n_full = len(rows)
    sample_n = int(cfg["pilot"]["micro_size"] if args.micro else cfg["pilot"]["size"])
    sample = stratified_sample(
        [(a, b, c) for a, b, c in rows],
        size=sample_n,
        seed=int(cfg["pilot"]["seed"]),
    )

    out_dir = ROOT / cfg["paths"]["pilot_out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.json").write_text(
        json.dumps(
            [{"image_id": a, "archive": b, "path": c} for a, b, c in sample],
            indent=2,
        )
    )

    # Mark legacy equirect pilot as superseded
    legacy = ROOT / cfg["paths"]["pilot_equirect_legacy"] / "gate2_estimate.json"
    if legacy.is_file():
        blob = json.loads(legacy.read_text())
        blob["superseded"] = True
        blob["superseded_reason"] = (
            "360° amendment: equirect pilots treated panoramas as flat; "
            "see outputs/pilot_cubemap/gate2_cubemap_estimate.json"
        )
        legacy.write_text(json.dumps(blob, indent=2))
        log.info("Marked %s superseded=true", legacy)

    ref_img = ROOT / cfg["references"]["paths"]["images_dir"]
    ref_msk = ROOT / cfg["references"]["paths"]["masks_dir"]
    extract = ROOT / cfg["paths"]["extract_root"]
    single_refs = list(cfg["references"]["single"])
    face_size = int(cfg["baseline_full_run"]["face_size"])
    fov = float(cfg["baseline_full_run"]["fov_deg"])
    margin = float(cfg["baseline_full_run"]["margin_deg"])
    merge_deg = float(cfg["angular_merge_deg"])
    area_warn_sr = float(proj["angular_size"]["area_flag_solid_angle_steradians"])

    sam_cache = SamEmbeddingCache(
        ROOT / cfg["paths"]["sam_face_cache"],
        enabled=bool(proj["sam_face_cache"]["enabled"]),
    )

    # Config cells
    cells: list[dict[str, object]] = []
    for model in cfg["baseline_full_run"]["models"]:
        cells.append(
            {
                "config_id": f"cubemap_{model}_f{face_size}_m{margin}_ensemble",
                "model": model,
                "mode": "cubemap",
                "face_size": face_size,
                "margin_deg": margin,
                "subtile": False,
                "refs": single_refs,
            }
        )
    if not args.skip_ablations:
        if cfg["pilot_grid"].get("equirect_whole"):
            for model in cfg["models"]:
                cells.append(
                    {
                        "config_id": f"equirect_whole_{model}_ensemble",
                        "model": model,
                        "mode": "equirect_whole",
                        "face_size": None,
                        "margin_deg": 0.0,
                        "subtile": False,
                        "refs": single_refs,
                    }
                )
        if cfg["pilot_grid"].get("persam_face_subtile"):
            cells.append(
                {
                    "config_id": f"cubemap_persam_f{face_size}_subtile512_ensemble",
                    "model": "persam",
                    "mode": "cubemap",
                    "face_size": face_size,
                    "margin_deg": margin,
                    "subtile": True,
                    "refs": single_refs,
                }
            )

    store = ResultsStore(out_dir / "pilot_cubemap.sqlite")
    done_path = out_dir / "completed_cells.json"
    completed: set[str] = set()
    if done_path.is_file():
        completed = set(json.loads(done_path.read_text()))

    timings: list[dict[str, object]] = []
    timings_path = out_dir / "timings_partial.json"
    if timings_path.is_file():
        timings = json.loads(timings_path.read_text())

    for c in cells:
        config_id = str(c["config_id"])
        if config_id in completed:
            log.info("SKIP completed %s", config_id)
            continue
        model = str(c["model"])
        log.info("=== %s ===", config_id)
        use_cache = c["mode"] == "cubemap" and model.startswith("persam")
        seg = build_segmenter(
            model,
            models_cfg,
            repo_root=ROOT,
            tight_prompt_crop=False,
            multi_instance=True,
            feature_ensemble=True,
            sam_embedding_cache=sam_cache if use_cache else None,
        )
        refs = build_prepared_refs(ref_img, ref_msk, list(c["refs"]))  # type: ignore[arg-type]
        t_prep = time.perf_counter()
        seg.prepare(refs)
        prep_s = time.perf_counter() - t_prep
        runtimes: list[float] = []
        fails = 0
        for image_id, archive, rel in sample:
            path = extract / archive / rel
            rgb = load_rgb_exif(path)
            if c["mode"] == "cubemap":
                res = segment_cubemap_image(
                    seg,
                    rgb,
                    image_id=image_id,
                    model=model,
                    config_id=config_id,
                    face_size=int(c["face_size"]),  # type: ignore[arg-type]
                    fov_deg=fov,
                    margin_deg=float(c["margin_deg"]),
                    yaw_offset_deg=0.0,
                    instance_mode=True,
                    min_cc_area=int(models_yaml["seggpt_instances"]["min_cc_area_px"]),
                    nms_iou=float(models_yaml["multi_instance"]["nms_iou"]),
                    area_warn_solid_angle_sr=area_warn_sr,
                    merge_deg=merge_deg,
                    subtile=bool(c["subtile"]),
                    subtile_px=int(proj["subtile_ablation"]["tile_px"]),
                    subtile_overlap=float(proj["subtile_ablation"]["overlap"]),
                )
            else:
                res = segment_image(
                    seg,
                    rgb,
                    image_id=image_id,
                    model=model,
                    config_id=config_id,
                    tiling=False,
                    tile_px=None,
                    overlap=0.0,
                    instance_mode=True,
                    min_cc_area=int(models_yaml["seggpt_instances"]["min_cc_area_px"]),
                    nms_iou=float(models_yaml["multi_instance"]["nms_iou"]),
                    area_warn_pct=float(models_yaml["mask_area"]["warn_pct"]),
                    oom_max_retries=int(models_yaml["oom"]["max_retries"]),
                    oom_fallback_tile=int(models_yaml["oom"]["fallback_tile_px"]),
                )
            store.write_image_result(config_id, res)
            if res.status != "succeeded":
                fails += 1
                log.error("fail %s %s", config_id, image_id)
            if res.runtime_s is not None:
                runtimes.append(res.runtime_s)
                log.info(
                    "%s %s %.2fs ndet=%d",
                    config_id,
                    image_id,
                    res.runtime_s,
                    len(res.detections),
                )
        seg.close()
        mean_s = float(np.mean(runtimes)) if runtimes else float("nan")
        est_full_h = (prep_s + mean_s * n_full) / 3600.0
        row = {
            "config_id": config_id,
            "model": model,
            "mode": c["mode"],
            "subtile": c["subtile"],
            "n_pilot": len(sample),
            "n_failed": fails,
            "prepare_s": round(prep_s, 3),
            "mean_s_per_image": round(mean_s, 3),
            "est_full_dataset_hours": round(est_full_h, 3),
        }
        timings.append(row)
        timings_path.write_text(json.dumps(timings, indent=2))
        completed.add(config_id)
        done_path.write_text(json.dumps(sorted(completed), indent=2))
        log.info(
            "DONE %s mean=%.3fs est_full=%.2fh fails=%d",
            config_id,
            mean_s,
            est_full_h,
            fails,
        )

    baseline_hours = sum(
        float(t["est_full_dataset_hours"])
        for t in timings
        if str(t["config_id"]).startswith("cubemap_")
        and not bool(t.get("subtile"))
        and "_subtile" not in str(t["config_id"])
    )
    # Prefer exact baseline cell ids
    baseline_ids = {
        f"cubemap_{m}_f{face_size}_m{margin}_ensemble"
        for m in cfg["baseline_full_run"]["models"]
    }
    baseline_hours = sum(
        float(t["est_full_dataset_hours"])
        for t in timings
        if t["config_id"] in baseline_ids
    )
    budget = float(cfg["time_budget_hours"])
    gate = {
        "schema_version": "2",
        "representation": "cubemap",
        "dataset_size": n_full,
        "pilot_size": len(sample),
        "micro": bool(args.micro),
        "face_size": face_size,
        "fov_deg": fov,
        "margin_deg": margin,
        "budget_hours": budget,
        "baseline_estimated_hours": round(baseline_hours, 3),
        "proceed_automatically": baseline_hours <= budget,
        "sam_face_cache_enabled": bool(proj["sam_face_cache"]["enabled"]),
        "equirect_baseline_superseded": True,
        "timings": timings,
        "note": (
            "Baseline = sum of 3 cubemap ensemble configs (6 faces/frame). "
            "Equirect whole and PerSAM face-subtile are ablations only."
        ),
    }
    out_gate = out_dir / "gate2_cubemap_estimate.json"
    out_gate.write_text(json.dumps(gate, indent=2))
    summary = {k: gate[k] for k in gate if k != "timings"}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_gate}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
