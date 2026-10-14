#!/usr/bin/env python3
"""Real-model yaw-sweep margin ablation (follow-up 2 item 1). Run detached."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from soilnail.projection.sphere_ops import angular_distance_rad  # noqa: E402
from soilnail.runner import build_prepared_refs, load_rgb_exif, segment_cubemap_image  # noqa: E402
from soilnail.schemas import Detection  # noqa: E402

log = logging.getLogger("margin_yaw")


def world_dets(dets: list[Detection]) -> list[list[float]]:
    out: list[list[float]] = []
    for d in dets:
        if d.spherical is None:
            continue
        out.append([d.spherical.lon_rad, d.spherical.lat_rad])
    return out


def match_counts(
    baseline: list[list[float]],
    other: list[list[float]],
    match_deg: float,
) -> dict:
    used: set[int] = set()
    retained = 0
    pos_err: list[float] = []
    for blon, blat in baseline:
        best_i, best_d = -1, 1e9
        for i, (olon, olat) in enumerate(other):
            if i in used:
                continue
            d = math.degrees(angular_distance_rad(blon, blat, olon, olat))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= match_deg:
            used.add(best_i)
            retained += 1
            pos_err.append(best_d)
    lost = len(baseline) - retained
    extra = len(other) - len(used)
    return {
        "n_baseline": len(baseline),
        "n_other": len(other),
        "retained": retained,
        "lost": lost,
        "extra": extra,
        "retention_frac": retained / max(len(baseline), 1),
        "mean_pos_err_deg": float(np.mean(pos_err)) if pos_err else None,
    }


def pick_candidates(
    rows: list[tuple[str, str, str]],
    n_per_walk: dict[str, int],
    seed: int,
) -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(seed)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    out: list[tuple[str, str, str]] = []
    for arch, n in n_per_walk.items():
        pool = by.get(arch, [])
        if not pool:
            continue
        k = min(n * 4, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        out.extend(pool[int(i)] for i in idx)
    return out


def run_one(seg, rgb, *, image_id, model, face_size, margin, yaw, models_yaml, area_warn, merge_deg):
    t0 = time.perf_counter()
    res = segment_cubemap_image(
        seg,
        rgb,
        image_id=image_id,
        model=model,
        config_id=f"yaw{yaw}_m{margin}",
        face_size=face_size,
        fov_deg=90.0,
        margin_deg=margin,
        yaw_offset_deg=yaw,
        instance_mode=True,
        min_cc_area=int(models_yaml["seggpt_instances"]["min_cc_area_px"]),
        nms_iou=float(models_yaml["multi_instance"]["nms_iou"]),
        area_warn_solid_angle_sr=area_warn,
        merge_deg=merge_deg,
    )
    dt = time.perf_counter() - t0
    if res.status != "succeeded":
        log.error("fail %s: %s", image_id, res.failure)
        return [], dt
    return res.detections, dt


def load_jsonl_index(path: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    if not path.is_file():
        return idx
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        idx[rec["key"]] = rec
    return idx


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/margin_ablation.yaml")
    ap.add_argument("--exp", default="configs/experiments.yaml")
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    abl = load_yaml(ROOT / args.config)
    exp = load_yaml(ROOT / args.exp)
    proj = load_yaml(ROOT / exp["paths"]["projection_config"])
    models_yaml = load_yaml(ROOT / exp["paths"]["models_config"])
    models = args.models.split(",") if args.models else list(abl["models"])
    margins = [float(m) for m in abl["margins_deg"]]
    baseline_m = float(abl["baseline_margin_deg"])
    yaws = [float(y) for y in abl["yaw_offsets_deg"]]
    face_size = int(abl["face_size"])
    match_deg = float(abl["match_angular_deg"])
    area_warn = float(proj["angular_size"]["area_flag_solid_angle_steradians"])
    merge_deg = float(exp["angular_merge_deg"])

    out_dir = ROOT / abl["paths"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    index = load_jsonl_index(results_path)

    conn = sqlite3.connect(ROOT / exp["paths"]["manifest_sqlite"])
    rows = [(a, b, c) for a, b, c in conn.execute(
        "SELECT image_id, archive, relative_path FROM images WHERE status='ok'"
    )]
    candidates = pick_candidates(rows, dict(abl["n_frames_per_walk"]), int(abl["seed"]))
    extract = ROOT / exp["paths"]["extract_root"]
    refs = build_prepared_refs(
        ROOT / exp["references"]["paths"]["images_dir"],
        ROOT / exp["references"]["paths"]["masks_dir"],
        list(exp["references"]["single"]),
    )
    cache = SamEmbeddingCache(ROOT / exp["paths"]["sam_face_cache"], enabled=True)

    # Phase A — select frames with baseline dets
    sel_path = out_dir / "selected_frames.json"
    if sel_path.is_file():
        selected = [
            (d["image_id"], d["archive"], d["path"]) for d in json.loads(sel_path.read_text())
        ]
        log.info("reuse selected %s", [s[0] for s in selected])
    else:
        seg = build_segmenter(
            "persam",
            ROOT / exp["paths"]["models_config"],
            repo_root=ROOT,
            tight_prompt_crop=False,
            multi_instance=True,
            feature_ensemble=True,
            sam_embedding_cache=cache,
        )
        seg.prepare(refs)
        need = dict(abl["n_frames_per_walk"])
        got: dict[str, int] = defaultdict(int)
        selected = []
        for image_id, archive, rel in candidates:
            if got[archive] >= need.get(archive, 0):
                continue
            rgb = load_rgb_exif(extract / archive / rel)
            dets, _ = run_one(
                seg, rgb, image_id=image_id, model="persam", face_size=face_size,
                margin=baseline_m, yaw=0.0, models_yaml=models_yaml,
                area_warn=area_warn, merge_deg=merge_deg,
            )
            log.info("screen %s ndet=%d", image_id, len(dets))
            if dets:
                selected.append((image_id, archive, rel))
                got[archive] += 1
            if all(got[a] >= need.get(a, 0) for a in need):
                break
        seg.close()
        sel_path.write_text(json.dumps(
            [{"image_id": a, "archive": b, "path": c} for a, b, c in selected], indent=2
        ))
    if not selected:
        log.error("no frames selected")
        return 1

    # Face px/deg at each margin (face_size fixed → FOV grows)
    face_px = {}
    for m in margins:
        fov = 90.0 + 2.0 * m
        face_px[str(m)] = {
            "face_size": face_size,
            "effective_fov_deg": fov,
            "px_per_deg_centre_approx": face_size / fov,
        }

    summary: dict = {
        "face_size": face_size,
        "face_size_basis": abl.get("face_size_basis"),
        "face_px_per_margin": face_px,
        "margins_deg": margins,
        "yaw_offsets_deg": yaws,
        "models": models,
        "selected": [s[0] for s in selected],
        "per_model": {},
    }

    for model in models:
        log.info("=== %s ===", model)
        seg = build_segmenter(
            model,
            ROOT / exp["paths"]["models_config"],
            repo_root=ROOT,
            tight_prompt_crop=False,
            multi_instance=True,
            feature_ensemble=True,
            sam_embedding_cache=cache if model.startswith("persam") else None,
        )
        seg.prepare(refs)
        rows_out: list[dict] = []
        costs: dict[float, list[float]] = defaultdict(list)

        for image_id, archive, rel in selected:
            rgb = load_rgb_exif(extract / archive / rel)
            for yaw in yaws:
                # baseline
                key_b = f"{model}|{image_id}|yaw{yaw}|m{baseline_m}"
                if key_b in index:
                    base_world = index[key_b]["world"]
                    dt_b = float(index[key_b]["runtime_s"])
                else:
                    dets, dt_b = run_one(
                        seg, rgb, image_id=image_id, model=model, face_size=face_size,
                        margin=baseline_m, yaw=yaw, models_yaml=models_yaml,
                        area_warn=area_warn, merge_deg=merge_deg,
                    )
                    base_world = world_dets(dets)
                    rec = {
                        "key": key_b, "model": model, "image_id": image_id,
                        "yaw": yaw, "margin": baseline_m, "n_det": len(dets),
                        "runtime_s": dt_b, "world": base_world,
                    }
                    with results_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    index[key_b] = rec
                costs[baseline_m].append(dt_b)

                for margin in margins:
                    key = f"{model}|{image_id}|yaw{yaw}|m{margin}"
                    if abs(margin - baseline_m) < 1e-9:
                        stats = match_counts(base_world, base_world, match_deg)
                        stats.update({
                            "model": model, "image_id": image_id, "yaw": yaw,
                            "margin": margin, "runtime_s": dt_b, "is_baseline": True,
                        })
                        rows_out.append(stats)
                        continue
                    if key in index:
                        other_world = index[key]["world"]
                        dt = float(index[key]["runtime_s"])
                    else:
                        dets, dt = run_one(
                            seg, rgb, image_id=image_id, model=model, face_size=face_size,
                            margin=margin, yaw=yaw, models_yaml=models_yaml,
                            area_warn=area_warn, merge_deg=merge_deg,
                        )
                        other_world = world_dets(dets)
                        rec = {
                            "key": key, "model": model, "image_id": image_id,
                            "yaw": yaw, "margin": margin, "n_det": len(dets),
                            "runtime_s": dt, "world": other_world,
                        }
                        with results_path.open("a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        index[key] = rec
                    costs[margin].append(dt)
                    stats = match_counts(base_world, other_world, match_deg)
                    stats.update({
                        "model": model, "image_id": image_id, "yaw": yaw,
                        "margin": margin, "runtime_s": dt, "is_baseline": False,
                    })
                    rows_out.append(stats)
                    log.info(
                        "%s %s yaw=%g m=%g ret=%.2f lost=%d extra=%d %.1fs",
                        model, image_id, yaw, margin,
                        stats["retention_frac"], stats["lost"], stats["extra"], dt,
                    )

        by_m: dict[str, dict] = {}
        for m in margins:
            rs = [r for r in rows_out if abs(float(r["margin"]) - m) < 1e-9 and not r.get("is_baseline")]
            if abs(m - baseline_m) < 1e-9:
                by_m[str(m)] = {
                    "mean_retention_frac": 1.0,
                    "mean_runtime_s": float(np.mean(costs[m])) if costs[m] else None,
                    "is_baseline": True,
                    "n": len(costs[m]),
                    "px_per_deg": face_px[str(m)],
                }
            else:
                by_m[str(m)] = {
                    "mean_retention_frac": float(np.mean([r["retention_frac"] for r in rs])) if rs else 0.0,
                    "mean_lost": float(np.mean([r["lost"] for r in rs])) if rs else 0.0,
                    "mean_extra": float(np.mean([r["extra"] for r in rs])) if rs else 0.0,
                    "mean_pos_err_deg": float(np.nanmean([
                        r["mean_pos_err_deg"] for r in rs if r["mean_pos_err_deg"] is not None
                    ])) if rs else None,
                    "mean_runtime_s": float(np.mean(costs[m])) if costs[m] else None,
                    "n": len(rs),
                    "px_per_deg": face_px[str(m)],
                }
        summary["per_model"][model] = {"by_margin": by_m, "n_rows": len(rows_out)}
        (out_dir / "summary_partial.json").write_text(json.dumps(summary, indent=2))
        seg.close()

    # Decision
    per_model_chosen = {}
    for model, blob in summary["per_model"].items():
        chosen = baseline_m
        for m in sorted(margins):
            if abs(m - baseline_m) < 1e-9:
                continue
            v = blob["by_margin"].get(str(m), {})
            if v.get("mean_retention_frac", 0) >= 0.95:
                chosen = m
                break
        per_model_chosen[model] = chosen
    global_m = max(per_model_chosen.values()) if per_model_chosen else baseline_m
    summary["decision"] = {
        "chosen_margin_deg": global_m,
        "per_model_chosen": per_model_chosen,
        "rule": (
            "smallest margin in {0,5,15} with mean retention≥0.95 vs margin=15 baseline "
            "across yaw×frames; global=max(models)"
        ),
        "face_size": face_size,
        "face_size_basis": "rationale_sam_native",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    dec_path = ROOT / "outputs/pilot_cubemap/ablation_decision.json"
    prev = json.loads(dec_path.read_text()) if dec_path.is_file() else {}
    prev.update({
        "chosen_margin_deg": global_m,
        "chosen_face_size": face_size,
        "face_size_basis": "rationale_sam_native",
        "reason_margin": summary["decision"]["rule"] + f" → {global_m}",
        "reason_face_size": "1024 = SAM-native (rationale-based until labels)",
        "yaw_sweep": summary["decision"],
        "yaw_sweep_path": str(out_dir / "summary.json"),
    })
    dec_path.write_text(json.dumps(prev, indent=2))
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
