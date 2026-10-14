#!/usr/bin/env python3
"""Full-dataset cubemap baseline (Gate 2 passed). Resumable; launch detached."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from soilnail.config import load_yaml  # noqa: E402
from soilnail.models.factory import build_segmenter  # noqa: E402
from soilnail.models.sam_cache import SamEmbeddingCache  # noqa: E402
from soilnail.runner import build_prepared_refs, load_rgb_exif, segment_cubemap_image  # noqa: E402
from soilnail.schemas import RunManifest  # noqa: E402
from soilnail.store import ResultsStore  # noqa: E402

log = logging.getLogger("full_cubemap")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--model", action="append", dest="models", default=None)
    ap.add_argument(
        "--resume-dir",
        default=None,
        help="Continue an incomplete run directory (keeps progress + sqlite).",
    )
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    proj = load_yaml(ROOT / cfg["paths"]["projection_config"])
    models_cfg = ROOT / cfg["paths"]["models_config"]
    models_yaml = load_yaml(models_cfg)

    models = args.models or list(cfg["baseline_full_run"]["models"])
    face_size = int(cfg["baseline_full_run"]["face_size"])
    fov = float(cfg["baseline_full_run"]["fov_deg"])
    margin = float(cfg["baseline_full_run"]["margin_deg"])
    merge_deg = float(cfg["angular_merge_deg"])
    area_warn_sr = float(proj["angular_size"]["area_flag_solid_angle_steradians"])

    conn = sqlite3.connect(ROOT / cfg["paths"]["manifest_sqlite"])
    rows = conn.execute(
        "SELECT image_id, archive, relative_path FROM images WHERE status='ok' ORDER BY image_id"
    ).fetchall()

    import hashlib

    config_hash = hashlib.sha256(
        json.dumps(
            {"experiments": cfg, "projection": proj, "face_size": face_size, "margin": margin},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        meta = json.loads((out_dir / "run_meta.json").read_text())
        run_id = meta["run_id"]
        meta["config_hash"] = config_hash
        meta["face_size"] = face_size
        meta["margin_deg"] = margin
        meta["resumed_at"] = datetime.now(timezone.utc).isoformat()
        meta["store_pk"] = "(run_id, image_id, model)"
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        incomplete = out_dir / "INCOMPLETE.json"
        if incomplete.is_file():
            incomplete.rename(out_dir / "INCOMPLETE.json.bak")
    else:
        run_id = (
            f"full_cubemap_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        out_dir = ROOT / cfg["paths"]["runs_out"] / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "models": models,
                    "face_size": face_size,
                    "margin_deg": margin,
                    "merge_deg": merge_deg,
                    "fov_deg": fov,
                    "config_hash": config_hash,
                    "equirect_polygons": True,
                    "sphere_merge": True,
                    "store_pk": "(run_id, image_id, model)",
                    "supersedes_run": "full_cubemap_20261008T073139Z_f6edda62",
                },
                indent=2,
            )
        )
    print(f"CONFIG_HASH={config_hash} face_size={face_size} margin_deg={margin} run_id={run_id}")

    store = ResultsStore(out_dir / "results.sqlite")
    sam_cache = SamEmbeddingCache(ROOT / cfg["paths"]["sam_face_cache"], enabled=True)
    ref_img = ROOT / cfg["references"]["paths"]["images_dir"]
    ref_msk = ROOT / cfg["references"]["paths"]["masks_dir"]
    extract = ROOT / cfg["paths"]["extract_root"]
    refs = build_prepared_refs(ref_img, ref_msk, list(cfg["references"]["single"]))

    progress_path = out_dir / "progress.json"
    progress: dict[str, list[str]] = {m: [] for m in models}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())

    for model in models:
        config_id = f"cubemap_{model}_f{face_size}_m{margin}_ensemble"
        done = set(progress.get(model, []))
        log.info("=== model %s (%d/%d done) ===", model, len(done), len(rows))
        seg = build_segmenter(
            model,
            models_cfg,
            repo_root=ROOT,
            tight_prompt_crop=False,
            multi_instance=True,
            feature_ensemble=True,
            sam_embedding_cache=sam_cache if model.startswith("persam") else None,
        )
        seg.prepare(refs)
        n_ok = n_fail = 0
        for image_id, archive, rel in rows:
            key = f"{config_id}::{image_id}"
            if key in done:
                continue
            rgb = load_rgb_exif(extract / archive / rel)
            res = segment_cubemap_image(
                seg,
                rgb,
                image_id=image_id,
                model=model,
                config_id=config_id,
                face_size=face_size,
                fov_deg=fov,
                margin_deg=margin,
                yaw_offset_deg=0.0,
                instance_mode=True,
                min_cc_area=int(models_yaml["seggpt_instances"]["min_cc_area_px"]),
                nms_iou=float(models_yaml["multi_instance"]["nms_iou"]),
                area_warn_solid_angle_sr=area_warn_sr,
                merge_deg=merge_deg,
            )
            store.write_image_result(run_id, res)
            if res.status == "succeeded":
                n_ok += 1
            else:
                n_fail += 1
                log.error("FAIL %s %s", model, image_id)
            done.add(key)
            progress[model] = sorted(done)
            if (n_ok + n_fail) % 10 == 0:
                progress_path.write_text(json.dumps(progress, indent=2))
                log.info(
                    "%s progress %d/%d last=%.2fs ndet=%d",
                    model,
                    len(done),
                    len(rows),
                    res.runtime_s or -1,
                    len(res.detections),
                )
        progress_path.write_text(json.dumps(progress, indent=2))
        seg.close()
        log.info("model %s finished ok=%d fail=%d", model, n_ok, n_fail)

    # Count from store / progress rather than inventing zeros
    n_done = sum(len(v) for v in progress.values())
    n_inputs = len(rows) * len(models)
    n_skip = n_inputs - n_done
    # Per-image success is already in sqlite; approximate succeeded=done for manifest.
    n_fail_total = 0
    try:
        cur = store._conn.execute(
            "SELECT COUNT(*) FROM image_results WHERE run_id=? AND status!='succeeded'",
            (run_id,),
        )
        n_fail_total = int(cur.fetchone()[0])
    except Exception:
        pass
    n_ok_total = max(0, n_done - n_fail_total)
    manifest = RunManifest(
        schema_version="2",
        run_id=run_id,
        created_utc=datetime.now(timezone.utc),
        git_commit=_git_commit(),
        config_hash=config_hash,
        config_path=str(args.config),
        seeds={"global": int(cfg["pilot"]["seed"])},
        environment={"cuda": True},
        input_file_sha256={},
        n_inputs=n_inputs,
        n_succeeded=n_ok_total,
        n_failed=n_fail_total,
        n_skipped=n_skip,
    )
    store.write_manifest(manifest)
    (out_dir / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat())
    print(json.dumps({"run_id": run_id, "out_dir": str(out_dir)}, indent=2))
    store.close()
    return 0


def _git_commit() -> str | None:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
