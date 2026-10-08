"""End-to-end ingest: validate/extract zips, audit images, profile CSVs, write manifest."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from soilnail.config import load_yaml
from soilnail.hashutil import sha256_file
from soilnail.ingest.csv_profile import profile_csv
from soilnail.ingest.images import audit_image
from soilnail.ingest.models import CsvProfile, ImageRecord, IngestConfig
from soilnail.ingest.safe_zip import safe_extract
from soilnail.ingest.stamp import probe_date_stamp

log = logging.getLogger(__name__)


def load_ingest_config(path: Path) -> IngestConfig:
    return IngestConfig.model_validate(load_yaml(path))


def _stable_image_id(archive: str, relative_path: str) -> str:
    # Include archive so duplicate filenames across zips never collide
    return f"{archive}::{relative_path.replace(chr(92), '/')}"


def run_ingest(cfg: IngestConfig, repo_root: Path) -> Path:
    raw_dir = (repo_root / cfg.paths.raw_dir).resolve()
    extract_root = (repo_root / cfg.paths.extract_root).resolve()
    manifest_dir = (repo_root / cfg.paths.manifest_dir).resolve()
    audit_md = (repo_root / cfg.paths.audit_md).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    all_records: list[ImageRecord] = []
    csv_profiles: list[CsvProfile] = []
    archive_summaries: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}

    for arch in cfg.archives:
        zip_path = raw_dir / arch.zip_filename
        csv_path = raw_dir / arch.csv_filename
        if not zip_path.is_file():
            raise FileNotFoundError(f"Missing zip: {zip_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing csv: {csv_path}")
        input_hashes[arch.zip_filename] = sha256_file(zip_path)
        input_hashes[arch.csv_filename] = sha256_file(csv_path)

        dest = extract_root / arch.name
        log.info("Extracting %s → %s", zip_path.name, dest)
        members = safe_extract(zip_path, dest, cfg.zip_limits)

        allowed = {e.lower() for e in cfg.image.allowed_extensions}
        image_members = []
        non_image = 0
        for m in members:
            ext = Path(m.archive_path).suffix.lower()
            if ext in allowed:
                image_members.append(m)
            else:
                non_image += 1

        records: list[ImageRecord] = []
        for m in image_members:
            rel = m.archive_path
            fname = Path(rel).name
            aud = audit_image(
                m.dest_path,
                max_pixels=cfg.image.max_pixels,
                apply_exif_transpose=cfg.image.apply_exif_transpose,
            )
            rec = ImageRecord(
                image_id=_stable_image_id(arch.name, rel),
                archive=arch.name,
                relative_path=rel,
                filename=fname,
                sha256=m.sha256,
                bytes=m.file_size,
                status=aud.status,
                failure_stage=aud.failure_stage,
                failure_type=aud.failure_type,
                failure_message=aud.failure_message,
                orig_width=aud.orig_width,
                orig_height=aud.orig_height,
                effective_width=aud.effective_width,
                effective_height=aud.effective_height,
                mode=aud.mode,
                format=aud.format,
                exif_orientation=aud.exif_orientation,
                transposed=aud.transposed,
            )
            records.append(rec)

        filenames = {r.filename for r in records if r.status == "ok"}
        csv_prof = profile_csv(
            csv_path, archive=arch.name, image_filenames=filenames, join_key="image_name"
        )
        csv_profiles.append(csv_prof)
        from soilnail.ingest.csv_profile import load_csv_rows

        _, rows = load_csv_rows(csv_path)
        csv_names = {str(r["image_name"]) for r in rows}
        for r in records:
            if r.filename in csv_names:
                r.csv_matched = True

        ok = sum(1 for r in records if r.status == "ok")
        failed = sum(1 for r in records if r.status == "failed")
        skipped = sum(1 for r in records if r.status == "skipped")
        if ok + failed + skipped != len(records):
            raise RuntimeError(
                f"reconciliation failed for {arch.name}: {ok}+{failed}+{skipped} != {len(records)}"
            )

        archive_summaries.append(
            {
                "name": arch.name,
                "zip": arch.zip_filename,
                "zip_sha256": input_hashes[arch.zip_filename],
                "csv": arch.csv_filename,
                "csv_sha256": input_hashes[arch.csv_filename],
                "members_extracted": len(members),
                "images": len(records),
                "non_image_files": non_image,
                "ok": ok,
                "failed": failed,
                "skipped": skipped,
                "csv_profile": csv_prof.model_dump(),
            }
        )
        all_records.extend(records)

    # Cross-archive duplicate filenames / byte-identical hashes
    name_map: dict[str, list[str]] = defaultdict(list)
    hash_map: dict[str, list[str]] = defaultdict(list)
    for r in all_records:
        name_map[r.filename].append(r.image_id)
        hash_map[r.sha256].append(r.image_id)
    dup_names = {k: v for k, v in name_map.items() if len(v) > 1}
    dup_hashes = {k: v for k, v in hash_map.items() if len(v) > 1}

    # Stamp probe across all ok images
    ok_paths = []
    for r in all_records:
        if r.status != "ok":
            continue
        ok_paths.append(extract_root / r.archive / r.relative_path)
    stamp = probe_date_stamp(
        ok_paths,
        sample_size=cfg.stamp_probe.sample_size,
        corner_frac_w=cfg.stamp_probe.corner_frac_w,
        corner_frac_h=cfg.stamp_probe.corner_frac_h,
        sat_delta_min=cfg.stamp_probe.sat_delta_min,
        sat_abs_min=cfg.stamp_probe.sat_abs_min,
        likely_fraction_threshold=cfg.stamp_probe.likely_fraction_threshold,
        seed=cfg.seeds.global_seed,
    )

    # Resolutions
    res_counter: Counter[str] = Counter()
    for r in all_records:
        if r.effective_width and r.effective_height:
            res_counter[f"{r.effective_width}x{r.effective_height}"] += 1

    total_ok = sum(1 for r in all_records if r.status == "ok")
    total_failed = sum(1 for r in all_records if r.status == "failed")
    total_skipped = sum(1 for r in all_records if r.status == "skipped")
    if total_ok + total_failed + total_skipped != len(all_records):
        raise RuntimeError("global reconciliation failed")

    # SQLite manifest
    db_path = manifest_dir / "dataset.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE images (
              image_id TEXT PRIMARY KEY,
              archive TEXT NOT NULL,
              relative_path TEXT NOT NULL,
              filename TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              bytes INTEGER NOT NULL,
              status TEXT NOT NULL,
              failure_stage TEXT,
              failure_type TEXT,
              failure_message TEXT,
              orig_width INTEGER,
              orig_height INTEGER,
              effective_width INTEGER,
              effective_height INTEGER,
              mode TEXT,
              format TEXT,
              exif_orientation INTEGER,
              transposed INTEGER NOT NULL,
              csv_matched INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany(
            """
            INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r.image_id,
                    r.archive,
                    r.relative_path,
                    r.filename,
                    r.sha256,
                    r.bytes,
                    r.status,
                    r.failure_stage,
                    r.failure_type,
                    r.failure_message,
                    r.orig_width,
                    r.orig_height,
                    r.effective_width,
                    r.effective_height,
                    r.mode,
                    r.format,
                    r.exif_orientation,
                    int(r.transposed),
                    int(r.csv_matched),
                )
                for r in all_records
            ],
        )
        meta = {
            "schema_version": "1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_images": str(len(all_records)),
            "n_ok": str(total_ok),
            "n_failed": str(total_failed),
            "n_skipped": str(total_skipped),
            "stamp_recommendation": stamp.recommendation,
        }
        conn.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
        conn.commit()
    finally:
        conn.close()

    # JSON sidecar summary
    summary = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hashes,
        "archives": archive_summaries,
        "totals": {
            "images": len(all_records),
            "ok": total_ok,
            "failed": total_failed,
            "skipped": total_skipped,
        },
        "duplicate_filenames_across_archives": {k: v for k, v in list(dup_names.items())[:20]},
        "n_duplicate_filename_groups": len(dup_names),
        "n_byte_identical_groups": len(dup_hashes),
        "resolutions": dict(res_counter.most_common()),
        "stamp_probe": {
            "sampled": stamp.sampled,
            "likely_stamp_count": stamp.likely_stamp_count,
            "mean_corner_saturation": stamp.mean_corner_saturation,
            "mean_full_saturation": stamp.mean_full_saturation,
            "recommendation": stamp.recommendation,
            "evidence": stamp.evidence,
        },
        "sqlite": str(db_path.relative_to(repo_root)),
    }
    summary_path = manifest_dir / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_audit_md(audit_md, summary, csv_profiles, stamp)
    log.info("Ingest complete: %s images (%s ok)", len(all_records), total_ok)
    return db_path


def _write_audit_md(
    path: Path,
    summary: dict[str, object],
    csv_profiles: list[CsvProfile],
    stamp: object,
) -> None:
    del stamp  # included inside summary["stamp_probe"]
    created = str(summary["created_utc"])
    input_sha256 = summary["input_sha256"]
    archives = summary["archives"]
    totals = summary["totals"]
    resolutions = summary["resolutions"]
    stamp_probe = summary["stamp_probe"]
    if not isinstance(input_sha256, dict):
        raise TypeError("input_sha256")
    if not isinstance(archives, list):
        raise TypeError("archives")
    if not isinstance(totals, dict):
        raise TypeError("totals")
    if not isinstance(resolutions, dict):
        raise TypeError("resolutions")
    if not isinstance(stamp_probe, dict):
        raise TypeError("stamp_probe")

    lines: list[str] = []
    lines.append("# Data audit (Phase 1)")
    lines.append("")
    lines.append(f"**Created (UTC):** {created}")
    lines.append("")
    lines.append("## Inputs (SHA-256)")
    lines.append("")
    for k, v in input_sha256.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Per-archive counts")
    lines.append("")
    lines.append(
        "| Archive | Zip members | Images | OK | Failed | Skipped | Non-image | "
        "CSV match (rows↔ok images) |"
    )
    lines.append(
        "|---------|-------------|--------|----|--------|---------|-----------|"
        "------------------------------|"
    )
    for a in archives:
        if not isinstance(a, dict):
            raise TypeError("archive summary entry")
        cp = a["csv_profile"]
        if not isinstance(cp, dict):
            raise TypeError("csv_profile")
        lines.append(
            f"| {a['name']} | {a['members_extracted']} | {a['images']} | {a['ok']} | "
            f"{a['failed']} | {a['skipped']} | {a['non_image_files']} | "
            f"matched={cp['rows_matched_to_images']}, "
            f"row_miss={cp['rows_unmatched']}, img_miss={cp['images_unmatched']} |"
        )
    lines.append("")
    lines.append(
        f"**Totals:** images={totals['images']}, ok={totals['ok']}, "
        f"failed={totals['failed']}, skipped={totals['skipped']} "
        f"(reconciles: {totals['ok']}+{totals['failed']}+{totals['skipped']}="
        f"{totals['images']})."
    )
    lines.append("")
    lines.append("## Resolutions")
    lines.append("")
    for res, n in resolutions.items():
        lines.append(f"- `{res}`: {n}")
    lines.append("")
    lines.append("## Duplicates")
    lines.append("")
    lines.append(
        f"- Duplicate filename groups across archives: "
        f"**{summary['n_duplicate_filename_groups']}** "
        f"(stable `image_id` = `archive::relative_path` prevents collisions)."
    )
    lines.append(f"- Byte-identical hash groups: **{summary['n_byte_identical_groups']}**")
    lines.append("")
    lines.append("## CSV profiles")
    lines.append("")
    for cp_obj in csv_profiles:
        d = cp_obj.model_dump()
        lines.append(f"### `{d['archive']}` — `{Path(str(d['path'])).name}`")
        lines.append("")
        lines.append(
            f"- encoding={d['encoding']}, delimiter=`{d['delimiter']}`, BOM={d['has_bom']}"
        )
        lines.append(f"- rows={d['n_rows']}, cols={d['n_cols']}: `{d['columns']}`")
        lines.append(f"- duplicate full rows: {d['duplicate_row_count']}")
        lines.append(f"- null rates: `{d['null_rates']}`")
        lines.append(
            f"- **coordinate_kind:** `{d['coordinate_kind']}` — {d['coordinate_rationale']}"
        )
        lines.append(
            f"- **join:** exact on `{d['join_key']}` → matched={d['rows_matched_to_images']}, "
            f"unmatched_rows={d['rows_unmatched']}, unmatched_images={d['images_unmatched']}"
        )
        if d["unmatched_row_keys"] or d["unmatched_image_names"]:
            lines.append(f"  - unmatched row keys (≤50): {d['unmatched_row_keys']}")
            lines.append(f"  - unmatched image names (≤50): {d['unmatched_image_names']}")
        lines.append("")
    lines.append("## Burned-in date stamp probe")
    lines.append("")
    lines.append(f"- sampled: {stamp_probe['sampled']}")
    lines.append(f"- likely stamp-like corners: {stamp_probe['likely_stamp_count']}")
    lines.append(
        f"- mean saturation corner/full: {float(stamp_probe['mean_corner_saturation']):.3f} / "
        f"{float(stamp_probe['mean_full_saturation']):.3f}"
    )
    lines.append(f"- **recommendation:** `{stamp_probe['recommendation']}`")
    lines.append(f"- evidence: {stamp_probe['evidence']}")
    lines.append("")
    lines.append("## Privacy")
    lines.append("")
    lines.append(
        "Site-walk photos may contain people. Data stays under "
        "`data/site_walk_*` (gitignored). No uploads."
    )
    lines.append("")
    lines.append("## Manifest")
    lines.append("")
    lines.append(f"- SQLite: `{summary['sqlite']}`")
    lines.append("- JSON summary: `data/manifests/ingest_summary.json`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
