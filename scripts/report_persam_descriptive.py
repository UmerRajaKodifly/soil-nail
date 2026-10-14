#!/usr/bin/env python3
"""PersAM-only descriptive report + privacy-redacted gallery overlays.

Unlabelled: no precision/recall. C55 and F159 reported side-by-side (never pooled
for transfer claims).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from soilnail.config import load_yaml  # noqa: E402
from soilnail.privacy import PRIVACY_NOTE, redact_equirect_for_display  # noqa: E402
from soilnail.store import ResultsStore  # noqa: E402

COLOUR = (0, 114, 178)  # Wong blue
RUN_DEFAULT = "outputs/runs/full_cubemap_20261008T073139Z_f6edda62"
MODEL = "persam"


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "p5": _pct(xs, 5),
        "p25": _pct(xs, 25),
        "p50": _pct(xs, 50),
        "p75": _pct(xs, 75),
        "p95": _pct(xs, 95),
        "max": float(a.max()),
        "sum": float(a.sum()),
    }


def _walk_meta(walks_cfg: dict) -> dict[str, dict]:
    return {w["archive"]: w for w in walks_cfg["walks"]}


def _draw_overlay(rgb: np.ndarray, detections: list, *, max_edge: int = 1600) -> Image.Image:
    rgb = redact_equirect_for_display(rgb)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_edge / float(w))
    dw, dh = int(w * scale), int(h * scale)
    im = Image.fromarray(rgb).resize((dw, dh), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(im, "RGBA")
    for d in detections:
        polys = d.get("equirect_polygons_xy") or (
            [d["equirect_polygon_xy"]] if d.get("equirect_polygon_xy") else []
        )
        for ring in polys:
            if not ring or len(ring) < 3:
                continue
            pts = [(float(x) * scale, float(y) * scale) for x, y in ring]
            draw.polygon(pts, outline=(*COLOUR, 255), fill=(*COLOUR, 45))
    return im


def _load_rgb(extract: Path, image_id: str, manifest: Path) -> np.ndarray | None:
    con = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    row = con.execute(
        "SELECT archive, relative_path FROM images WHERE image_id=?", (image_id,)
    ).fetchone()
    con.close()
    if row is None:
        return None
    p = extract / row[0] / row[1]
    if not p.is_file():
        return None
    return np.asarray(Image.open(p).convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gallery-per-walk", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_id = json.loads((run_dir / "run_meta.json").read_text())["run_id"]
    meta = json.loads((run_dir / "run_meta.json").read_text())
    walks_cfg = load_yaml(ROOT / "configs/walks.yaml")
    arch_meta = _walk_meta(walks_cfg)
    extract = ROOT / "data/site_walk_extracted"
    manifest = ROOT / "data/manifests/dataset.sqlite"

    out = ROOT / "outputs" / "reports" / f"persam_only_{run_id}"
    gal = out / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    gal.mkdir(exist_ok=True)

    store = ResultsStore(run_dir / "results.sqlite")
    image_ids = store.list_image_ids(run_id, model=MODEL)
    assert len(image_ids) == 591, f"expected 591 PersAM rows, got {len(image_ids)}"

    per_image: list[dict] = []
    by_walk: dict[str, list[dict]] = defaultdict(list)

    for iid in image_ids:
        res = store.get_image_result(run_id, iid, model=MODEL)
        assert res is not None
        arch = iid.split("::", 1)[0]
        w = arch_meta[arch]
        dets = [d.model_dump() for d in res.detections]
        faces = Counter(d.get("face_id") or "?" for d in dets)
        scores = [float(d["raw_score"]) for d in dets]
        row = {
            "image_id": iid,
            "archive": arch,
            "walk_id": w["walk_id"],
            "site_id": w["site_id"],
            "role": w["role"],
            "n_detections": len(dets),
            "n_area_flag": sum(1 for d in dets if d.get("area_flag")),
            "n_merged_faces": sum(1 for d in dets if d.get("merged_across_faces")),
            "n_touches_border": sum(1 for d in dets if d.get("touches_border")),
            "runtime_s": res.runtime_s,
            "face_counts": dict(faces),
            "score_mean": float(np.mean(scores)) if scores else None,
            "score_max": float(np.max(scores)) if scores else None,
        }
        per_image.append(row)
        by_walk[w["walk_id"]].append(row)

    def walk_summary(rows: list[dict]) -> dict:
        counts = [r["n_detections"] for r in rows]
        rts = [float(r["runtime_s"]) for r in rows if r["runtime_s"] is not None]
        face_tot: Counter[str] = Counter()
        for r in rows:
            face_tot.update(r["face_counts"])
        return {
            "n_images": len(rows),
            "detections_per_image": _stats([float(c) for c in counts]),
            "total_detections": int(sum(counts)),
            "runtime_s": _stats(rts),
            "zero_detection_images": int(sum(1 for c in counts if c == 0)),
            "area_flag_detections": int(sum(r["n_area_flag"] for r in rows)),
            "merged_across_faces_detections": int(sum(r["n_merged_faces"] for r in rows)),
            "face_counts": dict(face_tot),
            "images_with_any_detection": int(sum(1 for c in counts if c > 0)),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": MODEL,
        "config_hash": meta.get("config_hash"),
        "face_size": meta.get("face_size"),
        "margin_deg": meta.get("margin_deg"),
        "status": "descriptive_only_unlabelled",
        "caveat": (
            "No ground-truth labels — counts are model outputs, not accuracy. "
            "Inter-model agreement not applicable (PersAM-only). "
            "C55 = in-site vs references; F159 = cross-site (n=50, wide uncertainty)."
        ),
        "reference_site_id": walks_cfg["reference_site_id"],
        "privacy": PRIVACY_NOTE,
        "pooled_all_frames": walk_summary(per_image),
        "per_walk": {},
        "note_pooled": (
            "Pooled row is descriptive only. Do NOT use pooled numbers to claim "
            "cross-site transfer — report C55_W1 and F159_VID side by side."
        ),
    }
    for wid, rows in sorted(by_walk.items()):
        role = rows[0]["role"]
        site = rows[0]["site_id"]
        summary["per_walk"][wid] = {
            "site_id": site,
            "role": role,
            **walk_summary(rows),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_image.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_image) + "\n"
    )

    # ---- gallery: seeded extremes + random per walk ----
    rng = random.Random(args.seed)
    gallery_manifest: list[dict] = []

    def save_fig(tag: str, image_id: str, reason: str) -> Path | None:
        res = store.get_image_result(run_id, image_id, model=MODEL)
        if res is None:
            return None
        rgb = _load_rgb(extract, image_id, manifest)
        if rgb is None:
            return None
        dets = [d.model_dump() for d in res.detections]
        im = _draw_overlay(rgb, dets)
        # caption bar
        bar = Image.new("RGB", (im.width, 36), (20, 20, 20))
        draw = ImageDraw.Draw(bar)
        draw.text(
            (8, 8),
            f"{image_id} | n={len(dets)} | {reason} | PersAM | raw_score uncalibrated",
            fill=(230, 230, 230),
        )
        canvas = Image.new("RGB", (im.width, im.height + 36))
        canvas.paste(bar, (0, 0))
        canvas.paste(im, (0, 36))
        safe = image_id.replace("/", "_").replace("::", "__")
        path = gal / f"{tag}_{safe}_n{len(dets)}.jpg"
        canvas.save(path, quality=85)
        gallery_manifest.append(
            {
                "tag": tag,
                "image_id": image_id,
                "n_detections": len(dets),
                "reason": reason,
                "path": str(path.relative_to(ROOT)),
                "privacy_redaction": "nadir_equirect_band",
            }
        )
        return path

    for wid, rows in sorted(by_walk.items()):
        rows_sorted = sorted(rows, key=lambda r: r["n_detections"])
        picks: list[tuple[str, str, str]] = []
        # extremes
        picks.append(("low_count", rows_sorted[0]["image_id"], f"{wid} lowest count"))
        picks.append(("high_count", rows_sorted[-1]["image_id"], f"{wid} highest count"))
        mid = rows_sorted[len(rows_sorted) // 2]
        picks.append(("median_count", mid["image_id"], f"{wid} median count"))
        # random (exclude already picked)
        used = {p[1] for p in picks}
        pool = [r["image_id"] for r in rows if r["image_id"] not in used]
        rng.shuffle(pool)
        for iid in pool[: max(0, args.gallery_per_walk - 3)]:
            picks.append(("random", iid, f"{wid} seeded random"))
        for tag, iid, reason in picks:
            save_fig(f"{wid}_{tag}", iid, reason)

    (out / "gallery_manifest.json").write_text(json.dumps(gallery_manifest, indent=2))

    # ---- privacy figure review ledger ----
    redactions = [
        {
            "figure_id": g["tag"] + "::" + g["image_id"],
            "path": g["path"],
            "reason": "default nadir equirect blur (operator often in Down/nadir)",
            "method": "gaussian_blur_nadir_fraction_0.18",
        }
        for g in gallery_manifest
    ]
    (out / "privacy_figure_review.json").write_text(
        json.dumps(
            {
                "reviewed": True,
                "note": (
                    "All gallery figures use automatic nadir blur. No automatic person "
                    "detector — side-face people are NOT guaranteed redacted. Manual "
                    "page review still required before external PDF release."
                ),
                "redacted_figures": redactions,
            },
            indent=2,
        )
    )

    # ---- Markdown report ----
    lines: list[str] = []
    lines.append(f"# PersAM descriptive report (`{run_id}`)")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_utc']}`")
    lines.append("")
    lines.append("## Scope and limits")
    lines.append("")
    lines.append(
        "- **Model:** PersAM only (PersAM-F / SegGPT not included in this report)."
    )
    lines.append(
        f"- **Config:** face_size={meta.get('face_size')}, margin_deg={meta.get('margin_deg')}, "
        f"config_hash=`{meta.get('config_hash')}`."
    )
    lines.append(
        "- **Unlabelled:** detection counts are **descriptive**, not accuracy. "
        "No precision/recall/F1."
    )
    lines.append(
        f"- **References:** site `{walks_cfg['reference_site_id']}` QA photos "
        "(not walk frames — no pixel leakage)."
    )
    lines.append(
        "- **C55_W1** = in-site (same site as refs). **F159_VID** = cross-site "
        "(n=50 → wide uncertainty). Do not pool them to claim transfer."
    )
    lines.append(f"- **Privacy:** {PRIVACY_NOTE}")
    lines.append("")
    lines.append("## Per-walk detection counts")
    lines.append("")
    lines.append(
        "| Walk | Site | Role | n images | total dets | mean/img | p50 | p95 | max | zero-det images | mean runtime s |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for wid, blob in summary["per_walk"].items():
        d = blob["detections_per_image"]
        rt = blob["runtime_s"]
        lines.append(
            f"| `{wid}` | {blob['site_id']} | {blob['role']} | {blob['n_images']} | "
            f"{blob['total_detections']} | {d['mean']:.2f} | {d['p50']:.0f} | "
            f"{d['p95']:.0f} | {d['max']:.0f} | {blob['zero_detection_images']} | "
            f"{rt.get('mean', float('nan')):.2f} |"
        )
    pooled = summary["pooled_all_frames"]["detections_per_image"]
    lines.append("")
    lines.append(
        f"**Pooled (descriptive only, not for transfer):** "
        f"n={summary['pooled_all_frames']['n_images']}, "
        f"total={summary['pooled_all_frames']['total_detections']}, "
        f"mean/img={pooled['mean']:.2f}."
    )
    lines.append("")
    lines.append("## Face distribution (detection `face_id`)")
    lines.append("")
    for wid, blob in summary["per_walk"].items():
        lines.append(f"- **{wid}:** `{blob['face_counts']}`")
    lines.append("")
    lines.append("## Gallery (seeded)")
    lines.append("")
    lines.append(
        f"Seed={args.seed}. Low / median / high count plus random frames per walk. "
        "Polygons in Wong blue; nadir band blurred."
    )
    lines.append("")
    for g in gallery_manifest:
        rel_img = Path(g["path"])
        if rel_img.is_absolute():
            rel_img = rel_img.relative_to(out)
        else:
            rel_img = (ROOT / rel_img).relative_to(out)
        lines.append(
            f"- **{g['tag']}** (`n={g['n_detections']}`): `{g['image_id']}` — "
            f"![]({rel_img.as_posix()})"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{out.relative_to(ROOT)}/summary.json`")
    lines.append(f"- `{out.relative_to(ROOT)}/per_image.jsonl`")
    lines.append(f"- `{out.relative_to(ROOT)}/gallery/`")
    lines.append(f"- `{out.relative_to(ROOT)}/privacy_figure_review.json`")
    lines.append("")

    md_path = out / "PERSAM_REPORT.md"
    md_path.write_text("\n".join(lines))

    # ---- HTML index ----
    html_cards = []
    for g in gallery_manifest:
        p = Path(g["path"])
        rel = (p if p.is_absolute() else ROOT / p).relative_to(out).as_posix()
        html_cards.append(
            f'<figure><img src="{rel}" loading="lazy" style="max-width:100%"/>'
            f"<figcaption>{g['tag']} · {g['image_id']} · n={g['n_detections']}</figcaption>"
            f"</figure>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>PersAM report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#111;color:#eee}}
table{{border-collapse:collapse}} td,th{{border:1px solid #444;padding:.4rem .6rem}}
figure{{margin:1rem 0}} figcaption{{opacity:.8;font-size:.9rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1rem}}
.warn{{background:#332;padding:1rem;border-left:4px solid #c80}}
</style></head><body>
<h1>PersAM descriptive report</h1>
<p class="warn">{summary['caveat']}</p>
<p>Run <code>{run_id}</code> · config_hash <code>{meta.get('config_hash')}</code> ·
face_size={meta.get('face_size')} · margin_deg={meta.get('margin_deg')}</p>
<pre>{json.dumps({k: summary['per_walk'][k] for k in summary['per_walk']}, indent=2)}</pre>
<h2>Gallery</h2>
<div class="grid">{"".join(html_cards)}</div>
<p><small>{PRIVACY_NOTE}</small></p>
</body></html>
"""
    (out / "index.html").write_text(html)

    store.close()
    print(json.dumps({"out": str(out), "n_gallery": len(gallery_manifest), "summary_paths": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
