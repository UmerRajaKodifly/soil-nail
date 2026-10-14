"""Streamlit review UI — overview, gallery, viewer, map, labelling."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from soilnail.config import load_yaml
from soilnail.privacy import redact_equirect_for_display
from soilnail.ui.data import (
    image_path,
    list_runs,
    load_walks,
    open_store,
    summarize_run,
)
from soilnail.ui.labels import FP_TYPES, STAGE_TAGS, LabelStore, build_blocked_queue


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ui.yaml")
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model filter (overrides configs/ui.yaml models_filter).",
    )
    # Streamlit may inject its own args after --
    return p.parse_known_args()[0]


def _model_filter(cfg: dict[str, Any], args: argparse.Namespace) -> list[str] | None:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    filt = cfg.get("models_filter") or []
    return list(filt) if filt else None


def _filter_models(available: list[str], allowed: list[str] | None) -> list[str]:
    if not allowed:
        return available
    keep = [m for m in available if m in allowed]
    return keep if keep else allowed


@st.cache_resource
def _cfg(path: str) -> dict[str, Any]:
    root = _repo_root()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return load_yaml(p)


def _load_wgs84_csv(root: Path) -> dict[str, tuple[float, float]]:
    """filename → (lat, lon) for Walk 1 (C55)."""
    path = root / "data" / "site_walk_raw" / "Walk 1.csv"
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["image_name"]] = (float(row["latitude"]), float(row["longitude"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _thumb(rgb: np.ndarray, max_edge: int) -> Image.Image:
    im = Image.fromarray(rgb)
    im.thumbnail((max_edge, max_edge))
    return im


def _draw_polygons(
    im: Image.Image,
    polygons: list[list[list[float]]] | None,
    colour: str,
    scale: float,
    *,
    label: str | None = None,
) -> Image.Image:
    if not polygons:
        return im
    out = im.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    # parse hex
    r = int(colour[1:3], 16)
    g = int(colour[3:5], 16)
    b = int(colour[5:7], 16)
    cx = cy = None
    for ring in polygons:
        if len(ring) < 3:
            continue
        pts = [(float(x) * scale, float(y) * scale) for x, y in ring]
        draw.polygon(pts, outline=(r, g, b, 255), fill=(r, g, b, 40))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    if label is not None and cx is not None and cy is not None:
        draw.text((cx + 4, cy + 4), label, fill=(255, 255, 255, 255))
    return out


def _detection_table(detections: list[Any], *, model: str) -> None:
    """Always-visible score table (no expander)."""
    if not detections:
        st.info("No detections on this image.")
        return
    rows = []
    for d in detections:
        rows.append(
            {
                "model": model,
                "id": d.instance_id,
                "face": d.face_id,
                "raw_score": round(float(d.raw_score), 4),
                "area_px": d.area_px,
                "area_flag": d.area_flag,
                "merged_faces": d.merged_across_faces,
            }
        )
    st.markdown("**Detections** — `raw_score` is uncalibrated (not confidence %)")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def page_overview(cfg: dict[str, Any], root: Path) -> None:
    st.header("Run overview")
    runs = list_runs(root / cfg["runs_root"])
    if not runs:
        st.warning("No runs under outputs/runs yet.")
        return
    for r in runs:
        with st.expander(
            f"{r.run_id} — {'COMPLETE' if r.complete else 'INCOMPLETE' if r.incomplete else 'in progress'}",
            expanded=(r == runs[0]),
        ):
            st.json(r.meta)
            try:
                store = open_store(r)
                summary = summarize_run(store, r.run_id)
                store.close()
                st.write(summary)
            except Exception as exc:
                st.error(f"Could not read store: {exc}")


def _pick_run(runs: list, cfg: dict[str, Any]):
    default_id = cfg.get("default_run_id")
    idx = 0
    if default_id:
        for i, r in enumerate(runs):
            if r.run_id == default_id:
                idx = i
                break
    return st.selectbox("Run", runs, index=idx, format_func=lambda r: r.run_id)


def _nav_buttons(*, widget_key: str, index_key: str, idx: int, n: int) -> None:
    """Previous / Next row; updates session_state[index_key] and reruns."""
    c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1])
    with c1:
        if st.button("← Previous", key=f"{widget_key}_prev", disabled=(idx <= 0)):
            st.session_state[index_key] = max(0, idx - 1)
            st.rerun()
    with c2:
        if st.button("Next →", key=f"{widget_key}_next", disabled=(idx >= n - 1)):
            st.session_state[index_key] = min(n - 1, idx + 1)
            st.rerun()
    with c3:
        st.markdown(f"**{idx + 1} / {n}**")
    with c4:
        if st.button("⏮ First", key=f"{widget_key}_first", disabled=(idx <= 0)):
            st.session_state[index_key] = 0
            st.rerun()
    with c5:
        if st.button("Last ⏭", key=f"{widget_key}_last", disabled=(idx >= n - 1)):
            st.session_state[index_key] = n - 1
            st.rerun()


def page_gallery(cfg: dict[str, Any], root: Path, *, models_filter: list[str] | None) -> None:
    st.header("Gallery")
    if models_filter:
        st.caption(f"Model filter: {', '.join(models_filter)}")
    runs = list_runs(root / cfg["runs_root"])
    if not runs:
        st.warning("No runs.")
        return
    run = _pick_run(runs, cfg)
    store = open_store(run)
    models = _filter_models(
        store.list_models(run.run_id) or list(cfg["colours"].keys()), models_filter
    )
    model = models[0] if len(models) == 1 else st.selectbox("Model", models)
    if len(models) == 1:
        st.caption(f"Model: **{model}**")
    ids = store.list_image_ids(run.run_id, model=model)
    archive = st.radio("Archive", ["all", "walk1", "images"], horizontal=True)
    if archive != "all":
        ids = [i for i in ids if i.startswith(archive + "::")]
    if not ids:
        st.info("No images for this filter.")
        store.close()
        return

    # Continuous browse: one image at a time with prev/next (no pages / no dropdown).
    index_key = f"gallery_idx_{run.run_id}_{model}_{archive}"
    if index_key not in st.session_state:
        jump = st.session_state.get("viewer_image_id")
        st.session_state[index_key] = ids.index(jump) if jump in ids else 0
    idx = max(0, min(int(st.session_state[index_key]), len(ids) - 1))
    st.session_state[index_key] = idx

    _nav_buttons(widget_key=f"{index_key}_top", index_key=index_key, idx=idx, n=len(ids))
    image_id = ids[idx]

    extract = root / cfg["extract_root"]
    manifest = root / cfg["manifest_sqlite"]
    res = store.get_image_result(run.run_id, image_id, model=model)
    nd = len(res.detections) if res else 0
    st.subheader(image_id)
    st.caption(f"{nd} detections · raw scores uncalibrated")

    path = image_path(extract, image_id, manifest)
    if path is None:
        st.error("Image file missing.")
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        rgb = redact_equirect_for_display(rgb)
        h, w = rgb.shape[:2]
        display_w = 1400
        scale = display_w / w
        im = Image.fromarray(rgb).resize((display_w, int(h * scale)))
        colour = cfg["colours"].get(model, "#0072B2")
        if res is not None:
            for d in res.detections:
                im = _draw_polygons(
                    im,
                    d.equirect_polygons_xy,
                    colour,
                    scale,
                    label=f"#{d.instance_id} {d.raw_score:.2f}",
                )
        st.image(im, use_container_width=True)

    if res is not None:
        _detection_table(res.detections, model=model)

    _nav_buttons(widget_key=f"{index_key}_bot", index_key=index_key, idx=idx, n=len(ids))
    store.close()


def page_viewer(
    cfg: dict[str, Any], root: Path, *, models_filter: list[str] | None
) -> None:
    st.header("Image viewer")
    if models_filter:
        st.caption(f"Model filter: {', '.join(models_filter)}")
    runs = list_runs(root / cfg["runs_root"])
    if not runs:
        st.warning("No runs.")
        return
    default_run = st.session_state.get("viewer_run_id") or cfg.get("default_run_id")
    run_idx = max(0, next((i for i, r in enumerate(runs) if r.run_id == default_run), 0))
    run = runs[run_idx]
    if len(runs) > 1:
        run = st.selectbox(
            "Run",
            runs,
            index=run_idx,
            format_func=lambda r: r.run_id,
        )
    else:
        st.caption(f"Run: `{run.run_id}`")

    store = open_store(run)
    models = _filter_models(store.list_models(run.run_id), models_filter)
    if not models:
        st.info("No model results yet.")
        store.close()
        return
    show_models = (
        models
        if len(models) == 1
        else st.multiselect("Models", models, default=models)
    )
    if not show_models:
        st.warning("Select at least one model.")
        store.close()
        return

    ids = store.list_image_ids(run.run_id, model=show_models[0])
    archive = st.radio(
        "Archive",
        ["all", "walk1", "images"],
        horizontal=True,
        key="viewer_archive",
    )
    if archive != "all":
        ids = [i for i in ids if i.startswith(archive + "::")]
    if not ids:
        st.info("No results yet for this run (still processing?).")
        store.close()
        return

    index_key = f"viewer_idx_{run.run_id}_{archive}"
    if "viewer_image_id" in st.session_state and st.session_state["viewer_image_id"] in ids:
        st.session_state[index_key] = ids.index(st.session_state["viewer_image_id"])
        del st.session_state["viewer_image_id"]
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    idx = max(0, min(int(st.session_state[index_key]), len(ids) - 1))
    st.session_state[index_key] = idx

    _nav_buttons(widget_key=f"{index_key}_top", index_key=index_key, idx=idx, n=len(ids))
    image_id = ids[idx]
    st.session_state["viewer_run_id"] = run.run_id

    st.subheader(image_id)
    path = image_path(root / cfg["extract_root"], image_id, root / cfg["manifest_sqlite"])
    if path is None:
        st.error("Image file missing.")
        store.close()
        return
    rgb = np.asarray(Image.open(path).convert("RGB"))
    rgb = redact_equirect_for_display(rgb)
    h, w = rgb.shape[:2]
    display_w = 1400
    scale = display_w / w
    im = Image.fromarray(rgb).resize((display_w, int(h * scale)))
    for m in show_models:
        res = store.get_image_result(run.run_id, image_id, model=m)
        colour = cfg["colours"].get(m, "#FFFFFF")
        if res is None:
            continue
        for d in res.detections:
            im = _draw_polygons(
                im,
                d.equirect_polygons_xy,
                colour,
                scale,
                label=f"{m[0]}#{d.instance_id} {d.raw_score:.2f}",
            )
    st.image(im, caption="Nadir band blurred for display", use_container_width=True)
    st.caption(f"Raw scores are {cfg['raw_score_label']}.")

    _nav_buttons(widget_key=f"{index_key}_bot", index_key=index_key, idx=idx, n=len(ids))

    for m in show_models:
        res = store.get_image_result(run.run_id, image_id, model=m)
        if res is None:
            st.warning(f"No result for {m} on this image yet.")
            continue
        rt = f" · {res.runtime_s:.2f}s" if res.runtime_s is not None else ""
        st.markdown(f"### {m}{rt}")
        _detection_table(res.detections, model=m)
    store.close()

def page_map(cfg: dict[str, Any], root: Path) -> None:
    st.header("Map (C55 / WGS84 only)")
    st.info(
        "F159 uses local metres without a verified geographic origin — plan-view only, "
        "not shown on this basemap. C55 is in-site relative to references; F159 is cross-site."
    )
    walks = load_walks(root / cfg["walks_config"])
    gps = _load_wgs84_csv(root)
    if not gps:
        st.warning("Walk 1.csv not found — no map points.")
        return
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.error("folium / streamlit-folium not installed.")
        return
    # centre on median
    lats = [v[0] for v in gps.values()]
    lons = [v[1] for v in gps.values()]
    m = folium.Map(location=[float(np.median(lats)), float(np.median(lons))], zoom_start=17)
    # subsample for responsiveness
    items = list(gps.items())[::5]
    for name, (lat, lon) in items:
        folium.CircleMarker(
            [lat, lon],
            radius=3,
            popup=f"C55_W1::{name}",
            color="#0072B2",
            fill=True,
        ).add_to(m)
    st.caption(f"Showing every 5th of {len(gps)} C55_W1 points (walk_id={walks['walks'][0]['walk_id']}).")
    st_folium(m, width=900, height=500)


def page_labels(cfg: dict[str, Any], root: Path) -> None:
    st.header("Review / labelling")
    st.markdown(
        "C55 = **in-site** (same site as nail_ref_01..03). F159 = **cross-site** transfer test "
        "(n=50 frames → wide intervals). Reference photos are QA photos, not walk frames "
        "(no pixel leakage); physical nails 2D12/2D16/2D17 may appear in C55 walk imagery."
    )
    walks = load_walks(root / cfg["walks_config"])
    labels_path = root / cfg["labels_sqlite"]
    store_l = LabelStore(labels_path)
    queue = store_l.list_queue()
    if st.button("Rebuild label queue from config"):
        # Prefer images from latest complete-ish run; else all manifest ok images
        runs = list_runs(root / cfg["runs_root"])
        image_ids: list[str] = []
        if runs:
            s = open_store(runs[0])
            image_ids = s.list_image_ids(runs[0].run_id)
            s.close()
        if not image_ids:
            import sqlite3

            con = sqlite3.connect(root / cfg["manifest_sqlite"])
            image_ids = [
                r[0]
                for r in con.execute(
                    "SELECT image_id FROM images WHERE status='ok' ORDER BY image_id"
                )
            ]
            con.close()
        lq = cfg["label_queue"]
        items = build_blocked_queue(
            image_ids=image_ids,
            walks_cfg=walks,
            seed=int(lq["seed"]),
            quotas={k: int(v) for k, v in lq["per_walk_quotas"].items()},
            tuning_frac=float(lq["split"]["tuning_frac"]),
            buffer_gap_frames=int(lq["split"]["buffer_gap_frames"]),
        )
        store_l.replace_queue(items)
        queue = items
        st.success(f"Queue size {len(queue)}")
    if not queue:
        st.info("Queue empty — click Rebuild.")
        store_l.close()
        return
    by_walk: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for q in queue:
        by_walk[q.walk_id] = by_walk.get(q.walk_id, 0) + 1
        by_split[q.split] = by_split.get(q.split, 0) + 1
    st.write({"per_walk": by_walk, "per_split": by_split})
    split_f = st.selectbox("Split", ["tuning", "heldout", "all"])
    opts = [q for q in queue if split_f == "all" or q.split == split_f]
    q = st.selectbox(
        "Queue item",
        opts,
        format_func=lambda x: f"{x.split} · {x.walk_id} ({x.role}) · {x.image_id}",
    )
    existing = store_l.get_image_label(q.image_id) or {
        "no_nails": False,
        "unsure": False,
        "notes": "",
        "nails": [],
    }
    reviewer = st.text_input("Reviewer id", value="reviewer")
    no_nails = st.checkbox("No nails in this image", value=bool(existing["no_nails"]))
    unsure = st.checkbox("Unsure", value=bool(existing["unsure"]))
    notes = st.text_area("Notes", value=str(existing.get("notes") or ""))
    st.subheader("GT nails (boxes in equirect px: x,y,w,h)")
    nails_text = st.text_area(
        "JSON list of {bbox_xywh, stage_tag}",
        value=json.dumps(existing.get("nails") or [], indent=2),
        height=160,
    )
    if st.button("Save image label"):
        try:
            nails = json.loads(nails_text)
            for n in nails:
                if n.get("stage_tag") not in STAGE_TAGS and n.get("stage_tag") is not None:
                    raise ValueError(f"bad stage_tag {n.get('stage_tag')}")
            store_l.save_image_label(
                image_id=q.image_id,
                reviewer_id=reviewer,
                updated_utc=datetime.now(timezone.utc).isoformat(),
                no_nails=no_nails,
                unsure=unsure,
                notes=notes,
                nails=nails,
            )
            st.success("Saved.")
        except Exception as exc:
            st.error(str(exc))
    st.subheader("Detection accept / reject")
    runs = list_runs(root / cfg["runs_root"])
    if runs:
        rs = open_store(runs[0])
        for m in rs.list_models(runs[0].run_id):
            res = rs.get_image_result(runs[0].run_id, q.image_id, model=m)
            if res is None:
                continue
            st.markdown(f"**{m}**")
            for d in res.detections[:30]:
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    decision = st.selectbox(
                        f"{m}#{d.instance_id}",
                        ["unreviewed", "accept", "reject"],
                        key=f"dec_{m}_{d.instance_id}",
                    )
                with cols[1]:
                    fp = st.selectbox(
                        "FP type",
                        ["", *FP_TYPES],
                        key=f"fp_{m}_{d.instance_id}",
                    )
                with cols[2]:
                    if st.button("Save", key=f"save_{m}_{d.instance_id}"):
                        store_l.save_detection_review(
                            image_id=q.image_id,
                            model=m,
                            instance_id=d.instance_id,
                            decision=decision,
                            fp_type=fp or None,
                            reviewer_id=reviewer,
                            updated_utc=datetime.now(timezone.utc).isoformat(),
                        )
                        st.toast("saved")
        rs.close()
    if st.button("Export labels JSON"):
        out = root / "outputs" / "labels" / "export.json"
        store_l.export_json(out)
        st.success(f"Wrote {out}")
    store_l.close()


def main() -> None:
    args = _parse_args()
    root = _repo_root()
    cfg = _cfg(args.config)
    models_filter = _model_filter(cfg, args)
    st.set_page_config(page_title="Soil-nail review", layout="wide")
    title = "Soil-nail 360° review"
    if models_filter == ["persam"]:
        title = "Soil-nail 360° review — PersAM only"
    st.title(title)
    st.caption(
        "reference_site_id=C55 · privacy: nadir blur on display only · "
        "no automatic person detector"
    )
    if models_filter:
        st.sidebar.info("Models: " + ", ".join(models_filter))
    nav = st.sidebar.radio(
        "Page",
        ["Overview", "Gallery", "Viewer", "Map", "Labels"],
        index=["Overview", "Gallery", "Viewer", "Map", "Labels"].index(
            st.session_state.get("nav", "Gallery")
        )
        if st.session_state.get("nav", "Gallery")
        in ["Overview", "Gallery", "Viewer", "Map", "Labels"]
        else 1,
    )
    st.session_state["nav"] = nav
    if nav == "Overview":
        page_overview(cfg, root)
    elif nav == "Gallery":
        page_gallery(cfg, root, models_filter=models_filter)
    elif nav == "Viewer":
        page_viewer(cfg, root, models_filter=models_filter)
    elif nav == "Map":
        page_map(cfg, root)
    else:
        page_labels(cfg, root)


if __name__ == "__main__":
    main()
