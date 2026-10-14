"""Read-only helpers for the review UI."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soilnail.config import load_yaml
from soilnail.schemas import ImageResult
from soilnail.store import ResultsStore


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    path: Path
    meta: dict[str, Any]
    complete: bool
    incomplete: bool


def list_runs(runs_root: Path) -> list[RunInfo]:
    if not runs_root.is_dir():
        return []
    out: list[RunInfo] = []
    for d in sorted(runs_root.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "run_meta.json").is_file():
            continue
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        out.append(
            RunInfo(
                run_id=str(meta.get("run_id", d.name)),
                path=d,
                meta=meta,
                complete=(d / "COMPLETE").is_file(),
                incomplete=(d / "INCOMPLETE.json").is_file(),
            )
        )
    return out


def open_store(run: RunInfo) -> ResultsStore:
    return ResultsStore(run.path / "results.sqlite")


def load_walks(walks_cfg: Path) -> dict[str, Any]:
    return load_yaml(walks_cfg)


def archive_to_walk(walks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {w["archive"]: w for w in walks.get("walks", [])}


def image_path(extract_root: Path, image_id: str, manifest_sqlite: Path) -> Path | None:
    con = sqlite3.connect(f"file:{manifest_sqlite}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT archive, relative_path FROM images WHERE image_id=?",
            (image_id,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    p = extract_root / row[0] / row[1]
    return p if p.is_file() else None


def gps_for_image(manifest_sqlite: Path, image_id: str) -> tuple[float, float] | None:
    """Return (lat, lon) for WGS84 walk frames; None if local CRS or missing."""
    con = sqlite3.connect(f"file:{manifest_sqlite}?mode=ro", uri=True)
    try:
        # Prefer a joined csv table if present
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "csv_walk1" in tables:
            row = con.execute(
                """
                SELECT c.latitude, c.longitude
                FROM images i
                JOIN csv_walk1 c ON i.filename = c.image_name
                WHERE i.image_id=? AND i.archive='walk1'
                """,
                (image_id,),
            ).fetchone()
            if row and row[0] is not None and row[1] is not None:
                return float(row[0]), float(row[1])
        if "csv_rows" in tables:
            row = con.execute(
                """
                SELECT latitude, longitude FROM csv_rows
                WHERE image_id=? AND latitude IS NOT NULL
                """,
                (image_id,),
            ).fetchone()
            if row:
                return float(row[0]), float(row[1])
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return None


def summarize_run(store: ResultsStore, run_id: str) -> dict[str, Any]:
    models = store.list_models(run_id)
    by_model = {m: store.count_by_status(run_id, model=m) for m in models}
    n_images = len(store.list_image_ids(run_id))
    return {"models": models, "by_model": by_model, "n_distinct_images": n_images}


def load_result(
    store: ResultsStore, run_id: str, image_id: str, model: str
) -> ImageResult | None:
    return store.get_image_result(run_id, image_id, model=model)
