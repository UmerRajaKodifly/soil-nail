"""SQLite results store — atomic per-image writes, schema_versioned.

Primary key is (run_id, image_id, model) so multi-model runs do not overwrite
each other (D024).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from soilnail.schemas import Detection, Failure, ImageResult, RunManifest

SCHEMA_VERSION = "2"


class ResultsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_results (
              run_id TEXT NOT NULL,
              image_id TEXT NOT NULL,
              model TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              result_json TEXT NOT NULL,
              PRIMARY KEY (run_id, image_id, model)
            );
            INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION}');
            """
        )
        self._conn.commit()
        self._migrate_v1_to_v2()

    def _migrate_v1_to_v2(self) -> None:
        """Add model column / rebuild PK if an older single-model schema exists."""
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(image_results)").fetchall()
        }
        if "model" in cols:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
            return
        # Rebuild table with model in PK.
        self._conn.executescript(
            """
            CREATE TABLE image_results_v2 (
              run_id TEXT NOT NULL,
              image_id TEXT NOT NULL,
              model TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              result_json TEXT NOT NULL,
              PRIMARY KEY (run_id, image_id, model)
            );
            """
        )
        rows = self._conn.execute(
            "SELECT run_id, image_id, status, result_json FROM image_results"
        ).fetchall()
        for run_id, image_id, status, result_json in rows:
            model = _model_from_result_json(result_json)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO image_results_v2
                (run_id, image_id, model, status, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, image_id, model, status, result_json),
            )
        self._conn.executescript(
            """
            DROP TABLE image_results;
            ALTER TABLE image_results_v2 RENAME TO image_results;
            """
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def write_manifest(self, manifest: RunManifest) -> None:
        manifest.reconcile()
        self._conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, manifest_json) VALUES (?, ?)",
            (manifest.run_id, manifest.model_dump_json()),
        )
        self._conn.commit()

    def write_image_result(self, run_id: str, result: ImageResult) -> None:
        model = _model_from_image_result(result)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO image_results(run_id, image_id, model, status, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, result.image_id, model, result.status, result.model_dump_json()),
        )
        self._conn.commit()

    def get_image_result(
        self, run_id: str, image_id: str, *, model: str | None = None
    ) -> ImageResult | None:
        if model is None:
            cur = self._conn.execute(
                "SELECT result_json FROM image_results WHERE run_id=? AND image_id=? LIMIT 1",
                (run_id, image_id),
            )
        else:
            cur = self._conn.execute(
                "SELECT result_json FROM image_results WHERE run_id=? AND image_id=? AND model=?",
                (run_id, image_id, model),
            )
        row = cur.fetchone()
        if row is None:
            return None
        return ImageResult.model_validate_json(row[0])

    def list_models(self, run_id: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT model FROM image_results WHERE run_id=? ORDER BY model",
            (run_id,),
        )
        return [r[0] for r in cur.fetchall()]

    def list_image_ids(self, run_id: str, *, model: str | None = None) -> list[str]:
        if model is None:
            cur = self._conn.execute(
                "SELECT DISTINCT image_id FROM image_results WHERE run_id=? ORDER BY image_id",
                (run_id,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT image_id FROM image_results
                WHERE run_id=? AND model=? ORDER BY image_id
                """,
                (run_id, model),
            )
        return [r[0] for r in cur.fetchall()]

    def list_rows(
        self, run_id: str, *, model: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Return (image_id, model, status) rows."""
        if model is None:
            cur = self._conn.execute(
                """
                SELECT image_id, model, status FROM image_results
                WHERE run_id=? ORDER BY image_id, model
                """,
                (run_id,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT image_id, model, status FROM image_results
                WHERE run_id=? AND model=? ORDER BY image_id
                """,
                (run_id, model),
            )
        return [(str(a), str(b), str(c)) for a, b, c in cur.fetchall()]

    def export_detections_json(self, run_id: str, path: Path) -> None:
        cur = self._conn.execute(
            "SELECT result_json FROM image_results WHERE run_id=? ORDER BY image_id, model",
            (run_id,),
        )
        payload = [json.loads(r[0]) for r in cur.fetchall()]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def count_by_status(self, run_id: str, *, model: str | None = None) -> dict[str, int]:
        if model is None:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) FROM image_results WHERE run_id=? GROUP BY status",
                (run_id,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT status, COUNT(*) FROM image_results
                WHERE run_id=? AND model=? GROUP BY status
                """,
                (run_id, model),
            )
        return {r[0]: int(r[1]) for r in cur.fetchall()}


def _model_from_result_json(result_json: str) -> str:
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return ""
    if data.get("model"):
        return str(data["model"])
    dets = data.get("detections") or []
    if dets and isinstance(dets[0], dict) and dets[0].get("model"):
        return str(dets[0]["model"])
    fail = data.get("failure") or {}
    if isinstance(fail, dict) and fail.get("model"):
        return str(fail["model"])
    return ""


def _model_from_image_result(result: ImageResult) -> str:
    if result.model:
        return result.model
    if result.detections:
        return result.detections[0].model
    if result.failure is not None and result.failure.model:
        return result.failure.model
    return ""


def detection_from_mask(
    *,
    image_id: str,
    model: str,
    config_id: str,
    instance_id: int,
    mask: object,
    raw_score: float,
    area_warn_pct: float,
    touches_border: bool = False,
    merged_across_tiles: bool = False,
    source_tile_ids: list[str] | None = None,
) -> Detection:
    import numpy as np

    from soilnail.rle import encode_rle, mask_bbox_xywh

    m = np.asarray(mask).astype(bool)
    area = int(m.sum())
    h, w = m.shape
    frac = float(area / (h * w)) if h * w else 0.0
    return Detection(
        image_id=image_id,
        model=model,
        config_id=config_id,
        instance_id=instance_id,
        mask_rle=encode_rle(m),
        bbox_xywh=mask_bbox_xywh(m),
        area_px=area,
        area_fraction=frac,
        raw_score=float(raw_score),
        touches_border=touches_border,
        merged_across_tiles=merged_across_tiles,
        area_flag=frac * 100.0 > area_warn_pct,
        source_tile_ids=list(source_tile_ids or []),
    )


def failure_record(
    image_id: str,
    stage: str,
    exc: BaseException,
    *,
    model: str | None = None,
    config_id: str | None = None,
) -> Failure:
    return Failure(
        image_id=image_id,
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc),
        model=model,
        config_id=config_id,
    )
