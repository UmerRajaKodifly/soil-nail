"""Label store + blocked per-walk split for the review queue."""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FP_TYPES = (
    "scaffold",
    "whiteboard",
    "soil",
    "hose_or_pipe",
    "person_or_tool",
    "other",
)
STAGE_TAGS = ("hole_only", "bar_installed", "grouted", "unclear")
SplitName = Literal["tuning", "heldout", "unassigned"]


@dataclass(frozen=True)
class QueueItem:
    image_id: str
    walk_id: str
    site_id: str
    archive: str
    split: SplitName
    role: str  # in_site | cross_site


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS queue (
              image_id TEXT PRIMARY KEY,
              walk_id TEXT NOT NULL,
              site_id TEXT NOT NULL,
              archive TEXT NOT NULL,
              split TEXT NOT NULL,
              role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_labels (
              image_id TEXT PRIMARY KEY,
              reviewer_id TEXT NOT NULL,
              updated_utc TEXT NOT NULL,
              no_nails INTEGER NOT NULL DEFAULT 0,
              unsure INTEGER NOT NULL DEFAULT 0,
              notes TEXT,
              payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS detection_reviews (
              image_id TEXT NOT NULL,
              model TEXT NOT NULL,
              instance_id INTEGER NOT NULL,
              decision TEXT NOT NULL,
              fp_type TEXT,
              reviewer_id TEXT NOT NULL,
              updated_utc TEXT NOT NULL,
              PRIMARY KEY (image_id, model, instance_id)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def replace_queue(self, items: list[QueueItem]) -> None:
        self._conn.execute("DELETE FROM queue")
        self._conn.executemany(
            """
            INSERT INTO queue(image_id, walk_id, site_id, archive, split, role)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (i.image_id, i.walk_id, i.site_id, i.archive, i.split, i.role)
                for i in items
            ],
        )
        self._conn.commit()

    def list_queue(self, *, split: str | None = None) -> list[QueueItem]:
        if split:
            rows = self._conn.execute(
                "SELECT image_id, walk_id, site_id, archive, split, role FROM queue WHERE split=? ORDER BY image_id",
                (split,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT image_id, walk_id, site_id, archive, split, role FROM queue ORDER BY image_id"
            ).fetchall()
        return [QueueItem(*r) for r in rows]

    def save_image_label(
        self,
        *,
        image_id: str,
        reviewer_id: str,
        updated_utc: str,
        no_nails: bool,
        unsure: bool,
        notes: str,
        nails: list[dict[str, Any]],
    ) -> None:
        payload = {"nails": nails}
        self._conn.execute(
            """
            INSERT OR REPLACE INTO image_labels
            (image_id, reviewer_id, updated_utc, no_nails, unsure, notes, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                reviewer_id,
                updated_utc,
                int(no_nails),
                int(unsure),
                notes,
                json.dumps(payload),
            ),
        )
        self._conn.commit()

    def get_image_label(self, image_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT reviewer_id, updated_utc, no_nails, unsure, notes, payload_json FROM image_labels WHERE image_id=?",
            (image_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[5])
        return {
            "reviewer_id": row[0],
            "updated_utc": row[1],
            "no_nails": bool(row[2]),
            "unsure": bool(row[3]),
            "notes": row[4] or "",
            "nails": payload.get("nails", []),
        }

    def save_detection_review(
        self,
        *,
        image_id: str,
        model: str,
        instance_id: int,
        decision: str,
        fp_type: str | None,
        reviewer_id: str,
        updated_utc: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO detection_reviews
            (image_id, model, instance_id, decision, fp_type, reviewer_id, updated_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, model, instance_id, decision, fp_type, reviewer_id, updated_utc),
        )
        self._conn.commit()

    def export_json(self, path: Path) -> None:
        q = [
            {
                "image_id": i.image_id,
                "walk_id": i.walk_id,
                "site_id": i.site_id,
                "archive": i.archive,
                "split": i.split,
                "role": i.role,
            }
            for i in self.list_queue()
        ]
        labels = []
        for row in self._conn.execute(
            "SELECT image_id, reviewer_id, updated_utc, no_nails, unsure, notes, payload_json FROM image_labels"
        ):
            labels.append(
                {
                    "image_id": row[0],
                    "reviewer_id": row[1],
                    "updated_utc": row[2],
                    "no_nails": bool(row[3]),
                    "unsure": bool(row[4]),
                    "notes": row[5],
                    "payload": json.loads(row[6]),
                }
            )
        reviews = []
        for row in self._conn.execute(
            "SELECT image_id, model, instance_id, decision, fp_type, reviewer_id, updated_utc FROM detection_reviews"
        ):
            reviews.append(
                {
                    "image_id": row[0],
                    "model": row[1],
                    "instance_id": row[2],
                    "decision": row[3],
                    "fp_type": row[4],
                    "reviewer_id": row[5],
                    "updated_utc": row[6],
                }
            )
        path.write_text(
            json.dumps({"queue": q, "image_labels": labels, "detection_reviews": reviews}, indent=2),
            encoding="utf-8",
        )


def build_blocked_queue(
    *,
    image_ids: list[str],
    walks_cfg: dict[str, Any],
    seed: int,
    quotas: dict[str, int],
    tuning_frac: float,
    buffer_gap_frames: int,
) -> list[QueueItem]:
    """Sample per-walk quotas; split into contiguous blocks with a buffer gap."""
    rng = random.Random(seed)
    by_archive: dict[str, list[str]] = {}
    for iid in image_ids:
        arch = iid.split("::", 1)[0]
        by_archive.setdefault(arch, []).append(iid)
    for ids in by_archive.values():
        ids.sort()

    walk_by_arch = {w["archive"]: w for w in walks_cfg.get("walks", [])}
    items: list[QueueItem] = []

    for arch, ids in by_archive.items():
        w = walk_by_arch.get(arch)
        if w is None:
            continue
        walk_id = str(w["walk_id"])
        n_want = int(quotas.get(walk_id, 0))
        if n_want <= 0:
            continue
        n_want = min(n_want, len(ids))
        # Contiguous blocks: pick starting indices spaced apart, not independent frames.
        block_len = max(1, n_want // 4)
        starts = list(range(0, max(1, len(ids) - block_len + 1), block_len + buffer_gap_frames))
        rng.shuffle(starts)
        chosen: list[str] = []
        used_idx: set[int] = set()
        for s in starts:
            if len(chosen) >= n_want:
                break
            for k in range(block_len):
                idx = s + k
                if idx >= len(ids) or idx in used_idx:
                    continue
                # buffer: skip if neighbour already assigned from another block
                if any(abs(idx - u) <= buffer_gap_frames and u not in range(s, s + block_len) for u in used_idx):
                    continue
                chosen.append(ids[idx])
                used_idx.add(idx)
                if len(chosen) >= n_want:
                    break
        # Fill if short
        for iid in ids:
            if len(chosen) >= n_want:
                break
            if iid not in chosen:
                chosen.append(iid)

        chosen = chosen[:n_want]
        # Split chosen into contiguous tuning/heldout with buffer between.
        n_tune = max(1, int(round(len(chosen) * tuning_frac))) if len(chosen) > 1 else len(chosen)
        # Sort by frame order then cut once with buffer
        chosen_sorted = sorted(chosen)
        cut = n_tune
        # Apply buffer: drop buffer_gap frames around the cut from both sides into unassigned? Spec says
        # neighbours must not land in different splits — drop a gap from heldout/tuning boundary.
        tuning = chosen_sorted[:cut]
        heldout = chosen_sorted[cut:]
        if buffer_gap_frames > 0 and tuning and heldout:
            # remove last buffer from tuning and first buffer from heldout if they are adjacent in full list
            drop_t = tuning[-buffer_gap_frames:] if len(tuning) > buffer_gap_frames else []
            drop_h = heldout[:buffer_gap_frames] if len(heldout) > buffer_gap_frames else []
            # Only drop if they are near each other in the global sorted ids
            for x in drop_t:
                if x in tuning and len(tuning) > 1:
                    tuning.remove(x)
            for x in drop_h:
                if x in heldout and len(heldout) > 1:
                    heldout.remove(x)

        role = str(w.get("role", "unknown"))
        for iid in tuning:
            items.append(
                QueueItem(
                    image_id=iid,
                    walk_id=walk_id,
                    site_id=str(w["site_id"]),
                    archive=arch,
                    split="tuning",
                    role=role,
                )
            )
        for iid in heldout:
            items.append(
                QueueItem(
                    image_id=iid,
                    walk_id=walk_id,
                    site_id=str(w["site_id"]),
                    archive=arch,
                    split="heldout",
                    role=role,
                )
            )
    return items
