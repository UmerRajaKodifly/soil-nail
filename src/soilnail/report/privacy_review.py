"""Figure review ledger for PDF builds (follow-up 2 item 6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soilnail.config import load_yaml
from soilnail.privacy import PRIVACY_NOTE, redact_equirect_for_display, redact_nadir_face


def load_privacy_config(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def record_redaction(
    config_path: Path,
    *,
    figure_id: str,
    path: str,
    reason: str,
    method: str,
) -> None:
    cfg = load_privacy_config(config_path)
    entries = list(cfg.get("figure_review", {}).get("redacted_figures", []))
    entries.append(
        {
            "figure_id": figure_id,
            "path": path,
            "reason": reason,
            "method": method,
        }
    )
    # Persist sidecar ledger (do not rewrite YAML comments casually)
    ledger = config_path.with_suffix(".redactions.json")
    ledger.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def prepare_figure_rgb(
    rgb,
    *,
    kind: str,
    face_id: str | None = None,
    config_path: Path | None = None,
):
    """Apply display-time redaction. kind: equirect | face | other."""
    if kind == "equirect":
        return redact_equirect_for_display(rgb)
    if kind == "face" and face_id is not None:
        return redact_nadir_face(rgb, face_id)
    return rgb


def privacy_note_text(config_path: Path | None = None) -> str:
    if config_path and config_path.is_file():
        cfg = load_privacy_config(config_path)
        return str(cfg.get("privacy_note", PRIVACY_NOTE))
    return PRIVACY_NOTE
