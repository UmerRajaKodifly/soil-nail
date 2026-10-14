"""Local review UI (Streamlit)."""

from __future__ import annotations

__all__ = ["launch_ui"]


def launch_ui(config_path: str | None = None) -> None:
    """Launch Streamlit UI via subprocess so `make ui` works."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    app = root / "src" / "soilnail" / "ui" / "app.py"
    cfg = config_path or "configs/ui.yaml"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--",
        "--config",
        cfg,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))
