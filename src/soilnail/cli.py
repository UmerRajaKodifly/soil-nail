"""Command-line entrypoints (`python -m soilnail.cli …` / Makefile targets)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_ingest(args: argparse.Namespace) -> int:
    from soilnail.ingest.pipeline import load_ingest_config, run_ingest

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _repo_root() / cfg_path
    cfg = load_ingest_config(cfg_path)
    db = run_ingest(cfg, _repo_root())
    print(f"Ingest OK → {db}")
    return 0


def _unimplemented(name: str) -> Callable[[argparse.Namespace], int]:
    def _fn(_args: argparse.Namespace) -> int:
        raise SystemExit(f"{name}: not implemented yet (later phase)")

    return _fn


def cmd_ui(args: argparse.Namespace) -> int:
    from soilnail.ui import launch_ui

    launch_ui(args.config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soilnail")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Safe extract + data audit + manifest")
    ing.add_argument("--config", default="configs/ingest.yaml")
    ing.set_defaults(func=cmd_ingest)

    ui = sub.add_parser("ui", help="Launch local review UI (Streamlit)")
    ui.add_argument("--config", default="configs/ui.yaml")
    ui.set_defaults(func=cmd_ui)

    for name in ("pilot", "run", "report"):
        sp = sub.add_parser(name, help=f"{name} (later phase)")
        sp.add_argument("--config", required=False, default=None)
        sp.set_defaults(func=_unimplemented(name))

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
