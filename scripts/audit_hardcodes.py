#!/usr/bin/env python3
"""Flag unexplained angles/FOV/face-size constants outside configs/."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "soilnail"

# Literals that must live in config (or docs/PROJECTION.md defaults mirrored in CubemapConfig)
PATTERNS = [
    (re.compile(r"\b(90\.0|90)\b"), "possible FOV/right-angle literal"),
    (re.compile(r"\b(960|1024|1222)\b"), "possible face_size literal"),
    (re.compile(r"\b(45\.0|45)\b"), "possible yaw-offset literal"),
]

ALLOW_FILES = {
    "projection/cubemap.py",  # FOV math + defaults documented in PROJECTION.md
    "projection/geometry.py",
    "projection/sphere_ops.py",
}


def main() -> int:
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = str(path.relative_to(SRC))
        if rel in ALLOW_FILES or rel.endswith("sam_cache.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for pat, why in PATTERNS:
                if pat.search(line):
                    hits.append(f"{rel}:{i}: {why}: {line.strip()[:120]}")
    out = ROOT / "docs" / "HARDCODE_AUDIT.md"
    body = ["# Hardcode audit (angles / FOV / face size)\n", f"Hits outside allowlist: {len(hits)}\n"]
    if hits:
        body.append("\n## Findings\n")
        body.extend(f"- `{h}`\n" for h in hits)
    else:
        body.append("\nNo unexplained literals outside allowlisted projection maths.\n")
    out.write_text("".join(body))
    print(f"Wrote {out} ({len(hits)} hits)")
    return 0 if len(hits) < 40 else 1  # soft gate; review manually


if __name__ == "__main__":
    raise SystemExit(main())
