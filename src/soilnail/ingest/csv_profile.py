"""CSV profiling and exact image join (no fuzzy matching)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from soilnail.ingest.models import CsvProfile

CoordinateKind = Literal[
    "wgs84_latlon",
    "local_metres",
    "hk1980_grid_candidate",
    "unknown",
    "none",
]


def _detect_encoding(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()[:4096]
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            path.read_text(encoding=enc)
            return enc, has_bom
        except UnicodeDecodeError:
            continue
    return "latin-1", has_bom


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def infer_coordinate_kind(
    columns: list[str], rows: list[dict[str, str]]
) -> tuple[CoordinateKind, str]:
    cols = {c.lower(): c for c in columns}
    if "latitude" in cols and "longitude" in cols:
        lats = [
            float(r[cols["latitude"]]) for r in rows if r.get(cols["latitude"]) not in ("", None)
        ]
        lons = [
            float(r[cols["longitude"]]) for r in rows if r.get(cols["longitude"]) not in ("", None)
        ]
        if lats and lons:
            la_min, la_max = min(lats), max(lats)
            lo_min, lo_max = min(lons), max(lons)
            # WGS84 Hong Kong ~ lat 22, lon 114
            if la_min >= 22.0 and la_max <= 23.0 and lo_min >= 113.5 and lo_max <= 114.8:
                return (
                    "wgs84_latlon",
                    f"columns latitude/longitude; ranges lat=[{la_min:.6f},{la_max:.6f}] "
                    f"lon=[{lo_min:.6f},{lo_max:.6f}] consistent with WGS84 in Hong Kong",
                )
            if abs(la_max) <= 90 and abs(lo_max) <= 180:
                return (
                    "wgs84_latlon",
                    f"columns latitude/longitude; ranges lat=[{la_min},{la_max}] lon=[{lo_min},{lo_max}]",
                )
        return "unknown", "latitude/longitude columns present but ranges empty or unexpected"
    if "local_x_m" in cols and "local_y_m" in cols:
        return (
            "local_metres",
            "columns local_x_m/local_y_m/local_z_m — local metres, not geographic CRS",
        )
    # HK1980 Grid easting/northing typical ~800000E, ~800000N — only if named so
    if any(k in cols for k in ("easting", "northing", "hk80_e", "hk80_n")):
        return (
            "hk1980_grid_candidate",
            "easting/northing-style column names present — needs confirmation",
        )
    return "none", "no coordinate columns recognised"


def profile_csv(
    path: Path,
    *,
    archive: str,
    image_filenames: set[str],
    join_key: str = "image_name",
) -> CsvProfile:
    encoding, has_bom = _detect_encoding(path)
    text = path.read_text(encoding=encoding)
    delim = _sniff_delimiter(text[:4096])
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    columns = list(reader.fieldnames)
    rows: list[dict[str, str]] = [dict(r) for r in reader]
    n_rows = len(rows)
    null_rates: dict[str, float] = {}
    for c in columns:
        empty = sum(1 for r in rows if r.get(c) in (None, ""))
        null_rates[c] = float(empty / n_rows) if n_rows else 0.0
    # duplicate full rows
    row_tuples = [tuple(r.get(c, "") for c in columns) for r in rows]
    counts = Counter(row_tuples)
    dup_rows = sum(v - 1 for v in counts.values() if v > 1)

    if join_key not in columns:
        raise ValueError(f"Join key {join_key!r} not in CSV columns {columns} for {path.name}")

    csv_keys = [r[join_key] for r in rows]
    csv_set = set(csv_keys)
    img_set = set(image_filenames)
    matched = csv_set & img_set
    unmatched_rows = sorted(csv_set - img_set)
    unmatched_imgs = sorted(img_set - csv_set)

    kind, rationale = infer_coordinate_kind(columns, rows)

    # dtypes note: all read as str; numeric inference left as ranges in rationale
    return CsvProfile(
        archive=archive,
        path=str(path),
        encoding=encoding,
        delimiter=delim,
        has_bom=has_bom,
        n_rows=n_rows,
        n_cols=len(columns),
        columns=columns,
        null_rates=null_rates,
        duplicate_row_count=dup_rows,
        coordinate_kind=kind,
        coordinate_rationale=rationale,
        join_key=join_key,
        rows_matched_to_images=len(matched),
        rows_unmatched=len(unmatched_rows),
        images_unmatched=len(unmatched_imgs),
        unmatched_row_keys=unmatched_rows[:50],
        unmatched_image_names=unmatched_imgs[:50],
    )


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    encoding, _ = _detect_encoding(path)
    text = path.read_text(encoding=encoding)
    delim = _sniff_delimiter(text[:4096])
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    return list(reader.fieldnames), [dict(r) for r in reader]
