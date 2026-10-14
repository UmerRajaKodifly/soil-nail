# Data audit (Phase 1)

**Created (UTC):** 2026-10-08T04:59:35.717858+00:00

## Inputs (SHA-256)

- `images.zip`: `e4e9fa029d785be062b5c7cf87b2943af4eedec161f57723606015d37583b1af`
- `VID_20260808_104335_00_011.csv`: `d826aa4204443ca5b7f81c3f33751ab49758379a29d06b7953ebc157ea580a50`
- `Walk 1.zip`: `d25402badc29788bf7be82fa9603a51b168c86ce580025ec44a0293e254e1538`
- `Walk 1.csv`: `deb7f2a1b7b1fd1555b137744cd4c5f57284a6da537f20934c1cf27c508d93d4`

## Per-archive counts

| Archive | Zip members | Images | OK | Failed | Skipped | Non-image | CSV match (rows↔ok images) |
|---------|-------------|--------|----|--------|---------|-----------|------------------------------|
| images | 50 | 50 | 50 | 0 | 0 | 0 | matched=50, row_miss=0, img_miss=0 |
| walk1 | 541 | 541 | 541 | 0 | 0 | 0 | matched=541, row_miss=0, img_miss=0 |

**Totals:** images=591, ok=591, failed=0, skipped=0 (reconciles: 591+0+0=591).

## Resolutions

- `3840x1920`: 591

## Duplicates

- Duplicate filename groups across archives: **50** (stable `image_id` = `archive::relative_path` prevents collisions).
- Byte-identical hash groups: **0**

## CSV profiles

### `images` — `VID_20260808_104335_00_011.csv`

- encoding=utf-8-sig, delimiter=`,`, BOM=False
- rows=50, cols=9: `['image_name', 'local_x_m', 'local_y_m', 'local_z_m', 'video_offset_sec', 'pitch', 'roll', 'course', 'Time']`
- duplicate full rows: 0
- null rates: `{'image_name': 0.0, 'local_x_m': 0.0, 'local_y_m': 0.0, 'local_z_m': 0.0, 'video_offset_sec': 0.0, 'pitch': 0.0, 'roll': 0.0, 'course': 0.0, 'Time': 0.0}`
- **coordinate_kind:** `local_metres` — columns local_x_m/local_y_m/local_z_m — local metres, not geographic CRS
- **join:** exact on `image_name` → matched=50, unmatched_rows=0, unmatched_images=0

### `walk1` — `Walk 1.csv`

- encoding=utf-8-sig, delimiter=`,`, BOM=False
- rows=541, cols=8: `['image_name', 'latitude', 'longitude', 'altitude_m', 'video_offset_sec', 'pitch', 'roll', 'course']`
- duplicate full rows: 0
- null rates: `{'image_name': 0.0, 'latitude': 0.0, 'longitude': 0.0, 'altitude_m': 0.0, 'video_offset_sec': 0.0, 'pitch': 0.0, 'roll': 0.0, 'course': 0.0}`
- **coordinate_kind:** `wgs84_latlon` — columns latitude/longitude; ranges lat=[22.340049,22.340939] lon=[114.145265,114.145892] consistent with WGS84 in Hong Kong
- **join:** exact on `image_name` → matched=541, unmatched_rows=0, unmatched_images=0

## Burned-in date stamp probe

- sampled: 12
- likely stamp-like corners: 8
- mean saturation corner/full: 0.564 / 0.427
- **recommendation:** `measure_box_before_crop`
- evidence: 8/12 samples show elevated bottom-right saturation (mean corner S=0.564 vs full S=0.427). Do not apply reference-photo crop fractions without measuring the box.

### Measured consensus (script, not assumed)

Orange-ish high-saturation pixels in the bottom-right probe corner were boxed on the same seeded sample of 12 images (`data/manifests/stamp_measure.json`):

| Metric | Value | Source |
|--------|-------|--------|
| Samples with orange mask | 9 / 12 | measured |
| Consensus cover from bottom-right | `frac_w=0.25`, `frac_h=0.1203125` | max extents of those boxes |
| Prototype reference crop (Phase 1 smoke) | `frac_w=0.22`, `frac_h=0.08` | earlier code default — **not copied** |

**Decision:** stamp crop stays **OFF** by default. If enabled later, use the measured site-walk consensus (`0.25` × `0.1203125`), not the reference-photo values. Some orange detections may be soil/equipment rather than a date stamp — treat as optional ablation.

## 360° equirectangular verification (amendment)

**Question:** are the 591 site-walk frames full 360°×180° equirectangular panoramas?

| Check | Result | Evidence |
|-------|--------|----------|
| Aspect ratio | **All 591 are exactly `3840×1920` (2:1)** | `data/manifests/dataset.sqlite` + full scan |
| EXIF | **None** (0/591 have Exif APP1 marker) | byte scan of every frame |
| GPano / XMP `ProjectionType` | **None** (0/591) | byte scan for `GPano`, `ProjectionType`, `equirectangular` |
| Wrap seam | **All 591**: ratio p50=1.41 (descriptive). **Independent fail gate** (null-only, not the judged ratio sample): fail if `wrap_abs > adj_p95 + 0.25*(distant_med−adj_med)` = **25.42 grey-levels**; **0 fails**. Units: mean |\Delta| RGB over rows×channels (0–255). Null adj med=6.81 / distant med=68 / broken≈12 are grey-level column diffs (see `wrap_seam_review.json` units). Offset ±k: no dup/missing column. Visual: 10 worst + 10 typical ±64 px crops continuous (texture, not stitch tear). | `wrap_seam_all591.json`, `wrap_seam_review.json`, `outputs/pano360_probe/wrap_review/` |
| Level (pitch/roll) | CSV `pitch`/`roll`/`course` are **all 0.0** on both walks — **not usable** for leveling or heading | CSV profile in this section |
| Zenith / nadir (n=6 visual) | Zenith ≈ sky/clouds; nadir ≈ **operator clothing / mount** (people may appear) | `outputs/pano360_probe/*_{zenith,nadir}.jpg` |
| Burned-in stamp | Earlier stamp probe was on mixed sample; **no EXIF date stamp**. Orange corner heuristic may fire on soil/equipment — crop stays off | prior `stamp_measure.json` |
| Face-size arithmetic | Equator **10.666… px/deg**; 90° density match = **960** (= W/4). Draft “≈1222” is `W/π`, **not** used | measured |

**Verdict:** Geometry is consistent with **full equirectangular 2:1** panoramas. Metadata does **not** confirm `ProjectionType=equirectangular` (stripped JPEGs). We accept 2:1 + wrap continuity + pole content as operational evidence; reject any future frame with `W ≠ 2H`.

**CSV pose:** `course` exists but is identically zero — do **not** use as yaw/heading until a non-zero source is documented.

**Privacy:** nadir faces often show the operator; keep local-only.

## Privacy

Site-walk photos may contain people (especially nadir / down face). Data stays under `data/site_walk_*` (gitignored). No uploads.

## Manifest

- SQLite: `data/manifests/dataset.sqlite`
- JSON summary: `data/manifests/ingest_summary.json`

