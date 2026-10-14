# Soil-nail 360° detection (PerSAM / PerSAM-F / SegGPT)

SmartEye experiment for CEDD GEO GE/2024/21: detect soil-nail heads in **360° equirectangular** site-walk panoramas using one-shot segmentation (PerSAM, PerSAM-F, SegGPT) with cubemap projection, cross-face merge, and a local review UI.

**Inputs:** reference nail photos + masks; site-walk zips + GPS/pose CSVs (not in git).  
**Outputs:** SQLite results store, overlay galleries, descriptive reports, Streamlit review UI.

## Status (handover)

| Phase | Status |
|-------|--------|
| 0–2 | Done (package, LOO on refs, annotator) |
| 3 Gate 2 | Done — cubemap baseline ≤8 h → proceed |
| 4 Full run + UI | PersAM + PersAM-F complete on 591 frames; SegGPT was in progress (verify); UI works |
| 5 Eval + PDF | Scaffolded; needs labelled walk frames |
| 6 Hardening | Partial |

Walks: **C55_W1** (in-site vs refs) and **F159_VID** (cross-site, n=50). See `configs/walks.yaml`.

## Setup

```bash
bash scripts/setup.sh          # venv, torch cu128, weights, Personalize-SAM
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements.txt
# UI extras (if not already installed):
pip install 'streamlit>=1.28' 'plotly>=5.18' 'folium>=0.15' 'streamlit-folium>=0.15'
```

### Large files (not in git)

| Asset | Where / how |
|-------|-------------|
| SAM ViT-H | `weights/sam_vit_h_4b8939.pth` via `scripts/setup.sh` (SHA in `configs/models.yaml`) |
| MobileSAM | `weights/mobile_sam.pt` |
| SegGPT | Hugging Face `BAAI/seggpt-vit-large` (cached under HF hub) |
| Site-walk zips + CSVs | Place under `data/site_walk_raw/` then `make ingest` (gitignored; privacy) |
| Run outputs | `outputs/runs/` (gitignored; regenerable) |

## Run

```bash
# Ingest site-walk archives (after placing zips/CSVs in data/site_walk_raw/)
make ingest

# Full cubemap baseline (resumable; ~hours on 8 GB GPU)
.venv/bin/python -u scripts/run_full_cubemap.py --config configs/experiments.yaml
# Resume:
.venv/bin/python -u scripts/run_full_cubemap.py --resume-dir outputs/runs/<run_id>

# Review UI
make ui
# or: .venv/bin/python -m soilnail.cli ui --config configs/ui.yaml
# http://localhost:8501

# PersAM-only descriptive report (example)
.venv/bin/python scripts/report_persam_descriptive.py

# Classic LOO on reference photos
python scripts/run_loo_matrix.py --out outputs/loo_b
```

Active full-run log (if present): `logs/full_cubemap_m15.log` (not `logs/full_cubemap.log`).

## Test

```bash
make check          # ruff + mypy --strict + pytest + bandit + pip-audit
# or
.venv/bin/python -m pytest -q
```

## Config pointers

- `configs/experiments.yaml` — baseline face_size=1024, margin_deg=15 (yaw-sweep evidence)
- `configs/projection.yaml` — cubemap + overlap-primary merge
- `configs/walks.yaml` — C55 / F159 site ids; `reference_site_id: C55`
- `configs/ui.yaml` — review UI
- `docs/DEVIATIONS.md` — intentional shortcuts / evidence links
- `docs/LABELLING_PROTOCOL.md` — in-site vs cross-site labelling rules

## Licences

PerSAM: MIT · Segment Anything: Apache-2.0 · SegGPT/Painter: check upstream before redistribution. See `docs/LICENSES.md` (flag for legal review).
