# QA report — 360° cubemap amendment

**Date:** 2026-10-08  
**Branch:** `feature/site-walk-production`

## Projection tests

```
.venv/bin/python -m pytest tests/test_projection_*.py -q
# 26 passed
```

| Suite | Result |
|-------|--------|
| Geometry / hypothesis | pass |
| py360convert cross-check (dirs ≤1e-6; faces MAE&lt;0.5, PSNR≥45) | pass |
| Round-trip / seam-straddle / subtile arithmetic (9×6=54) | pass |

## Independent review (re-derived from docs/PROJECTION.md)

- Face centres F/R/B/L: lon/lat match axes to &lt;1e-14°.
- U/D: latitude ±90° exact; longitude singular at poles (expected).
- Visual (`outputs/pano360_probe/face_*_0003_512.jpg`, `cross_frame_0003_512.jpg`):
  - No mirrored faces; F shows rectilinear site; U = canopy/sky; D = operator/hard-hat/nadir.
  - Side faces L–F–R–B form a continuous horizon band in the cross layout.
  - Privacy: people visible on D / side faces — data stays local.

## Data verification contradictions vs amendment draft

| Draft claim | Measured |
|-------------|----------|
| Face size ≈1222 (`W/π`) | **Wrong for “match equator density @ 90° FOV”.** Correct: `3840/360×90 = **960**` (=W/4). `W/π≈1222` is a different formula. |
| GPano / ProjectionType present | **Absent on all 591 frames** |
| Usable pitch/roll/course | **All zeros** in both CSVs |

## Gate 2 (cubemap micro n=4)

| Model | mean s/frame | est full h |
|-------|--------------|------------|
| PerSAM | 9.625 | 1.582 |
| PerSAM-F (shared SAM embeds) | 4.134 | 0.681 |
| SegGPT | 5.970 | 0.980 |
| **Baseline sum** | | **3.243 h ≤ 8 h → proceed** |

Evidence: `outputs/pilot_cubemap/gate2_cubemap_estimate.json`.  
Equirect pilot marked `superseded=true` in `outputs/pilot/gate2_estimate.json`.

## Process hygiene

Long jobs: `setsid nohup … > logs/….log`, PID in `outputs/pilot_cubemap/PID` / `outputs/runs/full_cubemap.pid`. Never pipe into `head`/`tail` (killed earlier LOO attempt).

## make check

`make check` passed after amendment (ruff, mypy --strict, pytest 38, bandit, pip-audit, licenses).

## Golden LOO (post SAM-cache)

`logs/golden_loo_post_cache.log`: `failures=0` at measured tolerance 0.01.

## Full run

Detached: `logs/full_cubemap.log`, PID `outputs/runs/full_cubemap.pid`, run dir under `outputs/runs/full_cubemap_*`. Expected ~3.2 h.
