# Deviations from the production spec

Every intentional shortcut or deferred item is listed here with reason and impact.
Silent deviations are bugs.

| ID | Phase | Deviation | Reason | Impact | Status |
|----|-------|-----------|--------|--------|--------|
| D001 | 0 | `mypy --strict` applies to `src/soilnail/` only; legacy `src/*.py` excluded until refactor | Prototype is untyped and will be replaced in Phase 2; forcing ignores would violate "no type: ignore without reason" | Legacy code not type-checked until moved | Open until Phase 2 complete |
| D002 | 0 | `bandit` / `pip-audit` may warn on transitive deps we do not control | Record findings; fix only actionable high/critical in our code | Residual supply-chain risk | Track in QA |
| D003 | 0 | Third-party `Personalize-SAM` and `.venv` / `weights/` not committed | Size + licence; record SHAs in manifests | Clone needs `scripts/setup.sh` | Intentional |
| D004 | 0 | No CI remote yet | Local `make check` first | No remote gate | Open |
| D005 | 0 | `setuptools` capped at `<82` (installed ≥78.1.1) | `torch==2.11.0+cu128` requires `setuptools<82`; newer advisory fix wants ≥83 | Residual setuptools advisory risk until torch relaxes pin; recorded in pip-audit skips/notes | Open |
| D006 | 0 | `pip-audit` cannot audit `torch`/`torchvision` cu128 wheels (not on PyPI) | Index wheels | Supply-chain blind spot for CUDA wheels | Open |
| D007 | 0 | `pytest` bumped to ≥9.0.3 for PYSEC-2026-1845 | Security fix | Dev-dep only | Closed |
| D008 | 2 | Model adapters wrap legacy `src/*_runner.py` instead of full rewrite | Preserve LOO behaviour first; golden test gates drift | Legacy still in tree until inlined | Open |
| D009 | 2 | Golden LOO tolerances | Dual runs measured: run-to-run diff 0.0; max vs LOO_REPORT 0.0005; tol set to 0.01 | Closed in `configs/golden_loo.yaml` | Closed |
| D010 | 1 | Gate 1 skipped — CSV CRS/joins unambiguous from audit | Walk1 WGS84 HK; VID local metres; exact `image_name` 100% match; duplicate filenames across zips handled by `archive::path` ids | None if audit correct | Closed |
| D011 | 3 | Equirect whole/tile pilots treated 3840×1920 as flat photos | Data are 360° equirect; amendment requires cubemap | Timings/detections **superseded** (`superseded=true`); keep only as equirect ablation evidence | Closed (labelled) |
| D012 | 3→4 | Gate 2 option **C** adapted: baseline = 6 cubemap faces × 3 models; face sub-tiling + raw equirect = pilot-only ablations | User amendment 2026-10-08 | Replaces prior option-C (equirect tile_1024) | Open until cubemap Gate 2 measured |
| D013 | 1 | No GPano/EXIF on any frame — cannot metadata-prove equirect | Stripped JPEGs | Rely on 2:1 + seam + poles; reject non-2:1 | Documented |
| D014 | 1 | CSV pitch/roll/course all zero | Device did not populate | No leveling/heading from CSV | Documented |
| D015 | — | Face-size rationale corrected | 960 = **W/4** (equirect 90° lon span), **not** centre-density match; centre match is **W/π≈1222**; 1024 = SAM-native. Evidence: `face_density_table.json` | Restart baseline uses **1024** | Closed |
| D016 | 3→4 | Stopped full run `full_cubemap_…a454b8b3` mid-job | Item-1 gaps: `equirect_polygon_xy` was always null; margin chosen without ablation evidence | Partial results kept under that run dir but **not** used for decisions; restart after fix | Closed |
| D017 | 4 | `margin_deg=15` after real yaw-sweep | Persam mean retention m0=0.287, m5=0.310 vs m15 baseline (n=32 yaw×frame); neither ≥0.95 → **15**. Stub-chosen 0 contradicted. | `ablation_decision.json` | Closed |
| D018 | 1 | Wrap criterion | Independent null-only: fail if `wrap_abs > adj_p95+0.25*(dist_med−adj_med)` (=**25.42** grey-levels); 0 fails. Ratio 1.5×p99.5 kept descriptive only. Offset±k: no dup/missing col | `wrap_seam_review.json` | Closed |
| D019 | 4 | Leveling not applied | n=64; \|roll\| p95≈6.3°, \|pitch_proxy\| p95≈11°; leveling OFF | `camera_tilt_estimate.json` | Closed |
| D020 | 4 | Walk/site ids | `C55_W1`/`C55`, `F159_VID`/`F159`; `reference_site_id=C55`; feature_no only for C55 | `configs/walks.yaml` | Closed |
| D021 | 4 | Merge rule | Overlap-primary (Jaccard≥0.05 on 1° equirect) + secondary centroid 2° only if both border; replaces centroid-only 5° | `merge.py`, `merge_sensitivity.json` | Closed |
| D022 | 4 | `face_size=1024` | **Rationale** (SAM-native), not label evidence | until labelled size study | Open |
| D023 | 4 | Privacy | Nadir blur + figure-review ledger; no person detector — side-face people not auto-covered | `privacy_figures.yaml` | Open |
| D024 | 4 | Store PK `(run_id, image_id, model)` | Prior PK `(run_id, image_id)` would overwrite model A when writing B; caught before persam_f | `store.py` schema_version=2 | Closed |
