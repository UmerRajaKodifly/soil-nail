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
