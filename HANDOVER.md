## Soil-nail 360° detection (SmartEye / CEDD GEO GE/2024/21)

One-shot detection of soil-nail heads in 360° equirectangular site-walk panoramas for slope works QA. Inputs are curated reference nail photos/masks plus site-walk image zips and pose CSVs; outputs are per-frame detections (masks, uncalibrated scores, equirect polygons) in SQLite, overlay galleries, and a local Streamlit review UI. Stack: PyTorch + Segment Anything (PerSAM / PerSAM-F) and SegGPT, with a six-face cubemap pipeline, cross-face merge, and walk/site split (C55 in-site vs F159 cross-site).

GitHub: TODO — publish to private `Kodifly/soil-nail` blocked on GitHub API connectivity from this machine (verify after `gh auth login` / network restore)
Confluence: TODO
Contact Person: TODO

Tasks: Active R&D / handover mid-pipeline — PersAM and PersAM-F full cubemap runs finished on 591 frames (`margin_deg=15`, `face_size=1024`); SegGPT full run was in progress at last check (verify `outputs/runs/full_cubemap_*` and `logs/full_cubemap_m15.log`). Unfinished: Phase 5 labelled eval + PDF report (needs walk-frame labels; acceptance targets still null), Phase 6 hardening, and SmartEye integration wiring. Likely next work: finish/verify SegGPT, label C55 vs F159 with blocked splits, threshold tuning on tuning split only, face-size study once labels exist, and product integration with other teams.

1. `feature/site-walk-production` / `cleanup` — production cubemap ingest→run→UI path (current working line).
2. Reference LOO path (`scripts/run_loo_matrix.py`, `outputs/loo_b`) — leave-one-out on three QA refs; useful regression, not site-walk transfer.
3. Annotator (`tools/annotate_mask.py`) — Gradio/OpenCV tool for reference mask variants A/B.

### Needs review (left in place; not deleted)

- Legacy prototype modules under `src/*.py` (`persam_runner.py`, `compare.py`, etc.) still wrapped by adapters — may be removable after full inline (verify).
- `scripts/audit_hardcodes.py` / Phase 6 checklist items — partially done.
- Local artefacts (gitignored): `weights/`, `data/site_walk_*`, `data/caches/`, `outputs/runs/` — keep on shared disk or re-download; do not commit (privacy + size).
