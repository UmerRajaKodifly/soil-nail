# Soil-nail one-shot segmentation (PerSAM vs SegGPT)

Experiment workspace for SmartEye (CEDD GEO GE/2024/21): compare **PerSAM / PerSAM-F** and **SegGPT** on site-walk photos of soil-nail heads.

## Status

| Phase | Status |
|-------|--------|
| 0 Recon | done |
| 1 Env + smoke | done |
| 2 Annotator + wrappers | code ready — **draw variant-B masks, then LOO** |
| 3 LOO matrix | blocked on masks for refs 02 & 03 |

## Quick start

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/smoke_all.py
```

### Annotate reference masks (variant B = default)

```bash
python tools/annotate_mask.py --variant B
# http://127.0.0.1:7860 — click the photo, Save, Next
```

See `data/refs/soil_nail/MASKS.md`.

### Compare / leave-one-out

```bash
# single run
python compare.py --cls soil_nail --variant B --ref nail_ref_02_2D16_rebar_install \
  --images data/refs/soil_nail/images/nail_ref_01_2D12_rebar_install.jpg \
  --out outputs/demo --models persam persam_f seggpt

# optional EXTENSIONS: --tight_prompt_crop --multi_instance --tile --crop_bottom_right

# full LOO matrix (needs B masks for refs 01, 02, 03)
python scripts/run_loo_matrix.py --out outputs/loo_b
```

## Pins

- PerSAM upstream: `third_party/Personalize-SAM` @ `a7e87245156e0b8efcba10a15a745e699c679d9d` (**untouched**)
- Python: 3.10 (3.11 OK)
- PyTorch: `2.11.0+cu128` · transformers: `5.19.0`
- SAM ViT-H: `weights/sam_vit_h_4b8939.pth` · MobileSAM: `weights/mobile_sam.pt`
- SegGPT: `BAAI/seggpt-vit-large`

**Important:** PerSAM inference uses vendored `per_segment_anything` (modified SAM). Official `segment_anything` is for the annotator only.

## Mask variants

| Variant | Path | Meaning |
|---------|------|---------|
| **B (default)** | `data/refs/<cls>/masks/` | sleeve + bar end + hose roots only |
| **A (comparison)** | `data/refs/<cls>/masks_A/` | hole + contents |

## Extensions (opt-in, not in original papers)

- `--tight_prompt_crop` — crop prompt image/mask around ref mask (+ margin)
- `--multi_instance` — PerSAM multi-peak + NMS
- `--tile` — sliding-window (scaffolded; full-frame still used until tiling merge is wired in compare)
- `--crop_bottom_right` — blank date stamp
- Multi-reference target-feature averaging
- Mask-area sanity flag if prediction or ref mask **> 25%** of image (`mask_area_flag` in `summary.csv`)
- SegGPT: saves `*_seggpt_raw.npz` + heatmap under `<out>/seggpt_raw/`

## Licences

- PerSAM: MIT · Segment Anything: Apache-2.0 · SegGPT/Painter: check upstream before redistribution
