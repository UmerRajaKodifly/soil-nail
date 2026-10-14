# Architecture (site-walk production)

## Input contract

- Site-walk frames: **full equirectangular 2:1 only** (`W == 2*H`). Anything else raises `NotFullEquirectError`.
- Reference / LOO images: ordinary perspective photos (unchanged).
- Conventions: `docs/PROJECTION.md`.

## Pipeline

```
equirect JPEG
  → assert 2:1
  → cubemap (6 faces, FOV=90° ± margin)
  → per-face detection (PerSAM / PerSAM-F / SegGPT)
  → SAM image embedding cached per face (shared across PerSAM/PerSAM-F/refs)
  → spherical merge (angular distance)
  → Detection{face_id, mask_face, spherical, area_frac_face, area_frac_solid_angle}
```

## Packages

| Path | Role |
|------|------|
| `src/soilnail/projection/` | Equirect ↔ cubemap (single implementation) |
| `src/soilnail/models/` | Adapters + SAM embedding cache |
| `src/soilnail/runner.py` | `segment_image` (flat/LOO) + `segment_cubemap_image` |
| `src/soilnail/schemas.py` | Detection / SphericalGeom |
| `configs/projection.yaml` | face_size, FOV, margins, angular thresholds |

## Gate 2 baseline (option C / cubemap)

3 models × 6 faces × 591 frames; PerSAM face sub-tiling and raw equirect = pilot ablations only.
