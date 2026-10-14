# Labelling protocol

## Sites and transfer

| Walk | `walk_id` | `site_id` | Role vs references |
|------|-----------|-----------|--------------------|
| Walk 1.zip | `C55_W1` | `C55` | **In-site** — same site as curated refs `nail_ref_01..03` (IDs 2D12, 2D16, 2D17). Optimistic for transfer. |
| images.zip | `F159_VID` | `F159` | **Cross-site** — real transfer test. Only 50 frames → wide intervals. |

- `reference_site_id` = **C55**. Reference photos are QA photos, **not** walk frames (no pixel leakage). The same physical nails may appear in C55 walk imagery.
- Do **not** pool C55 and F159 to claim transfer. Report side by side with their own n and CIs.
- Do **not** map these ids to FR52/FR53/FR94.

## What to mark

Per image:

1. Draw a box (equirect pixel `xywh`) for each visible soil nail, with a **stage tag**: `hole_only` | `bar_installed` | `grouted` | `unclear`.
2. If there are no nails, check **no nails in this image** (needed for false-positive rates).
3. Use **unsure** when the content is ambiguous; do not invent nails.
4. For each model detection: **accept** / **reject**. Rejected detections get an FP type: `scaffold`, `whiteboard`, `soil`, `hose_or_pipe`, `person_or_tool`, `other`.

## Split rules

- Sampling uses **per-walk quotas** (`configs/ui.yaml` / `configs/walks.yaml`): floor for F159 so it is not drowned.
- Split is **contiguous blocks within each walk**, with a **buffer gap** so near-duplicate neighbours do not land in opposite splits.
- Thresholds may be tuned **only** on `tuning`. Freeze config before `heldout`.

## Privacy

Display galleries apply nadir / Down-face blur. There is **no** automatic person detector — workers on platforms in side faces are not auto-redacted. Review every PDF figure before release.
