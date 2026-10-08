# Prototype audit (Phase 0)

**Scope:** existing code under `src/`, `compare.py`, `tools/`, `scripts/` as of branch
`feature/site-walk-production` (pre-`src/soilnail` refactor).  
**Evidence:** file reads + greps + spot-checks against `third_party/Personalize-SAM`
(`a7e8724…`) and installed `transformers` SegGPT API.  
**Standard:** treat all of this as unreviewed prototype.

---

## 1. Inventory

| Path | Role | Lines (approx) |
|------|------|----------------|
| `src/device.py` | CUDA/MPS/CPU pick, VRAM helpers | 43 |
| `src/preprocess.py` | date-stamp crop, tight crop, overlays | 112 |
| `src/segmenter.py` | `Segmenter` ABC, `RefExample`, `Result` | 44 |
| `src/tiling.py` | sliding-window tiles + union merge | 42 |
| `src/persam_runner.py` | PerSAM / PerSAM-F | 468 |
| `src/seggpt_runner.py` | HF SegGPT | 229 |
| `compare.py` | CLI compare + panels + CSV | 277 |
| `tools/annotate_mask.py` | FastAPI click annotator | 473 |
| `scripts/smoke_all.py` | Phase-1 smoke | 364 |
| `scripts/run_loo_matrix.py` | LOO driver | 124 |
| `requirements.txt` | partial pins (no lockfile) | — |

No tests, no configs, no schemas, no result store, no UI for site-walk review.

---

## 2. Hardcoded values (must move to config)

| Location | Value | Notes / intended `source:` |
|----------|-------|----------------------------|
| `preprocess.crop_bottom_right` | `frac_w=0.22`, `frac_h=0.08` | Prototype decision for *reference* photos with orange stamp — **not measured on site-walk**; Phase 1 must re-measure |
| `preprocess.mask_bbox` / runners | `margin=0.15` | Prototype decision (tight crop ablation) |
| `compare.MASK_AREA_WARN_PCT` | `25.0` | Prototype decision |
| `tiling.make_tiles` | `tile=1024`, `overlap=0.25` | SAM input size rationale; not validated on site-walk sizes |
| `persam_runner` | `train_epoch=1000`, `lr=1e-3`, `eps=1e-4` | Upstream default (`persam_f.py` CLI) |
| `persam_runner` | `multi_k=5`, `min_dist=40`, `ratio=0.5`, `nms_iou=0.5` | **Our extension**, not paper |
| `persam_runner` focal loss | `alpha=0.25`, `gamma=2` | Upstream-style |
| `seggpt_runner` | `model_id=BAAI/seggpt-vit-large`, `min_cc_area=50` | HF default id; CC area is ours |
| `seggpt_runner` diagnosis | `0.001` frac threshold | Ad-hoc diagnostic |
| `annotate_mask.py` | port `7860`, area warn `25%` | UX defaults |
| `compare.COLORS` | RGB tuples | Not colour-blind-safe audited |
| Weight paths | `weights/sam_vit_h_4b8939.pth` | Relative to repo root via `Path(__file__)` |

---

## 3. Validation / error handling

| Issue | Evidence |
|-------|----------|
| No schema validation of CLI args beyond argparse types | `compare.py` |
| `assert` used for readiness (`persam_runner.segment`, `seggpt_runner.segment`) | Not suitable for SaaS input validation |
| Bare failures raise; no structured `Failure` records | No per-image failure ledger |
| OOM: only `smoke_all.py` catches `torch.cuda.OutOfMemoryError` | `compare.py` / runners do not retry |
| Annotator: some HTTP paths return generic 400; no audit log | `tools/annotate_mask.py` |
| No zip-slip / bomb guards | N/A yet (site-walk not ingested) |
| `except` in prototype app code: smoke OOM only | Mostly fail-loud; good. Gradio vendored code has broad excepts (ignore). |

---

## 4. CUDA / device assumptions

- `pick_device` prefers CUDA → MPS → CPU — OK.
- Upstream `persam_f.py` uses `.cuda()` directly; our wrapper uses `self.device` — better.
- Peak VRAM helpers assume single GPU; no `cuda:N` selection beyond `prefer` string.
- Models run sequentially in `compare.py` (good for 8 GB); no enforced free between models beyond `close()` if caller invokes it — **compare does not always call `close()` between models** (check): models constructed per name in loop; previous object may linger until GC. Risk of OOM on 8 GB when loading SegGPT after PerSAM without explicit `del`/`close`.
- No seed control inside PerSAM-F training loop beyond caller `np.random.seed` in `compare.py` (torch/cudnn determinism unset).

---

## 5. Untested / unwired paths

| Path | Status |
|------|--------|
| `--tile` in `compare.py` | Logs warning; **runs full-frame** (`compare.py` ~186–187) |
| `merge_masks_union` | Implemented; no property tests; unused in live compare |
| Multi-instance PerSAM | Code present; not used in LOO matrix script |
| SegGPT `feature_ensemble=True` | Wired; LOO used single ref only |
| PerSAM-F weight cache | **None** — re-trains every `prepare()` |
| EXIF orientation | Not applied in loaders (`cv2.imread` / PIL without `exif_transpose`) |
| Golden LOO regression | No automated test; numbers only in `outputs/loo_b/` |
| Manifest / hash of inputs | Absent |

---

## 6. Duplicated logic

- IoU helpers in both `persam_runner._mask_iou` and `compare.py`
- Overlay colour constants in annotate + compare
- Mask area % in preprocess + annotate + compare
- Device pick only in `device.py` (good) but runners each call `pick_device` at init

---

## 7. Upstream behaviour deviations

### PerSAM (`persam.py`)

| Topic | Upstream | Ours | Risk |
|-------|----------|------|------|
| Single reference | Yes | Multi-ref feature average (EXTENSION) | Documented |
| Positive + negative points | Yes | Matched for single-instance | OK |
| Multi-instance peaks + NMS | No | EXTENSION | Must label as ours in report |
| Tight prompt crop | No | EXTENSION | Hurt LOO; ablation only |

### PerSAM-F (`persam_f.py`)

| Topic | Upstream | Ours | Risk |
|-------|----------|------|------|
| Train epochs / AdamW / cosine | Defaults 1000 / 1e-3 | Same defaults | OK |
| `.cuda()` | Hardcoded | `self.device` | OK |
| Train loop calls `predict` each epoch | Yes | **One** `predict`, then train on **detached** logits | Documented workaround because `predict_torch` is `@torch.no_grad()`; effective graph matches weighting detached masks. **Must golden-test against LOO IoUs** |
| Fine-tune on first ref only | One-shot | Same; extra refs refresh target feat after train | EXTENSION |
| Weight cache | N/A (script) | Missing — SaaS must cache | Phase 2 |

### SegGPT (installed `transformers` SegGPT)

| Topic | Verified | Ours |
|-------|----------|------|
| `feature_ensemble` kwarg on `forward` | Yes (`SegGptForImageSegmentation.forward`) | Passed when ≥2 refs and flag set |
| `num_labels=1` palette path | Used | Matches processor API |
| Connected-component instances | Not in HF | Our post-process (`min_cc_area`) |
| Raw distance diagnostics | Not in HF | Our diagnostic for empty preds |

**Pinned evidence for Phase 2 configs:**

- SAM ViT-H SHA-256: `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` (`weights/sam_vit_h_4b8939.pth`, measured `sha256sum`)
- SegGPT HF snapshot dir present: `77b179b5a1b50a48bd04ef3138804cfacc2ced73` (local hub cache)
- PerSAM tree: `third_party/Personalize-SAM` @ `a7e87245156e0b8efcba10a15a745e699c679d9d` (README claim; verify in git once third_party is tracked or recorded)

---

## 8. Reproducibility gaps

- No `requirements.lock` / `uv.lock`
- No config hash / run manifest
- SegGPT revision not pinned in code (floating `from_pretrained(model_id)`)
- README status table stale (says LOO blocked; LOO already ran)
- `.gitignore` omits client site-walk data paths (must add before ingest)
- No `make check` / CI

---

## 9. Security / privacy

- Annotator binds `127.0.0.1` by default — OK for local
- No ingestion of client zips yet
- Photos may contain people — no redaction tooling
- Gradio was briefly used then replaced by FastAPI; `gradio>=4.0` still in requirements (unused dependency risk)

---

## 10. Verdict for refactor (Phase 2)

Prototype is sufficient to prove GPU pipelines and LOO anecdotes. It is **not** SaaS-ready:

1. Move all constants into versioned YAML + pydantic.
2. Replace `Result` with typed `Detection` / RLE / failure ledger.
3. Wire tiling for real; cache PerSAM-F weights; pin model revisions.
4. Golden-test LOO IoUs before full site-walk spend.
5. Exclude legacy modules from `mypy --strict` until moved into `src/soilnail/` (see `docs/DEVIATIONS.md`).
