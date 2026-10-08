#!/usr/bin/env python3
"""Phase-1 smoke tests: official SAM embed, PerSAM (vendored), SegGPT (HF)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Personalize-SAM"))

OUT = ROOT / "outputs" / "smoke"
REF_IMG = ROOT / "data/refs/soil_nail/images/nail_ref_02_2D16_rebar_install.jpg"
TEST_IMG = ROOT / "data/refs/soil_nail/images/nail_ref_01_2D12_rebar_install.jpg"
PLACEHOLDER_MASK = OUT / "placeholder_ref_mask.png"
SAM_CKPT = ROOT / "weights" / "sam_vit_h_4b8939.pth"
MOBILE_CKPT = ROOT / "weights" / "mobile_sam.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    out = image_rgb.copy()
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.any(axis=-1)
    tint = np.zeros_like(out)
    tint[m] = color
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    return out


def ensure_placeholder_mask() -> Path:
    """Rough dark-region circle on ref_02 — NOT a curated annotation."""
    OUT.mkdir(parents=True, exist_ok=True)
    if PLACEHOLDER_MASK.exists():
        return PLACEHOLDER_MASK
    img = cv2.imread(str(REF_IMG))
    if img is None:
        raise FileNotFoundError(REF_IMG)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = np.zeros_like(gray)
    roi[int(0.15 * h) : int(0.65 * h), int(0.25 * w) : int(0.75 * w)] = 1
    dark = ((gray < 90) & (roi > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.morphologyEx(dark, cv2.MORPH_OPEN, k))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ph = np.zeros((h, w), np.uint8)
    if cnts:
        (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        cv2.circle(ph, (int(cx), int(cy)), int(r * 1.15), 255, -1)
    else:
        cv2.circle(ph, (w // 2, h // 2), min(w, h) // 8, 255, -1)
    cv2.imwrite(str(PLACEHOLDER_MASK), ph)
    ov = overlay_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ph)
    cv2.imwrite(str(OUT / "placeholder_ref_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    return PLACEHOLDER_MASK


def smoke_sam_official(device: torch.device) -> dict:
    """(a) Official segment_anything loads and embeds an image."""
    from segment_anything import SamPredictor, sam_model_registry

    if not SAM_CKPT.is_file():
        raise FileNotFoundError(f"Missing {SAM_CKPT}")
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    image = cv2.cvtColor(cv2.imread(str(REF_IMG)), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    feat = predictor.get_image_embedding()
    elapsed = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None
    )
    info = {
        "ok": True,
        "device": str(device),
        "embedding_shape": list(feat.shape),
        "runtime_s": round(elapsed, 3),
        "peak_vram_mb": None if peak is None else round(peak, 1),
    }
    (OUT / "sam_embed.json").write_text(json.dumps(info, indent=2))
    print("[SAM official]", info)
    del sam, predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def point_selection(mask_sim: torch.Tensor, topk: int = 1):
    """Copied from upstream persam.py (positive top-k + negative bottom-k)."""
    _w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0).cpu().numpy()
    topk_label = np.array([1] * topk)

    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0).cpu().numpy()
    last_label = np.array([0] * topk)
    return topk_xy, topk_label, last_xy, last_label


def smoke_persam(device: torch.device, sam_type: str = "vit_h") -> dict:
    """(b) Vanilla single-instance PerSAM pipeline on ref_02 -> test_01."""
    from torch.nn import functional as F

    from per_segment_anything import SamPredictor, sam_model_registry

    mask_path = ensure_placeholder_mask()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if sam_type == "vit_h":
        ckpt = SAM_CKPT
    elif sam_type == "vit_t":
        ckpt = MOBILE_CKPT
    else:
        raise ValueError(sam_type)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    # Upstream hard-codes .cuda() for vit_h; we move explicitly.
    sam = sam_model_registry[sam_type](checkpoint=str(ckpt))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    ref_image = cv2.cvtColor(cv2.imread(str(REF_IMG)), cv2.COLOR_BGR2RGB)
    # Upstream expects 3-channel mask; convert our 0/255 single-channel to RGB-like.
    ref_mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ref_mask_rgb = np.zeros((*ref_mask_gray.shape, 3), dtype=np.uint8)
    ref_mask_rgb[ref_mask_gray > 0] = (128, 0, 0)  # matches PerSeg after BGR2RGB

    ref_mask_t = predictor.set_image(ref_image, ref_mask_rgb)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask_t = F.interpolate(ref_mask_t, size=ref_feat.shape[0:2], mode="bilinear")
    ref_mask_t = ref_mask_t.squeeze()[0]

    target_feat = ref_feat[ref_mask_t > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    test_image = cv2.cvtColor(cv2.imread(str(TEST_IMG)), cv2.COLOR_BGR2RGB)
    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()
    c, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(c, h * w)
    sim = target_feat @ test_feat
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim, input_size=predictor.input_size, original_size=predictor.original_size
    ).squeeze()

    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    sim_n = (sim - sim.mean()) / torch.std(sim)
    sim_n = F.interpolate(sim_n.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim_n.sigmoid_().unsqueeze(0).flatten(3)

    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=False,
        attn_sim=attn_sim,
        target_embedding=target_embedding,
    )
    best_idx = 0
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        mask_input=logits[best_idx : best_idx + 1, :, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    y, x = np.nonzero(masks[best_idx])
    input_box = np.array([x.min(), y.min(), x.max(), y.max()])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logits[best_idx : best_idx + 1, :, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    final = masks[best_idx].astype(bool)

    elapsed = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None
    )

    mask_u8 = (final.astype(np.uint8) * 255)
    cv2.imwrite(str(OUT / "persam_test_mask.png"), mask_u8)
    ov = overlay_mask(test_image, final)
    cv2.imwrite(str(OUT / "persam_test_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    info = {
        "ok": True,
        "model": "persam",
        "sam_type": sam_type,
        "device": str(device),
        "ref_image": str(REF_IMG.name),
        "test_image": str(TEST_IMG.name),
        "ref_mask": str(mask_path.name),
        "score": float(scores[best_idx]),
        "mask_area_pct": float(round(100.0 * float(final.mean()), 3)),
        "runtime_s": float(round(elapsed, 3)),
        "peak_vram_mb": None if peak is None else float(round(peak, 1)),
        "note": "placeholder mask (auto dark circle), not curated",
    }
    (OUT / "persam.json").write_text(json.dumps(info, indent=2))
    print("[PerSAM]", info)
    del sam, predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def _seggpt_predict(model, processor, image_input, image_prompt, prompt_mask, device):
    """Verified against transformers 5.19 SegGptImageProcessor / SegGptForImageSegmentation."""
    inputs = processor(
        images=image_input,
        prompt_images=image_prompt,
        prompt_masks=prompt_mask,
        num_labels=1,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    w, h = image_input.size
    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(h, w)], num_labels=1
    )[0]
    pred_np = pred.detach().cpu().numpy()
    binary = (pred_np > 0).astype(np.uint8) * 255
    return pred_np, binary


def smoke_seggpt(device: torch.device) -> dict:
    """(c) HF SegGPT: API check (same-image) + cross-image experiment on refs."""
    from transformers import SegGptForImageSegmentation, SegGptImageProcessor

    mask_path = ensure_placeholder_mask()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model_id = "BAAI/seggpt-vit-large"
    processor = SegGptImageProcessor.from_pretrained(model_id)
    model = SegGptForImageSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()

    image_prompt = Image.open(REF_IMG).convert("RGB")
    image_input = Image.open(TEST_IMG).convert("RGB")
    # Single-channel map 0=bg, 1=nail; num_labels=1 → palette [(0,0,0),(255,255,255)]
    prompt_mask = Image.fromarray(
        (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    )

    # API / pipeline verification: prompt and query are the same image.
    recon_np, recon_bin = _seggpt_predict(
        model, processor, image_prompt, image_prompt, prompt_mask, device
    )
    cv2.imwrite(str(OUT / "seggpt_sameimage_mask.png"), recon_bin)
    ov_r = overlay_mask(np.array(image_prompt), recon_bin > 0, color=(0, 255, 0))
    cv2.imwrite(
        str(OUT / "seggpt_sameimage_overlay.jpg"), cv2.cvtColor(ov_r, cv2.COLOR_RGB2BGR)
    )

    # Experiment: ref_02 prompt → ref_01 query (may fail with crude placeholder).
    pred_np, binary = _seggpt_predict(
        model, processor, image_input, image_prompt, prompt_mask, device
    )
    cv2.imwrite(str(OUT / "seggpt_test_mask.png"), binary)
    ov = overlay_mask(np.array(image_input), binary > 0, color=(0, 255, 0))
    cv2.imwrite(str(OUT / "seggpt_test_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    elapsed = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None
    )
    info = {
        "ok": True,
        "api_ok": bool((recon_bin > 0).any()),
        "model": "seggpt",
        "checkpoint": model_id,
        "transformers": __import__("transformers").__version__,
        "device": str(device),
        "ref_image": str(REF_IMG.name),
        "test_image": str(TEST_IMG.name),
        "sameimage_mask_area_pct": float(round(100.0 * float((recon_bin > 0).mean()), 3)),
        "sameimage_unique_labels": sorted(int(x) for x in np.unique(recon_np)),
        "crossimage_unique_labels": sorted(int(x) for x in np.unique(pred_np)),
        "crossimage_mask_area_pct": float(round(100.0 * float((binary > 0).mean()), 3)),
        "runtime_s": float(round(elapsed, 3)),
        "peak_vram_mb": None if peak is None else float(round(peak, 1)),
        "processor_size": dict(processor.size) if getattr(processor, "size", None) else None,
        "feature_ensemble": "model.forward kwarg (default False); multi-prompt later",
        "note": "placeholder mask (auto dark circle), not curated",
    }
    (OUT / "seggpt.json").write_text(json.dumps(info, indent=2))
    print("[SegGPT]", info)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device}")
    ensure_placeholder_mask()
    print(f"Placeholder mask: {PLACEHOLDER_MASK}")

    results = {}
    results["sam"] = smoke_sam_official(device)

    try:
        results["persam"] = smoke_persam(device, sam_type="vit_h")
    except torch.cuda.OutOfMemoryError:
        print("ViT-H OOM — retrying PerSAM with MobileSAM (vit_t)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results["persam"] = smoke_persam(device, sam_type="vit_t")
        results["persam"]["fallback"] = "vit_t after vit_h OOM"

    results["seggpt"] = smoke_seggpt(device)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll smoke outputs written under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
