"""SegGPT via Hugging Face transformers (verified against installed version)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.device import free_cuda, peak_vram_mb, pick_device, reset_peak_vram
from src.preprocess import crop_prompt_pair
from src.segmenter import RefExample, Result, Segmenter

log = logging.getLogger(__name__)


def connected_components(mask: np.ndarray, min_area: int = 50) -> list[np.ndarray]:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out.append(((labels == i).astype(np.uint8) * 255))
    if not out and m.any():
        out = [m * 255]
    return out


class SegGPTSegmenter(Segmenter):
    def __init__(
        self,
        model_id: str = "BAAI/seggpt-vit-large",
        device: str | None = None,
        tight_prompt_crop: bool = False,
        crop_margin: float = 0.15,
        feature_ensemble: bool = False,
        min_cc_area: int = 50,
        save_raw_dir: str | Path | None = None,
    ):
        self.model_id = model_id
        self.device = pick_device(device)
        self.tight_prompt_crop = tight_prompt_crop
        self.crop_margin = crop_margin
        self.feature_ensemble = feature_ensemble
        self.min_cc_area = min_cc_area
        self.save_raw_dir = Path(save_raw_dir) if save_raw_dir else None
        self.processor = None
        self.model = None
        self.refs: list[RefExample] = []

    def prepare(self, refs: list[RefExample]) -> None:
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor

        self.processor = SegGptImageProcessor.from_pretrained(self.model_id)
        self.model = SegGptForImageSegmentation.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.refs = refs
        log.info("SegGPT prepared with %d ref(s), feature_ensemble=%s", len(refs), self.feature_ensemble)

    def close(self) -> None:
        self.model = None
        self.processor = None
        free_cuda()

    def _prompt_pil(self, ref: RefExample) -> tuple[Image.Image, Image.Image]:
        img, msk = ref.image_rgb, ref.mask
        if self.tight_prompt_crop:
            img, msk, box = crop_prompt_pair(img, msk, margin=self.crop_margin)
            log.info("SegGPT tight_prompt_crop box=%s", None if box is None else box.as_tuple())
        prompt_image = Image.fromarray(img)
        prompt_mask = Image.fromarray((msk > 0).astype(np.uint8))
        return prompt_image, prompt_mask

    def _raw_distance_maps(self, outputs, target_hw: tuple[int, int]) -> dict:
        """
        Pre-threshold diagnostics from post_process_semantic_segmentation logic
        (transformers SegGptImageProcessor): unnormalize pred half, L2 distance
        to palette colours. Lower distance → closer to that class colour.
        """
        masks = outputs.pred_masks
        # Predicted half (bottom after vertical concat)
        masks = masks[:, :, masks.shape[2] // 2 :, :]
        std = torch.tensor(self.processor.image_std, device=masks.device)
        mean = torch.tensor(self.processor.image_mean, device=masks.device)
        masks = masks.permute(0, 2, 3, 1) * std + mean
        masks = masks.permute(0, 3, 1, 2)
        masks = torch.clip(masks * 255, 0, 255)

        h, w = target_hw
        mask = torch.nn.functional.interpolate(masks[:1], size=(h, w), mode="nearest")[0]
        # RGB float [3,H,W]
        rgb = mask.permute(1, 2, 0).detach().cpu().numpy()

        palette = self.processor.get_palette(1)  # [(0,0,0), (255,255,255)]
        dist_bg = np.sum((rgb - np.array(palette[0], dtype=np.float32)) ** 2, axis=-1)
        dist_fg = np.sum((rgb - np.array(palette[1], dtype=np.float32)) ** 2, axis=-1)
        # Soft score: higher → more FG-like (neg FG distance, or softmax-style)
        score_fg = dist_bg - dist_fg  # >0 means closer to white than black
        pred = (dist_fg < dist_bg).astype(np.uint8)

        return {
            "rgb_pred": rgb.astype(np.float32),
            "dist_bg": dist_bg.astype(np.float32),
            "dist_fg": dist_fg.astype(np.float32),
            "score_fg": score_fg.astype(np.float32),
            "argmax_binary": pred,
            "min_dist_fg": float(dist_fg.min()),
            "min_dist_bg": float(dist_bg.min()),
            "frac_closer_to_fg": float((dist_fg < dist_bg).mean()),
            "score_fg_max": float(score_fg.max()),
            "score_fg_mean": float(score_fg.mean()),
        }

    @staticmethod
    def _save_heatmap(score: np.ndarray, path: Path) -> None:
        # Normalize to 0-255 for visualization
        s = score.astype(np.float32)
        s = s - s.min()
        if s.max() > 0:
            s = s / s.max()
        heat = (s * 255).astype(np.uint8)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(path), color)

    def segment(self, image_rgb: np.ndarray, stem: str | None = None) -> Result:
        assert self.model is not None and self.processor is not None and self.refs
        t0 = time.perf_counter()
        reset_peak_vram(self.device)

        image_input = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]

        if self.feature_ensemble and len(self.refs) > 1:
            prompt_images = []
            prompt_masks = []
            for ref in self.refs:
                pi, pm = self._prompt_pil(ref)
                prompt_images.append(pi)
                prompt_masks.append(pm)
            inputs = self.processor(
                images=[image_input] * len(prompt_images),
                prompt_images=prompt_images,
                prompt_masks=prompt_masks,
                num_labels=1,
                return_tensors="pt",
            )
            use_ensemble = True
        else:
            pi, pm = self._prompt_pil(self.refs[0])
            inputs = self.processor(
                images=image_input,
                prompt_images=pi,
                prompt_masks=pm,
                num_labels=1,
                return_tensors="pt",
            )
            use_ensemble = False

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, feature_ensemble=use_ensemble)

        raw = self._raw_distance_maps(outputs, (h, w))
        # feature_ensemble may leave batch dim = n_prompts; match target_sizes to batch
        batch = int(outputs.pred_masks.shape[0])
        preds = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)] * batch, num_labels=1
        )
        if use_ensemble and batch > 1:
            stacked = torch.stack([(p > 0).to(torch.uint8) for p in preds], dim=0)
            pred_np = (stacked.float().mean(0) > 0.5).cpu().numpy().astype(np.uint8)
        else:
            pred_np = preds[0].detach().cpu().numpy()
        binary = ((pred_np > 0).astype(np.uint8) * 255)

        diagnosis = "empty_prediction"
        if binary.any():
            diagnosis = "has_foreground"
        elif raw["frac_closer_to_fg"] > 0.001 or raw["score_fg_max"] > 0:
            # Some pixels closer to FG colour, but argmax still bg everywhere
            # (shouldn't happen if frac>0) — or weak FG affinity below threshold
            if raw["frac_closer_to_fg"] > 0:
                diagnosis = "threshold_or_palette_mismatch"
            else:
                # All pixels closer to BG; max score_fg still negative ⇒ model prefers BG
                diagnosis = "predicts_nothing_all_bg"
        else:
            diagnosis = "predicts_nothing_all_bg"

        if self.save_raw_dir is not None:
            self.save_raw_dir.mkdir(parents=True, exist_ok=True)
            tag = stem or "pred"
            np.savez_compressed(
                self.save_raw_dir / f"{tag}_seggpt_raw.npz",
                dist_fg=raw["dist_fg"],
                dist_bg=raw["dist_bg"],
                score_fg=raw["score_fg"],
                rgb_pred=raw["rgb_pred"],
                argmax_binary=raw["argmax_binary"],
            )
            self._save_heatmap(raw["score_fg"], self.save_raw_dir / f"{tag}_seggpt_heatmap.jpg")
            # Also save dist_fg (lower = more FG-like) inverted for readability
            self._save_heatmap(-raw["dist_fg"], self.save_raw_dir / f"{tag}_seggpt_inv_dist_fg.jpg")

        instances = connected_components(binary, min_area=self.min_cc_area)
        scores = [float((m > 0).sum()) for m in instances]
        elapsed = time.perf_counter() - t0
        return Result(
            masks=instances,
            scores=scores,
            meta={
                "model": "seggpt",
                "device": str(self.device),
                "runtime_s": round(elapsed, 3),
                "peak_vram_mb": peak_vram_mb(self.device),
                "n_instances": len(instances),
                "tight_prompt_crop": self.tight_prompt_crop,
                "feature_ensemble": use_ensemble,
                "height": h,
                "width": w,
                "unique_labels": sorted(int(x) for x in np.unique(pred_np)),
                "diagnosis": diagnosis,
                "raw_frac_closer_to_fg": raw["frac_closer_to_fg"],
                "raw_score_fg_max": raw["score_fg_max"],
                "raw_score_fg_mean": raw["score_fg_mean"],
                "raw_min_dist_fg": raw["min_dist_fg"],
                "raw_min_dist_bg": raw["min_dist_bg"],
            },
        )
