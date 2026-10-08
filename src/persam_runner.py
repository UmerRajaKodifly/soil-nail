"""PerSAM / PerSAM-F wrappers. Reproduces upstream pipelines; does not edit third_party."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.device import free_cuda, peak_vram_mb, pick_device, reset_peak_vram
from src.preprocess import crop_prompt_pair, to_persam_rgb_mask
from src.segmenter import RefExample, Result, Segmenter

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PERSAM_ROOT = ROOT / "third_party" / "Personalize-SAM"
if str(PERSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PERSAM_ROOT))


def _point_selection_persam(mask_sim: torch.Tensor, topk: int = 1):
    """From upstream persam.py — positive top-k + negative bottom-k."""
    _w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy_np = torch.cat((topk_y, topk_x), dim=0).permute(1, 0).cpu().numpy()
    topk_label = np.array([1] * topk)

    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy_np = torch.cat((last_y, last_x), dim=0).permute(1, 0).cpu().numpy()
    last_label = np.array([0] * topk)
    return topk_xy_np, topk_label, last_xy_np, last_label


def _point_selection_f(mask_sim: torch.Tensor, topk: int = 1):
    """From upstream persam_f.py — positive only."""
    _w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy_np = torch.cat((topk_y, topk_x), dim=0).permute(1, 0).cpu().numpy()
    return topk_xy_np, np.array([1] * topk)


class Mask_Weights(nn.Module):
    """Upstream Mask_Weights: 2 learnable parameters."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def _dice_loss(inputs, targets, num_masks=1):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def _focal_loss(inputs, targets, num_masks=1, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


def _local_maxima(sim: np.ndarray, k: int, min_dist: int, ratio: float) -> list[tuple[int, int, float]]:
    """EXTENSION: top-k peaks for multi-instance mode."""
    from scipy.ndimage import maximum_filter

    thr = float(sim.max()) * ratio
    peaks = (sim >= thr) & (sim == maximum_filter(sim, size=max(3, min_dist)))
    ys, xs = np.where(peaks)
    vals = sim[ys, xs]
    order = np.argsort(-vals)
    chosen: list[tuple[int, int, float]] = []
    for i in order:
        y, x, v = int(ys[i]), int(xs[i]), float(vals[i])
        if any((y - cy) ** 2 + (x - cx) ** 2 < min_dist**2 for cy, cx, _ in chosen):
            continue
        chosen.append((y, x, v))
        if len(chosen) >= k:
            break
    return chosen


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 0.0


class _BasePerSAM(Segmenter):
    def __init__(
        self,
        sam_type: str = "vit_h",
        ckpt: str | Path | None = None,
        device: str | None = None,
        tight_prompt_crop: bool = False,
        crop_margin: float = 0.15,
        multi_instance: bool = False,
        multi_k: int = 5,
        multi_min_dist: int = 40,
        multi_score_ratio: float = 0.5,
        multi_nms_iou: float = 0.5,
        lr: float = 1e-3,
        train_epoch: int = 1000,
    ):
        self.sam_type = sam_type
        self.device = pick_device(device)
        self.tight_prompt_crop = tight_prompt_crop
        self.crop_margin = crop_margin
        self.multi_instance = multi_instance
        self.multi_k = multi_k
        self.multi_min_dist = multi_min_dist
        self.multi_score_ratio = multi_score_ratio
        self.multi_nms_iou = multi_nms_iou
        self.lr = lr
        self.train_epoch = train_epoch
        if ckpt is None:
            ckpt = (
                ROOT / "weights" / "sam_vit_h_4b8939.pth"
                if sam_type == "vit_h"
                else ROOT / "weights" / "mobile_sam.pt"
            )
        self.ckpt = Path(ckpt)
        self.sam = None
        self.predictor = None
        self.target_feat = None
        self.target_embedding = None
        self.weights = None
        self.weights_np = None
        self._mode = "persam"

    def _load_sam(self):
        from per_segment_anything import SamPredictor, sam_model_registry

        if not self.ckpt.is_file():
            raise FileNotFoundError(self.ckpt)
        sam = sam_model_registry[self.sam_type](checkpoint=str(self.ckpt))
        sam.to(device=self.device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        self.sam = sam
        self.predictor = SamPredictor(sam)

    def close(self) -> None:
        self.sam = None
        self.predictor = None
        free_cuda()

    def _avg_target_from_refs(self, refs: list[RefExample], use_mean_max: bool):
        """Single-ref upstream behaviour; multi-ref averages target features (EXTENSION)."""
        feats = []
        embeddings = []
        for ref in refs:
            img, msk = ref.image_rgb, ref.mask
            if self.tight_prompt_crop:
                img, msk, box = crop_prompt_pair(img, msk, margin=self.crop_margin)
                log.info("tight_prompt_crop=%s box=%s", True, None if box is None else box.as_tuple())
            ref_mask_rgb = to_persam_rgb_mask(msk)
            ref_mask_t = self.predictor.set_image(img, ref_mask_rgb)
            ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
            ref_mask_t = F.interpolate(ref_mask_t, size=ref_feat.shape[0:2], mode="bilinear")
            ref_mask_t = ref_mask_t.squeeze()[0]
            target_feat = ref_feat[ref_mask_t > 0]
            if target_feat.numel() == 0:
                raise ValueError(f"Empty reference mask after encode: {ref.name}")
            if use_mean_max:
                target = (torch.max(target_feat, dim=0)[0] / 2 + target_feat.mean(0) / 2).unsqueeze(0)
            else:
                target = target_feat.mean(0).unsqueeze(0)
            target_n = target / target.norm(dim=-1, keepdim=True)
            feats.append(target_n)
            embeddings.append(target.unsqueeze(0))
        self.target_feat = torch.stack(feats, dim=0).mean(0)
        self.target_feat = self.target_feat / self.target_feat.norm(dim=-1, keepdim=True)
        self.target_embedding = torch.stack(embeddings, dim=0).mean(0)

    def _similarity_map(self, image_rgb: np.ndarray) -> torch.Tensor:
        self.predictor.set_image(image_rgb)
        test_feat = self.predictor.features.squeeze()
        c, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(c, h * w)
        sim = self.target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()
        return sim

    def _predict_persam_at_points(self, sim: torch.Tensor, topk_xy, topk_label) -> tuple[np.ndarray, float]:
        sim_n = (sim - sim.mean()) / torch.std(sim)
        sim_n = F.interpolate(sim_n.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim_n.sigmoid_().unsqueeze(0).flatten(3)

        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=False,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding,
        )
        best_idx = 0
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        y, x = np.nonzero(masks[best_idx])
        if len(x) == 0:
            return masks[best_idx], float(scores[best_idx])
        input_box = np.array([x.min(), y.min(), x.max(), y.max()])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx], float(scores[best_idx])

    def _predict_persam_f_at_points(self, topk_xy, topk_label) -> tuple[np.ndarray, float]:
        masks, scores, logits, logits_high = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True,
        )
        logits_high = logits_high * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()
        logits_w = logits * self.weights_np[..., None]
        logit = logits_w.sum(0)

        y, x = np.nonzero(mask)
        if len(x) == 0:
            best_idx = int(np.argmax(scores))
            return masks[best_idx], float(scores[best_idx])
        input_box = np.array([x.min(), y.min(), x.max(), y.max()])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        y, x = np.nonzero(masks[best_idx])
        if len(x) == 0:
            return masks[best_idx], float(scores[best_idx])
        input_box = np.array([x.min(), y.min(), x.max(), y.max()])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx], float(scores[best_idx])

    def segment(self, image_rgb: np.ndarray) -> Result:
        assert self.predictor is not None and self.target_feat is not None
        t0 = time.perf_counter()
        reset_peak_vram(self.device)
        sim = self._similarity_map(image_rgb)
        h, w = image_rgb.shape[:2]
        masks_out: list[np.ndarray] = []
        scores_out: list[float] = []

        if self.multi_instance:
            # EXTENSION — not in original paper
            sim_np = sim.detach().cpu().numpy()
            peaks = _local_maxima(sim_np, self.multi_k, self.multi_min_dist, self.multi_score_ratio)
            if not peaks:
                y, x = np.unravel_index(int(sim_np.argmax()), sim_np.shape)
                peaks = [(int(y), int(x), float(sim_np[y, x]))]
            for y, x, _v in peaks:
                if self._mode == "persam":
                    # pos at peak, neg at global min (upstream-style pair)
                    _, _, last_xy, last_label = _point_selection_persam(sim, topk=1)
                    topk_xy = np.array([[x, y]], dtype=np.float32)
                    topk_label = np.array([1])
                    topk_xy = np.concatenate([topk_xy, last_xy], axis=0)
                    topk_label = np.concatenate([topk_label, last_label], axis=0)
                    m, s = self._predict_persam_at_points(sim, topk_xy, topk_label)
                else:
                    topk_xy = np.array([[x, y]], dtype=np.float32)
                    topk_label = np.array([1])
                    m, s = self._predict_persam_f_at_points(topk_xy, topk_label)
                masks_out.append(m.astype(np.uint8) * 255 if m.dtype == bool else (m > 0).astype(np.uint8) * 255)
                scores_out.append(s)
            # NMS
            keep = []
            order = np.argsort(-np.array(scores_out))
            suppressed = set()
            for i in order:
                if i in suppressed:
                    continue
                keep.append(int(i))
                for j in order:
                    if j in suppressed or j == i:
                        continue
                    if _mask_iou(masks_out[i] > 0, masks_out[j] > 0) >= self.multi_nms_iou:
                        suppressed.add(int(j))
            masks_out = [masks_out[i] for i in keep]
            scores_out = [scores_out[i] for i in keep]
        else:
            if self._mode == "persam":
                topk_xy_i, topk_label_i, last_xy_i, last_label_i = _point_selection_persam(sim, topk=1)
                topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
                topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
                m, s = self._predict_persam_at_points(sim, topk_xy, topk_label)
            else:
                topk_xy, topk_label = _point_selection_f(sim, topk=1)
                m, s = self._predict_persam_f_at_points(topk_xy, topk_label)
            masks_out = [(m > 0).astype(np.uint8) * 255]
            scores_out = [s]

        elapsed = time.perf_counter() - t0
        return Result(
            masks=masks_out,
            scores=scores_out,
            meta={
                "model": self._mode,
                "device": str(self.device),
                "runtime_s": round(elapsed, 3),
                "peak_vram_mb": peak_vram_mb(self.device),
                "n_instances": len(masks_out),
                "tight_prompt_crop": self.tight_prompt_crop,
                "multi_instance": self.multi_instance,
                "sam_type": self.sam_type,
                "height": h,
                "width": w,
            },
        )


class PerSAMSegmenter(_BasePerSAM):
    def prepare(self, refs: list[RefExample]) -> None:
        self._mode = "persam"
        self._load_sam()
        # Upstream uses single ref; multi-ref average is EXTENSION
        self._avg_target_from_refs(refs, use_mean_max=False)
        log.info("PerSAM prepared with %d ref(s)", len(refs))


class PerSAMFSegmenter(_BasePerSAM):
    def prepare(self, refs: list[RefExample]) -> None:
        self._mode = "persam_f"
        self._load_sam()
        # Fine-tune on first ref (upstream is one-shot); extra refs only for target feat avg
        ref = refs[0]
        img, msk = ref.image_rgb, ref.mask
        if self.tight_prompt_crop:
            img, msk, box = crop_prompt_pair(img, msk, margin=self.crop_margin)
            log.info("PerSAM-F tight_prompt_crop box=%s", None if box is None else box.as_tuple())

        ref_mask_rgb = to_persam_rgb_mask(msk)
        gt_mask = torch.tensor(ref_mask_rgb)[:, :, 0] > 0
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).to(self.device)

        ref_mask_t = self.predictor.set_image(img, ref_mask_rgb)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        ref_mask_t = F.interpolate(ref_mask_t, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask_t = ref_mask_t.squeeze()[0]
        target_feat = ref_feat[ref_mask_t > 0]
        target_feat_mean = target_feat.mean(0)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        target = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
        target = target / target.norm(dim=-1, keepdim=True)
        self.target_feat = target
        self.target_embedding = target.unsqueeze(0)

        # Location prior on the reference itself (upstream)
        h, w, c = ref_feat.shape
        ref_feat_n = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat_n = ref_feat_n.permute(2, 0, 1).reshape(c, h * w)
        sim = target @ ref_feat_n
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()
        topk_xy, topk_label = _point_selection_f(sim, topk=1)

        mask_weights = Mask_Weights().to(self.device)
        mask_weights.train()
        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=self.lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_epoch)
        # Upstream calls predict() each epoch; predict_torch is @torch.no_grad(), so the
        # decoder is frozen and only the 2 mask weights train against fixed multimask logits.
        self._finetune_mask_weights(mask_weights, optimizer, scheduler, topk_xy, topk_label, gt_mask)

        mask_weights.eval()
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        self.weights = weights.detach()
        self.weights_np = weights.detach().cpu().numpy()
        log.info("PerSAM-F mask weights:\n%s", self.weights_np)

        # If multiple refs: refresh target feat as mean-max average (EXTENSION)
        if len(refs) > 1:
            self._avg_target_from_refs(refs, use_mean_max=True)

    def _finetune_mask_weights(self, mask_weights, optimizer, scheduler, topk_xy, topk_label, gt_mask):
        """Reproduce PerSAM-F training despite @torch.no_grad on predict_torch.

        Upstream persam_f.py calls predictor.predict in the train loop; their
        predict_torch is decorated with @torch.no_grad(), so gradients would not
        flow through the decoder either. The 2 parameters still update if
        logits_high requires grad — in practice upstream relies on logits_high
        being a tensor that participates in the weighted sum with mask_weights.

        We keep mask_weights as the only trainable params and attach them to
        detached logits_high (same effective graph as weighting detached masks).
        """
        # One forward to get detached multimask logits at high res
        with torch.no_grad():
            _m, _s, logits, logits_high = self.predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True,
            )
        # logits_high: [3, H, W] tensor on device (may already be detached)
        base = logits_high.detach().flatten(1)  # [3, HW]

        for epoch in range(self.train_epoch):
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_w = base * weights
            logits_sum = logits_w.sum(0).unsqueeze(0)
            loss = _dice_loss(logits_sum, gt_mask) + _focal_loss(logits_sum, gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch % 200 == 0:
                log.info(
                    "PerSAM-F epoch %d/%d loss=%.4f",
                    epoch,
                    self.train_epoch,
                    float(loss.item()),
                )
