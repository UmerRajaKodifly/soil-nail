"""Adapters over legacy prototype runners.

DEVIATION D008: Phase 2 wraps `src/persam_runner.py` / `src/seggpt_runner.py`
rather than rewriting the SAM/SegGPT math in one step. Behaviour must match the
LOO golden test. A later commit can inline the implementations.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from soilnail.models.base import PreparedRefs, SegmentOutput

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _ref_hash(refs: PreparedRefs) -> str:
    h = hashlib.sha256()
    for name, img, msk in zip(refs.names, refs.images_rgb, refs.masks, strict=True):
        h.update(name.encode())
        h.update(img.tobytes())
        h.update(msk.tobytes())
    return h.hexdigest()[:16]


class PerSAMAdapter:
    def __init__(
        self,
        *,
        mode: str,
        sam_type: str,
        ckpt: Path,
        tight_prompt_crop: bool,
        crop_margin: float,
        multi_instance: bool,
        multi_k: int,
        multi_min_dist: int,
        multi_score_ratio: float,
        multi_nms_iou: float,
        train_epochs: int,
        lr: float,
        cache_dir: Path | None,
        config_hash: str,
        device: str | None = None,
    ) -> None:
        self.mode = mode
        self.sam_type = sam_type
        self.ckpt = ckpt
        self.tight_prompt_crop = tight_prompt_crop
        self.crop_margin = crop_margin
        self.multi_instance = multi_instance
        self.multi_k = multi_k
        self.multi_min_dist = multi_min_dist
        self.multi_score_ratio = multi_score_ratio
        self.multi_nms_iou = multi_nms_iou
        self.train_epochs = train_epochs
        self.lr = lr
        self.cache_dir = cache_dir
        self.config_hash = config_hash
        self.device = device
        self._inner: Any = None

    def prepare(self, refs: PreparedRefs) -> None:
        from src.persam_runner import PerSAMFSegmenter, PerSAMSegmenter
        from src.segmenter import RefExample

        if self.mode == "persam":
            seg: Any = PerSAMSegmenter(
                sam_type=self.sam_type,
                ckpt=self.ckpt,
                device=self.device,
                tight_prompt_crop=self.tight_prompt_crop,
                crop_margin=self.crop_margin,
                multi_instance=self.multi_instance,
                multi_k=self.multi_k,
                multi_min_dist=self.multi_min_dist,
                multi_score_ratio=self.multi_score_ratio,
                multi_nms_iou=self.multi_nms_iou,
            )
        elif self.mode == "persam_f":
            seg = PerSAMFSegmenter(
                sam_type=self.sam_type,
                ckpt=self.ckpt,
                device=self.device,
                tight_prompt_crop=self.tight_prompt_crop,
                crop_margin=self.crop_margin,
                multi_instance=self.multi_instance,
                multi_k=self.multi_k,
                multi_min_dist=self.multi_min_dist,
                multi_score_ratio=self.multi_score_ratio,
                multi_nms_iou=self.multi_nms_iou,
                lr=self.lr,
                train_epoch=self.train_epochs,
            )
        else:
            raise ValueError(self.mode)

        examples = [
            RefExample(image_rgb=img, mask=msk, name=name)
            for name, img, msk in zip(refs.names, refs.images_rgb, refs.masks, strict=True)
        ]

        cache_path = None
        if self.mode == "persam_f" and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{_ref_hash(refs)}_{self.config_hash}.pt"

        seg.prepare(examples)
        if self.mode == "persam_f" and cache_path is not None:
            if cache_path.is_file():
                # Local cache written by us in this method; not an untrusted checkpoint.
                blob = torch.load(  # nosec B614
                    cache_path, map_location="cpu", weights_only=False
                )
                seg.weights = blob["weights"].to(seg.device)
                seg.weights_np = blob["weights_np"]
                log.info("Loaded PerSAM-F weights from cache %s", cache_path)
            else:
                torch.save(
                    {
                        "weights": seg.weights.detach().cpu(),
                        "weights_np": seg.weights_np,
                    },
                    cache_path,
                )
                log.info("Cached PerSAM-F weights → %s", cache_path)

        self._inner = seg

    def segment(self, image_rgb: NDArray[np.uint8]) -> SegmentOutput:
        if self._inner is None:
            raise RuntimeError("prepare() not called")
        result = self._inner.segment(image_rgb)
        return SegmentOutput(
            masks=list(result.masks),
            scores=[float(s) for s in result.scores],
            meta=dict(result.meta),
        )

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
        self._inner = None


class SegGPTAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tight_prompt_crop: bool,
        crop_margin: float,
        feature_ensemble: bool,
        min_cc_area: int,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.tight_prompt_crop = tight_prompt_crop
        self.crop_margin = crop_margin
        self.feature_ensemble = feature_ensemble
        self.min_cc_area = min_cc_area
        self.device = device
        self._inner: Any = None

    def prepare(self, refs: PreparedRefs) -> None:
        from src.device import pick_device
        from src.seggpt_runner import SegGPTSegmenter
        from src.segmenter import RefExample
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor

        seg = SegGPTSegmenter(
            model_id=self.model_id,
            device=self.device,
            tight_prompt_crop=self.tight_prompt_crop,
            crop_margin=self.crop_margin,
            feature_ensemble=self.feature_ensemble,
            min_cc_area=self.min_cc_area,
        )
        # Pin HF revision (prototype did not). setattr: legacy fields typed as None.
        device = pick_device(self.device)
        # revision= pins HF commit (configs/models.yaml); bandit B615 false positive on kw.
        processor = SegGptImageProcessor.from_pretrained(  # nosec B615
            self.model_id, revision=self.revision
        )
        model_any: Any = SegGptForImageSegmentation.from_pretrained(  # nosec B615
            self.model_id, revision=self.revision
        )
        model_any.to(device)
        model_any.eval()
        legacy: Any = seg
        legacy.device = device
        legacy.processor = processor
        legacy.model = model_any
        legacy.refs = [
            RefExample(image_rgb=img, mask=msk, name=name)
            for name, img, msk in zip(refs.names, refs.images_rgb, refs.masks, strict=True)
        ]
        self._inner = legacy

    def segment(self, image_rgb: NDArray[np.uint8]) -> SegmentOutput:
        if self._inner is None:
            raise RuntimeError("prepare() not called")
        result = self._inner.segment(image_rgb)
        return SegmentOutput(
            masks=list(result.masks),
            scores=[float(s) for s in result.scores],
            meta=dict(result.meta),
        )

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
        self._inner = None
