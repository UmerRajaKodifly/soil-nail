"""Build segmenters from configs/models.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from soilnail.config import load_yaml
from soilnail.models.adapters import PerSAMAdapter, SegGPTAdapter


def _config_hash(blob: dict[str, Any]) -> str:
    raw = json.dumps(blob, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def build_segmenter(
    model_name: str,
    models_cfg_path: Path,
    *,
    repo_root: Path,
    tight_prompt_crop: bool | None = None,
    multi_instance: bool | None = None,
    feature_ensemble: bool = False,
    device: str | None = None,
) -> Any:
    cfg = load_yaml(models_cfg_path)
    chash = _config_hash(cfg)
    sam = cfg["sam"]
    ckpt = repo_root / sam["checkpoint"]
    crop_cfg = cfg["tight_prompt_crop"]
    mi = cfg["multi_instance"]
    use_crop = crop_cfg["enabled_default"] if tight_prompt_crop is None else tight_prompt_crop
    use_mi = mi["enabled_default"] if multi_instance is None else multi_instance

    if model_name in ("persam", "persam_f"):
        pf = cfg["persam_f"]
        cache_dir = repo_root / cfg["cache"]["persam_f_dir"]
        return PerSAMAdapter(
            mode=model_name,
            sam_type=sam["type"],
            ckpt=ckpt,
            tight_prompt_crop=bool(use_crop),
            crop_margin=float(crop_cfg["margin"]),
            multi_instance=bool(use_mi),
            multi_k=int(mi["top_k"]),
            multi_min_dist=int(mi["min_dist_px"]),
            multi_score_ratio=float(mi["score_ratio"]),
            multi_nms_iou=float(mi["nms_iou"]),
            train_epochs=int(pf["train_epochs"]),
            lr=float(pf["lr"]),
            cache_dir=cache_dir if model_name == "persam_f" else None,
            config_hash=chash,
            device=device,
        )
    if model_name == "seggpt":
        sg = cfg["seggpt"]
        return SegGPTAdapter(
            model_id=str(sg["model_id"]),
            revision=str(sg["revision"]),
            tight_prompt_crop=bool(use_crop),
            crop_margin=float(crop_cfg["margin"]),
            feature_ensemble=feature_ensemble,
            min_cc_area=int(cfg["seggpt_instances"]["min_cc_area_px"]),
            device=device,
        )
    raise ValueError(f"unknown model {model_name!r}")
