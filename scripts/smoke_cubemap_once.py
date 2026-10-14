#!/usr/bin/env python3
import time
from pathlib import Path

from soilnail.config import load_yaml
from soilnail.models.factory import build_segmenter
from soilnail.models.sam_cache import SamEmbeddingCache
from soilnail.runner import build_prepared_refs, load_rgb_exif, segment_cubemap_image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    exp = load_yaml(ROOT / "configs/experiments.yaml")
    cache = SamEmbeddingCache(ROOT / "data/caches/sam_face_embeddings", enabled=True)
    seg = build_segmenter(
        "persam",
        ROOT / "configs/models.yaml",
        repo_root=ROOT,
        tight_prompt_crop=False,
        multi_instance=True,
        feature_ensemble=True,
        sam_embedding_cache=cache,
    )
    refs = build_prepared_refs(
        ROOT / "data/refs/soil_nail/images",
        ROOT / "data/refs/soil_nail/masks",
        list(exp["references"]["single"]),
    )
    print("prepare start", flush=True)
    seg.prepare(refs)
    print("prepare done", flush=True)
    rgb = load_rgb_exif(ROOT / "data/site_walk_extracted/images/images/frame_0001.jpg")
    t0 = time.perf_counter()
    res = segment_cubemap_image(
        seg,
        rgb,
        image_id="t",
        model="persam",
        config_id="t",
        face_size=1024,
        fov_deg=90,
        margin_deg=15,
        yaw_offset_deg=0,
        instance_mode=True,
        min_cc_area=1,
        nms_iou=0.5,
        area_warn_solid_angle_sr=0.05,
        merge_deg=5,
    )
    print(
        "status",
        res.status,
        "ndet",
        len(res.detections),
        "dt",
        round(time.perf_counter() - t0, 2),
        flush=True,
    )
    seg.close()


if __name__ == "__main__":
    main()
