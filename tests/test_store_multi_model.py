"""ResultsStore must keep per-model rows (D024)."""

from __future__ import annotations

from pathlib import Path

from soilnail.schemas import Detection, ImageResult, RLE
from soilnail.store import ResultsStore


def _det(image_id: str, model: str, instance_id: int = 0) -> Detection:
    return Detection(
        image_id=image_id,
        model=model,
        config_id=f"cfg_{model}",
        instance_id=instance_id,
        mask_rle=RLE(counts=[0, 1], size=[1, 1]),
        bbox_xywh=[0, 0, 1, 1],
        area_px=1,
        area_fraction=1.0,
        raw_score=0.5,
    )


def test_multi_model_no_overwrite(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "r.sqlite")
    a = ImageResult(
        image_id="walk1::f.jpg",
        status="succeeded",
        detections=[_det("walk1::f.jpg", "persam")],
        model="persam",
        config_id="cfg_persam",
    )
    b = ImageResult(
        image_id="walk1::f.jpg",
        status="succeeded",
        detections=[_det("walk1::f.jpg", "seggpt")],
        model="seggpt",
        config_id="cfg_seggpt",
    )
    store.write_image_result("run1", a)
    store.write_image_result("run1", b)
    assert store.list_models("run1") == ["persam", "seggpt"]
    ra = store.get_image_result("run1", "walk1::f.jpg", model="persam")
    rb = store.get_image_result("run1", "walk1::f.jpg", model="seggpt")
    assert ra is not None and ra.model == "persam"
    assert rb is not None and rb.model == "seggpt"
    assert len(store.list_rows("run1")) == 2
    store.close()


def test_empty_detections_still_keyed_by_model(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "r.sqlite")
    empty = ImageResult(
        image_id="img",
        status="succeeded",
        detections=[],
        model="persam_f",
        config_id="cfg",
    )
    store.write_image_result("run1", empty)
    assert store.get_image_result("run1", "img", model="persam_f") is not None
    store.close()
