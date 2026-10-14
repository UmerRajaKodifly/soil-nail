"""Blocked per-walk label split: neighbours stay out of opposite splits."""

from __future__ import annotations

from soilnail.ui.labels import build_blocked_queue


def test_per_walk_quotas_and_roles() -> None:
    walks = {
        "walks": [
            {
                "archive": "walk1",
                "walk_id": "C55_W1",
                "site_id": "C55",
                "role": "in_site",
            },
            {
                "archive": "images",
                "walk_id": "F159_VID",
                "site_id": "F159",
                "role": "cross_site",
            },
        ]
    }
    ids = [f"walk1::frame_{i:04d}.jpg" for i in range(1, 101)]
    ids += [f"images::images/frame_{i:04d}.jpg" for i in range(1, 51)]
    items = build_blocked_queue(
        image_ids=ids,
        walks_cfg=walks,
        seed=42,
        quotas={"C55_W1": 20, "F159_VID": 10},
        tuning_frac=0.5,
        buffer_gap_frames=3,
    )
    assert sum(1 for i in items if i.walk_id == "C55_W1") <= 20
    assert sum(1 for i in items if i.walk_id == "F159_VID") <= 10
    assert all(i.role in ("in_site", "cross_site") for i in items)
    # No image in both splits
    tune = {i.image_id for i in items if i.split == "tuning"}
    hold = {i.image_id for i in items if i.split == "heldout"}
    assert tune.isdisjoint(hold)
