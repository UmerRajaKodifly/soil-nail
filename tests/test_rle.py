"""RLE round-trip property tests."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from soilnail.rle import decode_rle, encode_rle


@given(
    h=st.integers(1, 64),
    w=st.integers(1, 64),
    data=st.binary(min_size=1, max_size=64 * 64),
)
@settings(max_examples=40, deadline=None)
def test_rle_roundtrip(h: int, w: int, data: bytes) -> None:
    flat = np.frombuffer(data[: h * w], dtype=np.uint8)
    if flat.size < h * w:
        flat = np.pad(flat, (0, h * w - flat.size))
    mask = (flat.reshape(h, w) % 2).astype(bool)
    rle = encode_rle(mask)
    back = decode_rle(rle)
    assert back.shape == (h, w)
    assert np.array_equal(back, mask)
