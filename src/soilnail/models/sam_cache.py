"""Share SAM image embeddings across PerSAM / PerSAM-F / references.

SAM's image encoder output depends only on the target image (and weights),
not on the reference or PerSAM vs PerSAM-F. Cache key = face bytes hash +
weights file hash.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def face_content_hash(rgb: NDArray[np.uint8]) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()[:24]


_WEIGHTS_HASH_MEM: dict[str, str] = {}


def weights_hash(ckpt: Path) -> str:
    key = str(ckpt.resolve())
    if key in _WEIGHTS_HASH_MEM:
        return _WEIGHTS_HASH_MEM[key]
    h = hashlib.sha256()
    with ckpt.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()[:16]
    _WEIGHTS_HASH_MEM[key] = digest
    return digest


@dataclass
class CachedEmbedding:
    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    dtype: str = "float32"


class SamEmbeddingCache:
    def __init__(self, dir_path: Path, *, enabled: bool = True) -> None:
        self.dir_path = dir_path
        self.enabled = enabled
        self._mem: dict[str, CachedEmbedding] = {}
        if enabled:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _key(self, rgb: NDArray[np.uint8], ckpt: Path) -> str:
        return f"{face_content_hash(rgb)}_{weights_hash(ckpt)}"

    def get(self, rgb: NDArray[np.uint8], ckpt: Path) -> CachedEmbedding | None:
        if not self.enabled:
            return None
        key = self._key(rgb, ckpt)
        if key in self._mem:
            return self._mem[key]
        path = self.dir_path / f"{key}.pt"
        if not path.is_file():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
        emb = CachedEmbedding(
            features=blob["features"],
            original_size=tuple(blob["original_size"]),
            input_size=tuple(blob["input_size"]),
            dtype=str(blob.get("dtype", "float32")),
        )
        self._mem[key] = emb
        return emb

    def put(self, rgb: NDArray[np.uint8], ckpt: Path, emb: CachedEmbedding) -> None:
        if not self.enabled:
            return
        key = self._key(rgb, ckpt)
        self._mem[key] = emb
        path = self.dir_path / f"{key}.pt"
        torch.save(
            {
                "features": emb.features.detach().cpu(),
                "original_size": emb.original_size,
                "input_size": emb.input_size,
                "dtype": emb.dtype,
            },
            path,
        )


def apply_embedding_to_predictor(
    predictor: Any, emb: CachedEmbedding, device: torch.device
) -> None:
    """Restore a previously computed SAM image embedding onto a SamPredictor."""
    predictor.features = emb.features.to(device=device)
    predictor.original_size = emb.original_size
    predictor.input_size = emb.input_size
    predictor.is_image_set = True


def capture_embedding_from_predictor(predictor: Any) -> CachedEmbedding:
    return CachedEmbedding(
        features=predictor.features.detach().cpu().clone(),
        original_size=tuple(predictor.original_size),
        input_size=tuple(predictor.input_size),
        dtype=str(predictor.features.dtype).replace("torch.", ""),
    )


def ensure_image_embedding(
    predictor: Any,
    rgb: NDArray[np.uint8],
    ckpt: Path,
    cache: SamEmbeddingCache,
    device: torch.device,
) -> bool:
    """Set predictor image embedding from cache or by computing. Returns True if cache hit."""
    hit = cache.get(rgb, ckpt)
    if hit is not None:
        apply_embedding_to_predictor(predictor, hit, device)
        return True
    predictor.set_image(rgb)
    cache.put(rgb, ckpt, capture_embedding_from_predictor(predictor))
    return False
