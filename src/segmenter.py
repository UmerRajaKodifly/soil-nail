from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RefExample:
    image_rgb: np.ndarray
    mask: np.ndarray  # HxW uint8 0/255
    name: str = ""


@dataclass
class Result:
    masks: list[np.ndarray]  # list of HxW bool/uint8 instance masks
    scores: list[float]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def union_mask(self) -> np.ndarray:
        if not self.masks:
            h = self.meta.get("height", 0)
            w = self.meta.get("width", 0)
            return np.zeros((h, w), dtype=np.uint8)
        u = np.zeros_like(self.masks[0], dtype=np.uint8)
        for m in self.masks:
            u = np.maximum(u, (m > 0).astype(np.uint8) * 255)
        return u


class Segmenter(ABC):
    @abstractmethod
    def prepare(self, refs: list[RefExample]) -> None: ...

    @abstractmethod
    def segment(self, image_rgb: np.ndarray) -> Result: ...

    def close(self) -> None:
        """Release GPU resources."""
        pass
