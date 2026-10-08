"""Segmenter protocol — device-agnostic, no module-level model globals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PreparedRefs:
    """One or more reference examples (RGB + binary mask)."""

    names: tuple[str, ...]
    images_rgb: tuple[NDArray[np.uint8], ...]
    masks: tuple[NDArray[np.uint8], ...]

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.images_rgb) == len(self.masks) > 0):
            raise ValueError("PreparedRefs lengths must match and be non-empty")


@dataclass
class SegmentOutput:
    masks: list[NDArray[np.uint8]]
    scores: list[float]
    meta: dict[str, object]


class SegmenterProtocol(Protocol):
    def prepare(self, refs: PreparedRefs) -> None: ...

    def segment(self, image_rgb: NDArray[np.uint8]) -> SegmentOutput: ...

    def close(self) -> None: ...
