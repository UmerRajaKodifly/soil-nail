"""Segmenter implementations (PerSAM / PerSAM-F / SegGPT)."""

from __future__ import annotations

from soilnail.models.base import PreparedRefs, SegmenterProtocol
from soilnail.models.factory import build_segmenter

__all__ = ["PreparedRefs", "SegmenterProtocol", "build_segmenter"]
