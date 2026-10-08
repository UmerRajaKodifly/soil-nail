from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda → mps → cpu. Logs the choice."""
    if prefer:
        dev = torch.device(prefer)
        log.info("Using device (forced): %s", dev)
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Using device: %s", dev)
    return dev


def peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()
