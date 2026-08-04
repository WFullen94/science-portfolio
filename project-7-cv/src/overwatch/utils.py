"""Small shared helpers."""

from __future__ import annotations


def pick_device(requested=None) -> str:
    """Resolve the compute device string ultralytics/torch expect."""
    if requested:
        return str(requested)
    import torch
    if torch.cuda.is_available():
        return "0"          # first CUDA GPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
