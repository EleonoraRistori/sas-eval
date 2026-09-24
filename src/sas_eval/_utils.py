from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """
    Resolve a device specification.

    Supported examples are ``"auto"``, ``"cpu"``, ``"cuda"``, and
    ``"cuda:0"``.
    """
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-capable device is available."
        )

    return resolved_device


def default_num_workers() -> int:
    """Return a conservative default number of DataLoader workers."""
    return min(8, os.cpu_count() or 1)


def derive_ig_batch_size(batch_size: int) -> int:
    """
    Derive a conservative input batch size for Integrated Gradients.

    The original IG script used a batch size of 4. We retain this cap while
    ensuring that the result is never smaller than 1.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    return max(1, min(4, batch_size // 8))