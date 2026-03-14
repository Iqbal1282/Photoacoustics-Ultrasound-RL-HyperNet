"""
Picklable image transforms for PA and US B-scan images.
========================================================

Windows uses 'spawn' for multiprocessing, which requires all objects sent to
DataLoader worker processes to be fully picklable.  torch.nn.transforms.Lambda
wraps an arbitrary Python lambda which CANNOT be pickled across process
boundaries — causing 'Can't get local object ...<locals>.<lambda>' errors.

This module provides named, top-level callable classes that implement every
transform used in this project.  Being top-level classes they are always
picklable regardless of OS or multiprocessing start method.

Usage:
    from transforms import build_train_transform, build_val_transform
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2 as T
from typing import Tuple


# ── Individual picklable transforms ─────────────────────────────────────────

class ContrastBoost:
    """x * 1.5 - 0.2  (matches dataset.py default heuristic)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.5 - 0.2


class ClampUnit:
    """Clamp tensor values to [0, 1]."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


class MinMaxNorm:
    """Normalise tensor to [0, 1] using its own min/max."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lo = x.min()
        hi = x.max()
        return (x - lo) / (hi - lo + 1e-6)


class SpeckleNoise:
    """
    Mild multiplicative speckle noise for US images.
    x  ←  x * clip(1 + σ·N(0,1),  0,  2)
    """
    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = 1.0 + self.sigma * torch.randn_like(x)
        return x * noise.clamp(0.0, 2.0)


# ── Transform pipeline factories ─────────────────────────────────────────────

def build_train_transform(
    image_size: Tuple[int, int] = (512, 512),
    modality: str = "PA",
) -> T.Compose:
    """
    Return a picklable training transform pipeline.

    Args:
        image_size: (H, W) target spatial size.
        modality:   'PA' or 'US' — US adds mild speckle noise.
    """
    ops = [
        ContrastBoost(),
        ClampUnit(),
        T.RandomRotation(degrees=(-20, 20)),
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        MinMaxNorm(),
    ]
    if modality == "US":
        ops.append(SpeckleNoise(sigma=0.05))
    return T.Compose(ops)


def build_val_transform(
    image_size: Tuple[int, int] = (512, 512),
) -> T.Compose:
    """Return a picklable validation/test transform pipeline."""
    return T.Compose([
        ContrastBoost(),
        ClampUnit(),
        T.Resize(image_size),
        MinMaxNorm(),
    ])
