"""
Dataset Utilities for PA+US RL-HyperNet
=========================================

This module provides:
  1. PAUSBScanDataset  — thin wrapper around ArpamBScanDataset that always
     yields (pa_tensor, us_tensor, label) regardless of image_type variant.
  2. create_paus_dataloaders — patient-level stratified split → DataLoaders.
  3. create_dummy_paus_dataloaders — for quick unit tests without real data.

Relationship to dataset.py:
  ArpamBScanDataset supports image_type = "PAUSradial-pair" or
  "PAUSrect-pair" which returns (pa, us, y).  PAUSBScanDataset wraps this
  and handles the channel normalisation / augmentation consistently.

All image_type choices that include both PA and US:
  - PAUSradial-pair   → (1-ch PA radial, 1-ch US radial, y)
  - PAUSrect-pair     → (1-ch PA rect,   1-ch US rect,   y)
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Import ArpamBScanDataset from dataset_arpam.py (the original dataset.py
# shipped with this package, renamed to avoid a name clash with this file).
#
# Place dataset_arpam.py in the same directory as this file, then simply:
#   from dataset_arpam import ArpamBScanDataset, LABEL_MAP_BINARY, LABELS_BINARY
#
# We add the package directory to sys.path so the import works regardless of
# the working directory, and we do it without touching __file__ (which can
# trigger a Windows ntpath recursion bug).
# ──────────────────────────────────────────────────────────────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from dataset_arpam import (          # noqa: E402
    ArpamBScanDataset,
    LABEL_MAP_BINARY,
    LABELS_BINARY,
)
# Picklable transforms (no lambdas — required for Windows DataLoader workers)
from transforms import build_train_transform, build_val_transform  # noqa: E402


# ============================================================================
# Augmentation factories
# ============================================================================

def get_train_transform(image_size: Tuple[int, int] = (512, 512),
                        modality: str = "PA") -> Callable:
    """
    Return a picklable training transform pipeline.
    Delegates to transforms.build_train_transform (no lambdas).
    """
    return build_train_transform(image_size, modality)


def get_val_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    """Return a picklable validation/test transform pipeline."""
    return build_val_transform(image_size)


# ============================================================================
# PAUSBScanDataset
# ============================================================================

class PAUSBScanDataset(Dataset):
    """
    Wrapper around ArpamBScanDataset for the PA+US RL-HyperNet.

    Always yields:
        pa_tensor [C, H, W], us_tensor [C, H, W], label (int)

    Args:
        dataset_df:       DataFrame from bscan_dataset.csv.
        image_type:       Must be 'PAUSradial-pair' or 'PAUSrect-pair'.
        target_type:      'response' (binary) or 'pathology' (T, TRG).
        pa_transform:     Optional torchvision transform for PA images.
        us_transform:     Optional torchvision transform for US images.
        in_channels:      1 (grayscale) or 3 (force_3chan).
    """

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        image_type: Literal["PAUSradial-pair", "PAUSrect-pair"] = "PAUSradial-pair",
        target_type: Literal["response", "pathology"] = "response",
        pa_transform: Optional[Callable] = None,
        us_transform: Optional[Callable] = None,
        in_channels: int = 1,
    ):
        assert image_type in ("PAUSradial-pair", "PAUSrect-pair"), (
            f"image_type must be 'PAUSradial-pair' or 'PAUSrect-pair', got '{image_type}'"
        )

        self.pa_transform = pa_transform
        self.us_transform = us_transform
        self.in_channels  = in_channels
        self.image_type   = image_type

        force_3chan = (in_channels == 3)

        # Underlying dataset returns (PA_tensor, US_tensor, y)
        self._base = ArpamBScanDataset(
            dataset_df=dataset_df,
            transform=None,           # We apply modality-specific transforms below
            target_transform=None,
            image_type=image_type,
            target_type=target_type,
            force_3chan=force_3chan,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self._base[idx]

        # PAUSradial-pair returns (pa, us, y)  [numpy or tensor]
        if len(item) == 3:
            pa_raw, us_raw, y = item
        else:
            raise ValueError(f"Expected 3-tuple from base dataset, got {len(item)}")

        # Ensure tensors
        if not isinstance(pa_raw, torch.Tensor):
            pa_raw = torch.tensor(pa_raw, dtype=torch.float32)
        if not isinstance(us_raw, torch.Tensor):
            us_raw = torch.tensor(us_raw, dtype=torch.float32)

        # Apply modality-specific transforms
        if self.pa_transform is not None:
            pa_raw = self.pa_transform(pa_raw)
        if self.us_transform is not None:
            us_raw = self.us_transform(us_raw)

        label = int(y) if not isinstance(y, int) else y
        return pa_raw, us_raw, label


# ============================================================================
# DataLoader factory
# ============================================================================

def create_paus_dataloaders(
    csv_path: str,
    image_type: Literal["PAUSradial-pair", "PAUSrect-pair"] = "PAUSradial-pair",
    target_type: Literal["response", "pathology"] = "response",
    image_size: Tuple[int, int] = (512, 512),
    in_channels: int = 1,
    batch_size: int = 16,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create stratified patient-level train / val / test DataLoaders.

    Stratification is done at the PATIENT level (column 'pid') to prevent
    data leakage across splits.  Within each patient all B-scans go to the
    same split.

    Args:
        csv_path:       Path to bscan_dataset.csv.
        image_type:     'PAUSradial-pair' or 'PAUSrect-pair'.
        target_type:    'response' or 'pathology'.
        image_size:     Spatial (H, W) to resize images to.
        in_channels:    1 or 3.
        batch_size:     Batch size.
        num_workers:    DataLoader workers.
        val_fraction:   Fraction of patients for validation.
        test_fraction:  Fraction of patients for test.
        seed:           Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)

    # Patient-level split
    pids = sorted(df["pid"].unique().tolist())
    rng.shuffle(pids)

    n_val  = int(len(pids) * val_fraction)
    n_test = int(len(pids) * test_fraction)

    test_pids  = set(pids[:n_test])
    val_pids   = set(pids[n_test:n_test + n_val])
    train_pids = set(pids[n_test + n_val:])

    train_df = df[df["pid"].isin(train_pids)].reset_index(drop=True)
    val_df   = df[df["pid"].isin(val_pids)].reset_index(drop=True)
    test_df  = df[df["pid"].isin(test_pids)].reset_index(drop=True)

    print(f"Split summary (patients / B-scans):")
    print(f"  Train: {len(train_pids)} pts / {len(train_df)} scans")
    print(f"  Val:   {len(val_pids)} pts / {len(val_df)} scans")
    print(f"  Test:  {len(test_pids)} pts / {len(test_df)} scans")

    # Transforms
    pa_train_tf = get_train_transform(image_size, "PA")
    us_train_tf = get_train_transform(image_size, "US")
    val_tf      = get_val_transform(image_size)

    train_ds = PAUSBScanDataset(
        train_df, image_type=image_type, target_type=target_type,
        pa_transform=pa_train_tf, us_transform=us_train_tf,
        in_channels=in_channels,
    )
    val_ds = PAUSBScanDataset(
        val_df, image_type=image_type, target_type=target_type,
        pa_transform=val_tf, us_transform=val_tf,
        in_channels=in_channels,
    )
    test_ds = PAUSBScanDataset(
        test_df, image_type=image_type, target_type=target_type,
        pa_transform=val_tf, us_transform=val_tf,
        in_channels=in_channels,
    )

    # Windows uses 'spawn': num_workers > 0 requires picklable workers.
    # All transforms here are picklable, but set 0 as a safe default on Windows.
    nw = 0 if platform.system() == "Windows" else num_workers
    if nw != num_workers:
        print("  [Windows] num_workers forced to 0")

    _dl_kwargs = dict(num_workers=nw, pin_memory=(nw > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **_dl_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, **_dl_kwargs)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, **_dl_kwargs)

    return train_loader, val_loader, test_loader


# ============================================================================
# Dummy loaders for unit tests
# ============================================================================

class _DummyPAUSDataset(Dataset):
    """Random tensors; mimics PAUSBScanDataset output."""

    def __init__(self, n: int = 100, in_channels: int = 1,
                 image_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 2):
        self.n = n
        self.C = in_channels
        self.H, self.W = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pa = torch.rand(self.C, self.H, self.W)
        us = torch.rand(self.C, self.H, self.W)
        y  = torch.randint(0, self.num_classes, ()).item()
        return pa, us, y


def create_dummy_paus_dataloaders(
    batch_size: int = 8,
    num_train: int = 200,
    num_val: int = 50,
    num_test: int = 50,
    in_channels: int = 1,
    image_size: Tuple[int, int] = (512, 512),
    num_classes: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Dummy data loaders for quick testing without real data."""
    kwargs = dict(in_channels=in_channels, image_size=image_size,
                  num_classes=num_classes)
    train_loader = DataLoader(
        _DummyPAUSDataset(num_train, **kwargs),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        _DummyPAUSDataset(num_val, **kwargs),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        _DummyPAUSDataset(num_test, **kwargs),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    print("=== Dummy DataLoader self-test ===")
    train_l, val_l, test_l = create_dummy_paus_dataloaders(
        batch_size=4, num_train=20, num_val=8, num_test=8,
        in_channels=1, image_size=(256, 256),
    )
    pa, us, y = next(iter(train_l))
    print(f"PA batch:    {pa.shape}")
    print(f"US batch:    {us.shape}")
    print(f"Labels:      {y}")
    print("Dummy DataLoader OK.")
