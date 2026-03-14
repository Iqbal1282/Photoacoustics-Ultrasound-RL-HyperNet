"""
Patient-Specific Normalisation from Normal Tissue Scans
=========================================================

WHY
---
PA and US signal intensities vary across patients due to laser coupling
efficiency, skin melanin, probe pressure, and bowel gas.  Global min-max
normalisation (the existing default) cannot remove these offsets — a model
trained this way learns patient-level intensity differences instead of
tumour-specific features.

Using each patient's own Normal-tissue scans as a reference baseline is the
clinically correct approach: the tumour signal becomes a *deviation* from
that patient's healthy tissue, which is exactly the contrast a radiologist
uses when reading the scan.

WHAT THIS MODULE PROVIDES
--------------------------
1.  compute_normal_stats()  — walks the data root, loads every Normal scan for
    each patient, and computes per-patient per-modality mean (μ) and std (σ)
    on normalised [0,1] pixel values.  Results are saved to a JSON file.

2.  NormalTissueNorm        — a picklable transform class that applies
        x ← clamp( (x − μ) / (σ + ε),  clip_lo,  clip_hi )
    then rescales to [0, 1].  Fully picklable; safe on Windows DataLoader.

3.  FallbackNorm            — used when a patient has no Normal scans.
    Applies the standard ContrastBoost → ClampUnit → MinMaxNorm pipeline.

4.  build_normalised_transform()  — drop-in replacement for
    build_train_transform / build_val_transform that inserts the correct
    NormalTissueNorm (or FallbackNorm) as the first step.

5.  NormalisedPAUSDataset   — wraps ArpamBScanDataset and applies
    patient-specific transforms looked up per sample from the stats dict.

6.  create_normalised_dataloaders()  — drop-in replacement for
    create_paus_dataloaders() in dataset.py.

EXPECTED FOLDER STRUCTURE
--------------------------
    root_dir/
        <date>/
            <patient_id>/
                Normal/
                    <scan_001>/
                        USradial.tif
                        PAradial.tif
                    <scan_002>/
                        ...
                Tumor/
                    <scan_003>/
                        ...

USAGE
-----
# Step 1 — compute stats once (run from project root)
python normal_normalisation.py ^
    --data_root  path/to/invivo ^
    --out        data/normal_stats.json

# Dry-run: just show which Normal folders would be found
python normal_normalisation.py --data_root path/to/invivo --dry_run

# Step 2 — use in training
python training.py ^
    --csv        data/bscan_dataset.csv ^
    --normal_stats data/normal_stats.json ^
    --pa_ckpt    checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt    checkpoints/US_encoder_supervised_best.pth ^
    --epochs 50

# Step 3 — use in inference
python predict.py ^
    --model_ckpt checkpoints/best_model.pth ^
    --data_root  path/to/invivo ^
    --normal_stats data/normal_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ── sys.path guard ────────────────────────────────────────────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)


# =============================================================================
# Image finder helpers
# =============================================================================

_US_STEMS = ["USradial", "usradial", "US_radial", "USrect", "US"]
_PA_STEMS = ["PAradial", "paradial", "PA_radial", "PArect", "PA"]
_EXTS     = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG",
             ".jpg", ".jpeg", ".JPG", ".JPEG"]


def _find(folder: Path, stems: List[str]) -> Optional[Path]:
    for stem in stems:
        for ext in _EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def find_normal_images(scan_folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (us_path, pa_path) from a single scan folder, either may be None."""
    return _find(scan_folder, _US_STEMS), _find(scan_folder, _PA_STEMS)


# =============================================================================
# Statistics computation
# =============================================================================

def compute_normal_stats(
    data_root: str,
    csv_path: Optional[str] = None,
    normal_subdir: str = "Normal",
    resize_to: Tuple[int, int] = (512, 512),
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Walk the data root and compute per-patient μ and σ from Normal scans.

    Discovery modes
    ---------------
    A) CSV mode  (csv_path provided):
       Uses the 'pid' column from bscan_dataset.csv to find patients, then
       searches for  <data_root>/**/<pid>/Normal/  at up to 2 levels deep.

    B) Filesystem mode  (csv_path = None):
       Walks  <data_root>/<date>/<patient>/Normal/  directly.

    Args:
        data_root:     Root directory.
        csv_path:      Optional path to bscan_dataset.csv.
        normal_subdir: Name of the normal-tissue subfolder (default: 'Normal').
        resize_to:     (H, W) to resize images before computing stats.
                       Must match the training image_size.
        verbose:       Print per-patient statistics.

    Returns:
        stats dict: { "<pid>": { "PA": {"mean": float, "std": float,
                                         "n_scans": int},
                                  "US": {...} } }
        The key "__fallback__" is NOT added here; call compute_global_fallback()
        separately to get it.
    """
    root = Path(data_root)

    # ── Auto-detect folder depth and discover patient Normal directories ──
    #
    # Supported layouts (detected automatically):
    #
    #   Shallow  (your data):
    #       <root>/<patient_id>/<normal_subdir>/<scan_folder>/
    #       e.g.  data/258/normal/20240509 invivo 258_.../PAradial.tiff
    #
    #   Deep  (original assumed layout):
    #       <root>/<date>/<patient_id>/<normal_subdir>/<scan_folder>/
    #
    patient_normal_dirs: Dict[str, Path] = {}

    if csv_path is not None:
        df   = pd.read_csv(csv_path)
        pids = df["pid"].unique().tolist()
        for pid in pids:
            pid_s = str(pid)
            # Try shallow first, then deep (one extra level)
            for candidate in [
                root / pid_s / normal_subdir,
                *[d / pid_s / normal_subdir
                  for d in root.iterdir() if d.is_dir()],
            ]:
                if candidate.exists():
                    patient_normal_dirs[pid_s] = candidate
                    break
    else:
        for level1 in sorted(root.iterdir()):
            if not level1.is_dir():
                continue
            # Shallow: <root>/<patient_id>/normal/
            ndir_shallow = level1 / normal_subdir
            if ndir_shallow.exists():
                patient_normal_dirs[level1.name] = ndir_shallow
                continue
            # Deep: <root>/<date>/<patient_id>/normal/
            for level2 in sorted(level1.iterdir()):
                if not level2.is_dir():
                    continue
                ndir_deep = level2 / normal_subdir
                if ndir_deep.exists():
                    pid = f"{level1.name}/{level2.name}"
                    patient_normal_dirs[pid] = ndir_deep

    if not patient_normal_dirs:
        raise RuntimeError(
            f"No '{normal_subdir}' folders found under {root}.\n"
            f"Tried both layouts:\n"
            f"  Shallow: <root>/<patient_id>/{normal_subdir}/<scan>/\n"
            f"  Deep:    <root>/<date>/<patient_id>/{normal_subdir}/<scan>/\n"
            f"Check --data_root and --normal_subdir."
        )

    if verbose:
        print(f"Found {len(patient_normal_dirs)} patients with "
              f"'{normal_subdir}' folders.")

    # ── Per-patient statistics ─────────────────────────────────────────────
    stats: Dict[str, Dict] = {}

    for pid, normal_dir in sorted(patient_normal_dirs.items()):
        pa_arrays: List[np.ndarray] = []
        us_arrays: List[np.ndarray] = []

        scan_folders = [d for d in sorted(normal_dir.iterdir()) if d.is_dir()]
        if not scan_folders:
            # Images directly in the Normal folder (no scan sub-folders)
            scan_folders = [normal_dir]

        for scan_folder in scan_folders:
            us_path, pa_path = find_normal_images(scan_folder)

            for path, arrays in [(pa_path, pa_arrays), (us_path, us_arrays)]:
                if path is not None:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (resize_to[1], resize_to[0]),
                                         interpolation=cv2.INTER_LINEAR)
                        arrays.append(img.astype(np.float32) / 255.0)

        pid_stats: Dict[str, Optional[Dict]] = {}

        for key, arrays in [("PA", pa_arrays), ("US", us_arrays)]:
            if arrays:
                stack = np.stack(arrays)   # (N, H, W)
                pid_stats[key] = {
                    "mean":    float(stack.mean()),
                    "std":     float(stack.std()),
                    "n_scans": len(arrays),
                    "n_pixels": int(stack.size),
                }
            else:
                pid_stats[key] = None

        stats[pid] = pid_stats

        if verbose:
            for key in ("PA", "US"):
                s = pid_stats[key]
                info = (f"μ={s['mean']:.4f}  σ={s['std']:.4f}  "
                        f"({s['n_scans']} scans)"
                        if s else "NO IMAGES")
                print(f"  {pid:35s}  {key}: {info}")

    return stats


def compute_global_fallback(stats: Dict) -> Dict[str, Dict]:
    """
    Average per-patient stats across all patients to produce a global fallback
    used for patients that have no Normal scans in the stats dict.
    """
    pa_means, pa_stds, us_means, us_stds = [], [], [], []

    for pid_stats in stats.values():
        if isinstance(pid_stats, dict):
            if pid_stats.get("PA"):
                pa_means.append(pid_stats["PA"]["mean"])
                pa_stds.append(pid_stats["PA"]["std"])
            if pid_stats.get("US"):
                us_means.append(pid_stats["US"]["mean"])
                us_stds.append(pid_stats["US"]["std"])

    fallback: Dict[str, Dict] = {}
    if pa_means:
        fallback["PA"] = {"mean": float(np.mean(pa_means)),
                           "std":  float(np.mean(pa_stds))}
    if us_means:
        fallback["US"] = {"mean": float(np.mean(us_means)),
                           "std":  float(np.mean(us_stds))}
    return fallback


def save_stats(stats: Dict, fallback: Dict, out_path: str) -> None:
    """Save stats + fallback to JSON.  Fallback stored under '__fallback__'."""
    payload = dict(stats)
    payload["__fallback__"] = fallback
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ Saved normal tissue stats → {out_path}")


def load_stats(stats_path: str) -> Tuple[Dict, Dict]:
    """
    Load stats JSON.

    Returns:
        stats:    per-patient stats dict (without __fallback__ key)
        fallback: global fallback dict
    """
    with open(stats_path) as f:
        payload = json.load(f)
    fallback = payload.pop("__fallback__", {})
    return payload, fallback


# =============================================================================
# Picklable transform classes
# =============================================================================

class NormalTissueNorm:
    """
    Z-score normalisation using patient-specific Normal-tissue statistics.

    Pipeline:
        1.  z  ← (x − μ) / (σ + ε)
        2.  z  ← clamp(z, clip_lo, clip_hi)    # default: [−3, 3]  (>99.7% of normal)
        3.  z  ← (z − clip_lo) / (clip_hi − clip_lo)   # rescale to [0, 1]

    The output is in [0, 1] and compatible with all downstream layers.
    Fully picklable — no lambdas, safe on Windows DataLoader workers.

    Args:
        mean:     Per-patient Normal-tissue mean (on [0,1]-normalised images).
        std:      Per-patient Normal-tissue std.
        clip_lo:  Lower sigma-clip bound (default: -3.0).
        clip_hi:  Upper sigma-clip bound (default:  3.0).
        eps:      Numerical stability term.
    """

    def __init__(self, mean: float, std: float,
                 clip_lo: float = -3.0, clip_hi: float = 3.0,
                 eps: float = 1e-6):
        self.mean    = float(mean)
        self.std     = float(std)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.eps     = float(eps)

    def __call__(self, x):
        import torch
        z = (x - self.mean) / (self.std + self.eps)
        z = torch.clamp(z, self.clip_lo, self.clip_hi)
        z = (z - self.clip_lo) / (self.clip_hi - self.clip_lo)
        return z

    def __repr__(self):
        return (f"NormalTissueNorm(mean={self.mean:.4f}, std={self.std:.4f}, "
                f"clip=[{self.clip_lo}, {self.clip_hi}])")


class FallbackNorm:
    """
    Fallback when no Normal stats are available for a patient.
    Replicates the original ContrastBoost → ClampUnit → MinMaxNorm pipeline.
    Fully picklable.
    """

    def __call__(self, x):
        import torch
        x = x * 1.5 - 0.2
        x = torch.clamp(x, 0.0, 1.0)
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-6)

    def __repr__(self):
        return "FallbackNorm(ContrastBoost+Clamp+MinMax)"


# =============================================================================
# Transform builder
# =============================================================================

def build_normalised_transform(
    pid: str,
    modality: str,
    stats: Dict,
    fallback: Dict,
    image_size: Tuple[int, int] = (512, 512),
    is_train: bool = False,
):
    """
    Build a complete picklable transform pipeline for one patient + modality
    using patient-specific Normal-tissue statistics.

    Pipeline:
        NormalTissueNorm(μ_patient, σ_patient)   ← 1st step; replaces
            ContrastBoost + MinMaxNorm
        [RandomRotation, RandomResizedCrop, Flips]  (training only)
        [SpeckleNoise]                              (US training only)
        Resize(image_size)                          (val/test only)

    The output is always in [0, 1].

    Args:
        pid:        Patient ID string (key into stats).
        modality:   'PA' or 'US'.
        stats:      Per-patient stats dict from load_stats().
        fallback:   Global fallback dict from load_stats() or compute_global_fallback().
        image_size: (H, W) spatial size.
        is_train:   Include training augmentations.
    """
    from torchvision.transforms import v2 as T
    from transforms import SpeckleNoise

    pid_stats = stats.get(str(pid), {})
    mod_stats = pid_stats.get(modality) if isinstance(pid_stats, dict) else None

    if mod_stats is None:
        mod_stats = fallback.get(modality)

    if mod_stats is not None:
        norm_op = NormalTissueNorm(mean=mod_stats["mean"], std=mod_stats["std"])
    else:
        norm_op = FallbackNorm()

    ops = [norm_op]

    if is_train:
        ops += [
            T.RandomRotation(degrees=(-20, 20)),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
        ]
        if modality == "US":
            ops.append(SpeckleNoise(sigma=0.05))
    else:
        ops.append(T.Resize(image_size))

    return T.Compose(ops)


# =============================================================================
# Dataset wrapper
# =============================================================================

class NormalisedPAUSDataset:
    """
    Wraps ArpamBScanDataset and applies patient-specific normalisation
    per sample, looked up from the stats dict via the 'pid' column.

    Replaces PAUSBScanDataset when normal_stats are available.

    Args:
        dataset_df:    DataFrame from bscan_dataset.csv.
        stats:         Per-patient stats dict from load_stats().
        fallback:      Global fallback dict.
        image_size:    (H, W) spatial size.
        image_type:    'PAUSradial-pair' or 'PAUSrect-pair'.
        is_train:      Apply training augmentations.
    """

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        stats: Dict,
        fallback: Dict,
        image_size: Tuple[int, int] = (512, 512),
        image_type: str = "PAUSradial-pair",
        is_train: bool = False,
    ):
        from dataset_arpam import ArpamBScanDataset

        self.df         = dataset_df.reset_index(drop=True)
        self.stats      = stats
        self.fallback   = fallback
        self.image_size = image_size
        self.image_type = image_type
        self.is_train   = is_train

        self._base = ArpamBScanDataset(
            dataset_df=dataset_df,
            transform=None,
            image_type=image_type,
            target_type="response",
            force_3chan=False,
        )
        # Cache per-pid transform pairs  { pid → (pa_tf, us_tf) }
        self._tf_cache: Dict[str, tuple] = {}

    def _get_transforms(self, pid: str):
        if pid not in self._tf_cache:
            pa_tf = build_normalised_transform(
                pid, "PA", self.stats, self.fallback,
                self.image_size, self.is_train
            )
            us_tf = build_normalised_transform(
                pid, "US", self.stats, self.fallback,
                self.image_size, self.is_train
            )
            self._tf_cache[pid] = (pa_tf, us_tf)
        return self._tf_cache[pid]

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        import torch
        item = self._base[idx]
        if len(item) != 3:
            raise ValueError(f"Expected 3-tuple from base dataset, got {len(item)}")
        pa_raw, us_raw, y = item

        pid = str(self.df.iloc[idx]["pid"])
        pa_tf, us_tf = self._get_transforms(pid)

        if not isinstance(pa_raw, torch.Tensor):
            pa_raw = torch.tensor(pa_raw, dtype=torch.float32)
        if not isinstance(us_raw, torch.Tensor):
            us_raw = torch.tensor(us_raw, dtype=torch.float32)

        return pa_tf(pa_raw), us_tf(us_raw), int(y)


# =============================================================================
# DataLoader factory
# =============================================================================

def create_normalised_dataloaders(
    csv_path: str,
    stats_path: str,
    image_type: str = "PAUSradial-pair",
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 16,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Drop-in replacement for create_paus_dataloaders() that uses
    patient-specific Normal-tissue normalisation.

    Identical call signature except for the added stats_path argument.
    """
    import platform
    import random
    from torch.utils.data import DataLoader

    stats, fallback = load_stats(stats_path)

    df = pd.read_csv(csv_path)

    from dataset import stratified_patient_split
    train_df, val_df, test_df = stratified_patient_split(
        df, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed,
    )

    all_pids = df["pid"].unique().tolist()
    missing = [p for p in all_pids if str(p) not in stats]
    if missing:
        print(f"  [WARN] {len(missing)} patients missing Normal stats "
              f"→ global fallback applied: {missing}")

    train_ds = NormalisedPAUSDataset(train_df, stats, fallback,
                                     image_size, image_type, is_train=True)
    val_ds   = NormalisedPAUSDataset(val_df,   stats, fallback,
                                     image_size, image_type, is_train=False)
    test_ds  = NormalisedPAUSDataset(test_df,  stats, fallback,
                                     image_size, image_type, is_train=False)

    nw = 0 if platform.system() == "Windows" else num_workers
    if nw != num_workers:
        print("  [Windows] num_workers forced to 0")

    kw = dict(num_workers=nw, pin_memory=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-patient Normal-tissue normalisation statistics"
    )
    p.add_argument("--data_root",     type=str, required=True,
                   help="Root data directory  (<date>/<patient>/Normal/...)")
    p.add_argument("--csv",           type=str, default=None,
                   help="Path to bscan_dataset.csv (enables pid-based search)")
    p.add_argument("--out",           type=str, default="data/normal_stats.json",
                   help="Output JSON path  (default: data/normal_stats.json)")
    p.add_argument("--normal_subdir", type=str, default="Normal",
                   help="Normal-tissue subfolder name  (default: Normal)")
    p.add_argument("--image_size",    type=int, nargs=2, default=[512, 512],
                   metavar=("H", "W"))
    p.add_argument("--dry_run",       action="store_true",
                   help="List found Normal folders without computing stats")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("Normal Tissue Statistics Computation")
    print(f"{'='*60}")
    print(f"  Data root   : {args.data_root}")
    print(f"  CSV         : {args.csv or 'filesystem discovery'}")
    print(f"  Normal dir  : {args.normal_subdir}")
    print(f"  Image size  : {args.image_size}")
    print(f"  Output      : {args.out}")
    print()

    if args.dry_run:
        root = Path(args.data_root)
        count = 0
        for level1 in sorted(root.iterdir()):
            if not level1.is_dir():
                continue
            # Shallow layout: <root>/<patient>/normal/
            ndir = level1 / args.normal_subdir
            if ndir.exists():
                n_scans = sum(1 for d in ndir.iterdir() if d.is_dir())
                print(f"  [shallow] {level1.name:30s} → {n_scans} scan folder(s)")
                count += 1
                continue
            # Deep layout: <root>/<date>/<patient>/normal/
            for level2 in sorted(level1.iterdir()):
                if not level2.is_dir():
                    continue
                ndir = level2 / args.normal_subdir
                if ndir.exists():
                    n_scans = sum(1 for d in ndir.iterdir() if d.is_dir())
                    print(f"  [deep]    {level1.name}/{level2.name:20s} "
                          f"→ {n_scans} scan folder(s)")
                    count += 1
        print(f"\n{count} patient(s) with '{args.normal_subdir}' folders found.")
        return

    stats = compute_normal_stats(
        data_root=args.data_root,
        csv_path=args.csv,
        normal_subdir=args.normal_subdir,
        resize_to=tuple(args.image_size),
        verbose=True,
    )

    fallback = compute_global_fallback(stats)

    # Print summary
    pa_means = [v["PA"]["mean"] for v in stats.values() if v.get("PA")]
    us_means = [v["US"]["mean"] for v in stats.values() if v.get("US")]
    pa_stds  = [v["PA"]["std"]  for v in stats.values() if v.get("PA")]
    us_stds  = [v["US"]["std"]  for v in stats.values() if v.get("US")]

    print(f"\n{'─'*60}")
    print(f"Dataset summary ({len(stats)} patients):")
    if pa_means:
        print(f"  PA mean range : [{min(pa_means):.4f}, {max(pa_means):.4f}]  "
              f"std range: [{min(pa_stds):.4f}, {max(pa_stds):.4f}]")
    if us_means:
        print(f"  US mean range : [{min(us_means):.4f}, {max(us_means):.4f}]  "
              f"std range: [{min(us_stds):.4f}, {max(us_stds):.4f}]")
    print(f"\nGlobal fallback (used for patients without Normal scans):")
    for mod, fb in fallback.items():
        print(f"  {mod}: μ={fb['mean']:.4f}  σ={fb['std']:.4f}")

    save_stats(stats, fallback, args.out)


if __name__ == "__main__":
    main()
