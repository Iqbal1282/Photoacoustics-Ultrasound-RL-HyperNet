"""
Leave-One-Patient-Out (LOPO) Cross-Validation
===============================================

With only 25 patients a standard 80/20 split is too unreliable — the
val/test AUC depends heavily on which 4-5 patients happen to land in
the held-out set. A single bad split gives AUC=0.98 (epoch 1 lucky) or
AUC=0.50 (all test patients are pure-Normal).

LOPO fixes this by training 25 separate models, each time holding out
exactly 1 patient as the test set and training on the remaining 24.
The final reported AUC is the mean ± std across all 25 folds — a much
more reliable estimate of true generalisation.

What this script does
----------------------
For each fold k (patient k is test):
    1. Train PAUSFusionClassifier on patients {all} \ {k}
       - Inner split: 1 patient withheld from training as early-stop val
    2. Evaluate on patient k
    3. Collect per-scan probabilities and labels

After all folds:
    - Compute per-fold AUC, F1, sensitivity, specificity
    - Compute overall (pooled) AUC across all scans
    - Save fold-level CSV and overall CSV
    - Print summary table

Usage
-----
python lopo_cv.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --epochs 30 --batch_size 8 ^
    --out results/lopo

# Faster run (no encoder checkpoints — use ImageNet features)
python lopo_cv.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --normal_stats data/normal_stats.json ^
    --epochs 20 --batch_size 8 ^
    --out results/lopo_imagenet
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── sys.path guard ────────────────────────────────────────────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder
from train_fusion_ablation import PAUSFusionClassifier
from dataset import PAUSBScanDataset, get_train_transform, get_val_transform
from transforms import build_train_transform, build_val_transform


# =============================================================================
# Per-fold data builder
# =============================================================================

def build_fold_loaders(
    df: pd.DataFrame,
    test_pid,
    val_pid,
    image_size: Tuple[int, int],
    image_type: str,
    batch_size: int,
    in_channels: int,
    normal_stats: Optional[Dict],
    fallback: Optional[Dict],
    nw: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test loaders for one LOPO fold.

    test_pid : patient held out as test
    val_pid  : patient held out from train as early-stopping val
               (chosen as the mixed-patient most similar to test_pid)
    train = all other patients
    """
    test_df  = df[df["pid"] == test_pid].reset_index(drop=True)
    val_df   = df[df["pid"] == val_pid].reset_index(drop=True)
    train_df = df[~df["pid"].isin([test_pid, val_pid])].reset_index(drop=True)

    if normal_stats is not None:
        from normal_normalisation import NormalisedPAUSDataset
        train_ds = NormalisedPAUSDataset(train_df, normal_stats, fallback,
                                         image_size, image_type, is_train=True)
        val_ds   = NormalisedPAUSDataset(val_df,   normal_stats, fallback,
                                         image_size, image_type, is_train=False)
        test_ds  = NormalisedPAUSDataset(test_df,  normal_stats, fallback,
                                         image_size, image_type, is_train=False)
    else:
        pa_tf = build_train_transform(image_size, "PA")
        us_tf = build_train_transform(image_size, "US")
        v_tf  = build_val_transform(image_size)
        train_ds = PAUSBScanDataset(train_df, image_type=image_type,
                                    pa_transform=pa_tf, us_transform=us_tf,
                                    in_channels=in_channels)
        val_ds   = PAUSBScanDataset(val_df,   image_type=image_type,
                                    pa_transform=v_tf, us_transform=v_tf,
                                    in_channels=in_channels)
        test_ds  = PAUSBScanDataset(test_df,  image_type=image_type,
                                    pa_transform=v_tf, us_transform=v_tf,
                                    in_channels=in_channels)

    kw = dict(num_workers=nw, pin_memory=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# =============================================================================
# Single-fold training
# =============================================================================

def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    pa_ckpt: Optional[str],
    us_ckpt: Optional[str],
    feat_dim: int,
    in_channels: int,
    fusion_type: str,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    unfreeze_epoch: int,
    encoder_lr_scale: float,
    device: torch.device,
    model_type: str = "fusion",
    hypernet_hidden: int = 64,
    ctx_dim: int = 64,
) -> PAUSFusionClassifier:
    """Train one fold and return the best-val-AUC model."""

    # Build fresh model for each fold
    pa_enc = PAEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=(pa_ckpt is None))
    us_enc = USEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=(us_ckpt is None))

    if pa_ckpt:
        ckpt = torch.load(pa_ckpt, map_location="cpu", weights_only=False)
        pa_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        pa_enc.detach_projection_head()
    if us_ckpt:
        ckpt = torch.load(us_ckpt, map_location="cpu", weights_only=False)
        us_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        us_enc.detach_projection_head()

    if model_type == "fusion_hypernet_v2":
        from train_fusion_hypernet_v2 import PAUSFusionHyperNetV2
        model = PAUSFusionHyperNetV2(
            pa_encoder=pa_enc, us_encoder=us_enc,
            feat_dim=feat_dim, num_classes=2,
            fusion_type=fusion_type,
            ctx_dim=hypernet_hidden,
            hypernet_hidden=hypernet_hidden,
            dropout=dropout,
        ).to(device)
    elif model_type == "fusion_hypernet":
        from train_fusion_hypernet import PAUSFusionHyperNet
        model = PAUSFusionHyperNet(
            pa_encoder=pa_enc, us_encoder=us_enc,
            feat_dim=feat_dim, num_classes=2,
            fusion_type=fusion_type,
            hypernet_hidden=hypernet_hidden,
            dropout=dropout,
        ).to(device)
    else:
        model = PAUSFusionClassifier(
            pa_encoder=pa_enc, us_encoder=us_enc,
            feat_dim=feat_dim, num_classes=2,
            fusion_type=fusion_type, dropout=dropout,
        ).to(device)

    # Weighted CE loss
    try:
        all_labels = []
        for batch in train_loader:
            lbl = batch[-1]
            all_labels.extend(lbl.tolist() if isinstance(lbl, torch.Tensor) else lbl)
        labels_t  = torch.tensor(all_labels, dtype=torch.long)
        counts    = torch.bincount(labels_t, minlength=2).float()
        weights   = labels_t.shape[0] / (2 * counts.clamp(min=1))
        weights   = weights.to(device)
    except Exception:
        weights = torch.ones(2, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Optimizer: different param groups depending on model type
    if model_type == "fusion_hypernet_v2":
        trainable = [
            {"params": model.gate.parameters(),           "lr": lr},
            {"params": model.fusion.parameters(),          "lr": lr},
            {"params": model.context_encoder.parameters(), "lr": lr},
            {"params": model.hypernet.parameters(),        "lr": lr},
            {"params": model.adaptive_cls.parameters(),    "lr": lr},
            {"params": model.base_cls.parameters(),        "lr": lr * 2},
        ]
    elif model_type == "fusion_hypernet":
        trainable = [
            {"params": model.fusion.parameters(),     "lr": lr},
            {"params": model.pre_hyper.parameters(),  "lr": lr},
            {"params": model.hypernet.parameters(),   "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    else:
        trainable = [
            {"params": model.fusion.parameters(), "lr": lr},
            {"params": model.head.parameters(),   "lr": lr},
        ]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc    = -1.0
    best_state  = None
    unfrozen    = False

    for epoch in range(1, epochs + 1):

        # Unfreeze encoder heads at unfreeze_epoch
        if not unfrozen and epoch >= unfreeze_epoch:
            model.unfreeze_encoder_head()
            existing  = {id(p) for g in optimizer.param_groups for p in g["params"]}
            new_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in existing]
            if new_params:
                optimizer.add_param_group({
                    "params":       new_params,
                    "lr":           lr * encoder_lr_scale,
                    "weight_decay": weight_decay,
                })
            unfrozen = True

        # Train
        model.train()
        for batch in train_loader:
            pa, us, labels = (t.to(device) for t in batch)
            logits = model(pa, us)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        probs_v, labels_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                pa, us, labels = (t.to(device) for t in batch)
                logits = model(pa, us)
                p = F.softmax(logits, dim=1)[:, 1]
                probs_v.extend(p.cpu().tolist())
                labels_v.extend(labels.cpu().tolist())

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_v, probs_v)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =============================================================================
# Evaluate one fold
# =============================================================================

@torch.no_grad()
def evaluate_fold(
    model: PAUSFusionClassifier,
    test_loader: DataLoader,
    test_pid,
    threshold: float,
    device: torch.device,
) -> List[Dict]:
    model.eval()
    rows = []
    for batch in test_loader:
        pa, us, labels = (t.to(device) for t in batch)
        logits = model(pa, us)
        probs  = F.softmax(logits, dim=1)[:, 1]
        for p, l in zip(probs.cpu().tolist(), labels.cpu().tolist()):
            rows.append({
                "pid":        test_pid,
                "true_label": int(l),
                "tumor_prob": round(p, 4),
                "pred_label": int(p >= threshold),
            })
    return rows


# =============================================================================
# Pick val patient for each fold
# =============================================================================

def pick_val_pid(df: pd.DataFrame, test_pid, rng: random.Random):
    """
    Pick a val patient from the remaining patients.
    Prefer mixed patients (have both tumor and normal scans) so the val
    set always has both classes. Fall back to any non-test patient.
    """
    remaining = df[df["pid"] != test_pid]
    mixed = []
    for pid, grp in remaining.groupby("pid"):
        if 0 < grp["has_tumor"].sum() < len(grp):
            mixed.append(pid)
    if mixed:
        return rng.choice(mixed)
    # fallback: any remaining patient
    return rng.choice(remaining["pid"].unique().tolist())


# =============================================================================
# Main LOPO loop
# =============================================================================

def run_lopo(args) -> pd.DataFrame:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng    = random.Random(args.seed)
    nw     = 0 if platform.system() == "Windows" else args.num_workers
    image_size = tuple(args.image_size)

    df = pd.read_csv(args.csv)
    all_pids = sorted(df["pid"].unique().tolist())

    # Load normal stats if provided
    normal_stats = fallback = None
    if args.normal_stats:
        from normal_normalisation import load_stats
        normal_stats, fallback = load_stats(args.normal_stats)

    all_rows: List[Dict] = []
    fold_results: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"Leave-One-Patient-Out Cross-Validation")
    print(f"  Patients   : {len(all_pids)}")
    model_label = (
        "PAUSFusionHyperNetV2" if args.model == "fusion_hypernet_v2" else
        "PAUSFusionHyperNet"   if args.model == "fusion_hypernet" else
        "PAUSFusionClassifier")
    print(f"  Model      : {model_label} ({args.fusion_type})")
    print(f"  Epochs/fold: {args.epochs}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    for fold_idx, test_pid in enumerate(all_pids):
        print(f"\n── Fold {fold_idx+1:>2}/{len(all_pids)}  "
              f"test_pid={test_pid} ──────────────────")

        val_pid = pick_val_pid(df, test_pid, rng)
        n_test  = len(df[df["pid"] == test_pid])
        n_train = len(df[~df["pid"].isin([test_pid, val_pid])])
        t_test  = df[df["pid"] == test_pid]["has_tumor"].sum()
        print(f"  Val pid: {val_pid}  |  "
              f"Train scans: {n_train}  |  "
              f"Test scans: {n_test}  (tumor={t_test})")

        if t_test == 0:
            print(f"  [NOTE] test patient {test_pid} has no tumor scans "
                  f"— fold AUC will be undefined, using accuracy only")

        train_loader, val_loader, test_loader = build_fold_loaders(
            df=df, test_pid=test_pid, val_pid=val_pid,
            image_size=image_size, image_type=args.image_type,
            batch_size=args.batch_size, in_channels=args.in_channels,
            normal_stats=normal_stats, fallback=fallback, nw=nw,
        )

        model = train_one_fold(
            train_loader=train_loader, val_loader=val_loader,
            pa_ckpt=args.pa_ckpt, us_ckpt=args.us_ckpt,
            feat_dim=args.feat_dim, in_channels=args.in_channels,
            fusion_type=args.fusion_type, dropout=args.dropout,
            epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, threshold=args.threshold,
            unfreeze_epoch=args.unfreeze_epoch,
            encoder_lr_scale=args.encoder_lr_scale,
            device=device,
            model_type=args.model,
            hypernet_hidden=args.hypernet_hidden,
            ctx_dim=args.ctx_dim,
        )

        rows = evaluate_fold(model, test_loader, test_pid,
                             args.threshold, device)
        all_rows.extend(rows)

        # Per-fold metrics
        fold_df = pd.DataFrame(rows)
        trues   = fold_df["true_label"].tolist()
        probs   = fold_df["tumor_prob"].tolist()
        preds   = fold_df["pred_label"].tolist()

        try:
            from sklearn.metrics import (roc_auc_score, f1_score,
                                          confusion_matrix)
            auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else float("nan")
            f1  = f1_score(trues, preds, average="weighted", zero_division=0)
            cm  = confusion_matrix(trues, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
        except Exception:
            auc = f1 = sens = spec = float("nan")
            tn = fp = fn = tp = 0

        fold_results.append({
            "fold":       fold_idx + 1,
            "test_pid":   test_pid,
            "val_pid":    val_pid,
            "n_test":     n_test,
            "n_tumor":    int(t_test),
            "AUC":        round(auc, 4) if not np.isnan(auc) else "nan",
            "F1":         round(f1,  4),
            "Sensitivity":round(sens,4),
            "Specificity":round(spec,4),
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        })

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A (pure class)"
        print(f"  AUC={auc_str}  sens={sens:.3f}  spec={spec:.3f}  "
              f"TP={tp} TN={tn} FP={fp} FN={fn}")

    return pd.DataFrame(all_rows), pd.DataFrame(fold_results)


# =============================================================================
# Summary
# =============================================================================

def print_summary(fold_df: pd.DataFrame, all_rows_df: pd.DataFrame,
                  threshold: float):
    print(f"\n{'='*60}")
    print("LOPO CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nPer-fold results:")
    print(fold_df[["fold","test_pid","n_test","n_tumor",
                   "AUC","Sensitivity","Specificity","TP","FP","FN"]
                  ].to_string(index=False))

    # Only compute mean/std over folds with valid AUC
    valid = fold_df[fold_df["AUC"] != "nan"].copy()
    valid["AUC"]         = valid["AUC"].astype(float)
    valid["Sensitivity"] = valid["Sensitivity"].astype(float)
    valid["Specificity"] = valid["Specificity"].astype(float)

    print(f"\n{'─'*60}")
    print(f"Summary over {len(valid)}/{len(fold_df)} folds with both classes:")
    for col in ["AUC", "Sensitivity", "Specificity"]:
        print(f"  {col:12s}: mean={valid[col].mean():.4f}  "
              f"std={valid[col].std():.4f}  "
              f"[{valid[col].min():.4f}, {valid[col].max():.4f}]")

    # Overall pooled AUC (all scans together)
    trues = all_rows_df["true_label"].tolist()
    probs = all_rows_df["tumor_prob"].tolist()
    preds = all_rows_df["pred_label"].tolist()
    try:
        from sklearn.metrics import (roc_auc_score, f1_score,
                                      confusion_matrix, roc_curve)
        pooled_auc = roc_auc_score(trues, probs)
        pooled_f1  = f1_score(trues, preds, average="weighted", zero_division=0)
        cm         = confusion_matrix(trues, preds, labels=[0, 1])
        tn,fp,fn,tp = cm.ravel()

        fpr, tpr, _ = roc_curve(trues, probs)
        idx = np.where(1 - fpr >= 0.90)[0]
        sens90 = float(tpr[idx[-1]]) if len(idx) else float("nan")

        print(f"\n{'─'*60}")
        print(f"Pooled across ALL {len(all_rows_df)} scans:")
        print(f"  AUC               : {pooled_auc:.4f}")
        print(f"  F1 (weighted)     : {pooled_f1:.4f}")
        print(f"  Sensitivity@90spec: {sens90:.4f}")
        print(f"  Confusion matrix  :")
        print(f"  [[TN={tn} FP={fp}]")
        print(f"   [FN={fn} TP={tp}]]")
        print(f"  Total scans: {len(trues)}  "
              f"Tumor={sum(trues)}  Normal={len(trues)-sum(trues)}")
    except Exception as e:
        print(f"  [Pooled AUC failed: {e}]")

    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-One-Patient-Out CV for PAUSFusionClassifier"
    )
    p.add_argument("--csv",              type=str, required=True)
    p.add_argument("--pa_ckpt",          type=str, default=None)
    p.add_argument("--us_ckpt",          type=str, default=None)
    p.add_argument("--normal_stats",     type=str, default=None)
    p.add_argument("--image_type",       type=str, default="PAUSradial-pair",
                   choices=["PAUSradial-pair", "PAUSrect-pair"])
    p.add_argument("--image_size",       type=int, nargs=2, default=[512, 512])
    p.add_argument("--in_channels",      type=int, default=1)
    p.add_argument("--feat_dim",         type=int, default=256)
    p.add_argument("--fusion_type",      type=str, default="cross_attention",
                   choices=["cross_attention", "concat"])
    p.add_argument("--dropout",          type=float, default=0.3)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-3)
    p.add_argument("--threshold",        type=float, default=0.3)
    p.add_argument("--unfreeze_epoch",   type=int,   default=10)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--model",            type=str,   default="fusion",
                   choices=["fusion", "fusion_hypernet", "fusion_hypernet_v2"],
                   help="Model type: 'fusion' (linear head) or "
                        "'fusion_hypernet' (HyperNet head, no RL)")
    p.add_argument("--hypernet_hidden",  type=int,   default=64)
    p.add_argument("--ctx_dim",          type=int,   default=64,
                   help="Context bottleneck dim for v2 model")
    p.add_argument("--gate_hidden",      type=int,   default=32)
    p.add_argument("--device",           type=str,   default="cuda")
    p.add_argument("--out",              type=str,   default="results/lopo",
                   help="Output directory for CSV results")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    all_rows_df, fold_df = run_lopo(args)

    print_summary(fold_df, all_rows_df, args.threshold)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows_df.to_csv(out / "lopo_scan_predictions.csv",  index=False)
    fold_df.to_csv(    out / "lopo_fold_summary.csv",      index=False)
    print(f"Results saved to {out}/")
    print(f"  lopo_scan_predictions.csv  — per-scan probabilities")
    print(f"  lopo_fold_summary.csv      — per-fold AUC/sens/spec")


if __name__ == "__main__":
    main()
