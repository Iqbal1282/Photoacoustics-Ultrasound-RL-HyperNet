"""
Supervised Fusion + HyperNetwork  (no RL)
==========================================

Architecture
------------
Frozen PAEncoder  ──┐
                    ├─► CrossModalAttentionFusion ──► fused  [B, feat_dim]
Frozen USEncoder  ──┘                                   │
                                                        ├──► HyperNetwork(fused)
                                                        │         │
                                                        │         ▼
                                                        └──► AdaptiveClassifier(fused, W)
                                                                  │
                                                                  ▼
                                                              logits [B, 2]

Key difference from RL-HyperNet
---------------------------------
In the RL-HyperNet the HyperNetwork input is a latent code z produced by an
RL policy network.  Here the HyperNetwork is fed the fused feature vector
directly, making the whole pipeline purely supervised end-to-end.

This lets us answer:  does the HyperNetwork itself help over a plain linear
head?  If this beats PAUSFusionClassifier, the adaptive weights are adding
value.  If it does not, a fixed linear head is sufficient.

Ablation comparison table
--------------------------
Model                          | Head type      | Policy | This script
PAUSFusionClassifier           | linear (fixed) | none   | train_fusion_ablation.py
PAUSFusionHyperNet  (NEW)      | HyperNet       | none   | this file
PAUSRLHyperNet                 | HyperNet       | RL PPO | training.py

Usage
-----
# With trained encoders + Normal normalisation (recommended)
python train_fusion_hypernet.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --epochs 50 --batch_size 8

# LOPO cross-validation version (recommended for 25-patient dataset)
python lopo_cv.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --model fusion_hypernet ^
    --epochs 30 --batch_size 8 ^
    --out results/lopo_hypernet


python train_fusion_hypernet.py --csv  data/arpam_roi_select_286_all/bscan_dataset.csv --pa_ckpt checkpoints/PA_encoder_supervised_best.pth --us_ckpt checkpoints/US_encoder_supervised_best.pth --normal_stats data/normal_stats.json --epochs 50 --batch_size 8
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

from encoders import PAEncoder, USEncoder, PAUSEncoderPair
from models import (CrossModalAttentionFusion, ConcatFusion,
                    HyperNetwork, AdaptiveClassifier)
from dataset import PAUSBScanDataset, stratified_patient_split
from transforms import build_train_transform, build_val_transform
from train_fusion_ablation import FusionTrainer     # reuse trainer as-is


# =============================================================================
# Fusion + HyperNet model  (no RL)
# =============================================================================

class PAUSFusionHyperNet(nn.Module):
    """
    PA+US classifier using a supervised HyperNetwork head.

    The HyperNetwork takes the fused embedding as input and generates a
    sample-specific weight matrix W.  The adaptive classifier then computes
    logits = einsum(fused, W) — the same operation as in the RL-HyperNet,
    but with no policy network and no reward signal involved.

    This means the HyperNetwork learns to produce good weights purely from
    the supervision signal (cross-entropy loss), which is much more stable
    than RL on a small dataset.

    Args:
        pa_encoder:   Pre-trained PAEncoder (will be frozen).
        us_encoder:   Pre-trained USEncoder (will be frozen).
        feat_dim:     Encoder embedding dimension (default: 256).
        num_classes:  Number of output classes (default: 2).
        fusion_type:  'cross_attention' or 'concat'.
        hypernet_hidden: Hidden size of HyperNetwork MLP.
        dropout:      Dropout before HyperNetwork input.
    """

    def __init__(
        self,
        pa_encoder: PAEncoder,
        us_encoder: USEncoder,
        feat_dim: int = 256,
        num_classes: int = 2,
        fusion_type: str = "cross_attention",
        hypernet_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Frozen encoders
        self.encoder_pair = PAUSEncoderPair(pa_encoder, us_encoder, freeze=True)

        # Fusion  (same as RL-HyperNet)
        if fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(feat_dim)
        else:
            self.fusion = ConcatFusion(feat_dim)

        # Pre-HyperNet normalisation + dropout
        self.pre_hyper = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
        )

        # HyperNetwork: fused [B, feat_dim] → W [B, num_classes, feat_dim]
        # Input dim = feat_dim (fused features, not an RL latent code)
        self.hypernet = HyperNetwork(
            z_dim=feat_dim,
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden_dim=hypernet_hidden,
        )

        # Adaptive classifier  (identical to RL-HyperNet)
        self.classifier = AdaptiveClassifier()

        self.feat_dim    = feat_dim
        self.num_classes = num_classes

    # ── unfreeze helper (identical to PAUSFusionClassifier) ──────────────────
    def unfreeze_encoder_head(self):
        for enc in [self.encoder_pair.pa_encoder,
                    self.encoder_pair.us_encoder]:
            for name, param in enc.backbone.named_parameters():
                if "6" in name or "7" in name:
                    param.requires_grad = True
            for param in enc.embed_head.parameters():
                param.requires_grad = True
            for attr in ["contrast_scale", "contrast_shift"]:
                p = getattr(enc, attr, None)
                if p is not None:
                    p.requires_grad = True
            stem = getattr(enc, "spatial_attn", None)
            if stem is not None:
                for param in stem.parameters():
                    param.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [unfreeze_encoder_head] trainable params: {n:,}")

    def forward(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pa, us: [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        f_pa, f_us = self.encoder_pair(pa, us)
        fused       = self.fusion(f_pa, f_us)          # [B, feat_dim]
        z           = self.pre_hyper(fused)             # [B, feat_dim]
        W           = self.hypernet(z)                  # [B, num_classes, feat_dim]
        logits      = self.classifier(fused, W)         # [B, num_classes]
        return logits

    def get_embeddings(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return intermediate tensors for debugging / inference."""
        f_pa, f_us = self.encoder_pair(pa, us)
        fused       = self.fusion(f_pa, f_us)
        z           = self.pre_hyper(fused)
        W           = self.hypernet(z)
        logits      = self.classifier(fused, W)
        return dict(
            f_pa=f_pa, f_us=f_us,
            fused=fused, W=W, logits=logits,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised Fusion + HyperNet (no RL)"
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
    p.add_argument("--num_classes",      type=int, default=2)
    p.add_argument("--fusion_type",      type=str, default="cross_attention",
                   choices=["cross_attention", "concat"])
    p.add_argument("--hypernet_hidden",  type=int, default=64,
                   help="HyperNetwork hidden size (default: 64). "
                        "Keep small to reduce overfit risk.")
    p.add_argument("--dropout",          type=float, default=0.3)
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-3)
    p.add_argument("--threshold",        type=float, default=0.3)
    p.add_argument("--val_fraction",     type=float, default=0.20)
    p.add_argument("--test_fraction",    type=float, default=0.15)
    p.add_argument("--unfreeze_epoch",   type=int,   default=20,
                   help="Epoch to unfreeze encoder heads (default: 20)")
    p.add_argument("--smoothing_window", type=int,   default=3,
                   help="AUC smoothing window for checkpoint selection (default: 3)")
    p.add_argument("--encoder_lr_scale", type=float, default=0.1)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="cuda")
    p.add_argument("--save_dir",         type=str,   default="./checkpoints")
    p.add_argument("--use_wandb",        action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nw     = 0 if platform.system() == "Windows" else args.num_workers

    print(f"\n{'='*60}")
    print("Supervised Fusion + HyperNet  (no RL)")
    print(f"{'='*60}")
    print(f"  Fusion type     : {args.fusion_type}")
    print(f"  HyperNet hidden : {args.hypernet_hidden}")
    print(f"  Encoder PA      : {args.pa_ckpt or 'ImageNet'}")
    print(f"  Encoder US      : {args.us_ckpt or 'ImageNet'}")
    print(f"  Normal stats    : {args.normal_stats or 'global stretch'}")
    print(f"  Device          : {device}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = stratified_patient_split(
        df, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed,
    )
    image_size = tuple(args.image_size)

    if args.normal_stats:
        from normal_normalisation import load_stats, NormalisedPAUSDataset
        stats, fallback = load_stats(args.normal_stats)
        train_ds = NormalisedPAUSDataset(train_df, stats, fallback,
                                         image_size, args.image_type,
                                         is_train=True)
        val_ds   = NormalisedPAUSDataset(val_df,   stats, fallback,
                                         image_size, args.image_type,
                                         is_train=False)
        test_ds  = NormalisedPAUSDataset(test_df,  stats, fallback,
                                         image_size, args.image_type,
                                         is_train=False)
    else:
        pa_tf = build_train_transform(image_size, "PA")
        us_tf = build_train_transform(image_size, "US")
        v_tf  = build_val_transform(image_size)
        train_ds = PAUSBScanDataset(train_df, image_type=args.image_type,
                                    pa_transform=pa_tf, us_transform=us_tf,
                                    in_channels=args.in_channels)
        val_ds   = PAUSBScanDataset(val_df,   image_type=args.image_type,
                                    pa_transform=v_tf, us_transform=v_tf,
                                    in_channels=args.in_channels)
        test_ds  = PAUSBScanDataset(test_df,  image_type=args.image_type,
                                    pa_transform=v_tf, us_transform=v_tf,
                                    in_channels=args.in_channels)

    kw = dict(num_workers=nw, pin_memory=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, **kw)

    # ── Model ──────────────────────────────────────────────────────────────
    pa_enc = PAEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.pa_ckpt is None))
    us_enc = USEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.us_ckpt is None))

    if args.pa_ckpt:
        ckpt = torch.load(args.pa_ckpt, map_location="cpu", weights_only=False)
        pa_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        pa_enc.detach_projection_head()
        print(f"  ✓ PA encoder loaded from {args.pa_ckpt}")

    if args.us_ckpt:
        ckpt = torch.load(args.us_ckpt, map_location="cpu", weights_only=False)
        us_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        us_enc.detach_projection_head()
        print(f"  ✓ US encoder loaded from {args.us_ckpt}")

    model = PAUSFusionHyperNet(
        pa_encoder=pa_enc, us_encoder=us_enc,
        feat_dim=args.feat_dim, num_classes=args.num_classes,
        fusion_type=args.fusion_type,
        hypernet_hidden=args.hypernet_hidden,
        dropout=args.dropout,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}  (fusion + hypernet)")
    print(f"  Frozen params   : {total - trainable:,}  (encoders)")

    # ── Trainer — reuse FusionTrainer, just swap model ────────────────────
    # FusionTrainer uses model.fusion and model.head for the initial
    # param groups. For HyperNet we need fusion + pre_hyper + hypernet +
    # classifier instead. We subclass and override the param-group setup.

    class HyperNetTrainer(FusionTrainer):
        def __init__(self, *args, **kwargs):
            # Call grandparent __init__ manually to avoid FusionTrainer's
            # fixed param-group setup
            nn.Module.__init__  # not needed — we call super carefully
            import torch.nn as _nn
            super(FusionTrainer, self).__init__()  # object.__init__

            # Copy everything FusionTrainer.__init__ does, but with
            # the correct param groups for PAUSFusionHyperNet
            model_      = kwargs.get("model") or args[0]
            train_ld    = kwargs.get("train_loader") or args[1]
            val_ld      = kwargs.get("val_loader")   or args[2]
            device_     = kwargs.get("device",           "cuda")
            lr_         = kwargs.get("lr",               1e-4)
            wd_         = kwargs.get("weight_decay",     1e-3)
            threshold_  = kwargs.get("threshold",        0.3)
            unfreeze_e  = kwargs.get("unfreeze_epoch",   10)
            enc_scale   = kwargs.get("encoder_lr_scale", 0.1)
            wandb_      = kwargs.get("use_wandb",        False)

            self.model          = model_.to(device_)
            self.train_loader   = train_ld
            self.val_loader     = val_ld
            self.device         = device_
            self.threshold      = threshold_
            self.unfreeze_epoch = unfreeze_e
            self.encoder_lr_scale = enc_scale
            self.use_wandb      = wandb_
            self._unfrozen      = False
            self._base_lr       = lr_

            # Param groups: fusion + pre_hyper + hypernet + classifier
            trainable_params = [
                {"params": self.model.fusion.parameters(),     "lr": lr_},
                {"params": self.model.pre_hyper.parameters(),  "lr": lr_},
                {"params": self.model.hypernet.parameters(),   "lr": lr_},
                {"params": self.model.classifier.parameters(), "lr": lr_},
            ]
            self.optimizer = AdamW(trainable_params, lr=lr_, weight_decay=wd_)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

            class_weights  = self._compute_class_weights(train_ld, device_)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights,
                                                  label_smoothing=0.1)
            print(f"  Class weights: Normal={class_weights[0]:.3f}  "
                  f"Tumor={class_weights[1]:.3f}")
            print(f"  Classification threshold: {threshold_}")

        def train_epoch(self, epoch: int) -> Dict:
            self.model.train()
            tot_loss = 0.0
            correct = total = 0
            pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
            for batch in pbar:
                pa, us, labels = self._unpack(batch)
                logits = self.model(pa, us)
                loss   = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tot_loss += loss.item()
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = (probs >= self.threshold).long()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 acc=f"{100.*correct/total:.1f}%")
            return {"train/loss":     tot_loss / len(self.train_loader),
                    "train/accuracy": 100.0 * correct / total}

    if args.use_wandb:
        import wandb
        wandb.init(project="paus-fusion-hypernet", config=vars(args))

    trainer = HyperNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        unfreeze_epoch=args.unfreeze_epoch,
        encoder_lr_scale=args.encoder_lr_scale,
        use_wandb=args.use_wandb,
    )
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        ckpt_name="best_fusion_hypernet_model.pth",
        smoothing_window=args.smoothing_window,
    )

    # ── Test ──────────────────────────────────────────────────────────────
    best_path = Path(args.save_dir) / "best_fusion_hypernet_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n✓ Loaded best model  epoch={ckpt['epoch']}  "
              f"AUC={ckpt['val_auc']:.4f}")

    model.eval(); model.to(device)
    all_probs, all_labels, all_preds = [], [], []
    correct = total = 0

    with torch.no_grad():
        for pa, us, labels in tqdm(test_loader, desc="Test"):
            pa, us  = pa.to(device), us.to(device)
            labels  = labels.to(device)
            logits  = model(pa, us)
            probs   = F.softmax(logits, dim=1)[:, 1]
            preds   = (probs >= args.threshold).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    test_acc = 100.0 * correct / total
    try:
        from sklearn.metrics import (roc_auc_score, f1_score,
                                      confusion_matrix, roc_curve)
        test_auc = roc_auc_score(all_labels, all_probs)
        test_f1  = f1_score(all_labels, all_preds, average="weighted",
                            zero_division=0)
        cm       = confusion_matrix(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        idx = np.where(1 - fpr >= 0.90)[0]
        sens90 = float(tpr[idx[-1]]) if len(idx) else float("nan")
    except Exception:
        test_auc = test_f1 = sens90 = float("nan"); cm = None

    print(f"\n{'='*60}")
    print(f"Test Results — Fusion + HyperNet (no RL)")
    print(f"  Accuracy          : {test_acc:.2f}%")
    print(f"  AUC               : {test_auc:.4f}")
    print(f"  F1 (weighted)     : {test_f1:.4f}")
    print(f"  Sensitivity@90spec: {sens90:.4f}")
    if cm is not None:
        print(f"  Confusion matrix  :\n{cm}")
    print(f"{'='*60}")

    if args.use_wandb:
        import wandb
        wandb.log({"test/accuracy": test_acc, "test/auc": test_auc,
                   "test/f1": test_f1, "test/sens_at_90spec": sens90})
        wandb.finish()

    # ── Comparison reminder ───────────────────────────────────────────────
    print("\nModel comparison (run all three to compare):")
    print("  Fusion only     : python train_fusion_ablation.py  ...")
    print("  Fusion+HyperNet : python train_fusion_hypernet.py  ...")
    print("  Full RL-HyperNet: python training.py               ...")
    print("\nFor robust AUC with 25 patients, use LOPO:")
    print("  python lopo_cv.py --model fusion_hypernet ...")


if __name__ == "__main__":
    main()
