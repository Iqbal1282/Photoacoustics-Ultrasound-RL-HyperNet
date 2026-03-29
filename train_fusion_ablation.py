"""
Fusion Ablation — PA+US without RL-HyperNet
=============================================

Replaces the entire RL policy + HyperNetwork + AdaptiveClassifier stack
with a simple fixed linear head on top of the same frozen encoders and
the same CrossModalAttentionFusion / ConcatFusion modules.

Architecture:
    Frozen PAEncoder  →  f_pa  [B, feat_dim]
    Frozen USEncoder  →  f_us  [B, feat_dim]
                              ↓
                   CrossModalAttentionFusion  OR  ConcatFusion
                              ↓  fused [B, feat_dim]
                      Dropout(0.3)
                      Linear(feat_dim → num_classes)
                              ↓  logits

This is the minimal possible model on top of the encoders.
If this also collapses → encoders are the problem.
If this works → RL-HyperNet training is the problem.

Usage
-----
# Train fusion ablation (recommended first step)
python train_fusion_ablation.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --epochs 30 --batch_size 8

# With concat fusion (even simpler)
python train_fusion_ablation.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --fusion_type concat ^
    --epochs 30 --batch_size 8

# No encoder checkpoints — test raw ImageNet features
python train_fusion_ablation.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --epochs 30 --batch_size 8
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
from models import CrossModalAttentionFusion, ConcatFusion
from dataset import stratified_patient_split, PAUSBScanDataset
from dataset import get_train_transform, get_val_transform
from transforms import build_train_transform, build_val_transform


# =============================================================================
# Fusion-only model
# =============================================================================

class PAUSFusionClassifier(nn.Module):
    """
    Minimal PA+US classifier: frozen encoders → fusion → linear head.

    No RL, no HyperNetwork, no adaptive weights.
    Pure supervised baseline to isolate encoder quality.

    Args:
        pa_encoder:   Pre-trained PAEncoder (frozen).
        us_encoder:   Pre-trained USEncoder (frozen).
        feat_dim:     Encoder embedding dimension.
        num_classes:  Output classes (default: 2).
        fusion_type:  'cross_attention' or 'concat'.
        dropout:      Dropout before the linear head.
    """

    def __init__(
        self,
        pa_encoder: PAEncoder,
        us_encoder: USEncoder,
        feat_dim: int = 256,
        num_classes: int = 2,
        fusion_type: str = "cross_attention",
        dropout: float = 0.3,
    ):
        super().__init__()

        # Frozen encoder pair
        self.encoder_pair = PAUSEncoderPair(
            pa_encoder, us_encoder, freeze=True
        )

        # Fusion module — same as used in RL-HyperNet
        if fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(feat_dim)
        else:
            self.fusion = ConcatFusion(feat_dim)

        # Simple fixed classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self.feat_dim    = feat_dim
        self.num_classes = num_classes

    def unfreeze_encoder_head(self):
        """
        Unfreeze only the last ResNet layer (layer4) and embedding head of
        both encoders.  Everything else stays frozen.
        Calling this after a few warm-up epochs lets the encoder features
        fine-tune toward the downstream classification task without destroying
        the pretrained representations.
        """
        for enc in [self.encoder_pair.pa_encoder,
                    self.encoder_pair.us_encoder]:
            # Unfreeze layer4 of the backbone
            for name, param in enc.backbone.named_parameters():
                if "6" in name or "7" in name:  # layer4 = index 6, avgpool = 7
                    param.requires_grad = True
            # Always unfreeze the embedding projection head
            for param in enc.embed_head.parameters():
                param.requires_grad = True
            # Unfreeze modality-specific stems too
            for attr in ["contrast_scale", "contrast_shift"]:
                p = getattr(enc, attr, None)
                if p is not None:
                    p.requires_grad = True
            for stem_name in ["spatial_attn"]:
                stem = getattr(enc, stem_name, None)
                if stem is not None:
                    for param in stem.parameters():
                        param.requires_grad = True
        newly_trainable = sum(p.numel() for p in self.parameters()
                              if p.requires_grad)
        print(f"  [unfreeze_encoder_head] trainable params: "
              f"{newly_trainable:,}")

    def forward(self, pa: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pa, us: [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        f_pa, f_us = self.encoder_pair(pa, us)
        fused       = self.fusion(f_pa, f_us)
        return self.head(fused)

    def get_embeddings(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return intermediate tensors for debugging."""
        f_pa, f_us = self.encoder_pair(pa, us)
        fused       = self.fusion(f_pa, f_us)
        logits      = self.head(fused)
        return dict(f_pa=f_pa, f_us=f_us, fused=fused, logits=logits)


# =============================================================================
# Trainer
# =============================================================================

class FusionTrainer:
    """
    Supervised trainer for PAUSFusionClassifier.

    Uses:
    - Inverse-frequency weighted CrossEntropyLoss (handles imbalance)
    - Label smoothing = 0.1
    - Classification threshold = 0.3 (tumour-sensitive)
    - AUC as primary model-selection metric
    """

    def __init__(
        self,
        model: PAUSFusionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        threshold: float = 0.3,
        unfreeze_epoch: int = 10,
        encoder_lr_scale: float = 0.1,
        use_wandb: bool = False,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.threshold      = threshold
        self.unfreeze_epoch = unfreeze_epoch   # epoch at which to unfreeze encoder heads
        self.encoder_lr_scale = encoder_lr_scale
        self.use_wandb      = use_wandb
        self._unfrozen      = False
        self._base_lr       = lr

        # Stage 1: train only fusion + classification head
        trainable_params = [
            {"params": self.model.fusion.parameters(), "lr": lr},
            {"params": self.model.head.parameters(),   "lr": lr},
        ]
        self.optimizer = AdamW(trainable_params, lr=lr,
                               weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        # Weighted CE loss
        class_weights = self._compute_class_weights(train_loader, device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )
        print(f"  Class weights: Normal={class_weights[0]:.3f}  "
              f"Tumor={class_weights[1]:.3f}")
        print(f"  Classification threshold: {threshold}")

    @staticmethod
    def _compute_class_weights(loader: DataLoader,
                               device: str) -> torch.Tensor:
        try:
            labels = []
            for batch in loader:
                lbl = batch[-1]
                labels.extend(lbl.tolist() if isinstance(lbl, torch.Tensor)
                              else lbl)
            labels    = torch.tensor(labels, dtype=torch.long)
            n_classes = int(labels.max().item()) + 1
            counts    = torch.bincount(labels, minlength=n_classes).float()
            weights   = labels.shape[0] / (n_classes * counts.clamp(min=1))
            return weights.to(device)
        except Exception as e:
            print(f"  [WARN] Class weight computation failed ({e})")
            return torch.ones(2, device=device)

    def _maybe_unfreeze(self, epoch: int):
        """
        At unfreeze_epoch, unfreeze the encoder heads and add their
        parameters to the optimizer at a much lower LR (encoder_lr_scale × base LR).
        This prevents destroying pretrained features while allowing fine-tuning.
        """
        if self._unfrozen or epoch < self.unfreeze_epoch:
            return
        print(f"\n  [Epoch {epoch}] Unfreezing encoder last layer + embed heads "
              f"(LR = {self._base_lr * self.encoder_lr_scale:.2e})")
        self.model.unfreeze_encoder_head()
        # Collect newly unfrozen params not yet in the optimizer
        existing = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        new_params = [p for p in self.model.parameters()
                      if p.requires_grad and id(p) not in existing]
        if new_params:
            self.optimizer.add_param_group({
                "params":       new_params,
                "lr":           self._base_lr * self.encoder_lr_scale,
                "weight_decay": 1e-3,
            })
        self._unfrozen = True

    def _unpack(self, batch):
        pa, us, labels = batch
        return pa.to(self.device), us.to(self.device), labels.to(self.device)

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
            probs     = F.softmax(logits, dim=1)[:, 1]
            preds     = (probs >= self.threshold).long()
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.*correct/total:.1f}%")

        return {"train/loss":     tot_loss / len(self.train_loader),
                "train/accuracy": 100.0 * correct / total}

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        tot_loss = 0.0
        correct = total = 0
        all_probs, all_preds, all_labels = [], [], []

        for batch in tqdm(self.val_loader, desc="  Val"):
            pa, us, labels = self._unpack(batch)
            logits  = self.model(pa, us)
            tot_loss += self.criterion(logits, labels).item()

            probs    = F.softmax(logits, dim=1)[:, 1]
            preds    = (probs >= self.threshold).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        try:
            from sklearn.metrics import (roc_auc_score, f1_score,
                                          confusion_matrix)
            auc = roc_auc_score(all_labels, all_probs)
            f1  = f1_score(all_labels, all_preds, average="weighted",
                           zero_division=0)
            cm  = confusion_matrix(all_labels, all_preds)
        except Exception:
            auc, f1, cm = float("nan"), float("nan"), None

        return {
            "val/loss":     tot_loss / len(self.val_loader),
            "val/accuracy": 100.0 * correct / total,
            "val/auc":      auc,
            "val/f1":       f1,
            "_cm":          cm,
        }

    def train(self, num_epochs: int, save_dir: str = "./checkpoints",
             ckpt_name: str = "best_fusion_model.pth",
             smoothing_window: int = 3):
        """
        Args:
            smoothing_window: Number of recent epochs to average AUC over
                              before deciding to save a checkpoint.
                              Default 3 reduces sensitivity to single lucky
                              epochs on small val sets.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        best_smoothed_auc = 0.0
        auc_history: List[float] = []

        print(f"\n{'='*60}")
        print(f"Fusion Training  ({num_epochs} epochs, "
              f"AUC smoothing window={smoothing_window})")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            self._maybe_unfreeze(epoch)
            train_m = self.train_epoch(epoch)
            val_m   = self.validate()
            self.scheduler.step()

            cm  = val_m.pop("_cm")
            auc = val_m["val/auc"]
            auc_history.append(auc)

            # Smoothed AUC = mean of last `smoothing_window` epochs
            smoothed = float(np.mean(auc_history[-smoothing_window:]))
            val_m["val/auc_smoothed"] = smoothed

            print(f"  E{epoch:>2}  "
                  f"loss={train_m['train/loss']:.4f}  "
                  f"acc={train_m['train/accuracy']:.1f}%  ||  "
                  f"val_loss={val_m['val/loss']:.4f}  "
                  f"val_acc={val_m['val/accuracy']:.1f}%  "
                  f"AUC={auc:.4f}  smooth={smoothed:.4f}  "
                  f"F1={val_m['val/f1']:.4f}")
            if cm is not None:
                print(f"       CM: {cm.ravel().tolist()}  "
                      f"[TN FP FN TP]")

            if self.use_wandb:
                import wandb
                wandb.log({**train_m, **val_m,
                           "lr": self.optimizer.param_groups[0]["lr"]},
                          step=epoch)

            # Save based on smoothed AUC — prevents saving on single lucky epochs
            if smoothed > best_smoothed_auc and epoch >= smoothing_window:
                best_smoothed_auc = smoothed
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc":          auc,
                    "val_auc_smoothed": smoothed,
                    "val_accuracy":     val_m["val/accuracy"],
                    "metrics":          {**train_m, **val_m},
                }, save_path / ckpt_name)
                print(f"  ✓ New best smoothed AUC={smoothed:.4f}  "
                      f"(raw={auc:.4f})  acc={val_m['val/accuracy']:.1f}%")

        print(f"\nDone.  Best smoothed val AUC: {best_smoothed_auc:.4f}")
        print(f"Checkpoint: {save_path/ckpt_name}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fusion ablation — PA+US without RL-HyperNet"
    )
    p.add_argument("--csv",          type=str, required=True)
    p.add_argument("--pa_ckpt",      type=str, default=None,
                   help="PA encoder checkpoint. Omit to use raw ImageNet weights.")
    p.add_argument("--us_ckpt",      type=str, default=None,
                   help="US encoder checkpoint.")
    p.add_argument("--normal_stats", type=str, default=None,
                   help="normal_stats.json for patient-specific normalisation.")
    p.add_argument("--image_type",   type=str, default="PAUSradial-pair",
                   choices=["PAUSradial-pair", "PAUSrect-pair"])
    p.add_argument("--image_size",   type=int, nargs=2, default=[512, 512])
    p.add_argument("--in_channels",  type=int, default=1)
    p.add_argument("--feat_dim",     type=int, default=256)
    p.add_argument("--num_classes",  type=int, default=2)
    p.add_argument("--fusion_type",  type=str, default="cross_attention",
                   choices=["cross_attention", "concat"])
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="Learning rate for fusion+head (default: 1e-4). "
                        "Lower than before to prevent epoch-1 overfit.")
    p.add_argument("--weight_decay",    type=float, default=1e-3,
                   help="L2 regularisation (default: 1e-3, stronger than before).")
    p.add_argument("--unfreeze_epoch",  type=int,   default=20,
                   help="Epoch at which to unfreeze encoder last layer "
                        "(default: 20, after fusion head has converged). "
                        "Set to 0 to keep encoders fully frozen.")
    p.add_argument("--smoothing_window",type=int,   default=3,
                   help="Epochs to average AUC over for checkpoint selection "
                        "(default: 3). Higher = less sensitive to lucky epochs.")
    p.add_argument("--encoder_lr_scale",type=float, default=0.1,
                   help="Encoder LR = lr × scale (default: 0.1 = 10× lower).")
    p.add_argument("--threshold",    type=float, default=0.3)
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--test_fraction",type=float, default=0.15)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--use_wandb",    action="store_true")
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

    print(f"\n{'='*60}")
    print("PA+US Fusion Ablation (no RL-HyperNet)")
    print(f"{'='*60}")
    print(f"  Fusion type  : {args.fusion_type}")
    print(f"  Encoder PA   : {args.pa_ckpt or 'ImageNet init (no checkpoint)'}")
    print(f"  Encoder US   : {args.us_ckpt or 'ImageNet init (no checkpoint)'}")
    print(f"  Normal stats : {args.normal_stats or 'none (global contrast stretch)'}")
    print(f"  Device       : {device}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = stratified_patient_split(
        df, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed,
    )

    image_size = tuple(args.image_size)
    nw = 0 if platform.system() == "Windows" else args.num_workers

    if args.normal_stats is not None:
        from normal_normalisation import (
            load_stats, NormalisedPAUSDataset
        )
        stats, fallback = load_stats(args.normal_stats)
        print(f"  Using Normal-tissue normalisation from {args.normal_stats}")
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
        pa_train_tf = build_train_transform(image_size, "PA")
        us_train_tf = build_train_transform(image_size, "US")
        val_tf      = build_val_transform(image_size)
        train_ds = PAUSBScanDataset(train_df, image_type=args.image_type,
                                    pa_transform=pa_train_tf,
                                    us_transform=us_train_tf,
                                    in_channels=args.in_channels)
        val_ds   = PAUSBScanDataset(val_df,   image_type=args.image_type,
                                    pa_transform=val_tf, us_transform=val_tf,
                                    in_channels=args.in_channels)
        test_ds  = PAUSBScanDataset(test_df,  image_type=args.image_type,
                                    pa_transform=val_tf, us_transform=val_tf,
                                    in_channels=args.in_channels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=nw, pin_memory=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=nw, pin_memory=(nw > 0))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=nw, pin_memory=(nw > 0))

    # ── Model ──────────────────────────────────────────────────────────────
    pa_enc = PAEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.pa_ckpt is None))
    us_enc = USEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.us_ckpt is None))

    if args.pa_ckpt:
        ckpt = torch.load(args.pa_ckpt, map_location="cpu", weights_only=False)
        pa_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        pa_enc.detach_projection_head()
        print(f"  ✓ Loaded PA encoder from {args.pa_ckpt}")

    if args.us_ckpt:
        ckpt = torch.load(args.us_ckpt, map_location="cpu", weights_only=False)
        us_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        us_enc.detach_projection_head()
        print(f"  ✓ Loaded US encoder from {args.us_ckpt}")

    model = PAUSFusionClassifier(
        pa_encoder=pa_enc,
        us_encoder=us_enc,
        feat_dim=args.feat_dim,
        num_classes=args.num_classes,
        fusion_type=args.fusion_type,
        dropout=args.dropout,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  "
          f"(fusion + head only)")
    print(f"  Frozen params:    {total - trainable:,}  (encoders)")

    if args.use_wandb:
        import wandb
        wandb.init(project="paus-fusion-ablation", config=vars(args))

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = FusionTrainer(
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
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir,
                 smoothing_window=args.smoothing_window)

    # ── Test ──────────────────────────────────────────────────────────────
    best_path = Path(args.save_dir) / "best_fusion_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded best model  epoch={ckpt['epoch']}  "
              f"AUC={ckpt['val_auc']:.4f}")

    model.eval(); model.to(device)
    correct = total = 0
    all_probs, all_labels, all_preds = [], [], []

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
    print(f"Test Results (threshold={args.threshold}):")
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

    # ── What the results mean ─────────────────────────────────────────────
    print("\nDIAGNOSIS:")
    if test_auc > 0.75:
        print("  ✓ Encoders are working — fusion alone gives reasonable AUC.")
        print("    → RL-HyperNet training pipeline is the issue.")
        print("    → Retrain RL-HyperNet with the fixed stratified split.")
    elif test_auc > 0.55:
        print("  ~ Encoders have some signal but are weak.")
        print("    → Consider retraining encoders with --normal_stats.")
    else:
        print("  ✗ Encoders produce near-random features (AUC ≈ 0.5).")
        print("    → Encoders need retraining from scratch.")
        print("    → Use: python train_encoders.py --normal_stats data/normal_stats.json")


if __name__ == "__main__":
    main()
