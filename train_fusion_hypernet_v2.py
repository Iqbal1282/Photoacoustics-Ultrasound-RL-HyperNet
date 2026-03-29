"""
Improved Supervised Fusion + HyperNetwork  (no RL)  — v2
==========================================================

Seven improvements over v1 (train_fusion_hypernet.py)
------------------------------------------------------

① Gated Modality Weighting
   Before fusion, learn per-sample importance weights α_pa and α_us.
   A small MLP maps (f_pa ‖ f_us) → softmax([α_pa, α_us]).
   The scaled features α_pa·f_pa and α_us·f_us enter the cross-attention
   fusion.  This lets the model de-emphasise a noisy US channel for one
   patient while upweighting PA, or vice-versa.

② Context Bottleneck for HyperNet input
   In v1 the HyperNet received fused [256] as input — the same vector
   used for classification.  Now a small MLP projects
       f_pa ‖ f_us ‖ fused  [768] → context [64]
   The HyperNet sees richer multi-modal signal (all three vectors) but
   through a tight 64-d bottleneck that regularises the adaptation.

③ Residual classification
   A simple fixed linear head ("base classifier") runs in parallel with
   the HyperNet head.
       logits = base_logits + delta_logits
   Benefits:
     - Training is more stable: base_logits converge early via plain CE,
       giving the HyperNet a well-behaved residual to refine.
     - At epoch 0 the HyperNet output is near-zero (small init), so the
       model starts as a linear classifier and gradually becomes adaptive.
     - Prevents catastrophic overfitting: even if the HyperNet overfits
       the base path still produces sensible predictions.

④ MC-Dropout uncertainty
   Proper predictive uncertainty via N stochastic forward passes at
   inference time (Monte-Carlo Dropout).  Returns mean probability and
   standard deviation — both stored in the output CSV.

⑤ Temperature scaling calibration
   After training, finds the optimal temperature T on the val set using
   NLL minimisation (Guo et al., 2017).  Calibrated probabilities are
   more clinically meaningful than raw softmax outputs.

⑥ Focal loss option
   Optional focal loss (α=0.25, γ=2.0) instead of weighted CE.
   Focal loss down-weights easy correct predictions and focuses training
   on hard borderline cases — useful when normal tissue scans are "easy"
   and tumour scans are "hard".

⑦ Early stopping
   Monitors smoothed val AUC and stops if it does not improve for
   `patience` epochs, saving compute and preventing late-epoch overfit.

Usage
-----
# Recommended
python train_fusion_hypernet_v2.py ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --epochs 60 --batch_size 8 --patience 15

# With focal loss
python train_fusion_hypernet_v2.py ... --focal_loss

# LOPO cross-validation (most reliable with 25 patients)
python lopo_cv.py --model fusion_hypernet_v2 ^
    --csv  data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt checkpoints/US_encoder_supervised_best.pth ^
    --normal_stats data/normal_stats.json ^
    --epochs 30 --out results/lopo_v2
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

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder, PAUSEncoderPair
from encoders_v2 import PAEncoderV2 as PAEncoder, USEncoderV2 as USEncoder
from models import (CrossModalAttentionFusion, ConcatFusion,
                    HyperNetwork, AdaptiveClassifier)
from dataset import PAUSBScanDataset, stratified_patient_split
from transforms import build_train_transform, build_val_transform


# =============================================================================
# ① Gated Modality Weighting
# =============================================================================

class GatedModalityWeighting(nn.Module):
    """
    Learn per-sample importance weights for PA and US before fusion.

    Given f_pa [B, D] and f_us [B, D], produces scalar gates α_pa and α_us
    that sum to 1.0 (softmax).  The gated features are:
        f_pa_gated = α_pa * f_pa
        f_us_gated = α_us * f_us

    A high α_pa means the model found PA more informative for this sample.
    The gate values are returned so they can be logged/visualised.

    Args:
        feat_dim:   Encoder embedding dimension.
        hidden_dim: MLP hidden size (keep small — just 2 outputs needed).
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # → [logit_pa, logit_us]
        )
        # Initialise near-equal gates so training starts balanced
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self, f_pa: torch.Tensor, f_us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_pa_gated [B, D], f_us_gated [B, D], gates [B, 2]
        """
        gates    = F.softmax(self.gate_mlp(torch.cat([f_pa, f_us], dim=1)), dim=1)
        alpha_pa = gates[:, 0:1]   # [B, 1]
        alpha_us = gates[:, 1:2]   # [B, 1]
        return alpha_pa * f_pa, alpha_us * f_us, gates


# =============================================================================
# ② Context Bottleneck
# =============================================================================

class ContextEncoder(nn.Module):
    """
    Project f_pa ‖ f_us ‖ fused [3*feat_dim] → context [ctx_dim].

    The context vector carries richer multi-modal information than fused
    alone, but is compressed to ctx_dim to regularise the HyperNet.

    Args:
        feat_dim: Encoder embedding dimension.
        ctx_dim:  Output context dimension (default: 64).
    """

    def __init__(self, feat_dim: int, ctx_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, ctx_dim),
            nn.Tanh(),   # bound context in [-1, 1] — same as RL latent code
        )

    def forward(
        self, f_pa: torch.Tensor, f_us: torch.Tensor, fused: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(torch.cat([f_pa, f_us, fused], dim=1))


# =============================================================================
# ⑥ Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) with class weighting.

    Down-weights easy correct predictions, focusing training on hard
    borderline cases.  Particularly useful when one class (Normal) is
    systematically easier to classify than the other (Tumour).

    Args:
        alpha:   Per-class weight tensor [num_classes].
        gamma:   Focusing parameter (default: 2.0).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             reduction="none", label_smoothing=0.05)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


# =============================================================================
# Main model — PAUSFusionHyperNetV2
# =============================================================================

class PAUSFusionHyperNetV2(nn.Module):
    """
    Improved PA+US Fusion + HyperNetwork classifier.

    Forward pass:
        1. Encode  → f_pa, f_us  [B, feat_dim]
        2. Gate    → α_pa·f_pa, α_us·f_us  (modality importance weighting)
        3. Fuse    → fused  [B, feat_dim]  via cross-modal attention
        4. Context → ctx  [B, ctx_dim]  from f_pa ‖ f_us ‖ fused
        5. HyperNet(ctx)  → W  [B, 2, feat_dim]
        6. delta_logits   = einsum(fused, W)
        7. base_logits    = Linear(fused)   ← residual base path
        8. logits         = base_logits + delta_logits  ← residual add
        (post-training: logits / T for calibration)

    Args:
        pa_encoder:      Pre-trained PAEncoder (frozen).
        us_encoder:      Pre-trained USEncoder (frozen).
        feat_dim:        Encoder embedding dimension (default: 256).
        num_classes:     Number of output classes (default: 2).
        fusion_type:     'cross_attention' or 'concat'.
        ctx_dim:         Context bottleneck dimension (default: 64).
        hypernet_hidden: HyperNet MLP hidden size (default: 64).
        gate_hidden:     Gating MLP hidden size (default: 32).
        dropout:         Dropout in context encoder and head (default: 0.3).
    """

    def __init__(
        self,
        pa_encoder: PAEncoder,
        us_encoder: USEncoder,
        feat_dim: int = 256,
        num_classes: int = 2,
        fusion_type: str = "cross_attention",
        ctx_dim: int = 64,
        hypernet_hidden: int = 64,
        gate_hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Frozen encoders
        self.encoder_pair = PAUSEncoderPair(pa_encoder, us_encoder, freeze=True)

        # ① Gated modality weighting
        self.gate = GatedModalityWeighting(feat_dim, hidden_dim=gate_hidden)

        # ② Cross-modal attention fusion (on gated features)
        if fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(feat_dim)
        else:
            self.fusion = ConcatFusion(feat_dim)

        # ③ Context bottleneck
        self.context_encoder = ContextEncoder(feat_dim, ctx_dim, dropout=dropout)

        # ④ HyperNetwork (input = context, not raw fused)
        self.hypernet = HyperNetwork(
            z_dim=ctx_dim,
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden_dim=hypernet_hidden,
        )

        # ⑤ Adaptive classifier
        self.adaptive_cls = AdaptiveClassifier()

        # ③ Base classifier (residual path)
        self.base_cls = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        # Initialise base classifier near zero so it starts balanced
        nn.init.zeros_(self.base_cls[-1].bias)

        # ⑤ Learnable temperature for post-training calibration
        # Starts at 1.0 (no scaling), calibrate_temperature() updates it
        self.register_buffer("temperature", torch.ones(1))

        self.feat_dim    = feat_dim
        self.num_classes = num_classes
        self.ctx_dim     = ctx_dim

    # ── Encoder unfreeze helper (same as v1) ─────────────────────────────────
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
        self, pa: torch.Tensor, us: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple:
        """
        Args:
            pa, us:              [B, C, H, W]
            return_intermediates: If True, return (logits, intermediates_dict)

        Returns:
            logits [B, num_classes]  (scaled by temperature)
        """
        # Encode
        f_pa, f_us = self.encoder_pair(pa, us)

        # ① Gate
        f_pa_g, f_us_g, gates = self.gate(f_pa, f_us)

        # ② Fuse gated features
        fused = self.fusion(f_pa_g, f_us_g)

        # ③ Context bottleneck (uses original f_pa, f_us, fused)
        ctx = self.context_encoder(f_pa, f_us, fused)

        # HyperNet → adaptive delta
        W            = self.hypernet(ctx)
        delta_logits = self.adaptive_cls(fused, W)

        # Base path → base logits
        base_logits  = self.base_cls(fused)

        # ③ Residual add
        logits = base_logits + delta_logits

        # ⑤ Temperature scaling
        logits = logits / self.temperature.clamp(min=0.1)

        if return_intermediates:
            return logits, {
                "f_pa": f_pa, "f_us": f_us,
                "gates": gates,
                "fused": fused, "ctx": ctx,
                "W": W,
                "base_logits": base_logits,
                "delta_logits": delta_logits,
            }
        return logits

    def get_embeddings(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full intermediate dict for inference scripts."""
        logits, inter = self.forward(pa, us, return_intermediates=True)
        inter["logits"] = logits
        return inter


# =============================================================================
# ⑤ Temperature scaling calibration
# =============================================================================

def calibrate_temperature(
    model: PAUSFusionHyperNetV2,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Find optimal temperature T on the val set by minimising NLL.

    Sets model.temperature in-place and returns the final T value.
    Call once after training is complete.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """
    model.eval()

    # Collect all logits and labels (before temperature scaling)
    all_logits, all_labels = [], []
    with torch.no_grad():
        model.temperature.fill_(1.0)   # temporarily reset
        for batch in val_loader:
            pa, us, labels = (t.to(device) for t in batch)
            logits, _ = model(pa, us, return_intermediates=True)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Optimise T via LBFGS
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter,
                              line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled_logits = all_logits / T.clamp(min=0.1)
        loss = F.cross_entropy(scaled_logits, all_labels)
        loss.backward()
        return loss

    opt.step(closure)
    T_final = float(T.clamp(min=0.1).item())
    model.temperature.fill_(T_final)

    # Compute calibration improvement
    nll_before = F.cross_entropy(all_logits, all_labels).item()
    nll_after  = F.cross_entropy(all_logits / T_final, all_labels).item()
    print(f"  Temperature calibration: T={T_final:.4f}  "
          f"NLL {nll_before:.4f} → {nll_after:.4f}")
    return T_final


# =============================================================================
# ④ MC-Dropout inference
# =============================================================================

@torch.no_grad()
def predict_with_uncertainty(
    model: PAUSFusionHyperNetV2,
    pa: torch.Tensor,
    us: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-Dropout predictive uncertainty.

    Runs N stochastic forward passes with dropout active, returns mean
    probability and standard deviation across passes.

    Args:
        model:    Must have dropout layers (standard for this model).
        n_samples: Number of MC samples (default: 20).

    Returns:
        mean_probs [B, num_classes], std_probs [B, num_classes]
    """
    model.train()   # enable dropout
    all_probs = []
    for _ in range(n_samples):
        logits = model(pa, us)
        all_probs.append(F.softmax(logits, dim=1))
    model.eval()

    probs_stack = torch.stack(all_probs, dim=0)   # [N, B, C]
    return probs_stack.mean(dim=0), probs_stack.std(dim=0)


# =============================================================================
# Trainer
# =============================================================================

class FusionHyperNetV2Trainer:
    """
    Supervised trainer for PAUSFusionHyperNetV2.

    Two-phase training:
        Phase 1 (epochs 1 .. unfreeze_epoch-1):
            Train fusion + context + gate + hypernet + base_cls only.
            Base path converges first, HyperNet learns the residual.

        Phase 2 (epoch unfreeze_epoch onwards):
            Add encoder last-layer parameters at a 10× lower LR.

    Early stopping: halts when smoothed val AUC has not improved for
    `patience` consecutive epochs.
    """

    def __init__(
        self,
        model: PAUSFusionHyperNetV2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        threshold: float = 0.3,
        unfreeze_epoch: int = 20,
        encoder_lr_scale: float = 0.1,
        focal_loss: bool = False,
        smoothing_window: int = 3,
        patience: int = 15,
        use_wandb: bool = False,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.threshold      = threshold
        self.unfreeze_epoch = unfreeze_epoch
        self.encoder_lr_scale = encoder_lr_scale
        self.smoothing_window = smoothing_window
        self.patience       = patience
        self.use_wandb      = use_wandb
        self._unfrozen      = False
        self._base_lr       = lr

        # Param groups — base path gets 2× higher LR to converge first
        trainable = [
            {"params": model.gate.parameters(),           "lr": lr},
            {"params": model.fusion.parameters(),          "lr": lr},
            {"params": model.context_encoder.parameters(), "lr": lr},
            {"params": model.hypernet.parameters(),        "lr": lr},
            {"params": model.adaptive_cls.parameters(),    "lr": lr},
            {"params": model.base_cls.parameters(),        "lr": lr * 2},
        ]
        self.optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        # Loss function
        class_weights = self._compute_class_weights(train_loader, device)
        if focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            print(f"  Loss: Focal (γ=2.0)")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=0.1)
            print(f"  Loss: weighted CE + label smoothing")
        print(f"  Class weights: Normal={class_weights[0]:.3f}  "
              f"Tumor={class_weights[1]:.3f}")
        print(f"  Threshold: {threshold}  |  patience: {patience}")

    @staticmethod
    def _compute_class_weights(loader, device):
        try:
            labels = []
            for batch in loader:
                lbl = batch[-1]
                labels.extend(lbl.tolist() if isinstance(lbl, torch.Tensor) else lbl)
            labels    = torch.tensor(labels, dtype=torch.long)
            n_classes = int(labels.max().item()) + 1
            counts    = torch.bincount(labels, minlength=n_classes).float()
            weights   = labels.shape[0] / (n_classes * counts.clamp(min=1))
            return weights.to(device)
        except Exception as e:
            print(f"  [WARN] Class weight failed ({e})")
            return torch.ones(2, device=device)

    def _maybe_unfreeze(self, epoch: int):
        if self._unfrozen or epoch < self.unfreeze_epoch:
            return
        print(f"\n  [Epoch {epoch}] Unfreezing encoder heads "
              f"(LR = {self._base_lr * self.encoder_lr_scale:.2e})")
        self.model.unfreeze_encoder_head()
        existing   = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
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
        gate_pa_sum = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for batch in pbar:
            pa, us, labels = self._unpack(batch)
            logits, inter  = self.model(pa, us, return_intermediates=True)
            loss           = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            tot_loss    += loss.item()
            probs        = F.softmax(logits, dim=1)[:, 1]
            preds        = (probs >= self.threshold).long()
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            gate_pa_sum += inter["gates"][:, 0].mean().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.*correct/total:.1f}%",
                             αpa=f"{gate_pa_sum/(total/labels.size(0)):.2f}")

        n = len(self.train_loader)
        return {
            "train/loss":     tot_loss / n,
            "train/accuracy": 100.0 * correct / total,
            "train/gate_pa":  gate_pa_sum / n,   # avg PA gate weight
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        tot_loss = 0.0
        correct = total = 0
        all_probs, all_preds, all_labels = [], [], []
        gate_pa_sum = 0.0

        for batch in tqdm(self.val_loader, desc="  Val"):
            pa, us, labels = self._unpack(batch)
            logits, inter  = self.model(pa, us, return_intermediates=True)
            tot_loss      += self.criterion(logits, labels).item()

            probs    = F.softmax(logits, dim=1)[:, 1]
            preds    = (probs >= self.threshold).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            gate_pa_sum += inter["gates"][:, 0].mean().item()

        try:
            from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
            auc = roc_auc_score(all_labels, all_probs)
            f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            cm  = confusion_matrix(all_labels, all_preds)
        except Exception:
            auc, f1, cm = float("nan"), float("nan"), None

        n = len(self.val_loader)
        return {
            "val/loss":     tot_loss / n,
            "val/accuracy": 100.0 * correct / total,
            "val/auc":      auc,
            "val/f1":       f1,
            "val/gate_pa":  gate_pa_sum / n,
            "_cm":          cm,
        }

    def train(self, num_epochs: int, save_dir: str = "./checkpoints",
              ckpt_name: str = "best_fusion_hypernet_v2_model.pth"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_smoothed_auc  = 0.0
        auc_history: List[float] = []
        epochs_without_improvement = 0
        best_epoch = 0

        print(f"\n{'='*60}")
        print(f"FusionHyperNetV2 Training  ({num_epochs} epochs)")
        print(f"  Smoothing window: {self.smoothing_window}  |  "
              f"Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            self._maybe_unfreeze(epoch)
            train_m = self.train_epoch(epoch)
            val_m   = self.validate()
            self.scheduler.step()

            cm  = val_m.pop("_cm")
            auc = val_m["val/auc"]
            auc_history.append(auc)
            smoothed = float(np.mean(auc_history[-self.smoothing_window:]))
            val_m["val/auc_smoothed"] = smoothed

            # Gate stats — tell us if the model is learning to weight modalities
            g_pa = train_m["train/gate_pa"]
            g_us = 1.0 - g_pa

            print(f"  E{epoch:>2}  loss={train_m['train/loss']:.4f}  "
                  f"acc={train_m['train/accuracy']:.1f}%  "
                  f"gate_PA={g_pa:.2f}/US={g_us:.2f}  ||  "
                  f"val_acc={val_m['val/accuracy']:.1f}%  "
                  f"AUC={auc:.4f}  smooth={smoothed:.4f}  "
                  f"F1={val_m['val/f1']:.4f}")
            if cm is not None:
                print(f"       CM: {cm.ravel().tolist()}  [TN FP FN TP]")

            if self.use_wandb:
                import wandb
                wandb.log({**train_m, **val_m,
                           "lr": self.optimizer.param_groups[0]["lr"]},
                          step=epoch)

            if smoothed > best_smoothed_auc:
                best_smoothed_auc = smoothed
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc":          auc,
                    "val_auc_smoothed": smoothed,
                    "val_accuracy":     val_m["val/accuracy"],
                    "metrics":          {**train_m, **val_m},
                    "temperature":      float(self.model.temperature.item()),
                }, save_path / ckpt_name)
                print(f"  ✓ Best smoothed AUC={smoothed:.4f}  "
                      f"(raw={auc:.4f})  epoch={epoch}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"\n  [Early stop] No improvement for {self.patience} "
                          f"epochs. Stopping at epoch {epoch}.")
                    break

        print(f"\nDone.  Best smoothed AUC={best_smoothed_auc:.4f}  "
              f"(epoch {best_epoch})")
        print(f"Checkpoint: {save_path/ckpt_name}\n")
        return best_epoch


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Improved Supervised Fusion + HyperNet v2 (no RL)"
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
    p.add_argument("--ctx_dim",          type=int, default=64,
                   help="Context bottleneck dimension (default: 64)")
    p.add_argument("--hypernet_hidden",  type=int, default=64)
    p.add_argument("--gate_hidden",      type=int, default=32,
                   help="Gating MLP hidden size (default: 32)")
    p.add_argument("--dropout",          type=float, default=0.3)
    p.add_argument("--epochs",           type=int,   default=60)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-3)
    p.add_argument("--threshold",        type=float, default=0.3)
    p.add_argument("--val_fraction",     type=float, default=0.20)
    p.add_argument("--test_fraction",    type=float, default=0.15)
    p.add_argument("--unfreeze_epoch",   type=int,   default=20)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1)
    p.add_argument("--focal_loss",       action="store_true",
                   help="Use focal loss instead of weighted CE")
    p.add_argument("--smoothing_window", type=int,   default=3)
    p.add_argument("--patience",         type=int,   default=15,
                   help="Early stopping patience (default: 15)")
    p.add_argument("--calibrate",        action="store_true", default=True,
                   help="Run temperature calibration after training (default: on)")
    p.add_argument("--no_calibrate",     dest="calibrate", action="store_false")
    p.add_argument("--mc_samples",       type=int,   default=20,
                   help="MC-Dropout samples for test uncertainty (default: 20)")
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
    print("Improved Fusion + HyperNet v2  (no RL)")
    print(f"{'='*60}")
    print(f"  Fusion type     : {args.fusion_type}")
    print(f"  Context dim     : {args.ctx_dim}")
    print(f"  HyperNet hidden : {args.hypernet_hidden}")
    print(f"  Loss            : {'focal' if args.focal_loss else 'weighted CE'}")
    print(f"  Calibrate       : {args.calibrate}")
    print(f"  MC samples      : {args.mc_samples}")
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
                                         image_size, args.image_type, is_train=True)
        val_ds   = NormalisedPAUSDataset(val_df,   stats, fallback,
                                         image_size, args.image_type, is_train=False)
        test_ds  = NormalisedPAUSDataset(test_df,  stats, fallback,
                                         image_size, args.image_type, is_train=False)
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
        print(f"  ✓ PA encoder: {args.pa_ckpt}")

    if args.us_ckpt:
        ckpt = torch.load(args.us_ckpt, map_location="cpu", weights_only=False)
        us_enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        us_enc.detach_projection_head()
        print(f"  ✓ US encoder: {args.us_ckpt}")

    model = PAUSFusionHyperNetV2(
        pa_encoder=pa_enc, us_encoder=us_enc,
        feat_dim=args.feat_dim, num_classes=args.num_classes,
        fusion_type=args.fusion_type,
        ctx_dim=args.ctx_dim,
        hypernet_hidden=args.hypernet_hidden,
        gate_hidden=args.gate_hidden,
        dropout=args.dropout,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params   : {total - trainable:,}")

    if args.use_wandb:
        import wandb
        wandb.init(project="paus-fusion-hypernet-v2", config=vars(args))

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = FusionHyperNetV2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        unfreeze_epoch=args.unfreeze_epoch,
        encoder_lr_scale=args.encoder_lr_scale,
        focal_loss=args.focal_loss,
        smoothing_window=args.smoothing_window,
        patience=args.patience,
        use_wandb=args.use_wandb,
    )
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    # ── Load best ─────────────────────────────────────────────────────────
    best_path = Path(args.save_dir) / "best_fusion_hypernet_v2_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded best model  epoch={ckpt['epoch']}  "
              f"AUC={ckpt['val_auc']:.4f}")

    model.to(device)

    # ⑤ Temperature calibration
    if args.calibrate:
        print("\nCalibrating temperature on val set...")
        calibrate_temperature(model, val_loader, device)
        # Re-save with updated temperature
        ckpt["temperature"] = float(model.temperature.item())
        torch.save(ckpt, best_path)
        print(f"  ✓ Temperature={model.temperature.item():.4f} saved to checkpoint")

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\nEvaluating on test set (with MC-Dropout uncertainty)...")
    model.eval()

    all_mean_probs, all_std_probs, all_labels, all_preds = [], [], [], []
    correct = total = 0

    for batch in tqdm(test_loader, desc="Test"):
        pa, us, labels = (t.to(device) for t in batch)
        mean_p, std_p = predict_with_uncertainty(
            model, pa, us, n_samples=args.mc_samples)
        tumor_prob = mean_p[:, 1]
        preds      = (tumor_prob >= args.threshold).long()
        correct   += (preds == labels).sum().item()
        total     += labels.size(0)
        all_mean_probs.extend(tumor_prob.cpu().tolist())
        all_std_probs.extend(std_p[:, 1].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    test_acc = 100.0 * correct / total
    try:
        from sklearn.metrics import (roc_auc_score, f1_score,
                                      confusion_matrix, roc_curve)
        test_auc = roc_auc_score(all_labels, all_mean_probs)
        test_f1  = f1_score(all_labels, all_preds,
                            average="weighted", zero_division=0)
        cm       = confusion_matrix(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_mean_probs)
        idx = np.where(1 - fpr >= 0.90)[0]
        sens90 = float(tpr[idx[-1]]) if len(idx) else float("nan")
    except Exception:
        test_auc = test_f1 = sens90 = float("nan"); cm = None

    mean_unc = float(np.mean(all_std_probs))

    print(f"\n{'='*60}")
    print(f"Test Results — FusionHyperNetV2 (threshold={args.threshold})")
    print(f"  Accuracy            : {test_acc:.2f}%")
    print(f"  AUC                 : {test_auc:.4f}")
    print(f"  F1 (weighted)       : {test_f1:.4f}")
    print(f"  Sensitivity@90spec  : {sens90:.4f}")
    print(f"  Mean MC uncertainty : {mean_unc:.4f}")
    if cm is not None:
        tn, fp, fn, tp = cm.ravel()
        print(f"  Sensitivity (TPR)   : {tp/(tp+fn+1e-9):.4f}")
        print(f"  Specificity (TNR)   : {tn/(tn+fp+1e-9):.4f}")
        print(f"  Confusion matrix    :\n{cm}")
    print(f"{'='*60}")

    if args.use_wandb:
        import wandb
        wandb.log({"test/accuracy": test_acc, "test/auc": test_auc,
                   "test/f1": test_f1, "test/sens90spec": sens90,
                   "test/mean_uncertainty": mean_unc})
        wandb.finish()


if __name__ == "__main__":
    main()
