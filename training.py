"""
Training Script — PA+US RL-HyperNet
======================================

Supports two training strategies:
  1. REINFORCE  — simple policy-gradient (lighter, faster to converge).
  2. PPO-style  — clipped policy gradient + value baseline (more stable
                  for longer training runs).

The encoders are expected to already be pre-trained (via train_encoders.py)
and are loaded frozen.  Only the fusion module, policy network, HyperNetwork,
and adaptive classifier are updated here.

Usage:
    # Quick test with dummy data
    python training.py --dummy --epochs 5

    # Real data (REINFORCE)
    python training.py \
        --csv ~/data/arpam_roi_select_281/bscan_dataset.csv \
        --pa_ckpt checkpoints/PA_encoder_supervised_best.pth \
        --us_ckpt checkpoints/US_encoder_supervised_best.pth \
        --image_type PAUSradial-pair \
        --epochs 50 --batch_size 16

    # Real data (PPO)
    python training.py \
        --csv ~/data/arpam_roi_select_281/bscan_dataset.csv \
        --pa_ckpt checkpoints/PA_encoder_supervised_best.pth \
        --us_ckpt checkpoints/US_encoder_supervised_best.pth \
        --image_type PAUSradial-pair \
        --epochs 50 --use_ppo
"""

from __future__ import annotations

import argparse
import os as _os
import random
import sys as _sys
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

# Ensure the package directory is on sys.path (Windows-safe, no pathlib).
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder                                  # noqa: E402
from models import PAUSRLHyperNet, compute_reward                          # noqa: E402
from dataset import create_paus_dataloaders, create_dummy_paus_dataloaders # noqa: E402


# ============================================================================
# REINFORCE trainer
# ============================================================================

class RLHyperNetTrainer:
    """
    Trains the PA+US RL-HyperNet with REINFORCE-style policy gradient.

    Loss = CE(logits, labels)  +  β · RL_loss
    RL_loss = −reward · mean(z²)    (penalises large latent codes on wrong
                                      predictions, rewards them on correct ones)

    Args:
        model:         PAUSRLHyperNet instance (encoders already frozen).
        train_loader:  DataLoader yielding (pa, us, label).
        val_loader:    DataLoader yielding (pa, us, label).
        device:        'cuda' or 'cpu'.
        lr:            Learning rate.
        beta:          Weight of RL loss relative to CE loss.
        use_wandb:     Log to Weights & Biases.
    """

    def __init__(
        self,
        model: PAUSRLHyperNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        beta: float = 0.1,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.beta = beta
        self.use_wandb = use_wandb

        # Only update non-encoder parameters
        trainable = [
            {"params": self.model.fusion.parameters(),     "lr": lr},
            {"params": self.model.policy.parameters(),     "lr": lr},
            {"params": self.model.hypernet.parameters(),   "lr": lr},
            {"params": self.model.classifier.parameters(), "lr": lr},
        ]
        self.optimizer = AdamW(trainable, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        self.ce_loss   = nn.CrossEntropyLoss()

        self.train_history: List[Dict] = []
        self.val_history:   List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        pa, us, labels = batch
        return (pa.to(self.device),
                us.to(self.device),
                labels.to(self.device))

    # ──────────────────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()

        tot_loss = tot_ce = tot_rl = tot_reward = 0.0
        correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for batch in pbar:
            pa, us, labels = self._unpack_batch(batch)

            logits, z = self.model(pa, us)

            ce_loss = self.ce_loss(logits, labels)

            with torch.no_grad():
                reward = compute_reward(logits, labels, confidence_weight=0.1)

            rl_loss = -reward * torch.mean(z ** 2)

            loss = ce_loss + self.beta * rl_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            tot_loss   += loss.item()
            tot_ce     += ce_loss.item()
            tot_rl     += rl_loss.item()
            tot_reward += reward.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.*correct/total:.1f}%",
                r=f"{reward.item():.3f}",
            )

        n = len(self.train_loader)
        return {
            "train/loss":     tot_loss   / n,
            "train/ce_loss":  tot_ce     / n,
            "train/rl_loss":  tot_rl     / n,
            "train/reward":   tot_reward / n,
            "train/accuracy": 100.0 * correct / total,
        }

    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[Dict, list, list]:
        self.model.eval()

        tot_loss = tot_reward = 0.0
        correct = total = 0
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(self.val_loader, desc="  Val"):
            pa, us, labels = self._unpack_batch(batch)
            logits, _ = self.model(pa, us)

            tot_loss   += self.ce_loss(logits, labels).item()
            tot_reward += compute_reward(logits, labels).item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float("nan")

        n = len(self.val_loader)
        metrics = {
            "val/loss":     tot_loss   / n,
            "val/reward":   tot_reward / n,
            "val/accuracy": 100.0 * correct / total,
            "val/auc":      auc,
        }
        return metrics, all_preds, all_labels

    # ──────────────────────────────────────────────────────────────────────

    def train(self, num_epochs: int, save_dir: str = "./checkpoints"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        best_auc = 0.0

        if self.use_wandb:
            import wandb

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")

            train_m = self.train_epoch(epoch)
            val_m, _, _ = self.validate(epoch)

            self.scheduler.step()

            all_m = {**train_m, **val_m,
                     "lr": self.optimizer.param_groups[0]["lr"]}
            self.train_history.append(train_m)
            self.val_history.append(val_m)

            print(
                f"  Train — loss: {train_m['train/loss']:.4f} | "
                f"acc: {train_m['train/accuracy']:.2f}% | "
                f"reward: {train_m['train/reward']:.3f}"
            )
            print(
                f"  Val   — loss: {val_m['val/loss']:.4f} | "
                f"acc: {val_m['val/accuracy']:.2f}% | "
                f"AUC: {val_m['val/auc']:.3f}"
            )

            if self.use_wandb:
                import wandb
                wandb.log(all_m, step=epoch)

            if val_m["val/auc"] > best_auc:
                best_auc = val_m["val/auc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auc": best_auc,
                    "metrics": all_m,
                }, save_path / "best_model.pth")
                print(f"  ✓ Saved best model (val_auc={best_auc:.2f}%)")

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "metrics": all_m,
                }, save_path / f"ckpt_epoch{epoch:04d}.pth")

        print(f"\n{'='*60}\nTraining done.  Best val auc: {best_auc:.2f}%\n{'='*60}")
        if self.use_wandb:
            import wandb
            wandb.finish()


# ============================================================================
# PPO-style trainer
# ============================================================================

class PPOStyleTrainer(RLHyperNetTrainer):
    """
    Enhanced trainer with PPO clipping + value baseline for stable RL.

    Extra components:
      - Value network V(z) estimates expected reward.
      - Advantage A = reward − V(z) reduces variance in policy gradient.
      - PPO clipping prevents overly large policy updates.
      - Entropy bonus encourages exploration of the latent space.

    Args:
        clip_epsilon:  PPO clipping range (default 0.2).
        value_coef:    Weight of value loss (default 0.5).
        entropy_coef:  Weight of entropy bonus (default 0.01).
    """

    def __init__(self, *args,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_epsilon  = clip_epsilon
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef

        # Value network takes z → scalar reward estimate
        z_dim = self.model.z_dim
        self.value_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        # Add value net to optimizer
        self.optimizer.add_param_group({
            "params": self.value_net.parameters(),
            "lr": self.optimizer.param_groups[0]["lr"],
        })

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self.value_net.train()

        tot_loss = tot_ce = tot_pol = tot_val = tot_reward = 0.0
        correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"Train(PPO) E{epoch}")
        for batch in pbar:
            pa, us, labels = self._unpack_batch(batch)

            logits, z = self.model(pa, us)
            ce_loss = self.ce_loss(logits, labels)

            with torch.no_grad():
                reward = compute_reward(logits, labels, confidence_weight=0.1)

            # Value estimate and advantage
            value = self.value_net(z).squeeze(-1)   # (B,)
            advantage = reward - value.mean().detach()

            # Simplified ratio (no explicit old policy stored)
            ratio = torch.exp(-torch.mean(z ** 2))
            clipped = torch.clamp(ratio,
                                  1.0 - self.clip_epsilon,
                                  1.0 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantage,
                                     clipped * advantage)

            # Value loss
            val_target = torch.full((value.shape[0],), reward.item(),
                                    device=self.device)
            value_loss = F.mse_loss(value, val_target)

            # Entropy bonus (encourages z diversity)
            entropy = -torch.mean(z * torch.log(torch.abs(z) + 1e-8))

            loss = (ce_loss
                    + self.beta * policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.optimizer.step()

            tot_loss   += loss.item()
            tot_ce     += ce_loss.item()
            tot_pol    += policy_loss.item()
            tot_val    += value_loss.item()
            tot_reward += reward.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.*correct/total:.1f}%",
                r=f"{reward.item():.3f}",
            )

        n = len(self.train_loader)
        return {
            "train/loss":         tot_loss   / n,
            "train/ce_loss":      tot_ce     / n,
            "train/policy_loss":  tot_pol    / n,
            "train/value_loss":   tot_val    / n,
            "train/reward":       tot_reward / n,
            "train/accuracy":     100.0 * correct / total,
        }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train PA+US RL-HyperNet model"
    )

    # Data
    p.add_argument("--csv", type=str, default=None,
                   help="Path to bscan_dataset.csv")
    p.add_argument("--image_type", type=str, default="PAUSradial-pair",
                   choices=["PAUSradial-pair", "PAUSrect-pair"])
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--in_channels", type=int, default=1,
                   help="1 for grayscale, 3 for force_3chan")
    p.add_argument("--target_type", type=str, default="response",
                   choices=["response", "pathology"])
    p.add_argument("--val_fraction",  type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # Encoder checkpoints
    p.add_argument("--pa_ckpt", type=str, default=None,
                   help="PA encoder checkpoint (.pth)")
    p.add_argument("--us_ckpt", type=str, default=None,
                   help="US encoder checkpoint (.pth)")

    # Model
    p.add_argument("--feat_dim",    type=int, default=256)
    p.add_argument("--z_dim",       type=int, default=32)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--fusion_type", type=str, default="cross_attention",
                   choices=["concat", "cross_attention"])
    p.add_argument("--hidden_dim",  type=int, default=128)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--beta",   type=float, default=0.1,
                   help="RL loss weight")
    p.add_argument("--use_ppo",       action="store_true")
    p.add_argument("--clip_epsilon",  type=float, default=0.2)
    p.add_argument("--value_coef",    type=float, default=0.5)
    p.add_argument("--entropy_coef",  type=float, default=0.01)

    # Misc
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--save_dir",  type=str, default="./checkpoints")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str,
                   default="paus-rl-hypernet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dummy", action="store_true",
                   help="Use random dummy data (no CSV needed)")

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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("PA+US RL-HyperNet Training")
    print(f"{'='*60}")
    print(f"Device:  {device}")
    print(f"Mode:    {'PPO' if args.use_ppo else 'REINFORCE'}")
    print(f"Fusion:  {args.fusion_type}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    if args.dummy:
        print("Using dummy data for quick test...")
        train_loader, val_loader, test_loader = create_dummy_paus_dataloaders(
            batch_size=args.batch_size,
            in_channels=args.in_channels,
            image_size=tuple(args.image_size),
            num_classes=args.num_classes,
            num_workers=0,
        )
    else:
        assert args.csv is not None, "--csv is required for real data"
        train_loader, val_loader, test_loader = create_paus_dataloaders(
            csv_path=args.csv,
            image_type=args.image_type,
            target_type=args.target_type,
            image_size=tuple(args.image_size),
            in_channels=args.in_channels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    # ── Encoders ───────────────────────────────────────────────────────────
    print("Building encoders...")
    pa_enc = PAEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.pa_ckpt is None))
    us_enc = USEncoder(in_channels=args.in_channels, feat_dim=args.feat_dim,
                       pretrained=(args.us_ckpt is None))

    # ── Model ──────────────────────────────────────────────────────────────
    model = PAUSRLHyperNet(
        pa_encoder=pa_enc,
        us_encoder=us_enc,
        feat_dim=args.feat_dim,
        num_classes=args.num_classes,
        z_dim=args.z_dim,
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
    )

    # Load encoder weights (PAUSEncoderPair loads them internally)
    if args.pa_ckpt:
        ckpt = torch.load(args.pa_ckpt, map_location="cpu", weights_only=False)
        model.encoder_pair.pa_encoder.load_state_dict(
            ckpt["encoder_state_dict"]
        )
        print(f"✓ Loaded PA encoder from {args.pa_ckpt}")

    if args.us_ckpt:
        ckpt = torch.load(args.us_ckpt, map_location="cpu", weights_only=False)
        model.encoder_pair.us_encoder.load_state_dict(
            ckpt["encoder_state_dict"]
        )
        print(f"✓ Loaded US encoder from {args.us_ckpt}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel summary:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {total - trainable:,}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer_cls = PPOStyleTrainer if args.use_ppo else RLHyperNetTrainer

    trainer_kwargs: dict = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        lr=args.lr,
        beta=args.beta,
        use_wandb=args.use_wandb,
    )
    if args.use_ppo:
        trainer_kwargs.update(
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
        )

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    # ── Test evaluation ───────────────────────────────────────────────────
    best_ckpt = Path(args.save_dir) / "best_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n✓ Loaded best model from epoch {ckpt['epoch']}")

    print("\nEvaluating on test set...")
    model.eval()
    model.to(device)

    correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for pa, us, labels in tqdm(test_loader, desc="Test"):
            pa, us = pa.to(device), us.to(device)
            labels = labels.to(device)
            logits, _ = model(pa, us)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = 100.0 * correct / total
    try:
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        test_auc = float("nan")

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  AUC:      {test_auc:.3f}")

    if args.use_wandb:
        import wandb
        wandb.log({"test/accuracy": test_acc, "test/auc": test_auc})
        wandb.finish()


if __name__ == "__main__":
    main()
