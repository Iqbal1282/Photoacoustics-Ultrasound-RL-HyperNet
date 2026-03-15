"""
Encoder v2 Pre-training Script
================================

Trains PAEncoderV2 and USEncoderV2 using either:

  Mode A — SupCon (Supervised Contrastive)   ← NEW, replaces SimCLR
      Uses tumour/normal labels to pull same-class scans together and
      push different-class scans apart.  Much more label-efficient than
      SimCLR with only 25 patients.

  Mode B — Supervised (cross-entropy)
      Standard CE on binary tumour/normal label.
      Use after SupCon pre-training for fine-tuning.

Usage
-----
# SupCon pre-training (recommended first step)
python train_encoders_v2.py --modality PA --mode supcon ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 100 ^
    --normal_stats data/normal_stats.json

python train_encoders_v2.py --modality US --mode supcon ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 100 ^
    --normal_stats data/normal_stats.json

# Supervised fine-tune from SupCon checkpoint
python train_encoders_v2.py --modality PA --mode supervised ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 30 ^
    --resume checkpoints/PA_encoder_v2_supcon_best.pth ^
    --normal_stats data/normal_stats.json

python train_encoders_v2.py --modality US --mode supervised ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 30 ^
    --resume checkpoints/US_encoder_v2_supcon_best.pth ^
    --normal_stats data/normal_stats.json
"""

from __future__ import annotations

import argparse
import math
import os as _os
import platform
import random
import sys as _sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from encoders_v2 import PAEncoderV2, USEncoderV2, SupConLoss
from dataset_arpam import ArpamBScanDataset
from dataset import stratified_patient_split
from transforms import build_train_transform, build_val_transform


def _safe_nw(n: int) -> int:
    return 0 if platform.system() == "Windows" else n


# =============================================================================
# SupCon dataset wrapper — yields two views per scan + label
# =============================================================================

class SupConWrapper(Dataset):
    """
    Yields (view1, view2, label) for SupCon pre-training.

    Two independently augmented views of the same scan share the same
    label.  SupConLoss uses these labels to define positives/negatives.
    """

    def __init__(self, base_dataset, augment: Callable, modality: str = "PA"):
        self.base     = base_dataset
        self.augment  = augment
        self.modality = modality

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) == 2:
            x, y = item
        elif len(item) == 3:
            pa, us, y = item
            x = pa if self.modality == "PA" else us
        else:
            raise ValueError(f"Unexpected item length {len(item)}")

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        return self.augment(x.clone()), self.augment(x.clone()), int(y)


# =============================================================================
# Classification head
# =============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# =============================================================================
# Encoder trainer v2
# =============================================================================

class EncoderTrainerV2:
    """
    Unified trainer for PAEncoderV2 and USEncoderV2.

    Mode 'supcon':     SupCon loss on two views per scan.
    Mode 'supervised': Weighted CE + focal loss on single view.
    """

    def __init__(
        self,
        encoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        mode: str = "supcon",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        modality: str = "PA",
        num_classes: int = 2,
        temperature: float = 0.07,
        use_wandb: bool = False,
    ):
        self.encoder      = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.mode         = mode
        self.num_epochs   = num_epochs
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.modality     = modality
        self.use_wandb    = use_wandb
        self.head: Optional[ClassificationHead] = None
        self.best_metric  = -math.inf

        if mode == "supcon":
            self.encoder.attach_projection_head(hidden_dim=256, out_dim=128)
            self.criterion = SupConLoss(temperature=temperature)
        else:
            # Supervised: compute class weights
            self.head = ClassificationHead(encoder.feat_dim,
                                           num_classes).to(device)
            class_weights = self._compute_weights(train_loader, device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights,
                                                  label_smoothing=0.1)
            print(f"  Class weights: {class_weights.tolist()}")

        params = list(self.encoder.parameters())
        if self.head:
            params += list(self.head.parameters())
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=5)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(1, num_epochs - 5))
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[5])

    @staticmethod
    def _compute_weights(loader, device):
        try:
            labels = []
            for batch in loader:
                lbl = batch[-1]
                labels.extend(lbl.tolist() if isinstance(lbl, torch.Tensor)
                              else lbl)
            t = torch.tensor(labels, dtype=torch.long)
            c = torch.bincount(t, minlength=2).float()
            return (t.shape[0] / (2 * c.clamp(min=1))).to(device)
        except Exception:
            return torch.ones(2, device=device)

    def _to(self, *ts):
        return [t.to(self.device) for t in ts]

    def _save(self, epoch: int, metric: float, tag: str) -> Path:
        ckpt = {
            "epoch":                epoch,
            "modality":             self.modality,
            "mode":                 self.mode,
            "encoder_state_dict":   self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric":               metric,
        }
        if self.head:
            ckpt["head_state_dict"] = self.head.state_dict()
        name = f"{self.modality}_encoder_v2_{self.mode}_{tag}.pth"
        p    = self.save_dir / name
        torch.save(ckpt, p)
        return p

    # ── SupCon epoch ─────────────────────────────────────────────────────

    def _supcon_epoch(self, epoch: int) -> dict:
        self.encoder.train()
        total = 0.0
        pbar  = tqdm(self.train_loader,
                     desc=f"[{self.modality}] SupCon E{epoch}")
        for view1, view2, labels in pbar:
            view1, view2, labels = self._to(view1, view2, labels)
            # Two views → two projections → batch them for SupCon
            _, z1 = self.encoder(view1, return_projection=True)
            _, z2 = self.encoder(view2, return_projection=True)
            # Concatenate both views; labels repeated accordingly
            feats  = torch.cat([z1, z2], dim=0)
            lbls   = torch.cat([labels, labels], dim=0)
            loss   = self.criterion(feats, lbls)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return {"train/supcon_loss": total / len(self.train_loader)}

    @torch.no_grad()
    def _supcon_val(self) -> dict:
        """Linear probe AUC on frozen features."""
        self.encoder.eval()
        embs, lbls = [], []
        for batch in self.val_loader:
            x = batch[0].to(self.device)
            embs.append(self.encoder(x).cpu())
            lbls.append(batch[-1] if isinstance(batch[-1], torch.Tensor)
                        else torch.tensor(batch[-1]))
        X = torch.cat(embs).numpy()
        y = torch.cat(lbls).numpy()
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            X    = StandardScaler().fit_transform(X)
            split = max(1, int(0.7 * len(X)))
            clf  = LogisticRegression(max_iter=500, random_state=42,
                                      class_weight="balanced")
            clf.fit(X[:split], y[:split])
            probs = clf.predict_proba(X[split:])[:, 1]
            auc   = roc_auc_score(y[split:], probs) if len(set(y[split:])) > 1 else 0.5
            return {"val/linear_probe_auc": auc}
        except Exception:
            return {"val/linear_probe_auc": 0.5}

    # ── Supervised epoch ──────────────────────────────────────────────────

    def _supervised_epoch(self, epoch: int) -> dict:
        self.encoder.train()
        self.head.train()
        total_loss = 0.0
        correct = total = 0
        pbar = tqdm(self.train_loader,
                    desc=f"[{self.modality}] Supervised E{epoch}")
        for batch in pbar:
            if len(batch) == 2:
                x, y = self._to(*batch)
                emb  = self.encoder(x)
            else:
                pa, us, y = self._to(*batch)
                emb = self.encoder(pa if self.modality == "PA" else us)
            logits      = self.head(emb)
            y           = y.squeeze().long()
            loss        = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.*correct/total:.1f}%")
        return {"train/loss":     total_loss / len(self.train_loader),
                "train/accuracy": 100.0 * correct / total}

    @torch.no_grad()
    def _supervised_val(self) -> dict:
        self.encoder.eval()
        self.head.eval()
        total_loss = 0.0
        correct = total = 0
        probs, lbls = [], []
        for batch in tqdm(self.val_loader, desc="  Val"):
            if len(batch) == 2:
                x, y = self._to(*batch)
                emb  = self.encoder(x)
            else:
                pa, us, y = self._to(*batch)
                emb = self.encoder(pa if self.modality == "PA" else us)
            logits      = self.head(emb)
            y           = y.squeeze().long()
            total_loss += self.criterion(logits, y).item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
            lbls.extend(y.cpu().tolist())
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(lbls, probs)
        except Exception:
            auc = float("nan")
        return {"val/loss":     total_loss / len(self.val_loader),
                "val/accuracy": 100.0 * correct / total,
                "val/auc":      auc}

    # ── Main train loop ───────────────────────────────────────────────────

    def train(self):
        monitor = ("val/linear_probe_auc" if self.mode == "supcon"
                   else "val/auc")
        print(f"\n{'='*60}")
        print(f"Training {self.modality} encoder v2  |  mode: {self.mode}")
        print(f"Monitor metric: {monitor}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            import wandb
            wandb.init(project="paus-encoders-v2",
                       name=f"{self.modality}_{self.mode}")

        for epoch in range(1, self.num_epochs + 1):
            train_m = (self._supcon_epoch(epoch) if self.mode == "supcon"
                       else self._supervised_epoch(epoch))
            val_m   = (self._supcon_val()        if self.mode == "supcon"
                       else self._supervised_val())
            self.scheduler.step()

            print("  E{:>3}/{}  {}".format(
                epoch, self.num_epochs,
                "  |  ".join(f"{k}={v:.4f}"
                             for k, v in {**train_m, **val_m}.items())))

            if self.use_wandb:
                import wandb
                wandb.log({**train_m, **val_m,
                           "lr": self.optimizer.param_groups[0]["lr"]},
                          step=epoch)

            mv = val_m[monitor]
            if not math.isnan(mv) and mv > self.best_metric:
                self.best_metric = mv
                p = self._save(epoch, mv, "best")
                print(f"  ✓ Best {monitor}={mv:.4f}  →  {p}")

            if epoch % 10 == 0:
                self._save(epoch, mv, f"e{epoch:04d}")

        self._save(self.num_epochs, self.best_metric, "final")
        if self.use_wandb:
            import wandb
            wandb.finish()
        print(f"\nDone.  Best {monitor}: {self.best_metric:.4f}")
        print(f"Checkpoints → {self.save_dir}\n")


# =============================================================================
# DataLoader builder
# =============================================================================

def _build_loaders(
    csv_path: str,
    image_type: str,
    target_type: str,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    mode: str,
    modality: str,
    normal_stats: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    df = pd.read_csv(csv_path)
    train_df, val_df, _ = stratified_patient_split(
        df, val_fraction=val_fraction, test_fraction=0.0, seed=42)

    nw = _safe_nw(num_workers)
    if nw != num_workers:
        print("  [Windows] num_workers set to 0")

    if normal_stats and "pair" in image_type:
        from normal_normalisation import load_stats, NormalisedPAUSDataset
        stats, fallback = load_stats(normal_stats)
        train_base = NormalisedPAUSDataset(train_df, stats, fallback,
                                            image_size, image_type,
                                            is_train=True)
        val_base   = NormalisedPAUSDataset(val_df,   stats, fallback,
                                            image_size, image_type,
                                            is_train=False)
    else:
        train_tf = build_train_transform(image_size, modality)
        val_tf   = build_val_transform(image_size)
        train_base = ArpamBScanDataset(train_df, transform=train_tf,
                                        image_type=image_type,
                                        target_type=target_type)
        val_base   = ArpamBScanDataset(val_df,   transform=val_tf,
                                        image_type=image_type,
                                        target_type=target_type)

    aug = build_train_transform(image_size, modality)

    if mode == "supcon":
        train_ds = SupConWrapper(train_base, aug, modality)
        val_ds   = SupConWrapper(val_base,   aug, modality)
    else:
        train_ds = train_base
        val_ds   = val_base

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=nw,
                              pin_memory=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=(nw > 0))
    return train_loader, val_loader


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-train PA/US v2 encoders with SupCon or supervised CE"
    )
    p.add_argument("--csv",          type=str,   required=True)
    p.add_argument("--image_type",   type=str,   default="PAUSradial-pair",
                   choices=["PAradial","USradial","PAUSradial-pair","PAUSrect-pair"])
    p.add_argument("--image_size",   type=int,   nargs=2, default=[512, 512])
    p.add_argument("--target_type",  type=str,   default="response")
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--modality",     type=str,   required=True,
                   choices=["PA", "US", "PAUS"])
    p.add_argument("--mode",         type=str,   default="supcon",
                   choices=["supcon", "supervised"])
    p.add_argument("--in_channels",  type=int,   default=1)
    p.add_argument("--feat_dim",     type=int,   default=256)
    p.add_argument("--pretrained",   action="store_true", default=True)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature",  type=float, default=0.07,
                   help="SupCon temperature (default: 0.07)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Resume from checkpoint (supcon → supervised fine-tune)")
    p.add_argument("--normal_stats", type=str,   default=None)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  OS: {platform.system()}")
    if args.normal_stats:
        print(f"Normal-tissue normalisation: {args.normal_stats}")

    image_size = tuple(args.image_size)
    modalities = ["PA", "US"] if args.modality == "PAUS" else [args.modality]

    image_type = args.image_type
    if args.modality == "PAUS" and "pair" not in image_type:
        image_type = "PAUSradial-pair"

    for mod in modalities:
        print(f"\n{'='*50}\nTraining {mod} encoder v2  ({args.mode})\n{'='*50}")

        train_loader, val_loader = _build_loaders(
            csv_path=args.csv, image_type=image_type,
            target_type=args.target_type, image_size=image_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            val_fraction=args.val_fraction, mode=args.mode,
            modality=mod, normal_stats=args.normal_stats,
        )

        enc_cls  = PAEncoderV2 if mod == "PA" else USEncoderV2
        enc      = enc_cls(in_channels=args.in_channels,
                           feat_dim=args.feat_dim,
                           pretrained=args.pretrained)

        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu",
                              weights_only=False)
            enc.load_state_dict(ckpt["encoder_state_dict"], strict=False)
            enc.detach_projection_head()
            print(f"✓ Resumed from {args.resume}")

        EncoderTrainerV2(
            encoder=enc, train_loader=train_loader, val_loader=val_loader,
            device=str(device), mode=args.mode, lr=args.lr,
            weight_decay=args.weight_decay, num_epochs=args.epochs,
            save_dir=args.save_dir, modality=mod,
            temperature=args.temperature, use_wandb=args.use_wandb,
        ).train()


if __name__ == "__main__":
    main()
