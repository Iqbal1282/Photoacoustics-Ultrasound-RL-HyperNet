"""
Encoder Pre-training Script
============================

Trains the PAEncoder and USEncoder independently (or jointly) using either:

  Mode A — Supervised classification
      Standard cross-entropy on the binary tumor / no-tumor label.
      Uses the full ArpamBScanDataset from dataset_arpam.py.

  Mode B — Contrastive pre-training (SimCLR-style)
      Self-supervised: two augmented views of the same B-scan are pushed
      together, while views from different B-scans are pushed apart.
      No labels required; embeddings are learned from image structure alone.
      After contrastive pre-training run Mode A (fine-tune) for a few epochs.

Usage (Windows):
    python train_encoders.py --modality PA --mode supervised ^
        --csv data/bscan_dataset.csv --image_type PAUSradial-pair --epochs 50

    python train_encoders.py --modality PAUS --mode contrastive ^
        --csv data/bscan_dataset.csv --image_type PAUSradial-pair --epochs 100

Windows note
------------
Windows uses the 'spawn' multiprocessing start method, which requires every
object sent to a DataLoader worker to be fully picklable.
  * All transforms use named classes from transforms.py (no lambdas).
  * num_workers is forced to 0 on Windows automatically.
  * The if __name__ == '__main__': guard is required and is present.
"""

from __future__ import annotations

import argparse
import math
import os as _os
import platform
import random
import sys as _sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── sys.path guard (Windows-safe, no pathlib on import) ─────────────────────
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder                          # noqa: E402
from dataset_arpam import ArpamBScanDataset                        # noqa: E402
from transforms import build_train_transform, build_val_transform  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_num_workers(requested: int) -> int:
    """Return 0 on Windows (spawn mode cannot pickle lambdas)."""
    return 0 if platform.system() == "Windows" else requested


# ============================================================================
# Contrastive dataset wrapper
# ============================================================================

class ContrastiveWrapper(Dataset):
    """
    Wraps ArpamBScanDataset to yield two independently-augmented views per
    B-scan for SimCLR-style contrastive pre-training.

    Returns: (view1, view2, label)
    The augment callable must be a picklable object (no lambdas).
    """

    def __init__(self, base_dataset: ArpamBScanDataset, augment: Callable):
        self.base    = base_dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) == 2:
            x, y = item
        elif len(item) == 3:
            x, _, y = item          # pair mode — take first channel (PA)
        else:
            raise ValueError(f"Unexpected item length {len(item)}")

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        view1 = self.augment(x.clone())
        view2 = self.augment(x.clone())
        return view1, view2, int(y)


# ============================================================================
# NT-Xent loss (SimCLR)
# ============================================================================

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.shape[0]
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        sim.masked_fill_(torch.eye(2 * B, device=z.device).bool(), float("-inf"))
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0,     B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ============================================================================
# Classification head (supervised mode)
# ============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ============================================================================
# Encoder trainer
# ============================================================================

class EncoderTrainer:
    """
    Unified trainer for PA and US encoders.

    Args:
        encoder:      PAEncoder or USEncoder instance.
        train_loader: DataLoader yielding (x, y) or (view1, view2, y).
        val_loader:   DataLoader.
        device:       'cuda' or 'cpu'.
        mode:         'supervised' or 'contrastive'.
        lr:           Learning rate.
        weight_decay: L2 regularisation.
        num_epochs:   Training epochs.
        save_dir:     Directory for checkpoint files.
        modality:     'PA' or 'US' — used in checkpoint filenames.
        num_classes:  Output classes (supervised mode only).
        temperature:  NT-Xent τ (contrastive mode only).
        use_wandb:    Log metrics to Weights & Biases.
    """

    def __init__(
        self,
        encoder: PAEncoder | USEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        mode: Literal["supervised", "contrastive"] = "supervised",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
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

        if mode == "contrastive":
            self.encoder.attach_projection_head(hidden_dim=256, out_dim=128)
            self.criterion: nn.Module = NTXentLoss(temperature)
        else:
            self.head      = ClassificationHead(encoder.feat_dim, num_classes).to(device)
            self.criterion = nn.CrossEntropyLoss()

        params = list(self.encoder.parameters())
        if self.head is not None:
            params += list(self.head.parameters())
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=5)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, num_epochs - 5))
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[5])
        self.best_metric = -math.inf

    # ── Internal helpers ──────────────────────────────────────────────────

    def _to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _save_checkpoint(self, epoch: int, metric: float, tag: str = "best") -> Path:
        ckpt = {
            "epoch":                epoch,
            "modality":             self.modality,
            "mode":                 self.mode,
            "encoder_state_dict":   self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric":               metric,
        }
        if self.head is not None:
            ckpt["head_state_dict"] = self.head.state_dict()
        path = self.save_dir / f"{self.modality}_encoder_{self.mode}_{tag}.pth"
        torch.save(ckpt, path)
        return path

    # ── Contrastive training ──────────────────────────────────────────────

    def _contrastive_epoch(self, epoch: int) -> dict:
        self.encoder.train()
        total = 0.0
        pbar  = tqdm(self.train_loader, desc=f"[{self.modality}] Contrastive E{epoch}")
        for view1, view2, _ in pbar:
            view1, view2 = self._to_device(view1, view2)
            _, z1 = self.encoder(view1, return_projection=True)
            _, z2 = self.encoder(view2, return_projection=True)
            loss  = self.criterion(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return {"train/contrastive_loss": total / len(self.train_loader)}

    @torch.no_grad()
    def _contrastive_val(self) -> dict:
        self.encoder.eval()
        embeddings, all_labels = [], []
        for batch in self.val_loader:
            x = batch[0].to(self.device)
            embeddings.append(self.encoder(x).cpu())
            lbl = batch[-1]
            if isinstance(lbl, torch.Tensor):
                all_labels.append(lbl)
            else:
                all_labels.append(torch.tensor(lbl))

        X   = torch.cat(embeddings).numpy()
        y   = torch.cat(all_labels).numpy()

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X     = StandardScaler().fit_transform(X)
        split = max(1, int(0.7 * len(X)))
        clf   = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X[:split], y[:split])
        acc   = clf.score(X[split:], y[split:]) * 100
        return {"val/linear_probe_acc": acc}

    # ── Supervised training ───────────────────────────────────────────────

    def _supervised_epoch(self, epoch: int) -> dict:
        self.encoder.train()
        self.head.train()
        total_loss = 0.0
        correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.modality}] Supervised E{epoch}")
        for batch in pbar:
            if len(batch) == 2:
                x, y = self._to_device(*batch)
                emb  = self.encoder(x)
            else:
                pa, us, y = self._to_device(*batch)
                emb = self.encoder(pa if self.modality == "PA" else us)

            logits     = self.head(emb)
            y          = y.squeeze().long()
            loss       = self.criterion(logits, y)
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
                x, y = self._to_device(*batch)
                emb  = self.encoder(x)
            else:
                pa, us, y = self._to_device(*batch)
                emb = self.encoder(pa if self.modality == "PA" else us)

            logits     = self.head(emb)
            y          = y.squeeze().long()
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

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training {self.modality} encoder  |  mode: {self.mode}")
        print(f"Epochs: {self.num_epochs}  |  Device: {self.device}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            import wandb
            wandb.init(project="paus-encoders",
                       name=f"{self.modality}_{self.mode}",
                       config={"modality": self.modality, "mode": self.mode})

        monitor_key = ("val/linear_probe_acc" if self.mode == "contrastive"
                       else "val/accuracy")

        for epoch in range(1, self.num_epochs + 1):
            if self.mode == "contrastive":
                train_m = self._contrastive_epoch(epoch)
                val_m   = self._contrastive_val()
            else:
                train_m = self._supervised_epoch(epoch)
                val_m   = self._supervised_val()

            self.scheduler.step()

            row = (f"  Epoch {epoch:>3}/{self.num_epochs}  |  "
                   + "  |  ".join(f"{k}={v:.4f}"
                                  for k, v in {**train_m, **val_m}.items()))
            print(row)

            if self.use_wandb:
                import wandb
                wandb.log({**train_m, **val_m,
                           "lr": self.optimizer.param_groups[0]["lr"]},
                          step=epoch)

            mv = val_m[monitor_key]
            if mv > self.best_metric:
                self.best_metric = mv
                p = self._save_checkpoint(epoch, mv, tag="best")
                print(f"  ✓ Best {monitor_key}={mv:.4f}  →  {p}")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, mv, tag=f"epoch{epoch:04d}")

        self._save_checkpoint(self.num_epochs, self.best_metric, tag="final")

        if self.use_wandb:
            import wandb
            wandb.finish()

        print(f"\nDone.  Best {monitor_key}: {self.best_metric:.4f}")
        print(f"Checkpoints → {self.save_dir}\n")


# ============================================================================
# DataLoader builder
# ============================================================================

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
) -> Tuple[DataLoader, DataLoader]:
    df   = pd.read_csv(csv_path)
    pids = sorted(df["pid"].unique().tolist())
    random.seed(42)
    random.shuffle(pids)

    n_val    = max(1, int(len(pids) * val_fraction))
    val_pids = set(pids[:n_val])
    trn_pids = set(pids[n_val:])

    train_df = df[df["pid"].isin(trn_pids)].reset_index(drop=True)
    val_df   = df[df["pid"].isin(val_pids)].reset_index(drop=True)

    print(f"  Train: {len(trn_pids)} patients / {len(train_df)} scans")
    print(f"  Val:   {len(val_pids)} patients / {len(val_df)} scans")

    train_tf = build_train_transform(image_size, modality)
    val_tf   = build_val_transform(image_size)

    train_base = ArpamBScanDataset(train_df, transform=train_tf,
                                   image_type=image_type,
                                   target_type=target_type)
    val_base   = ArpamBScanDataset(val_df,   transform=val_tf,
                                   image_type=image_type,
                                   target_type=target_type)

    if mode == "contrastive":
        train_ds = ContrastiveWrapper(train_base, train_tf)
        val_ds   = ContrastiveWrapper(val_base,   val_tf)
    else:
        train_ds = train_base
        val_ds   = val_base

    nw = _safe_num_workers(num_workers)
    if nw != num_workers:
        print(f"  [Windows] num_workers set to 0 (spawn limitation)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(nw > 0),
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=(nw > 0))
    return train_loader, val_loader


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          type=str,   required=True)
    p.add_argument("--image_type",   type=str,   default="PAUSradial-pair",
                   choices=["PAradial", "USradial", "PArect", "USrect",
                             "PAUSradial-pair", "PAUSrect-pair"])
    p.add_argument("--image_size",   type=int,   nargs=2, default=[512, 512])
    p.add_argument("--target_type",  type=str,   default="response",
                   choices=["response", "pathology"])
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--modality",     type=str,   required=True,
                   choices=["PA", "US", "PAUS"])
    p.add_argument("--mode",         type=str,   default="supervised",
                   choices=["supervised", "contrastive"])
    p.add_argument("--in_channels",     type=int,   default=1)
    p.add_argument("--feat_dim",        type=int,   default=256)
    p.add_argument("--pretrained",      action="store_true", default=True)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature",  type=float, default=0.07)
    p.add_argument("--num_classes",  type=int,   default=2)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args  = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  OS: {platform.system()}")

    image_size = tuple(args.image_size)
    image_type = args.image_type

    if args.modality == "PAUS" and "pair" not in image_type:
        image_type = (image_type
                      .replace("PAradial", "PAUSradial-pair")
                      .replace("USradial", "PAUSradial-pair")
                      .replace("PArect",   "PAUSrect-pair")
                      .replace("USrect",   "PAUSrect-pair"))
        print(f"  Auto-selected image_type={image_type}")

    print("\nBuilding data loaders...")
    train_loader, val_loader = _build_loaders(
        csv_path=args.csv,
        image_type=image_type,
        target_type=args.target_type,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        mode=args.mode,
        modality="PA" if args.modality == "PAUS" else args.modality,
    )

    def _make_encoder(mod: str) -> PAEncoder | USEncoder:
        kw = dict(in_channels=args.in_channels, feat_dim=args.feat_dim,
                  pretrained=args.pretrained, freeze=args.freeze_backbone)
        return PAEncoder(**kw) if mod == "PA" else USEncoder(**kw)

    def _run(mod: str):
        enc = _make_encoder(mod)
        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu")
            # The contrastive checkpoint includes projection_head weights.
            # Load with strict=False so extra keys are ignored, then
            # explicitly drop the projection head for supervised fine-tuning.
            missing, unexpected = enc.load_state_dict(
                ckpt["encoder_state_dict"], strict=False
            )
            enc.detach_projection_head()   # ensure head is removed
            if unexpected:
                print(f"  (ignored checkpoint keys: {unexpected})")
            if missing:
                print(f"  (missing keys filled randomly: {missing})")
            print(f"✓ Resumed {mod} encoder from {args.resume}")
        EncoderTrainer(
            encoder=enc, train_loader=train_loader, val_loader=val_loader,
            device=str(device), mode=args.mode, lr=args.lr,
            weight_decay=args.weight_decay, num_epochs=args.epochs,
            save_dir=args.save_dir, modality=mod,
            num_classes=args.num_classes, temperature=args.temperature,
            use_wandb=args.use_wandb,
        ).train()

    if args.modality in ("PA", "US"):
        _run(args.modality)
    else:
        _run("PA")
        _run("US")


# ── Required on Windows: protects the multiprocessing entry point ────────────
if __name__ == "__main__":
    main()
