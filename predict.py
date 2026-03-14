"""
Inference Script — PA+US RL-HyperNet
======================================

Runs the trained PAUSRLHyperNet on new PA+US B-scan images and produces a
per-scan and per-patient summary CSV.

Data folder structure (mirrors predict_invivo_generic.py):
    root_dir/
        <date>/
            <patient_id>/
                Normal/         ← used for normalisation stats (optional)
                Tumor/          ← or whatever --scan_subdir is
                    <scan_001>/
                        USradial.tif
                        PAradial.tif
                    <scan_002>/
                        ...

Flat mode (--flat):
    root_dir/
        <scan_001>/
            USradial.tif
            PAradial.tif

Usage
-----
# Standard — global normalisation
python predict.py ^
    --model_ckpt  checkpoints/best_model.pth ^
    --pa_ckpt     checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt     checkpoints/US_encoder_supervised_best.pth ^
    --data_root   \\\\10.229.121.108\\Workspace\\ARPAM\\System2\\invivo ^
    --scan_subdir Tumor ^
    --out         results/predictions.csv

# With patient-specific Normal-tissue normalisation (recommended)
python predict.py ^
    --model_ckpt   checkpoints/best_model.pth ^
    --data_root    path/to/invivo ^
    --normal_stats data/normal_stats.json ^
    --scan_subdir  Tumor ^
    --out          results/predictions.csv

# With visualisations
python predict.py ... --visualise --vis_dir results/maps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── sys.path guard ────────────────────────────────────────────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder
from models import PAUSRLHyperNet
from transforms import ContrastBoost, ClampUnit, MinMaxNorm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# =============================================================================
# Preprocessing
# =============================================================================

class InferencePreprocessor:
    """
    Preprocesses a single grayscale image for one modality.

    Two modes
    ---------
    Standard (no normal_stats):
        ContrastBoost → ClampUnit → Resize → MinMaxNorm

    Patient-specific (normal_stats provided):
        NormalTissueNorm(μ_patient, σ_patient) → Resize
        Falls back to FallbackNorm when pid not in stats.

    Call set_patient(pid) before processing each patient's scans.

    Args:
        image_size:    (H, W).
        modality:      'PA' or 'US'.
        normal_stats:  Per-patient stats dict from load_stats(), or None.
        fallback:      Global fallback dict, or None.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        modality: str = "PA",
        normal_stats: Optional[Dict] = None,
        fallback: Optional[Dict] = None,
    ):
        self.H, self.W       = image_size
        self.modality        = modality
        self.normal_stats    = normal_stats
        self.fallback        = fallback or {}
        self._pid: Optional[str] = None

        # Standard pipeline components (used when no normal_stats)
        self._boost = ContrastBoost()
        self._clamp = ClampUnit()
        self._norm  = MinMaxNorm()

    def set_patient(self, pid: str):
        """Call once per patient before processing that patient's scans."""
        self._pid = str(pid)

    def _resolve_norm_op(self):
        """Return the normalisation callable for the current patient."""
        if self.normal_stats is None or self._pid is None:
            return None   # use standard pipeline

        from normal_normalisation import NormalTissueNorm, FallbackNorm

        pid_stats = self.normal_stats.get(self._pid, {})
        mod_stats = pid_stats.get(self.modality) if isinstance(pid_stats, dict) else None

        if mod_stats is None:
            mod_stats = self.fallback.get(self.modality)

        if mod_stats is not None:
            return NormalTissueNorm(mean=mod_stats["mean"], std=mod_stats["std"])
        return FallbackNorm()

    def __call__(self, img_gray: np.ndarray) -> torch.Tensor:
        """
        Args:
            img_gray: uint8 grayscale array (H, W) from cv2.imread.
        Returns:
            Tensor [1, 1, H, W] float32 in [0, 1].
        """
        img = cv2.resize(img_gray, (self.W, self.H),
                         interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        norm_op = self._resolve_norm_op()
        if norm_op is not None:
            x = norm_op(x)
        else:
            x = self._boost(x)
            x = self._clamp(x)
            x = self._norm(x)

        return x   # (1, 1, H, W)


# =============================================================================
# Image finder
# =============================================================================

_US_STEMS  = ["USradial", "usradial", "US_radial", "us_radial",
               "USrect",   "usrect",   "US",         "us"]
_PA_STEMS  = ["PAradial", "paradial", "PA_radial", "pa_radial",
               "PArect",   "parect",   "PA",         "pa"]
_EXTS      = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG",
               ".jpg", ".jpeg", ".JPG", ".JPEG"]


def _find_file(folder: Path, stems: List[str]) -> Optional[Path]:
    for stem in stems:
        for ext in _EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def find_paus_images(scan_folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Returns (us_path, pa_path); either may be None."""
    return (_find_file(scan_folder, _US_STEMS),
            _find_file(scan_folder, _PA_STEMS))


# =============================================================================
# Model loader
# =============================================================================

def load_model(
    model_ckpt: str,
    pa_ckpt: Optional[str],
    us_ckpt: Optional[str],
    feat_dim: int = 256,
    z_dim: int = 32,
    num_classes: int = 2,
    in_channels: int = 1,
    fusion_type: str = "cross_attention",
    hidden_dim: int = 128,
    device: torch.device = torch.device("cpu"),
) -> PAUSRLHyperNet:
    """Load PAUSRLHyperNet from checkpoints."""
    pa_enc = PAEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)
    us_enc = USEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)

    model = PAUSRLHyperNet(
        pa_encoder=pa_enc, us_encoder=us_enc,
        feat_dim=feat_dim, num_classes=num_classes,
        z_dim=z_dim, fusion_type=fusion_type, hidden_dim=hidden_dim,
    )

    ckpt  = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"✓ Loaded RL-HyperNet  epoch={ckpt.get('epoch','?')}  "
          f"AUC={ckpt.get('val_auc','?')}")

    for attr, path, name in [(model.encoder_pair.pa_encoder, pa_ckpt, "PA"),
                              (model.encoder_pair.us_encoder, us_ckpt, "US")]:
        if path:
            ec = torch.load(path, map_location="cpu", weights_only=False)
            attr.load_state_dict(ec["encoder_state_dict"], strict=False)
            attr.detach_projection_head()
            print(f"✓ Overrode {name} encoder from {path}")

    model.to(device).eval()
    return model


# =============================================================================
# Single-scan inference
# =============================================================================

def predict_scan(
    model: PAUSRLHyperNet,
    us_path: Path,
    pa_path: Path,
    pa_preprocessor: InferencePreprocessor,
    us_preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
) -> Optional[Dict]:
    """Run inference on one PA+US scan pair."""
    img_us = cv2.imread(str(us_path), cv2.IMREAD_GRAYSCALE)
    img_pa = cv2.imread(str(pa_path), cv2.IMREAD_GRAYSCALE)

    if img_us is None:
        print(f"  [WARN] Cannot read US: {us_path}")
        return None
    if img_pa is None:
        print(f"  [WARN] Cannot read PA: {pa_path}")
        return None

    pa_t = pa_preprocessor(img_pa).to(device)
    us_t = us_preprocessor(img_us).to(device)

    with torch.no_grad():
        inter = model.get_intermediate(pa_t, us_t)

    probs      = F.softmax(inter["logits"], dim=1)[0]
    pred_class = int(probs.argmax().item())
    tumor_prob = float(probs[1].item()) if len(class_names) == 2 \
                 else float(probs.max().item())

    return {
        "pred_class":     pred_class,
        "pred_label":     class_names[pred_class],
        "tumor_prob":     round(tumor_prob, 4),
        "z_norm":         round(float(inter["z"].norm(dim=1).item()), 4),
        "uncertainty_PA": round(float(inter["u_pa"].item()), 6),
        "uncertainty_US": round(float(inter["u_us"].item()), 6),
        "_pa_np":  img_pa,
        "_us_np":  img_us,
        "_probs":  probs.cpu().numpy(),
    }


# =============================================================================
# Optional visualisation
# =============================================================================

def save_attention_vis(pa_np, us_np, probs, pred_label,
                       save_path: Path, class_names: List[str]):
    if not _MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Prediction: {pred_label}  "
        + (f"(Tumor prob: {probs[1]:.3f})" if len(class_names) == 2 else ""),
        fontsize=13,
    )
    axes[0].imshow(pa_np, cmap="hot");  axes[0].set_title("PA"); axes[0].axis("off")
    axes[1].imshow(us_np, cmap="gray"); axes[1].set_title("US"); axes[1].axis("off")

    colors = ["steelblue"] * len(class_names)
    colors[probs.argmax()] = "tomato"
    axes[2].barh(class_names, probs, color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Probability")
    axes[2].set_title("Class probabilities")
    for i, v in enumerate(probs):
        axes[2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Batch runners
# =============================================================================

def run_nested(
    root_dir: Path,
    model: PAUSRLHyperNet,
    pa_preprocessor: InferencePreprocessor,
    us_preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    scan_subdir: str = "Tumor",
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Walk the data root looking for  <scan_subdir>/<scan>/  folders.

    Supports two layouts automatically:

      Shallow  (your data):
          <root>/<patient_id>/<scan_subdir>/<scan>/
          e.g.  data/258/tumor/20240509 invivo 258_.../

      Deep:
          <root>/<date>/<patient_id>/<scan_subdir>/<scan>/

    The patient_id used for normalisation lookup is always the bare
    patient folder name (e.g. "258") so it matches the pid in normal_stats.json.
    """
    results = []

    for level1 in sorted(root_dir.iterdir()):
        if not level1.is_dir():
            continue

        # ── Shallow layout: <root>/<patient_id>/<scan_subdir>/ ───────────
        subdir_shallow = level1 / scan_subdir
        if subdir_shallow.exists():
            patient_id = level1.name          # e.g. "258"
            print(f"\nPatient: {patient_id}")
            pa_preprocessor.set_patient(patient_id)
            us_preprocessor.set_patient(patient_id)
            results.extend(_process_scan_folder(
                subdir_shallow, patient_id, model,
                pa_preprocessor, us_preprocessor,
                device, class_names, visualise, vis_dir,
            ))
            continue

        # ── Deep layout: <root>/<date>/<patient_id>/<scan_subdir>/ ───────
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            subdir_deep = level2 / scan_subdir
            if subdir_deep.exists():
                patient_id = level2.name      # bare patient name
                print(f"\nPatient: {patient_id}  (date: {level1.name})")
                pa_preprocessor.set_patient(patient_id)
                us_preprocessor.set_patient(patient_id)
                results.extend(_process_scan_folder(
                    subdir_deep, f"{level1.name}/{level2.name}", model,
                    pa_preprocessor, us_preprocessor,
                    device, class_names, visualise, vis_dir,
                ))

    return results


def run_flat(
    root_dir: Path,
    model: PAUSRLHyperNet,
    pa_preprocessor: InferencePreprocessor,
    us_preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """Flat structure — scan folders directly under root_dir."""
    print(f"\nFlat mode: {root_dir}")
    return _process_scan_folder(
        root_dir, "batch", model,
        pa_preprocessor, us_preprocessor,
        device, class_names, visualise, vis_dir,
    )


def _process_scan_folder(
    scan_parent: Path,
    patient_id: str,
    model: PAUSRLHyperNet,
    pa_preprocessor: InferencePreprocessor,
    us_preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    visualise: bool,
    vis_dir: Optional[Path],
) -> List[Dict]:
    rows: List[Dict] = []
    per_patient_probs: List[float] = []

    for scan_folder in sorted(scan_parent.iterdir()):
        if not scan_folder.is_dir():
            continue
        us_path, pa_path = find_paus_images(scan_folder)
        if us_path is None or pa_path is None:
            print(f"  [SKIP] No PA+US pair in {scan_folder.name}")
            continue

        print(f"  Scan: {scan_folder.name}  "
              f"US={us_path.name}  PA={pa_path.name}", end=" ... ")

        res = predict_scan(model, us_path, pa_path,
                           pa_preprocessor, us_preprocessor,
                           device, class_names)
        if res is None:
            print("FAILED")
            continue

        print(f"{res['pred_label']}  (p_tumor={res['tumor_prob']:.3f})")

        if visualise and vis_dir is not None:
            save_attention_vis(
                res["_pa_np"], res["_us_np"], res["_probs"],
                res["pred_label"],
                vis_dir / patient_id.replace("/", "_") / f"{scan_folder.name}.png",
                class_names,
            )

        rows.append({
            "Patient_ID":     patient_id,
            "Scan":           scan_folder.name,
            "Pred_Class":     res["pred_class"],
            "Pred_Label":     res["pred_label"],
            "Tumor_Prob":     res["tumor_prob"],
            "Z_Norm":         res["z_norm"],
            "Uncertainty_PA": res["uncertainty_PA"],
            "Uncertainty_US": res["uncertainty_US"],
            "US_Path":        str(us_path),
            "PA_Path":        str(pa_path),
            "Type":           "Individual",
        })
        per_patient_probs.append(res["tumor_prob"])

    if per_patient_probs:
        mean_prob  = float(np.mean(per_patient_probs))
        mean_class = int(mean_prob >= 0.5)
        rows.append({
            "Patient_ID":     patient_id,
            "Scan":           "AVERAGE",
            "Pred_Class":     mean_class,
            "Pred_Label":     class_names[mean_class],
            "Tumor_Prob":     round(mean_prob, 4),
            "Z_Norm":         "",
            "Uncertainty_PA": "",
            "Uncertainty_US": "",
            "US_Path":        "",
            "PA_Path":        "",
            "Type":           "MEAN",
        })
        print(f"  → Patient mean  Tumor_Prob={mean_prob:.3f}  "
              f"({len(per_patient_probs)} scans)")

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for PA+US RL-HyperNet")

    p.add_argument("--model_ckpt",   type=str, required=True)
    p.add_argument("--data_root",    type=str, required=True,
                   help="Root directory containing patient/scan folders")
    p.add_argument("--out",          type=str, default="predictions.csv")

    p.add_argument("--pa_ckpt",      type=str, default=None)
    p.add_argument("--us_ckpt",      type=str, default=None)

    p.add_argument("--feat_dim",     type=int,   default=256)
    p.add_argument("--z_dim",        type=int,   default=32)
    p.add_argument("--num_classes",  type=int,   default=2)
    p.add_argument("--in_channels",  type=int,   default=1)
    p.add_argument("--fusion_type",  type=str,   default="cross_attention",
                   choices=["concat", "cross_attention"])
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--image_size",   type=int,   nargs=2, default=[512, 512],
                   metavar=("H", "W"))

    p.add_argument("--flat",         action="store_true")
    p.add_argument("--scan_subdir",  type=str,   default="Tumor")

    p.add_argument("--class_names",  type=str, nargs="+",
                   default=["Normal", "Tumor"])

    # Normal-tissue normalisation
    p.add_argument("--normal_stats", type=str, default=None,
                   help="Path to normal_stats.json.  When provided, each "
                        "patient's scans are z-scored against their own "
                        "Normal-tissue baseline (same as training).")

    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--visualise",    action="store_true")
    p.add_argument("--vis_dir",      type=str, default="vis_output")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("PA+US RL-HyperNet Inference")
    print(f"{'='*60}")
    print(f"Device     : {device}")
    print(f"Data root  : {args.data_root}")
    print(f"Output     : {args.out}")
    print(f"Norm mode  : {'Normal-tissue z-score' if args.normal_stats else 'global contrast stretch'}")
    if not args.flat:
        print(f"Scan subdir: {args.scan_subdir}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(
        model_ckpt=args.model_ckpt,
        pa_ckpt=args.pa_ckpt,
        us_ckpt=args.us_ckpt,
        feat_dim=args.feat_dim,
        z_dim=args.z_dim,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    # ── Load normal stats (optional) ──────────────────────────────────────
    normal_stats = fallback = None
    if args.normal_stats:
        from normal_normalisation import load_stats
        normal_stats, fallback = load_stats(args.normal_stats)
        print(f"✓ Loaded Normal stats for {len(normal_stats)} patients")

    # ── Build per-modality preprocessors ──────────────────────────────────
    image_size = tuple(args.image_size)
    pa_pre = InferencePreprocessor(image_size, modality="PA",
                                   normal_stats=normal_stats,
                                   fallback=fallback)
    us_pre = InferencePreprocessor(image_size, modality="US",
                                   normal_stats=normal_stats,
                                   fallback=fallback)

    # ── Run inference ─────────────────────────────────────────────────────
    root_dir = Path(args.data_root)
    vis_dir  = Path(args.vis_dir) if args.visualise else None

    print("\n--- STARTING INFERENCE ---")
    if args.flat:
        results = run_flat(root_dir, model, pa_pre, us_pre,
                           device, args.class_names, args.visualise, vis_dir)
    else:
        results = run_nested(root_dir, model, pa_pre, us_pre,
                             device, args.class_names, args.scan_subdir,
                             args.visualise, vis_dir)

    if not results:
        print("\n[WARN] No results — check --data_root and folder structure.")
        return

    df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    individual = df[df["Type"] == "Individual"]
    n_scans    = len(individual)
    n_tumor    = (individual["Pred_Class"] == 1).sum()

    print(f"\n{'='*60}")
    print(f"Inference complete.")
    print(f"  Scans processed : {n_scans}")
    print(f"  Predicted Tumor : {n_tumor}  ({100*n_tumor/max(n_scans,1):.1f}%)")
    print(f"  Predicted Normal: {n_scans - n_tumor}")
    print(f"  Results saved   : {out_path}")
    print(f"{'='*60}\n")

    means = df[df["Type"] == "MEAN"][
        ["Patient_ID", "Pred_Label", "Tumor_Prob"]
    ].reset_index(drop=True)
    if not means.empty:
        print("Per-patient summary:")
        print(means.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
