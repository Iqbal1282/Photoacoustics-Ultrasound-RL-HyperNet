"""
Inference Script — PA+US RL-HyperNet
======================================

Runs the trained PAUSRLHyperNet on new PA+US B-scan images and produces a
per-scan and per-patient summary CSV, mirroring the structure of the
reference inference file (predict_invivo_generic.py).

Data folder structure expected (mirrors predict_invivo_generic.py):
    root_dir/
        <date_or_cohort>/
            <patient_id>/
                Tumor/          ← or Normal/, or any subfolder name
                    <scan_001>/
                        USradial.tif   (or .tiff / .TIF / .TIFF / .png)
                        PAradial.tif
                    <scan_002>/
                        ...

If your data is a flat folder of scans (no date/patient nesting), use
--flat mode:
    root_dir/
        <scan_001>/
            USradial.tif
            PAradial.tif
        <scan_002>/
            ...

Usage
-----
# Standard nested structure
python predict.py ^
    --model_ckpt  checkpoints/best_model.pth ^
    --pa_ckpt     checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt     checkpoints/US_encoder_supervised_best.pth ^
    --data_root   path/to/invivo/root ^
    --scan_subdir Tumor ^
    --out         results/predictions.csv

# Flat structure (one level of scan folders directly under root)
python predict.py ^
    --model_ckpt  checkpoints/best_model.pth ^
    --pa_ckpt     checkpoints/PA_encoder_supervised_best.pth ^
    --us_ckpt     checkpoints/US_encoder_supervised_best.pth ^
    --data_root   path/to/scans ^
    --flat ^
    --out         results/predictions.csv

# Visualise attention maps alongside predictions
python predict.py ... --visualise --vis_dir results/attention_maps


python predict.py --model_ckpt  checkpoints/best_model.pth --pa_ckpt     checkpoints/PA_encoder_supervised_best.pth --us_ckpt  checkpoints/US_encoder_supervised_best.pth --data_root 10.229.121.108/Workspace/ARPAM/System2/invivo --scan_subdir Tumor --out results/predictions.csv

Notes
-----
- The model is loaded from the RL-HyperNet checkpoint (best_model.pth).
- PA and US encoder checkpoints are loaded separately and injected into
  the model (same as training.py does).
- The preprocessing pipeline is identical to the val/test transform used
  during training (ContrastBoost → ClampUnit → Resize → MinMaxNorm).
- No augmentation is applied at inference.
- For each scan the script reports:
    * Predicted class (0 = Normal / No Tumor, 1 = Tumor)
    * Probability of Tumor class (softmax score)
    * RL latent code norm (proxy for model uncertainty)
    * Per-modality uncertainty (feature variance from encoder)
- Per-patient averages are appended as summary rows (Type = MEAN).
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

# ── optional matplotlib (only needed for --visualise) ─────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; safe on servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False


# =============================================================================
# Preprocessing
# =============================================================================

class InferencePreprocessor:
    """
    Replicates the val/test preprocessing from transforms.build_val_transform:
        1. ContrastBoost : x * 1.5 - 0.2
        2. ClampUnit     : clamp to [0, 1]
        3. Resize        : to (H, W)
        4. MinMaxNorm    : (x - min) / (max - min + 1e-6)

    Accepts raw uint8 grayscale numpy arrays (H, W) as returned by
    cv2.imread(..., cv2.IMREAD_GRAYSCALE).

    Returns:
        torch.Tensor  [1, 1, H, W]  float32, ready to feed to the encoder.
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.H, self.W = image_size
        self._boost  = ContrastBoost()
        self._clamp  = ClampUnit()
        self._norm   = MinMaxNorm()

    def __call__(self, img_gray: np.ndarray) -> torch.Tensor:
        # Resize with OpenCV (faster than torchvision for single images)
        img = cv2.resize(img_gray, (self.W, self.H),
                         interpolation=cv2.INTER_LINEAR)
        # uint8 → float32 in [0, 1]
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        x = self._boost(x)
        x = self._clamp(x)
        x = self._norm(x)
        return x                           # (1, 1, H, W)


# =============================================================================
# Image finder  (mirrors find_images() in predict_invivo_generic.py)
# =============================================================================

_US_STEMS = ["USradial", "usradial", "US_radial", "us_radial",
             "USrect",   "usrect",   "US",         "us"]
_PA_STEMS = ["PAradial", "paradial", "PA_radial", "pa_radial",
             "PArect",   "parect",   "PA",         "pa"]
_EXTENSIONS = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG",
               ".jpg", ".jpeg", ".JPG", ".JPEG"]


def _find_file(folder: Path, stems: List[str]) -> Optional[Path]:
    """Return the first existing file whose stem matches any of `stems`."""
    for stem in stems:
        for ext in _EXTENSIONS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def find_paus_images(scan_folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate USradial and PAradial images inside a scan folder.
    Returns (us_path, pa_path) — either may be None if not found.
    """
    us_path = _find_file(scan_folder, _US_STEMS)
    pa_path = _find_file(scan_folder, _PA_STEMS)
    return us_path, pa_path


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
    """
    Rebuild the PAUSRLHyperNet from its component checkpoints and move to device.

    The model_ckpt (best_model.pth from training.py) stores the full
    model state dict including encoder_pair weights.  The separate
    pa_ckpt / us_ckpt are optional — if provided they override the encoder
    weights stored in model_ckpt (useful if you want to swap encoders).
    """
    # ── Build skeleton ────────────────────────────────────────────────────
    pa_enc = PAEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)
    us_enc = USEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)

    model = PAUSRLHyperNet(
        pa_encoder=pa_enc,
        us_encoder=us_enc,
        feat_dim=feat_dim,
        num_classes=num_classes,
        z_dim=z_dim,
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
    )

    # ── Load full model weights ───────────────────────────────────────────
    ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    # training.py saves {"model_state_dict": ..., "epoch": ..., ...}
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"✓ Loaded RL-HyperNet from  {model_ckpt}  "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── Optional: override encoder weights ───────────────────────────────
    if pa_ckpt:
        enc_ckpt = torch.load(pa_ckpt, map_location="cpu", weights_only=False)
        model.encoder_pair.pa_encoder.load_state_dict(
            enc_ckpt["encoder_state_dict"], strict=False
        )
        model.encoder_pair.pa_encoder.detach_projection_head()
        print(f"✓ Overrode PA encoder from {pa_ckpt}")

    if us_ckpt:
        enc_ckpt = torch.load(us_ckpt, map_location="cpu", weights_only=False)
        model.encoder_pair.us_encoder.load_state_dict(
            enc_ckpt["encoder_state_dict"], strict=False
        )
        model.encoder_pair.us_encoder.detach_projection_head()
        print(f"✓ Overrode US encoder from {us_ckpt}")

    model.to(device)
    model.eval()
    return model


# =============================================================================
# Single-scan inference
# =============================================================================

def predict_scan(
    model: PAUSRLHyperNet,
    us_path: Path,
    pa_path: Path,
    preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
) -> Optional[Dict]:
    """
    Run inference on one PA+US scan pair.

    Returns a dict with prediction results, or None if images can't be read.
    """
    img_us = cv2.imread(str(us_path), cv2.IMREAD_GRAYSCALE)
    img_pa = cv2.imread(str(pa_path), cv2.IMREAD_GRAYSCALE)
    #print("path:", str(us_path), str(pa_path))

    if img_us is None:
        print(f"  [WARN] Cannot read US image: {us_path}")
        return None
    if img_pa is None:
        print(f"  [WARN] Cannot read PA image: {pa_path}")
        return None

    # Preprocess each modality independently
    pa_tensor = preprocessor(img_pa).to(device)   # (1, 1, H, W)
    us_tensor = preprocessor(img_us).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        intermediates = model.get_intermediate(pa_tensor, us_tensor)

    logits = intermediates["logits"]               # (1, num_classes)
    z      = intermediates["z"]                    # (1, z_dim)
    u_pa   = intermediates["u_pa"]                 # (1, 1)
    u_us   = intermediates["u_us"]                 # (1, 1)

    probs      = F.softmax(logits, dim=1)[0]       # (num_classes,)
    pred_class = int(probs.argmax().item())
    tumor_prob = float(probs[1].item()) if len(class_names) == 2 else float(probs.max().item())
    z_norm     = float(z.norm(dim=1).item())       # scalar — RL latent magnitude
    unc_pa     = float(u_pa.item())
    unc_us     = float(u_us.item())

    return {
        "pred_class":       pred_class,
        "pred_label":       class_names[pred_class],
        "tumor_prob":       round(tumor_prob, 4),
        "z_norm":           round(z_norm, 4),
        "uncertainty_PA":   round(unc_pa, 6),
        "uncertainty_US":   round(unc_us, 6),
        # Store raw tensors for optional visualisation
        "_pa_np":  img_pa,
        "_us_np":  img_us,
        "_probs":  probs.cpu().numpy(),
    }


# =============================================================================
# Optional attention-map visualisation
# =============================================================================

def save_attention_vis(
    pa_np: np.ndarray,
    us_np: np.ndarray,
    probs: np.ndarray,
    pred_label: str,
    save_path: Path,
    class_names: List[str],
):
    """Save a side-by-side PA / US image with prediction bar."""
    if not _MATPLOTLIB_OK:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Prediction: {pred_label}  "
                 f"(Tumor prob: {probs[1]:.3f})" if len(class_names) == 2
                 else f"Prediction: {pred_label}",
                 fontsize=13)

    axes[0].imshow(pa_np, cmap="hot")
    axes[0].set_title("PA (photoacoustic)")
    axes[0].axis("off")

    axes[1].imshow(us_np, cmap="gray")
    axes[1].set_title("US (ultrasound)")
    axes[1].axis("off")

    # Probability bar chart
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
    preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    scan_subdir: str = "Tumor",
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Walk  root_dir/<date>/<patient>/<scan_subdir>/<scan>/  structure.
    Mirrors the nested loop in predict_invivo_generic.py.
    """
    results = []

    for date_folder in sorted(root_dir.iterdir()):
        if not date_folder.is_dir():
            continue
        for patient_folder in sorted(date_folder.iterdir()):
            if not patient_folder.is_dir():
                continue

            subdir = patient_folder / scan_subdir
            if not subdir.exists():
                continue

            patient_id = f"{date_folder.name}/{patient_folder.name}"
            print(f"\nPatient: {patient_id}")

            scan_results = _process_scan_folder(
                subdir, patient_id, model, preprocessor, device, class_names,
                visualise, vis_dir,
            )
            results.extend(scan_results)

    return results


def run_flat(
    root_dir: Path,
    model: PAUSRLHyperNet,
    preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Walk  root_dir/<scan>/  — flat structure, no patient nesting.
    Patient_ID is set to the scan folder name.
    """
    results = []
    print(f"\nFlat mode: scanning {root_dir}")

    scan_results = _process_scan_folder(
        root_dir, patient_id="batch", model=model,
        preprocessor=preprocessor, device=device,
        class_names=class_names, visualise=visualise, vis_dir=vis_dir,
    )
    results.extend(scan_results)
    return results


def _process_scan_folder(
    scan_parent: Path,
    patient_id: str,
    model: PAUSRLHyperNet,
    preprocessor: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    visualise: bool,
    vis_dir: Optional[Path],
) -> List[Dict]:
    """
    Iterate over immediate sub-folders of scan_parent, each expected to
    contain one USradial + PAradial image pair.
    """
    rows = []
    per_patient_probs = []

    for scan_folder in sorted(scan_parent.iterdir()):
        if not scan_folder.is_dir():
            continue

        us_path, pa_path = find_paus_images(scan_folder)

        if us_path is None or pa_path is None:
            print(f"  [SKIP] No PA+US pair in {scan_folder.name}")
            continue

        print(f"  Scan: {scan_folder.name}  "
              f"US={us_path.name}  PA={pa_path.name}", end=" ... ")

        res = predict_scan(model, us_path, pa_path, preprocessor,
                           device, class_names)
        if res is None:
            print("FAILED")
            continue

        print(f"{res['pred_label']}  (p_tumor={res['tumor_prob']:.3f})")

        # Optional visualisation
        if visualise and vis_dir is not None:
            vis_path = (vis_dir
                        / patient_id.replace("/", "_")
                        / f"{scan_folder.name}.png")
            save_attention_vis(
                res["_pa_np"], res["_us_np"], res["_probs"],
                res["pred_label"], vis_path, class_names,
            )

        rows.append({
            "Patient_ID":       patient_id,
            "Scan":             scan_folder.name,
            "Pred_Class":       res["pred_class"],
            "Pred_Label":       res["pred_label"],
            "Tumor_Prob":       res["tumor_prob"],
            "Z_Norm":           res["z_norm"],
            "Uncertainty_PA":   res["uncertainty_PA"],
            "Uncertainty_US":   res["uncertainty_US"],
            "US_Path":          str(us_path),
            "PA_Path":          str(pa_path),
            "Type":             "Individual",
        })
        per_patient_probs.append(res["tumor_prob"])

    # Per-patient mean row (mirrors AVERAGE row in predict_invivo_generic.py)
    if per_patient_probs:
        mean_prob  = float(np.mean(per_patient_probs))
        mean_class = int(mean_prob >= 0.5)
        rows.append({
            "Patient_ID":       patient_id,
            "Scan":             "AVERAGE",
            "Pred_Class":       mean_class,
            "Pred_Label":       class_names[mean_class],
            "Tumor_Prob":       round(mean_prob, 4),
            "Z_Norm":           "",
            "Uncertainty_PA":   "",
            "Uncertainty_US":   "",
            "US_Path":          "",
            "PA_Path":          "",
            "Type":             "MEAN",
        })
        print(f"  → Patient mean  Tumor_Prob={mean_prob:.3f}  "
              f"({len(per_patient_probs)} scans)")

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for PA+US RL-HyperNet"
    )

    # ── Required paths ────────────────────────────────────────────────────
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to best_model.pth from training.py")
    p.add_argument("--data_root",  type=str, required=True,
                   help="Root directory containing patient/scan folders")
    p.add_argument("--out",        type=str, default="predictions.csv",
                   help="Output CSV file path (default: predictions.csv)")

    # ── Optional encoder overrides ────────────────────────────────────────
    p.add_argument("--pa_ckpt", type=str, default=None,
                   help="Override PA encoder with a separate checkpoint")
    p.add_argument("--us_ckpt", type=str, default=None,
                   help="Override US encoder with a separate checkpoint")

    # ── Model architecture (must match training) ──────────────────────────
    p.add_argument("--feat_dim",    type=int,   default=256)
    p.add_argument("--z_dim",       type=int,   default=32)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--in_channels", type=int,   default=1)
    p.add_argument("--fusion_type", type=str,   default="cross_attention",
                   choices=["concat", "cross_attention"])
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--image_size",  type=int,   nargs=2, default=[512, 512],
                   metavar=("H", "W"))

    # ── Data structure ────────────────────────────────────────────────────
    p.add_argument("--flat",        action="store_true",
                   help="Flat scan folder structure (no date/patient nesting)")
    p.add_argument("--scan_subdir", type=str,   default="Tumor",
                   help="Sub-folder name inside each patient folder "
                        "(default: Tumor). Ignored with --flat.")

    # ── Class names ───────────────────────────────────────────────────────
    p.add_argument("--class_names", type=str, nargs="+",
                   default=["Normal", "Tumor"],
                   help="Class names in label order (default: Normal Tumor)")

    # ── Output / misc ─────────────────────────────────────────────────────
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--visualise", action="store_true",
                   help="Save side-by-side PA/US + probability plots")
    p.add_argument("--vis_dir",   type=str, default="vis_output",
                   help="Directory for visualisation images (default: vis_output)")

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"PA+US RL-HyperNet Inference")
    print(f"{'='*60}")
    print(f"Device:      {device}")
    print(f"Data root:   {args.data_root}")
    print(f"Output:      {args.out}")
    print(f"Classes:     {args.class_names}")
    print(f"Flat mode:   {args.flat}")
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

    # ── Build preprocessor ────────────────────────────────────────────────
    preprocessor = InferencePreprocessor(image_size=tuple(args.image_size))

    # ── Run inference ─────────────────────────────────────────────────────
    #root_dir = Path(args.data_root)
    from pathlib import Path

    root_path_str = r'\\10.229.121.108\Workspace\ARPAM\System2\invivo'
    root_dir = Path(root_path_str)
    vis_dir  = Path(args.vis_dir) if args.visualise else None

    print(f"\n--- STARTING INFERENCE ---")
    if args.flat:
        results = run_flat(
            root_dir, model, preprocessor, device,
            args.class_names, args.visualise, vis_dir,
        )
    else:
        results = run_nested(
            root_dir, model, preprocessor, device,
            args.class_names, args.scan_subdir,
            args.visualise, vis_dir,
        )

    # ── Save CSV ──────────────────────────────────────────────────────────
    if not results:
        print("\n[WARN] No results generated — check data_root and folder structure.")
        return

    df = pd.DataFrame(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────────
    individual = df[df["Type"] == "Individual"]
    n_scans    = len(individual)
    n_tumor    = (individual["Pred_Class"] == 1).sum()

    print(f"\n{'='*60}")
    print(f"Inference complete.")
    print(f"  Scans processed : {n_scans}")
    print(f"  Predicted Tumor : {n_tumor}  ({100*n_tumor/max(n_scans,1):.1f}%)")
    print(f"  Predicted Normal: {n_scans - n_tumor}")
    print(f"  Results saved   : {out_path}")
    if args.visualise:
        print(f"  Visualisations  : {vis_dir}")
    print(f"{'='*60}\n")

    # Quick per-patient table
    means = df[df["Type"] == "MEAN"][
        ["Patient_ID", "Pred_Label", "Tumor_Prob"]
    ].reset_index(drop=True)
    if not means.empty:
        print("Per-patient summary:")
        print(means.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
