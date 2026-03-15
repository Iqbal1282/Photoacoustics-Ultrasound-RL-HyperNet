"""
Inference Script — PA+US RL-HyperNet
======================================

Normalisation modes (in priority order)
-----------------------------------------
1. ON-THE-FLY  (--normal_subdir, default)         ← RECOMMENDED
   For each patient the script looks for a  <patient>/normal/  subfolder,
   computes μ and σ from those scans on the spot, and z-scores the tumour
   scans against that patient's own baseline.  No pre-computed JSON needed.
   This is the clinically correct approach for new/prospective patients.

2. PRE-COMPUTED JSON  (--normal_stats path/to/normal_stats.json)
   Uses statistics computed from training patients.  Acceptable when
   inference patients are the same as training patients (retrospective).
   Falls back to global average stats for patients not in the JSON.

3. NONE  (omit both flags)
   Global contrast-stretch normalisation  (ContrastBoost → Clamp → MinMax).
   Use only when no Normal scans exist at the inference site.

Data folder structure
----------------------
Shallow  (your data):
    root/
        <patient_id>/          e.g. 258/
            normal/            ← Normal scans used for on-the-fly stats
                <scan_001>/
                    PAradial.tiff
                    USradial.tiff
            tumor/             ← Scans to classify (--scan_subdir tumor)
                <scan_002>/
                    PAradial.tiff
                    USradial.tiff

Deep  (also supported):
    root/
        <date>/
            <patient_id>/
                normal/ ...
                tumor/  ...

Usage
-----
# Recommended — on-the-fly Normal-tissue normalisation
python predict.py ^
    --model_ckpt checkpoints/best_model.pth ^
    --data_root  data/arpam_roi_select_286_all ^
    --scan_subdir tumor ^
    --normal_subdir normal ^
    --out results/predictions.csv

# Pre-computed stats (retrospective / same patients as training)
python predict.py ^
    --model_ckpt   checkpoints/best_model.pth ^
    --data_root    data/arpam_roi_select_286_all ^
    --scan_subdir  tumor ^
    --normal_stats data/normal_stats.json ^
    --out results/predictions.csv

# No Normal scans available
python predict.py ^
    --model_ckpt checkpoints/best_model.pth ^
    --data_root  path/to/invivo ^
    --scan_subdir tumor ^
    --out results/predictions.csv

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

# Global threshold — overwritten by main() from --threshold arg.
# Must match the value used during training (default 0.3).
_THRESHOLD: float = 0.3

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# =============================================================================
# On-the-fly Normal-tissue statistics
# =============================================================================

_US_STEMS = ["USradial", "usradial", "US_radial", "us_radial",
              "USrect",   "usrect",   "US",         "us"]
_PA_STEMS = ["PAradial", "paradial", "PA_radial", "pa_radial",
              "PArect",   "parect",   "PA",         "pa"]
_EXTS     = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG",
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


def compute_patient_stats_on_the_fly(
    normal_dir: Path,
    image_size: Tuple[int, int],
) -> Dict[str, Optional[Dict]]:
    """
    Compute per-modality μ and σ from all Normal scans in normal_dir.

    Walks  normal_dir/<scan>/  and loads every PAradial + USradial image.
    If scan images are directly in normal_dir (no sub-folders) those are
    loaded directly.

    Returns:
        { "PA": {"mean": float, "std": float, "n_scans": int} or None,
          "US": {"mean": float, "std": float, "n_scans": int} or None }
    """
    H, W = image_size
    pa_arrays: List[np.ndarray] = []
    us_arrays: List[np.ndarray] = []

    scan_folders = [d for d in sorted(normal_dir.iterdir()) if d.is_dir()]
    if not scan_folders:
        scan_folders = [normal_dir]   # images directly in normal_dir

    for scan_folder in scan_folders:
        us_path, pa_path = find_paus_images(scan_folder)
        for path, arrays in [(pa_path, pa_arrays), (us_path, us_arrays)]:
            if path is not None:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                    arrays.append(img.astype(np.float32) / 255.0)

    result: Dict[str, Optional[Dict]] = {}
    for key, arrays in [("PA", pa_arrays), ("US", us_arrays)]:
        if arrays:
            stack = np.stack(arrays)
            result[key] = {
                "mean":    float(stack.mean()),
                "std":     float(stack.std()),
                "n_scans": len(arrays),
            }
        else:
            result[key] = None

    return result


# =============================================================================
# InferencePreprocessor
# =============================================================================

class InferencePreprocessor:
    """
    Preprocesses a single grayscale image for one modality.

    Normalisation is resolved in this priority order each time
    set_patient_stats() or set_patient_from_json() is called:

        1. On-the-fly stats  (set_patient_stats called with live computation)
        2. Pre-computed JSON  (set_patient_from_json)
        3. FallbackNorm / global contrast-stretch  (if neither available)

    Args:
        image_size:  (H, W) spatial resolution.
        modality:    'PA' or 'US'.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        modality: str = "PA",
    ):
        self.H, self.W   = image_size
        self.modality    = modality
        self._norm_op    = None   # set per-patient

        self._boost = ContrastBoost()
        self._clamp = ClampUnit()
        self._norm  = MinMaxNorm()

    def set_patient_stats(self, mod_stats: Optional[Dict]):
        """
        Set normalisation from a dict with keys 'mean' and 'std'.
        Pass None to fall back to FallbackNorm.
        """
        from normal_normalisation import NormalTissueNorm, FallbackNorm
        if mod_stats is not None:
            self._norm_op = NormalTissueNorm(
                mean=mod_stats["mean"], std=mod_stats["std"]
            )
        else:
            self._norm_op = FallbackNorm()

    def set_patient_from_json(self, pid: str, stats: Dict, fallback: Dict):
        """
        Set normalisation from a pre-computed stats dict (JSON file).
        Falls back to global average when pid not found.
        """
        from normal_normalisation import NormalTissueNorm, FallbackNorm
        pid_stats = stats.get(str(pid), {})
        mod_stats = pid_stats.get(self.modality) if isinstance(pid_stats, dict) else None
        if mod_stats is None:
            mod_stats = fallback.get(self.modality)
        if mod_stats is not None:
            self._norm_op = NormalTissueNorm(
                mean=mod_stats["mean"], std=mod_stats["std"]
            )
        else:
            self._norm_op = FallbackNorm()

    def clear(self):
        """Reset to standard pipeline (no patient-specific normalisation)."""
        self._norm_op = None

    def __call__(self, img_gray: np.ndarray) -> torch.Tensor:
        """
        Args:
            img_gray: uint8 grayscale array (H, W).
        Returns:
            Tensor [1, 1, H, W] float32 in [0, 1].
        """
        img = cv2.resize(img_gray, (self.W, self.H),
                         interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        if self._norm_op is not None:
            x = self._norm_op(x)
        else:
            x = self._boost(x)
            x = self._clamp(x)
            x = self._norm(x)

        return x   # (1, 1, H, W)


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
    pa_enc = PAEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)
    us_enc = USEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)
    model  = PAUSRLHyperNet(
        pa_encoder=pa_enc, us_encoder=us_enc,
        feat_dim=feat_dim, num_classes=num_classes,
        z_dim=z_dim, fusion_type=fusion_type, hidden_dim=hidden_dim,
    )
    ckpt  = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"✓ Loaded RL-HyperNet  epoch={ckpt.get('epoch','?')}  "
          f"AUC={ckpt.get('val_auc', '?')}")

    for enc, path, name in [(model.encoder_pair.pa_encoder, pa_ckpt, "PA"),
                             (model.encoder_pair.us_encoder, us_ckpt, "US")]:
        if path:
            ec = torch.load(path, map_location="cpu", weights_only=False)
            enc.load_state_dict(ec["encoder_state_dict"], strict=False)
            enc.detach_projection_head()
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
    img_us = cv2.imread(str(us_path), cv2.IMREAD_GRAYSCALE)
    img_pa = cv2.imread(str(pa_path), cv2.IMREAD_GRAYSCALE)
    if img_us is None:
        print(f"  [WARN] Cannot read US: {us_path}"); return None
    if img_pa is None:
        print(f"  [WARN] Cannot read PA: {pa_path}"); return None

    pa_t = pa_preprocessor(img_pa).to(device)
    us_t = us_preprocessor(img_us).to(device)

    with torch.no_grad():
        inter = model.get_intermediate(pa_t, us_t)

    probs      = F.softmax(inter["logits"], dim=1)[0]
    tumor_prob = float(probs[1].item()) if len(class_names) == 2 \
                 else float(probs.max().item())
    pred_class = int(tumor_prob >= _THRESHOLD)

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
        f"Prediction: {pred_label}"
        + (f"  (Tumor prob: {probs[1]:.3f})" if len(class_names) == 2 else ""),
        fontsize=13,
    )
    axes[0].imshow(pa_np, cmap="hot");  axes[0].set_title("PA"); axes[0].axis("off")
    axes[1].imshow(us_np, cmap="gray"); axes[1].set_title("US"); axes[1].axis("off")
    colors = ["steelblue"] * len(class_names)
    colors[probs.argmax()] = "tomato"
    axes[2].barh(class_names, probs, color=colors)
    axes[2].set_xlim(0, 1); axes[2].set_xlabel("Probability")
    axes[2].set_title("Class probabilities")
    for i, v in enumerate(probs):
        axes[2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# Per-patient normalisation setup
# =============================================================================

def setup_patient_normalisation(
    patient_folder: Path,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    image_size: Tuple[int, int],
    normal_subdir: Optional[str],
    precomputed_stats: Optional[Dict],
    precomputed_fallback: Optional[Dict],
    patient_id: str,
) -> str:
    """
    Resolve normalisation for one patient and update both preprocessors.

    Priority:
        1. On-the-fly  — Normal folder exists at inference site
        2. Pre-computed JSON  — pid found in precomputed_stats
        3. FallbackNorm  — global average or ContrastBoost

    Returns a short string describing which mode was used (for logging).
    """

    # ── 1. On-the-fly from Normal folder ─────────────────────────────────
    if normal_subdir is not None:
        normal_dir = patient_folder / normal_subdir
        if normal_dir.exists():
            patient_stats = compute_patient_stats_on_the_fly(
                normal_dir, image_size
            )
            pa_pre.set_patient_stats(patient_stats.get("PA"))
            us_pre.set_patient_stats(patient_stats.get("US"))
            n_pa = (patient_stats["PA"] or {}).get("n_scans", 0)
            n_us = (patient_stats["US"] or {}).get("n_scans", 0)
            return f"on-the-fly  (PA: {n_pa} scans, US: {n_us} scans)"

    # ── 2. Pre-computed JSON ──────────────────────────────────────────────
    if precomputed_stats is not None:
        pa_pre.set_patient_from_json(
            patient_id, precomputed_stats, precomputed_fallback or {}
        )
        us_pre.set_patient_from_json(
            patient_id, precomputed_stats, precomputed_fallback or {}
        )
        in_json = str(patient_id) in precomputed_stats
        return f"pre-computed JSON  ({'found' if in_json else 'fallback'})"

    # ── 3. No normalisation stats at all ─────────────────────────────────
    pa_pre.clear()
    us_pre.clear()
    return "global contrast-stretch"


# =============================================================================
# Batch runners
# =============================================================================

def run_nested(
    root_dir: Path,
    model: PAUSRLHyperNet,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    scan_subdir: str = "tumor",
    normal_subdir: Optional[str] = "normal",
    precomputed_stats: Optional[Dict] = None,
    precomputed_fallback: Optional[Dict] = None,
    image_size: Tuple[int, int] = (512, 512),
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Walk the data root, resolve per-patient normalisation, then predict.

    Handles both layouts:
      Shallow:  <root>/<patient_id>/<scan_subdir>/<scan>/
      Deep:     <root>/<date>/<patient_id>/<scan_subdir>/<scan>/
    """
    results = []

    for level1 in sorted(root_dir.iterdir()):
        if not level1.is_dir():
            continue

        # ── Shallow: <root>/<patient_id>/<scan_subdir>/ ───────────────────
        subdir = level1 / scan_subdir
        if subdir.exists():
            patient_id     = level1.name
            patient_folder = level1
            norm_mode = setup_patient_normalisation(
                patient_folder, pa_pre, us_pre, image_size,
                normal_subdir, precomputed_stats,
                precomputed_fallback, patient_id,
            )
            print(f"\nPatient: {patient_id}  [{norm_mode}]")
            results.extend(_process_scan_folder(
                subdir, patient_id, model, pa_pre, us_pre,
                device, class_names, visualise, vis_dir,
            ))
            continue

        # ── Deep: <root>/<date>/<patient_id>/<scan_subdir>/ ───────────────
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            subdir = level2 / scan_subdir
            if subdir.exists():
                patient_id     = level2.name
                patient_folder = level2
                norm_mode = setup_patient_normalisation(
                    patient_folder, pa_pre, us_pre, image_size,
                    normal_subdir, precomputed_stats,
                    precomputed_fallback, patient_id,
                )
                print(f"\nPatient: {patient_id}  (date: {level1.name})  "
                      f"[{norm_mode}]")
                results.extend(_process_scan_folder(
                    subdir, f"{level1.name}/{level2.name}", model,
                    pa_pre, us_pre, device, class_names, visualise, vis_dir,
                ))

    return results


def run_flat(
    root_dir: Path,
    model: PAUSRLHyperNet,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    """Flat structure — scan folders directly under root_dir."""
    print(f"\nFlat mode: {root_dir}  [global contrast-stretch]")
    pa_pre.clear(); us_pre.clear()
    return _process_scan_folder(
        root_dir, "batch", model, pa_pre, us_pre,
        device, class_names, visualise, vis_dir,
    )


def _process_scan_folder(
    scan_parent: Path,
    patient_id: str,
    model: PAUSRLHyperNet,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
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
                           pa_pre, us_pre, device, class_names)
        if res is None:
            print("FAILED"); continue

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
        print(f"  → Mean  Tumor_Prob={mean_prob:.3f}  "
              f"({len(per_patient_probs)} scans)")

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference — PA+US RL-HyperNet")

    p.add_argument("--model_ckpt",  type=str, required=True)
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--out",         type=str, default="predictions.csv")

    p.add_argument("--pa_ckpt",     type=str, default=None)
    p.add_argument("--us_ckpt",     type=str, default=None)

    p.add_argument("--feat_dim",    type=int,   default=256)
    p.add_argument("--z_dim",       type=int,   default=32)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--in_channels", type=int,   default=1)
    p.add_argument("--fusion_type", type=str,   default="cross_attention",
                   choices=["concat", "cross_attention"])
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--image_size",  type=int,   nargs=2, default=[512, 512],
                   metavar=("H", "W"))

    p.add_argument("--flat",         action="store_true",
                   help="Flat scan structure — all scans directly under data_root")
    p.add_argument("--scan_subdir",  type=str, default="tumor",
                   help="Subfolder containing scans to classify (default: tumor)")

    # ── Normalisation options (mutually exclusive; on-the-fly takes priority) ──
    norm_grp = p.add_argument_group(
        "Normalisation",
        "On-the-fly is recommended when Normal scans exist at the inference "
        "site. Pre-computed JSON is for retrospective use. Omit both for "
        "global contrast-stretch only."
    )
    norm_grp.add_argument(
        "--normal_subdir", type=str, default="normal",
        help="Subfolder name containing Normal scans for on-the-fly stats "
             "(default: normal). Set to empty string '' to disable on-the-fly."
    )
    norm_grp.add_argument(
        "--normal_stats", type=str, default=None,
        help="Path to normal_stats.json (pre-computed, retrospective use). "
             "Used only when on-the-fly Normal folder is not found."
    )

    p.add_argument("--class_names", type=str, nargs="+",
                   default=["Normal", "Tumor"])
    p.add_argument("--threshold",   type=float, default=0.3,
                   help="Tumour probability threshold (default: 0.3). "
                        "Must match the value used during training.")
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--visualise",   action="store_true")
    p.add_argument("--vis_dir",     type=str, default="vis_output")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Treat empty string as "disabled"
    normal_subdir = args.normal_subdir if args.normal_subdir else None

    print(f"\n{'='*60}")
    print("PA+US RL-HyperNet Inference")
    print(f"{'='*60}")
    print(f"Device         : {device}")
    print(f"Data root      : {args.data_root}")
    print(f"Scan subfolder : {args.scan_subdir}")
    print(f"Norm priority  :")
    if normal_subdir:
        print(f"  1. On-the-fly from <patient>/{normal_subdir}/  ← primary")
    if args.normal_stats:
        n = 2 if normal_subdir else 1
        print(f"  {n}. Pre-computed JSON: {args.normal_stats}")
    print(f"  {'3' if normal_subdir and args.normal_stats else ('2' if normal_subdir or args.normal_stats else '1')}. Global contrast-stretch  ← last resort")
    print(f"Output         : {args.out}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(
        model_ckpt=args.model_ckpt,
        pa_ckpt=args.pa_ckpt, us_ckpt=args.us_ckpt,
        feat_dim=args.feat_dim, z_dim=args.z_dim,
        num_classes=args.num_classes, in_channels=args.in_channels,
        fusion_type=args.fusion_type, hidden_dim=args.hidden_dim,
        device=device,
    )

    # ── Load pre-computed stats (optional) ────────────────────────────────
    precomputed_stats = precomputed_fallback = None
    if args.normal_stats:
        from normal_normalisation import load_stats
        precomputed_stats, precomputed_fallback = load_stats(args.normal_stats)
        print(f"✓ Loaded pre-computed stats for "
              f"{len(precomputed_stats)} patients (fallback)")

    # ── Build preprocessors ───────────────────────────────────────────────
    image_size = tuple(args.image_size)
    pa_pre = InferencePreprocessor(image_size, modality="PA")
    us_pre = InferencePreprocessor(image_size, modality="US")

    # ── Run inference ─────────────────────────────────────────────────────
    root_dir = Path(args.data_root)
    vis_dir  = Path(args.vis_dir) if args.visualise else None

    global _THRESHOLD
    _THRESHOLD = args.threshold
    print(f"Classification threshold : {args.threshold}")
    print("\n--- STARTING INFERENCE ---")
    if args.flat:
        results = run_flat(root_dir, model, pa_pre, us_pre,
                           device, args.class_names, args.visualise, vis_dir)
    else:
        results = run_nested(
            root_dir=root_dir,
            model=model,
            pa_pre=pa_pre, us_pre=us_pre,
            device=device,
            class_names=args.class_names,
            scan_subdir=args.scan_subdir,
            normal_subdir=normal_subdir,
            precomputed_stats=precomputed_stats,
            precomputed_fallback=precomputed_fallback,
            image_size=image_size,
            visualise=args.visualise,
            vis_dir=vis_dir,
        )

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
