"""
Inference Script — PA+US Fusion + HyperNet v2 (no RL)
=======================================================

Runs the trained PAUSFusionHyperNetV2 (from train_fusion_hypernet_v2.py).

Key additions over predict_fusion_hypernet.py
----------------------------------------------
  - Loads PAUSFusionHyperNetV2 (gated weighting + context bottleneck +
    residual classification + temperature scaling)
  - MC-Dropout uncertainty estimation per scan (mean ± std probability)
  - Reports per-scan modality gate weights (α_pa, α_us) — tells you
    which modality the model relied on for each prediction
  - W_Norm: magnitude of HyperNet-generated weight matrix

Output CSV columns
-------------------
  Patient_ID, Scan, Pred_Class, Pred_Label,
  Tumor_Prob,       ← mean MC-Dropout probability
  Uncertainty,      ← std across MC-Dropout passes (higher = less confident)
  Gate_PA,          ← PA modality importance weight (0–1)
  Gate_US,          ← US modality importance weight (0–1)
  W_Norm,           ← HyperNet adaptation magnitude
  US_Path, PA_Path, Type

Usage
-----
python predict_fusion_hypernet_v2.py ^
    --model_ckpt checkpoints/best_fusion_hypernet_v2_model.pth ^
    --data_root  \\\\10.229.121.108\\Workspace\\ARPAM\\System2\\invivo ^
    --scan_subdir Tumor ^
    --normal_subdir normal ^
    --out results/invivo_hypernet_v2.csv

# Training data (sanity check)
python predict_fusion_hypernet_v2.py ^
    --model_ckpt checkpoints/best_fusion_hypernet_v2_model.pth ^
    --data_root  data/arpam_roi_select_286_all ^
    --scan_subdir tumor ^
    --normal_subdir normal ^
    --out results/train_data_hypernet_v2.csv

# With visualisations
python predict_fusion_hypernet_v2.py ... --visualise --vis_dir results/v2_maps

# Disable MC-Dropout (faster, single forward pass)
python predict_fusion_hypernet_v2.py ... --mc_samples 1
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

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from encoders import PAEncoder, USEncoder
from encoders_v2 import PAEncoderV2 as PAEncoder, USEncoderV2 as USEncoder
from train_fusion_hypernet_v2 import (
    PAUSFusionHyperNetV2,
    predict_with_uncertainty,
)
from transforms import ContrastBoost, ClampUnit, MinMaxNorm

_THRESHOLD: float = 0.3

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# =============================================================================
# Image finder
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
    return (_find_file(scan_folder, _US_STEMS),
            _find_file(scan_folder, _PA_STEMS))


# =============================================================================
# On-the-fly Normal stats
# =============================================================================

def compute_patient_stats_on_the_fly(
    normal_dir: Path,
    image_size: Tuple[int, int],
) -> Dict[str, Optional[Dict]]:
    H, W = image_size
    pa_arrays: List[np.ndarray] = []
    us_arrays: List[np.ndarray] = []

    scan_folders = [d for d in sorted(normal_dir.iterdir()) if d.is_dir()]
    if not scan_folders:
        scan_folders = [normal_dir]

    for sf in scan_folders:
        us_p, pa_p = find_paus_images(sf)
        for path, arrays in [(pa_p, pa_arrays), (us_p, us_arrays)]:
            if path is not None:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                    arrays.append(img.astype(np.float32) / 255.0)

    result: Dict[str, Optional[Dict]] = {}
    for key, arrays in [("PA", pa_arrays), ("US", us_arrays)]:
        if arrays:
            stack = np.stack(arrays)
            result[key] = {"mean": float(stack.mean()),
                           "std":  float(stack.std()),
                           "n_scans": len(arrays)}
        else:
            result[key] = None
    return result


# =============================================================================
# InferencePreprocessor
# =============================================================================

class InferencePreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (512, 512),
                 modality: str = "PA"):
        self.H, self.W = image_size
        self.modality  = modality
        self._norm_op  = None
        self._boost    = ContrastBoost()
        self._clamp    = ClampUnit()
        self._norm     = MinMaxNorm()

    def set_patient_stats(self, mod_stats: Optional[Dict]):
        from normal_normalisation import NormalTissueNorm, FallbackNorm
        if mod_stats is not None:
            self._norm_op = NormalTissueNorm(mean=mod_stats["mean"],
                                              std=mod_stats["std"])
        else:
            self._norm_op = FallbackNorm()

    def set_patient_from_json(self, pid: str, stats: Dict, fallback: Dict):
        from normal_normalisation import NormalTissueNorm, FallbackNorm
        ps = stats.get(str(pid), {})
        ms = ps.get(self.modality) if isinstance(ps, dict) else None
        if ms is None:
            ms = fallback.get(self.modality)
        if ms is not None:
            self._norm_op = NormalTissueNorm(mean=ms["mean"], std=ms["std"])
        else:
            self._norm_op = FallbackNorm()

    def clear(self):
        self._norm_op = None

    def __call__(self, img_gray: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_gray, (self.W, self.H),
                         interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0)
        if self._norm_op is not None:
            x = self._norm_op(x)
        else:
            x = self._boost(x)
            x = self._clamp(x)
            x = self._norm(x)
        return x


# =============================================================================
# Model loader
# =============================================================================

def load_model(
    model_ckpt: str,
    pa_ckpt: Optional[str] = None,
    us_ckpt: Optional[str] = None,
    feat_dim: int = 256,
    num_classes: int = 2,
    in_channels: int = 1,
    fusion_type: str = "cross_attention",
    ctx_dim: int = 64,
    hypernet_hidden: int = 64,
    gate_hidden: int = 32,
    dropout: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> PAUSFusionHyperNetV2:
    pa_enc = PAEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)
    us_enc = USEncoder(in_channels=in_channels, feat_dim=feat_dim,
                       pretrained=False)

    model = PAUSFusionHyperNetV2(
        pa_encoder=pa_enc, us_encoder=us_enc,
        feat_dim=feat_dim, num_classes=num_classes,
        fusion_type=fusion_type,
        ctx_dim=ctx_dim,
        hypernet_hidden=hypernet_hidden,
        gate_hidden=gate_hidden,
        dropout=dropout,
    )

    ckpt  = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    # Restore calibrated temperature if present
    if "temperature" in ckpt:
        model.temperature.fill_(ckpt["temperature"])
        print(f"✓ Loaded FusionHyperNetV2  "
              f"epoch={ckpt.get('epoch','?')}  "
              f"AUC={ckpt.get('val_auc','?')}  "
              f"T={ckpt['temperature']:.4f}")
    else:
        print(f"✓ Loaded FusionHyperNetV2  "
              f"epoch={ckpt.get('epoch','?')}  "
              f"AUC={ckpt.get('val_auc','?')}")

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
    model: PAUSFusionHyperNetV2,
    us_path: Path,
    pa_path: Path,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    mc_samples: int = 20,
) -> Optional[Dict]:
    img_us = cv2.imread(str(us_path), cv2.IMREAD_GRAYSCALE)
    img_pa = cv2.imread(str(pa_path), cv2.IMREAD_GRAYSCALE)
    if img_us is None:
        print(f"  [WARN] Cannot read US: {us_path}"); return None
    if img_pa is None:
        print(f"  [WARN] Cannot read PA: {pa_path}"); return None

    pa_t = pa_pre(img_pa).to(device)
    us_t = us_pre(img_us).to(device)

    if mc_samples > 1:
        # MC-Dropout: mean ± std over N stochastic passes
        mean_p, std_p = predict_with_uncertainty(
            model, pa_t, us_t, n_samples=mc_samples)
        tumor_prob  = float(mean_p[0, 1].item())
        uncertainty = float(std_p[0, 1].item())
    else:
        with torch.no_grad():
            logits = model(pa_t, us_t)
        probs       = F.softmax(logits, dim=1)[0]
        tumor_prob  = float(probs[1].item())
        uncertainty = 0.0

    pred_class = int(tumor_prob >= _THRESHOLD)

    # Intermediates (single deterministic pass for gate weights / W_norm)
    with torch.no_grad():
        inter = model.get_embeddings(pa_t, us_t)

    gate_pa = float(inter["gates"][0, 0].item())
    gate_us = float(inter["gates"][0, 1].item())
    w_norm  = float(inter["W"].norm(dim=(1, 2)).item())

    return {
        "pred_class":  pred_class,
        "pred_label":  class_names[pred_class],
        "tumor_prob":  round(tumor_prob, 4),
        "uncertainty": round(uncertainty, 4),
        "gate_pa":     round(gate_pa, 4),
        "gate_us":     round(gate_us, 4),
        "w_norm":      round(w_norm, 4),
        "_pa_np":  img_pa,
        "_us_np":  img_us,
        "_probs":  np.array([1 - tumor_prob, tumor_prob]),
    }


# =============================================================================
# Visualisation
# =============================================================================

def save_vis(pa_np, us_np, probs, pred_label, gate_pa, gate_us,
             uncertainty, save_path: Path, class_names: List[str]):
    if not _MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Prediction: {pred_label}  "
        f"(p_tumor={probs[1]:.3f}  unc={uncertainty:.3f})  "
        f"Gate: PA={gate_pa:.2f} US={gate_us:.2f}",
        fontsize=11,
    )
    axes[0].imshow(pa_np, cmap="hot");  axes[0].set_title("PA"); axes[0].axis("off")
    axes[1].imshow(us_np, cmap="gray"); axes[1].set_title("US"); axes[1].axis("off")

    colors = ["steelblue", "steelblue"]
    colors[probs.argmax()] = "tomato"
    axes[2].barh(class_names, probs, color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Probability")
    axes[2].set_title("Class probabilities")

    # Annotate uncertainty
    axes[2].axvline(probs[1], color="gray", linewidth=1, linestyle="--")
    if uncertainty > 0:
        axes[2].barh(class_names[1], uncertainty * 2, left=probs[1] - uncertainty,
                     color="tomato", alpha=0.2, height=0.4)

    for i, v in enumerate(probs):
        axes[2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    if normal_subdir is not None:
        nd = patient_folder / normal_subdir
        if nd.exists():
            ps = compute_patient_stats_on_the_fly(nd, image_size)
            pa_pre.set_patient_stats(ps.get("PA"))
            us_pre.set_patient_stats(ps.get("US"))
            n_pa = (ps["PA"] or {}).get("n_scans", 0)
            n_us = (ps["US"] or {}).get("n_scans", 0)
            return f"on-the-fly  (PA:{n_pa} US:{n_us} scans)"

    if precomputed_stats is not None:
        pa_pre.set_patient_from_json(
            patient_id, precomputed_stats, precomputed_fallback or {})
        us_pre.set_patient_from_json(
            patient_id, precomputed_stats, precomputed_fallback or {})
        found = str(patient_id) in precomputed_stats
        return f"pre-computed JSON ({'found' if found else 'fallback'})"

    pa_pre.clear(); us_pre.clear()
    return "global contrast-stretch"


# =============================================================================
# Batch runners
# =============================================================================

def run_nested(
    root_dir: Path,
    model: PAUSFusionHyperNetV2,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    scan_subdir: str = "tumor",
    normal_subdir: Optional[str] = "normal",
    precomputed_stats: Optional[Dict] = None,
    precomputed_fallback: Optional[Dict] = None,
    image_size: Tuple[int, int] = (512, 512),
    mc_samples: int = 20,
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    results = []

    for level1 in sorted(root_dir.iterdir()):
        if not level1.is_dir():
            continue

        # Shallow layout
        subdir = level1 / scan_subdir
        if subdir.exists():
            pid = level1.name
            nm  = setup_patient_normalisation(
                level1, pa_pre, us_pre, image_size,
                normal_subdir, precomputed_stats, precomputed_fallback, pid)
            print(f"\nPatient: {pid}  [{nm}]")
            results.extend(_process_scan_folder(
                subdir, pid, model, pa_pre, us_pre,
                device, class_names, mc_samples, visualise, vis_dir))
            continue

        # Deep layout
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            subdir = level2 / scan_subdir
            if subdir.exists():
                pid = level2.name
                nm  = setup_patient_normalisation(
                    level2, pa_pre, us_pre, image_size,
                    normal_subdir, precomputed_stats, precomputed_fallback, pid)
                print(f"\nPatient: {pid}  (date:{level1.name})  [{nm}]")
                results.extend(_process_scan_folder(
                    subdir, f"{level1.name}/{level2.name}", model,
                    pa_pre, us_pre, device, class_names,
                    mc_samples, visualise, vis_dir))

    return results


def run_flat(
    root_dir: Path,
    model: PAUSFusionHyperNetV2,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    mc_samples: int = 20,
    visualise: bool = False,
    vis_dir: Optional[Path] = None,
) -> List[Dict]:
    print(f"\nFlat mode: {root_dir}  [global contrast-stretch]")
    pa_pre.clear(); us_pre.clear()
    return _process_scan_folder(root_dir, "batch", model, pa_pre, us_pre,
                                device, class_names, mc_samples,
                                visualise, vis_dir)


def _process_scan_folder(
    scan_parent: Path,
    patient_id: str,
    model: PAUSFusionHyperNetV2,
    pa_pre: InferencePreprocessor,
    us_pre: InferencePreprocessor,
    device: torch.device,
    class_names: List[str],
    mc_samples: int,
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
                           pa_pre, us_pre, device, class_names, mc_samples)
        if res is None:
            print("FAILED"); continue

        print(f"{res['pred_label']}  "
              f"p={res['tumor_prob']:.3f}  "
              f"unc={res['uncertainty']:.3f}  "
              f"PA={res['gate_pa']:.2f}/US={res['gate_us']:.2f}")

        if visualise and vis_dir is not None:
            save_vis(
                res["_pa_np"], res["_us_np"], res["_probs"],
                res["pred_label"], res["gate_pa"], res["gate_us"],
                res["uncertainty"],
                vis_dir / patient_id.replace("/", "_") / f"{scan_folder.name}.png",
                class_names,
            )

        rows.append({
            "Patient_ID":  patient_id,
            "Scan":        scan_folder.name,
            "Pred_Class":  res["pred_class"],
            "Pred_Label":  res["pred_label"],
            "Tumor_Prob":  res["tumor_prob"],
            "Uncertainty": res["uncertainty"],
            "Gate_PA":     res["gate_pa"],
            "Gate_US":     res["gate_us"],
            "W_Norm":      res["w_norm"],
            "US_Path":     str(us_path),
            "PA_Path":     str(pa_path),
            "Type":        "Individual",
        })
        per_patient_probs.append(res["tumor_prob"])

    if per_patient_probs:
        mean_prob  = float(np.mean(per_patient_probs))
        mean_class = int(mean_prob >= _THRESHOLD)
        rows.append({
            "Patient_ID":  patient_id,
            "Scan":        "AVERAGE",
            "Pred_Class":  mean_class,
            "Pred_Label":  class_names[mean_class],
            "Tumor_Prob":  round(mean_prob, 4),
            "Uncertainty": "",
            "Gate_PA":     "",
            "Gate_US":     "",
            "W_Norm":      "",
            "US_Path":     "",
            "PA_Path":     "",
            "Type":        "MEAN",
        })
        print(f"  → Mean  p_tumor={mean_prob:.3f}  ({len(per_patient_probs)} scans)")

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference — PA+US Fusion + HyperNet v2 (no RL)"
    )

    p.add_argument("--model_ckpt",     type=str, required=True,
                   help="Path to best_fusion_hypernet_v2_model.pth")
    p.add_argument("--data_root",      type=str, required=True)
    p.add_argument("--out",            type=str,
                   default="hypernet_v2_predictions.csv")

    p.add_argument("--pa_ckpt",        type=str, default=None)
    p.add_argument("--us_ckpt",        type=str, default=None)

    # Architecture — must match train_fusion_hypernet_v2.py
    p.add_argument("--feat_dim",       type=int,   default=256)
    p.add_argument("--num_classes",    type=int,   default=2)
    p.add_argument("--in_channels",    type=int,   default=1)
    p.add_argument("--fusion_type",    type=str,   default="cross_attention",
                   choices=["concat", "cross_attention"])
    p.add_argument("--ctx_dim",        type=int,   default=64,
                   help="Context bottleneck dim (default: 64)")
    p.add_argument("--hypernet_hidden",type=int,   default=64)
    p.add_argument("--gate_hidden",    type=int,   default=32)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--image_size",     type=int,   nargs=2, default=[512, 512],
                   metavar=("H", "W"))

    # Data structure
    p.add_argument("--flat",           action="store_true")
    p.add_argument("--scan_subdir",    type=str, default="tumor")

    # Normalisation
    norm = p.add_argument_group("Normalisation")
    norm.add_argument("--normal_subdir", type=str, default="normal",
                      help="Normal scan subfolder for on-the-fly stats. "
                           "Pass '' to disable.")
    norm.add_argument("--normal_stats",  type=str, default=None,
                      help="Pre-computed normal_stats.json (retrospective fallback).")

    p.add_argument("--class_names",    type=str, nargs="+",
                   default=["Normal", "Tumor"])
    p.add_argument("--threshold",      type=float, default=0.3,
                   help="Classification threshold (default: 0.3)")
    p.add_argument("--mc_samples",     type=int,   default=20,
                   help="MC-Dropout samples per scan (default: 20). "
                        "Set 1 to disable uncertainty estimation.")
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--visualise",      action="store_true")
    p.add_argument("--vis_dir",        type=str, default="vis_v2_output")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    normal_subdir = args.normal_subdir if args.normal_subdir else None

    print(f"\n{'='*60}")
    print("PA+US Fusion + HyperNet v2 Inference  (no RL)")
    print(f"{'='*60}")
    print(f"Device          : {device}")
    print(f"Data root       : {args.data_root}")
    print(f"Scan subfolder  : {args.scan_subdir}")
    print(f"Fusion type     : {args.fusion_type}")
    print(f"Context dim     : {args.ctx_dim}")
    print(f"Threshold       : {args.threshold}")
    print(f"MC samples      : {args.mc_samples}")
    print(f"Norm priority   :")
    if normal_subdir:
        print(f"  1. On-the-fly from <patient>/{normal_subdir}/  ← primary")
    if args.normal_stats:
        n = 2 if normal_subdir else 1
        print(f"  {n}. Pre-computed JSON: {args.normal_stats}")
    last = 3 if (normal_subdir and args.normal_stats) else \
           2 if (normal_subdir or args.normal_stats) else 1
    print(f"  {last}. Global contrast-stretch  ← last resort")
    print(f"Output          : {args.out}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(
        model_ckpt=args.model_ckpt,
        pa_ckpt=args.pa_ckpt, us_ckpt=args.us_ckpt,
        feat_dim=args.feat_dim, num_classes=args.num_classes,
        in_channels=args.in_channels, fusion_type=args.fusion_type,
        ctx_dim=args.ctx_dim, hypernet_hidden=args.hypernet_hidden,
        gate_hidden=args.gate_hidden, dropout=args.dropout,
        device=device,
    )

    # ── Pre-computed stats ────────────────────────────────────────────────
    precomputed_stats = precomputed_fallback = None
    if args.normal_stats:
        from normal_normalisation import load_stats
        precomputed_stats, precomputed_fallback = load_stats(args.normal_stats)
        print(f"✓ Loaded pre-computed stats for "
              f"{len(precomputed_stats)} patients")

    # ── Preprocessors ─────────────────────────────────────────────────────
    image_size = tuple(args.image_size)
    pa_pre = InferencePreprocessor(image_size, modality="PA")
    us_pre = InferencePreprocessor(image_size, modality="US")

    global _THRESHOLD
    _THRESHOLD = args.threshold

    # ── Run inference ─────────────────────────────────────────────────────
    root_dir = Path(args.data_root)
    vis_dir  = Path(args.vis_dir) if args.visualise else None

    print("\n--- STARTING INFERENCE ---")
    if args.flat:
        results = run_flat(root_dir, model, pa_pre, us_pre, device,
                           args.class_names, args.mc_samples,
                           args.visualise, vis_dir)
    else:
        results = run_nested(
            root_dir=root_dir, model=model,
            pa_pre=pa_pre, us_pre=us_pre, device=device,
            class_names=args.class_names,
            scan_subdir=args.scan_subdir,
            normal_subdir=normal_subdir,
            precomputed_stats=precomputed_stats,
            precomputed_fallback=precomputed_fallback,
            image_size=image_size,
            mc_samples=args.mc_samples,
            visualise=args.visualise, vis_dir=vis_dir,
        )

    if not results:
        print("\n[WARN] No results — check --data_root and folder structure.")
        return

    df       = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    individual = df[df["Type"] == "Individual"]
    n_scans    = len(individual)
    n_tumor    = (individual["Pred_Class"] == 1).sum()

    # Gate weight summary — check if PA/US weighting is non-trivial
    gate_pa_vals = individual["Gate_PA"].astype(float)
    gate_us_vals = individual["Gate_US"].astype(float)

    print(f"\n{'='*60}")
    print(f"Inference complete.")
    print(f"  Scans processed : {n_scans}")
    print(f"  Predicted Tumor : {n_tumor}  ({100*n_tumor/max(n_scans,1):.1f}%)")
    print(f"  Predicted Normal: {n_scans - n_tumor}")
    print(f"  Results saved   : {out_path}")

    if args.mc_samples > 1:
        unc = individual["Uncertainty"].astype(float)
        print(f"  Uncertainty     : mean={unc.mean():.3f}  "
              f"std={unc.std():.3f}  max={unc.max():.3f}")
        high_unc = (unc > 0.15).sum()
        if high_unc > 0:
            print(f"  High-uncertainty scans (>0.15): {high_unc} "
                  f"— review these manually")

    print(f"  Gate PA/US      : PA={gate_pa_vals.mean():.2f}±{gate_pa_vals.std():.2f}  "
          f"US={gate_us_vals.mean():.2f}±{gate_us_vals.std():.2f}")
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
