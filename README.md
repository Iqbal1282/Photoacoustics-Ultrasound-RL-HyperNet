# PA+US RL-HyperNet — Multimodal Classification for ARPAM B-scans

This package adapts the original three-modality (MRI + PA + US) RL-HyperNet to
work exclusively with **PA (Photoacoustic)** and **US (Ultrasound)** B-scan
images from the ARPAM dataset, using the data structures from `dataset.py`.

---

## File Overview

| File | Purpose |
|---|---|
| `encoders.py` | `PAEncoder`, `USEncoder`, `PAUSEncoderPair` |
| `train_encoders.py` | Pre-trains encoders (supervised or SimCLR contrastive) |
| `models.py` | `PAUSRLHyperNet` — the full RL+HyperNet model |
| `dataset.py` | `PAUSBScanDataset`, `create_paus_dataloaders` |
| `training.py` | `RLHyperNetTrainer` (REINFORCE), `PPOStyleTrainer`, CLI |
| `config.yaml` | All hyper-parameters in one place |

---

## Recommended Workflow

### Step 1 — Pre-train encoders

Two training modes are available:

**A) Supervised** (fast; needs labels)
```bash
# PA encoder
python train_encoders.py \
    --modality PA --mode supervised \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --image_type PAUSradial-pair \
    --in_channels 1 --feat_dim 256 --epochs 50

# US encoder
python train_encoders.py \
    --modality US --mode supervised \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --image_type PAUSradial-pair \
    --in_channels 1 --feat_dim 256 --epochs 50
```

**B) Contrastive pre-training then fine-tune** (better generalisation)
```bash
# Contrastive pre-train
python train_encoders.py \
    --modality PA --mode contrastive \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --image_type PAUSradial-pair --epochs 100

# Fine-tune with labels
python train_encoders.py \
    --modality PA --mode supervised \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --image_type PAUSradial-pair --epochs 20 \
    --resume checkpoints/PA_encoder_contrastive_best.pth

python train_encoders.py --modality PA --mode supervised  --csv data/arpam_roi_select_286_all/bscan_dataset.csv   --image_type PAUSradial-pair --epochs 20 --resume checkpoints/PA_encoder_contrastive_best.pth

python train_encoders.py --modality PA --mode supervised  --csv data/arpam_roi_select_286_all/bscan_dataset.csv   --image_type PAUSradial-pair --epochs 20 --resume checkpoints/PA_encoder_contrastive_best.pth
```

Encoder checkpoints are saved to `./checkpoints/`.

---

### Step 2 — Train the RL-HyperNet

```bash
# REINFORCE (lighter)
python training.py \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth \
    --us_ckpt checkpoints/US_encoder_supervised_best.pth \
    --image_type PAUSradial-pair \
    --fusion_type cross_attention \
    --epochs 50 --batch_size 16

# PPO (more stable)
python training.py \
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv \
    --pa_ckpt checkpoints/PA_encoder_supervised_best.pth \
    --us_ckpt checkpoints/US_encoder_supervised_best.pth \
    --image_type PAUSradial-pair \
    --fusion_type cross_attention \
    --epochs 50 --use_ppo

python training.py --csv data/arpam_roi_select_286_all/bscan_dataset.csv  --pa_ckpt checkpoints/PA_encoder_supervised_best.pth --us_ckpt checkpoints/US_encoder_supervised_best.pth --image_type PAUSradial-pair --fusion_type cross_attention --epochs 50 --use_ppo

# Quick sanity check (dummy data, no CSV needed)
python training.py --dummy --epochs 3 --batch_size 4
```

```
# Standard nested structure (date/patient/Tumor/scan/)
python predict.py ^
    --model_ckpt checkpoints/best_model.pth ^
    --data_root  path/to/invivo ^
    --scan_subdir Tumor ^
    --out results/predictions.csv

# Flat structure (scans directly under root)
python predict.py ^
    --model_ckpt checkpoints/best_model.pth ^
    --data_root  path/to/scans ^
    --flat --out results/predictions.csv

# With PA+US visualisations saved
python predict.py ... --visualise --vis_dir results/maps
```

---

## Architecture

```
PA B-scan ──► PAEncoder (frozen)
                │  f_pa [B, feat_dim]
                ▼
             CrossModalAttentionFusion ──► fused [B, feat_dim]
                ▲                              │
US B-scan ──► USEncoder (frozen)              │
                │  f_us [B, feat_dim]          │
                                               ▼
State = [f_pa ‖ f_us ‖ u_pa ‖ u_us]   AdaptiveClassifier
          ▼                                    ▲
       RLPolicy ──► z [B, z_dim]              W
                          │                    │
                     HyperNetwork ─────────────┘
                    generates W [B, num_classes, feat_dim]

Logits = einsum("bf,bcf->bc", fused, W)
```

### Key design decisions

- **PAEncoder** — learnable contrast stem (`scale*x + shift`) handles the
  high dynamic range of PA signals; initialised to the `x*1.5 - 0.2` heuristic
  from `dataset.py`.
- **USEncoder** — lightweight spatial attention mask (2-layer CNN) suppresses
  ultrasound speckle before the backbone.
- **CrossModalAttentionFusion** — PA attends to US context and vice-versa via
  `nn.MultiheadAttention`. This is more expressive than simple concatenation
  because PA and US capture complementary physics (optical absorption vs.
  acoustic impedance contrast).
- **RL latent code z** — constrained to `[-1, 1]` via Tanh; state includes
  per-modality uncertainty (feature variance) so the policy can up-weight the
  more reliable modality per-patient.
- **HyperNetwork** — generates patient-specific classifier weights `W` from `z`,
  enabling personalisation without storing per-patient parameters.

---

## Dataset compatibility

The code uses `ArpamBScanDataset` (from your `dataset.py`) with:

| `image_type` | PA channel source | US channel source |
|---|---|---|
| `PAUSradial-pair` | `row["PAradial"]` | `row["USradial"]` |
| `PAUSrect-pair`   | `row["PArect"]`   | `row["USrect"]` |

For `in_channels=1` the encoders receive raw single-channel images.
For `in_channels=3` set `force_3chan=True` in the dataset; the first conv of
each ResNet-18 backbone is widened accordingly.

---

## Requirements

```
torch >= 2.0
torchvision >= 0.15
numpy
pandas
scikit-learn
opencv-python
tqdm
pyyaml
# optional
wandb
```
"# Photoacoustics-Ultrasound-RL-HyperNet" 
