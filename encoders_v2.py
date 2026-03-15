"""
Improved PA and US Encoders — v2
==================================

Four architectural improvements over encoders.py (v1):

① CBAM (Convolutional Block Attention Module)
   Applied to layer4 feature maps before pooling.  Two sequential steps:
     - Channel attention: sigmoid(MLP(GAP(F)) + MLP(GMP(F))) → which
       feature channels matter for tumour detection
     - Spatial attention: sigmoid(Conv([GAP_c(F), GMP_c(F)])) → where
       in the radial image to focus
   This is far more powerful than the current single Conv2d spatial gate
   in USEncoder.  Applies to both PA and US encoders.

② Multi-scale feature aggregation
   Instead of only using layer4 avgpool output [512], concatenate:
     GAP(layer2) [128]  +  GAP(layer3) [256]  +  GAP(layer4_cbam) [512]
   → input to embedding head = [896]
   Each layer captures different granularity:
     layer2 = low-level edges, speckle texture
     layer3 = mid-level shapes, absorption boundaries
     layer4 = high-level semantic tumour/tissue patterns

③ SupCon (Supervised Contrastive Loss) pre-training
   Replaces SimCLR (self-supervised, no labels) with SupCon (uses labels).
   SupCon pulls ALL tumour scans together and ALL normal scans together
   regardless of patient, while pushing tumour away from normal.
   With 25 patients this is far more label-efficient than CE:
     SimCLR: "make augmented views similar"  (ignores the tumour label)
     SupCon: "make all tumour scans similar" (directly uses the label)
   Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

④ Better modality-specific stems
   PA:  Learnable frequency filter (1D conv along depth dimension) before
        backbone.  PA images encode absorption depth along the radial axis,
        so a depth-wise filter can learn to highlight absorption spikes.
   US:  Full CBAM replaces the simple 2-layer spatial gate.  No separate
        stem needed — CBAM handles both channel and spatial suppression.

Usage (replaces train_encoders.py workflow)
--------------------------------------------
# SupCon pre-training (replaces contrastive mode)
python train_encoders_v2.py --modality PA --mode supcon ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 100 ^
    --normal_stats data/normal_stats.json

# Supervised fine-tune from SupCon checkpoint
python train_encoders_v2.py --modality PA --mode supervised ^
    --csv data/arpam_roi_select_286_all/bscan_dataset.csv ^
    --image_type PAUSradial-pair --epochs 30 ^
    --resume checkpoints/PA_encoder_v2_supcon_best.pth ^
    --normal_stats data/normal_stats.json

# Or drop-in replacement for existing train_fusion_hypernet_v2.py:
pa_enc = PAEncoderV2(in_channels=1, feat_dim=256, pretrained=True)
us_enc = USEncoderV2(in_channels=1, feat_dim=256, pretrained=True)
"""

from __future__ import annotations

import os as _os
import sys as _sys

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, Tuple


# =============================================================================
# Helper: patch first conv for grayscale
# =============================================================================

def _replace_first_conv(backbone: nn.Module, in_channels: int) -> nn.Module:
    old: nn.Conv2d = backbone.conv1
    new = nn.Conv2d(in_channels, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        if in_channels == 1:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        elif in_channels == 3:
            new.weight.copy_(old.weight)
        else:
            nn.init.xavier_uniform_(new.weight)
    backbone.conv1 = new
    return backbone


# =============================================================================
# ① CBAM — Convolutional Block Attention Module
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel attention: re-weights feature channels via GAP + GMP.

    Learns to emphasise feature channels that encode tumour contrast and
    suppress channels dominated by noise or irrelevant anatomy.

    Args:
        in_channels: Number of input feature channels.
        reduction:   Bottleneck reduction ratio (default 16 → smaller MLP).
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg_out = self.mlp(x.mean(dim=[2, 3]))         # GAP → MLP
        max_out = self.mlp(x.amax(dim=[2, 3]))         # GMP → MLP
        scale   = torch.sigmoid(avg_out + max_out)     # [B, C]
        return x * scale.unsqueeze(2).unsqueeze(3)     # broadcast


class SpatialAttention(nn.Module):
    """
    Spatial attention: highlights diagnostically relevant image regions.

    Aggregates channel information via GAP and GMP across the channel dim,
    then learns a spatial map showing where to focus in the radial B-scan.

    Args:
        kernel_size: Convolution kernel for spatial map (default 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_c = x.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        max_c = x.amax(dim=1, keepdim=True)    # [B, 1, H, W]
        cat   = torch.cat([avg_c, max_c], dim=1)
        scale = torch.sigmoid(self.conv(cat))   # [B, 1, H, W]
        return x * scale


class CBAM(nn.Module):
    """
    CBAM: channel attention followed by spatial attention.

    Applied to layer4 feature maps [B, 512, H, W] before pooling.
    Both PA and US encoders use this — for PA it helps focus on
    absorption hotspots, for US it suppresses speckle noise regions.

    Reference: Woo et al., ECCV 2018.
    """

    def __init__(self, in_channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# =============================================================================
# ④ PA-specific: Depth-wise frequency stem
# =============================================================================

class PAFrequencyStem(nn.Module):
    """
    Learnable 1D filter along the radial (depth) axis for PA images.

    PA signals encode optical absorption as a function of depth along the
    ultrasound propagation direction, which is the vertical axis in a
    radial B-scan.  A depth-wise 1D convolution learns to highlight
    absorption spikes at tumour depths while suppressing surface artefacts.

    Architecture:
        Depthwise Conv1D along H (depth) axis per pixel column
        → channel mixing Conv1x1
        → residual add with original input

    Args:
        in_channels: Input image channels (default 1).
        kernel_size: Depth filter size (default 9 — ~18 pixels at 512px).
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        # Depthwise: each channel filtered independently along depth axis
        self.depth_filter = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=in_channels,    # depthwise
            bias=False,
        )
        # Channel mixing after depth filter
        self.channel_mix = nn.Conv2d(in_channels, in_channels,
                                     kernel_size=1, bias=True)
        self.bn  = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

        # Initialise as identity so early training is stable
        nn.init.dirac_(self.depth_filter.weight)
        nn.init.zeros_(self.channel_mix.weight.data
                       - torch.eye(in_channels).unsqueeze(2).unsqueeze(3)
                       if in_channels > 1 else
                       self.channel_mix.weight.data)
        nn.init.zeros_(self.channel_mix.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.channel_mix(self.depth_filter(x))))
        return x + out   # residual


# =============================================================================
# Projection head (unchanged from v1)
# =============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# =============================================================================
# ③ SupCon Loss
# =============================================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls all samples of the same class together and pushes apart samples
    from different classes, regardless of augmentation view.

    For our dataset:
        Positives of sample i = all other tumour scans (label=1)
                              OR all other normal scans (label=0)
        Negatives = all scans with a different label

    This is strictly better than SimCLR for our use case because it uses
    the tumour/normal label — SimCLR treats randomly augmented same-image
    views as positives, which makes no use of the clinical label.

    Args:
        temperature: Logit scaling temperature (default 0.07).
        base_temperature: Reference temperature for loss normalisation.
    """

    def __init__(self, temperature: float = 0.07,
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalised embeddings  [B, feat_dim]
            labels:   Class labels              [B]  (0 or 1)

        Returns:
            Scalar loss.
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Mask out self-similarity
        eye_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim.masked_fill_(eye_mask, float("-inf"))

        # Positive mask: same label, not same sample
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~eye_mask   # [B, B]

        # For each anchor, compute log softmax over all non-self pairs
        log_probs = F.log_softmax(sim, dim=1)   # [B, B]

        # Mean over positives for each anchor
        # Guard: skip anchors with no positive pairs
        n_pos     = pos_mask.sum(dim=1).float().clamp(min=1)
        loss_per  = -(log_probs * pos_mask).sum(dim=1) / n_pos

        # Scale by temperature ratio (standard SupCon normalisation)
        loss = loss_per.mean() * (self.temperature / self.base_temperature)
        return loss


# =============================================================================
# Base encoder v2
# =============================================================================

class _BaseEncoderV2(nn.Module):
    """
    Base class for improved PA and US encoders.

    Architecture:
        [modality stem]  →  ResNet-18 layers 1-4
        →  CBAM(layer4 output)
        →  Multi-scale: GAP(layer2) ‖ GAP(layer3) ‖ GAP(layer4_cbam)
        →  embedding head  →  [B, feat_dim]

    The multi-scale output dimension is 128 + 256 + 512 = 896.
    The embedding head projects this to feat_dim (default 256).
    """

    MULTISCALE_DIM = 128 + 256 + 512   # layer2 + layer3 + layer4 dims

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
        cbam_reduction: int = 16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_dim    = feat_dim

        # Build ResNet-18 backbone
        weights   = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone  = resnet18(weights=weights)
        backbone  = _replace_first_conv(backbone, in_channels)

        # Split backbone into stages for multi-scale tapping
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2   # [B, 128, H/8,  W/8]
        self.layer3 = backbone.layer3   # [B, 256, H/16, W/16]
        self.layer4 = backbone.layer4   # [B, 512, H/32, W/32]

        # ① CBAM on layer4 output
        self.cbam = CBAM(512, reduction=cbam_reduction)

        # ② Multi-scale global average pooling (applied to l2, l3, l4_cbam)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ② Embedding head: 896 → feat_dim
        self.embed_head = nn.Sequential(
            nn.LayerNorm(self.MULTISCALE_DIM),
            nn.Dropout(dropout),
            nn.Linear(self.MULTISCALE_DIM, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Projection head for SupCon pre-training
        self.projection_head: Optional[ProjectionHead] = None

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = False

    def freeze_backbone(self):
        self._freeze_backbone()

    def unfreeze_backbone(self):
        for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = True

    def attach_projection_head(self, hidden_dim: int = 256, out_dim: int = 128):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.projection_head = ProjectionHead(self.feat_dim,
                                               hidden_dim, out_dim).to(device)

    def detach_projection_head(self):
        self.projection_head = None

    def _extract_multiscale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run backbone and return (f2, f3, f4_cbam) feature maps."""
        x  = self.stem(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)    # [B, 128, ...]
        f3 = self.layer3(f2)   # [B, 256, ...]
        f4 = self.layer4(f3)   # [B, 512, ...]
        f4 = self.cbam(f4)     # CBAM attention
        return f2, f3, f4

    def _embed(self, f2, f3, f4) -> torch.Tensor:
        """Pool + concatenate + project to feat_dim."""
        v2 = self.gap(f2).flatten(1)   # [B, 128]
        v3 = self.gap(f3).flatten(1)   # [B, 256]
        v4 = self.gap(f4).flatten(1)   # [B, 512]
        ms = torch.cat([v2, v3, v4], dim=1)   # [B, 896]
        return self.embed_head(ms)             # [B, feat_dim]

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement in subclass")


# =============================================================================
# PA Encoder v2
# =============================================================================

class PAEncoderV2(_BaseEncoderV2):
    """
    Improved PA B-scan encoder.

    Additions over PAEncoder (v1):
      - Depth-wise frequency stem along the radial axis
      - CBAM on layer4 feature maps
      - Multi-scale feature aggregation (layer2 + layer3 + layer4)
      - SupCon-compatible (attach projection head for pre-training)
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
        cbam_reduction: int = 16,
        use_freq_stem: bool = True,
    ):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim,
                         pretrained=pretrained, freeze=freeze,
                         dropout=dropout, cbam_reduction=cbam_reduction)
        self.modality = "PA"

        # Learnable contrast scale (kept from v1 — works well)
        self.contrast_scale = nn.Parameter(
            torch.ones(1, in_channels, 1, 1) * 1.5)
        self.contrast_shift = nn.Parameter(
            torch.ones(1, in_channels, 1, 1) * (-0.2))

        # ④ Depth-wise frequency stem along radial axis
        self.freq_stem = PAFrequencyStem(in_channels) if use_freq_stem else None

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Contrast normalisation
        x = torch.clamp(x * self.contrast_scale + self.contrast_shift,
                        0.0, 1.0)

        # Depth-wise frequency filter
        if self.freq_stem is not None:
            x = self.freq_stem(x)

        # Multi-scale feature extraction
        f2, f3, f4 = self._extract_multiscale(x)
        embedding   = self._embed(f2, f3, f4)

        if return_projection and self.projection_head is not None:
            return embedding, self.projection_head(embedding)
        return embedding


# =============================================================================
# US Encoder v2
# =============================================================================

class USEncoderV2(_BaseEncoderV2):
    """
    Improved US B-scan encoder.

    Additions over USEncoder (v1):
      - Full CBAM (channel + spatial attention) replaces the simple 2-layer
        spatial gate — properly handles structured US speckle suppression
      - Multi-scale feature aggregation (layer2 + layer3 + layer4)
      - SupCon-compatible

    No frequency stem — US speckle is broadband and doesn't have the
    depth-specific spectral structure that PA signals do.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
        cbam_reduction: int = 16,
    ):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim,
                         pretrained=pretrained, freeze=freeze,
                         dropout=dropout, cbam_reduction=cbam_reduction)
        self.modality = "US"

        # Early spatial attention gate for speckle suppression
        # (applied before backbone, lighter than CBAM — used as pre-filter)
        self.pre_attn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention speckle gate
        x = x * self.pre_attn(x)

        # Multi-scale feature extraction (with CBAM on layer4)
        f2, f3, f4 = self._extract_multiscale(x)
        embedding   = self._embed(f2, f3, f4)

        if return_projection and self.projection_head is not None:
            return embedding, self.projection_head(embedding)
        return embedding


# =============================================================================
# Pair wrapper (drop-in replacement for PAUSEncoderPair)
# =============================================================================

class PAUSEncoderPairV2(nn.Module):
    """
    Drop-in replacement for PAUSEncoderPair using v2 encoders.

    Usage in train_fusion_hypernet_v2.py:
        from encoders_v2 import PAEncoderV2, USEncoderV2, PAUSEncoderPairV2
        pa_enc = PAEncoderV2(...)
        us_enc = USEncoderV2(...)
        model.encoder_pair = PAUSEncoderPairV2(pa_enc, us_enc, freeze=True)
    """

    def __init__(
        self,
        pa_encoder: PAEncoderV2,
        us_encoder: USEncoderV2,
        pa_ckpt: Optional[str] = None,
        us_ckpt: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.pa_encoder = pa_encoder
        self.us_encoder = us_encoder

        for enc, path, name in [(pa_encoder, pa_ckpt, "PA"),
                                 (us_encoder, us_ckpt, "US")]:
            if path:
                state = torch.load(path, map_location="cpu",
                                   weights_only=False)
                enc.load_state_dict(state["encoder_state_dict"], strict=False)
                enc.detach_projection_head()
                print(f"✓ Loaded {name} encoder v2 from {path}")

        self.pa_encoder.detach_projection_head()
        self.us_encoder.detach_projection_head()

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def feat_dim(self) -> int:
        return self.pa_encoder.feat_dim

    def forward(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pa_encoder(pa), self.us_encoder(us)


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    print("=== Encoder v2 Self-Test ===\n")

    B, H, W = 4, 512, 512
    feat_dim = 256

    pa = torch.rand(B, 1, H, W)
    us = torch.rand(B, 1, H, W)

    pa_enc = PAEncoderV2(in_channels=1, feat_dim=feat_dim, pretrained=False)
    us_enc = USEncoderV2(in_channels=1, feat_dim=feat_dim, pretrained=False)

    f_pa = pa_enc(pa)
    f_us = us_enc(us)

    print(f"PA embedding: {f_pa.shape}")   # (4, 256)
    print(f"US embedding: {f_us.shape}")   # (4, 256)

    total_pa = sum(p.numel() for p in pa_enc.parameters())
    total_us = sum(p.numel() for p in us_enc.parameters())
    print(f"PA encoder params: {total_pa:,}")
    print(f"US encoder params: {total_us:,}")

    # SupCon pre-training test
    pa_enc.attach_projection_head(hidden_dim=256, out_dim=128)
    emb, proj = pa_enc(pa, return_projection=True)
    labels = torch.tensor([0, 1, 1, 0])
    supcon = SupConLoss(temperature=0.07)
    loss   = supcon(proj, labels)
    print(f"\nSupCon loss (random features): {loss.item():.4f}")
    print("  (should be ~ln(B-1) ≈ {:.2f} for random features)".format(
        __import__("math").log(B - 1)))

    # CBAM test
    cbam = CBAM(512)
    x    = torch.rand(B, 512, 16, 16)
    out  = cbam(x)
    print(f"\nCBAM: {x.shape} → {out.shape}")
    print("All tests passed.")
