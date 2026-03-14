"""
PA and US Modality Encoders
============================

Dedicated encoder architectures for Photoacoustic (PA) and Ultrasound (US)
single-channel grayscale images from the ARPAM B-scan dataset.

Key design choices driven by dataset.py:
  - PA and US images are single-channel grayscale (1, H, W), optionally 3-channel
    when force_3chan=True or PAUSradial mode stacks (US, US, PA).
  - The encoders accept 1-channel or 3-channel inputs via `in_channels` arg.
  - Encoders produce a fixed-length embedding vector used downstream by the
    RL-HyperNetwork.
  - Both encoders share the same ResNet-18 backbone architecture but are
    trained independently with modality-specific augmentations.
  - A ProjectionHead is attached for contrastive pre-training (SimCLR-style)
    and is removed before plugging into the RL-HyperNet.

Encoder variants:
  - PAEncoder  : for Photoacoustic images  (radial or rect)
  - USEncoder  : for Ultrasound images     (radial or rect)
  - PAUSEncoderPair : convenience wrapper holding both frozen encoders
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Ensure the package directory is importable from any working directory.
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_first_conv(backbone: nn.Module, in_channels: int) -> nn.Module:
    """
    Replace the first Conv2d of a ResNet so it accepts `in_channels` channels.

    When in_channels == 1 we initialise the new conv by averaging the pretrained
    RGB weights across the channel dimension so the pretrained representation is
    preserved as much as possible.
    """
    old_conv: nn.Conv2d = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels == 1:
            # Average pretrained RGB weights → single-channel weight
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        elif in_channels == 3:
            new_conv.weight.copy_(old_conv.weight)
        else:
            # Fallback: xavier init for unusual channel counts
            nn.init.xavier_uniform_(new_conv.weight)

    backbone.conv1 = new_conv
    return backbone


# ---------------------------------------------------------------------------
# Projection head (used only during contrastive pre-training)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive (SimCLR-style) pre-training.

    Args:
        in_dim:    Encoder output dimension (512 for ResNet-18).
        hidden_dim: Hidden layer size (default: 256).
        out_dim:   Projection space dimension (default: 128).
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256,
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


# ---------------------------------------------------------------------------
# Base modality encoder
# ---------------------------------------------------------------------------

class _BaseModalityEncoder(nn.Module):
    """
    Internal base class for both PA and US encoders.

    Architecture:
        ResNet-18 backbone  →  AdaptiveAvgPool2d(1,1)  →  Flatten
        → optional LayerNorm  →  embedding of size `feat_dim`

    The backbone's final FC layer is replaced by an Identity so the raw
    spatial pooled output (512-d for ResNet-18) flows through a small
    embedding MLP to `feat_dim`.

    Args:
        in_channels:  Number of input image channels (1 or 3).
        feat_dim:     Output embedding dimension.
        pretrained:   Load ImageNet weights for backbone initialisation.
        freeze:       Freeze backbone weights after initialisation.
        dropout:      Dropout probability in the embedding head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_dim = feat_dim

        # ---- Backbone ----
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone = _replace_first_conv(backbone, in_channels)

        # Remove the classification head; keep up to the pooling layer.
        # ResNet-18 ends with: layer4 → avgpool → flatten → fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,  # (B, 512, 1, 1)
        )
        backbone_out_dim = 512  # ResNet-18 fixed

        # ---- Embedding head ----
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(backbone_out_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, feat_dim),
            nn.GELU(),
            nn.LayerNorm(feat_dim),
        )

        # Optional: projection head for contrastive pre-training
        # Attached externally via `attach_projection_head()`.
        self.projection_head: Optional[ProjectionHead] = None

        if freeze:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        """Freeze backbone (call before RL-HyperNet training)."""
        self._freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze backbone (call during fine-tuning)."""
        self._unfreeze_backbone()

    def attach_projection_head(self, hidden_dim: int = 256,
                               out_dim: int = 128):
        """Attach a SimCLR projection head for contrastive pre-training.
        The head is automatically moved to the same device as the encoder."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.projection_head = ProjectionHead(
            self.feat_dim, hidden_dim, out_dim
        ).to(device)

    def detach_projection_head(self):
        """Remove projection head before downstream training."""
        self.projection_head = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                  Image tensor [B, C, H, W].
            return_projection:  If True and projection head is attached,
                                return (embedding, projection) tuple.

        Returns:
            embedding [B, feat_dim]  or  (embedding, projection) tuple.
        """
        features = self.backbone(x)       # (B, 512, 1, 1)
        embedding = self.embed_head(features)  # (B, feat_dim)

        if return_projection and self.projection_head is not None:
            projection = self.projection_head(embedding)
            return embedding, projection

        return embedding


# ---------------------------------------------------------------------------
# PA Encoder
# ---------------------------------------------------------------------------

class PAEncoder(_BaseModalityEncoder):
    """
    Encoder for Photoacoustic (PA) images.

    PA images in the ARPAM dataset are single-channel grayscale B-scans
    (PAradial or PArect from dataset.py).  When the dataset uses
    ``force_3chan=True`` or the ``PAUSradial`` image_type (which stacks
    channels as (US, US, PA)), set ``in_channels=3``.

    The encoder applies a mild contrast-enhancement stem (learnable scale/
    shift) to help the backbone handle the high-dynamic-range PA signal.

    Args:
        in_channels:  1 for raw grayscale PA, 3 for stacked/3-channel input.
        feat_dim:     Output embedding dimension (default 256).
        pretrained:   Use ImageNet-pretrained ResNet-18 backbone.
        freeze:       Freeze backbone after init (for downstream use).
        dropout:      Dropout in embedding head.
        contrast_stem: Whether to add a learnable contrast-adjustment stem.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
        contrast_stem: bool = True,
    ):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim,
                         pretrained=pretrained, freeze=freeze, dropout=dropout)
        self.modality = "PA"

        # Learnable contrast normalisation (scale + shift per channel)
        # Mimics the dataset.py transform: x * 1.5 - 0.2
        if contrast_stem:
            self.contrast_scale = nn.Parameter(
                torch.ones(1, in_channels, 1, 1) * 1.5
            )
            self.contrast_shift = nn.Parameter(
                torch.ones(1, in_channels, 1, 1) * (-0.2)
            )
        else:
            self.contrast_scale = None
            self.contrast_shift = None

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Apply learnable contrast adjustment
        if self.contrast_scale is not None:
            x = x * self.contrast_scale + self.contrast_shift
            x = torch.clamp(x, 0.0, 1.0)

        return super().forward(x, return_projection=return_projection)


# ---------------------------------------------------------------------------
# US Encoder
# ---------------------------------------------------------------------------

class USEncoder(_BaseModalityEncoder):
    """
    Encoder for Ultrasound (US) images.

    US images in the ARPAM dataset are single-channel grayscale B-scans
    (USradial or USrect).  When the dataset uses ``force_3chan=True`` or
    the ``PAUSradial`` image type (channels: US, US, PA), set
    ``in_channels=3``.

    US images tend to have more speckle noise; a spatial attention gate is
    added before the backbone to help the encoder focus on relevant anatomy.

    Args:
        in_channels:     1 for raw grayscale US, 3 for stacked input.
        feat_dim:        Output embedding dimension (default 256).
        pretrained:      Use ImageNet-pretrained ResNet-18 backbone.
        freeze:          Freeze backbone after init.
        dropout:         Dropout in embedding head.
        spatial_attn:    Whether to add a lightweight spatial attention stem.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        dropout: float = 0.2,
        spatial_attn: bool = True,
    ):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim,
                         pretrained=pretrained, freeze=freeze, dropout=dropout)
        self.modality = "US"

        # Lightweight spatial attention stem to suppress speckle
        if spatial_attn:
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.spatial_attn = None

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Apply spatial attention mask
        if self.spatial_attn is not None:
            attn_mask = self.spatial_attn(x)  # (B, 1, H, W)
            x = x * attn_mask

        return super().forward(x, return_projection=return_projection)


# ---------------------------------------------------------------------------
# Convenience wrapper: frozen PA+US encoder pair
# ---------------------------------------------------------------------------

class PAUSEncoderPair(nn.Module):
    """
    Frozen PA and US encoder pair, ready to be plugged into the
    RL-HyperNetwork.

    Loads both encoders from saved checkpoints (produced by
    ``train_encoders.py``) and freezes their weights.

    Args:
        pa_encoder:   PAEncoder instance.
        us_encoder:   USEncoder instance.
        pa_ckpt:      Optional path to PA encoder checkpoint.
        us_ckpt:      Optional path to US encoder checkpoint.
        freeze:       Whether to freeze both encoders.
    """

    def __init__(
        self,
        pa_encoder: PAEncoder,
        us_encoder: USEncoder,
        pa_ckpt: Optional[str] = None,
        us_ckpt: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.pa_encoder = pa_encoder
        self.us_encoder = us_encoder

        if pa_ckpt is not None:
            state = torch.load(pa_ckpt, map_location="cpu")
            self.pa_encoder.load_state_dict(
                state["encoder_state_dict"], strict=False
            )
            self.pa_encoder.detach_projection_head()
            print(f"✓ Loaded PA encoder from {pa_ckpt}")

        if us_ckpt is not None:
            state = torch.load(us_ckpt, map_location="cpu")
            self.us_encoder.load_state_dict(
                state["encoder_state_dict"], strict=False
            )
            self.us_encoder.detach_projection_head()
            print(f"✓ Loaded US encoder from {us_ckpt}")

        # Remove projection heads if any
        self.pa_encoder.detach_projection_head()
        self.us_encoder.detach_projection_head()

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def feat_dim(self) -> int:
        assert self.pa_encoder.feat_dim == self.us_encoder.feat_dim
        return self.pa_encoder.feat_dim

    def forward(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_pa: PA embeddings  [B, feat_dim]
            f_us: US embeddings  [B, feat_dim]
        """
        f_pa = self.pa_encoder(pa)
        f_us = self.us_encoder(us)
        return f_pa, f_us


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PA / US Encoder Self-Test ===\n")

    B, H, W = 4, 512, 512
    feat_dim = 256

    # --- 1-channel (raw grayscale) ---
    pa_enc = PAEncoder(in_channels=1, feat_dim=feat_dim, pretrained=False)
    us_enc = USEncoder(in_channels=1, feat_dim=feat_dim, pretrained=False)

    pa_img = torch.rand(B, 1, H, W)
    us_img = torch.rand(B, 1, H, W)

    f_pa = pa_enc(pa_img)
    f_us = us_enc(us_img)

    print(f"1-channel  PA embedding: {f_pa.shape}")   # (4, 256)
    print(f"1-channel  US embedding: {f_us.shape}")   # (4, 256)

    # --- 3-channel (PAUSradial stacked) ---
    pa_enc3 = PAEncoder(in_channels=3, feat_dim=feat_dim, pretrained=False)
    us_enc3 = USEncoder(in_channels=3, feat_dim=feat_dim, pretrained=False)

    paus_img = torch.rand(B, 3, H, W)   # (US, US, PA) stack from dataset.py

    f_pa3 = pa_enc3(paus_img)
    f_us3 = us_enc3(paus_img)

    print(f"3-channel  PA embedding: {f_pa3.shape}")  # (4, 256)
    print(f"3-channel  US embedding: {f_us3.shape}")  # (4, 256)

    # --- Projection head for contrastive pre-training ---
    pa_enc.attach_projection_head(hidden_dim=256, out_dim=128)
    emb, proj = pa_enc(pa_img, return_projection=True)
    print(f"\nContrastive pre-training:")
    print(f"  PA embedding:  {emb.shape}")
    print(f"  PA projection: {proj.shape}")

    # --- Frozen encoder pair ---
    pair = PAUSEncoderPair(pa_enc, us_enc, freeze=True)
    pair.pa_encoder.detach_projection_head()
    f_pa_pair, f_us_pair = pair(pa_img, us_img)
    print(f"\nFrozen pair  PA: {f_pa_pair.shape}  US: {f_us_pair.shape}")

    trainable = sum(p.numel() for p in pair.parameters() if p.requires_grad)
    print(f"Trainable parameters in frozen pair: {trainable}")
