"""
PA+US Multimodal RL-HyperNet Model
=====================================

Adapted from the original three-modality (MRI + PA + US) model to work
exclusively with PA and US B-scan images from the ARPAM dataset.

Key changes from the original models.py:
  - Removed MRI encoder entirely.
  - FusionBackbone now fuses 2 modalities instead of 3.
  - RLPolicy state dimension reduced accordingly (2×feat_dim + 2 uncertainties).
  - PAUSRLHyperNet replaces MultimodalRLHyperNet as the top-level model.
  - Added cross-modal attention fusion as an alternative to simple concat.
  - compute_reward() kept identical to original.

All encoders (PAEncoder / USEncoder) are expected to be pre-trained and
passed in frozen; only the fusion, policy, hypernet, and classifier are
trained during RL-HyperNet training.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Ensure the package directory is on sys.path so sibling imports work from any
# working directory on Windows (avoids pathlib ntpath recursion bugs).
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from encoders import PAUSEncoderPair, PAEncoder, USEncoder


# ============================================================================
# Fusion backbone variants
# ============================================================================

class ConcatFusion(nn.Module):
    """
    Simple concatenation + linear projection fusion for 2 modalities.

    Args:
        feat_dim:      Dimension of each modality embedding.
        dropout:       Dropout probability.
    """

    def __init__(self, feat_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(feat_dim * 2, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, f_pa: torch.Tensor, f_us: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_pa, f_us], dim=1)   # (B, 2*feat_dim)
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x                               # (B, feat_dim)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.

    PA features attend to US features (and vice-versa) via multi-head
    attention, then the attended representations are summed and projected.

    Rationale: PA captures optical-absorption contrast while US captures
    mechanical/acoustic structure.  Cross-attention lets each modality
    selectively pull complementary information from the other.

    Args:
        feat_dim:   Embedding dimension.
        num_heads:  Number of attention heads.
        dropout:    Dropout in attention and projection.
    """

    def __init__(self, feat_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # PA → US attention
        self.pa_attn_us = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        # US → PA attention
        self.us_attn_pa = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm_pa = nn.LayerNorm(feat_dim)
        self.norm_us = nn.LayerNorm(feat_dim)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, f_pa: torch.Tensor, f_us: torch.Tensor) -> torch.Tensor:
        # Reshape to sequence length 1 for MHA: (B, 1, feat_dim)
        q_pa = f_pa.unsqueeze(1)
        q_us = f_us.unsqueeze(1)

        # PA features attend to US context
        pa_cross, _ = self.pa_attn_us(q_pa, q_us, q_us)
        pa_cross = self.norm_pa(f_pa + pa_cross.squeeze(1))

        # US features attend to PA context
        us_cross, _ = self.us_attn_pa(q_us, q_pa, q_pa)
        us_cross = self.norm_us(f_us + us_cross.squeeze(1))

        # Concatenate and project
        fused = self.proj(torch.cat([pa_cross, us_cross], dim=1))
        return fused   # (B, feat_dim)


# ============================================================================
# HyperNetwork (unchanged from original)
# ============================================================================

class HyperNetwork(nn.Module):
    """
    Generates patient-specific classifier weight matrix from RL latent code z.

    Args:
        z_dim:      Latent code dimension.
        feat_dim:   Fused feature dimension (classifier input).
        num_classes: Number of output classes.
        hidden_dim: Hidden size of the HyperNet MLP.
    """

    def __init__(self, z_dim: int, feat_dim: int, num_classes: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            W: [B, num_classes, feat_dim]
        """
        W = self.net(z)
        return W.view(-1, self.num_classes, self.feat_dim)


# ============================================================================
# Adaptive classifier (unchanged)
# ============================================================================

class AdaptiveClassifier(nn.Module):
    """Uses HyperNet-generated weights for patient-specific classification."""

    def forward(self, features: torch.Tensor,
                W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, feat_dim]
            W:        [B, num_classes, feat_dim]
        Returns:
            logits:   [B, num_classes]
        """
        return torch.einsum("bf,bcf->bc", features, W)


# ============================================================================
# RL Policy
# ============================================================================

class RLPolicy(nn.Module):
    """
    Policy network that maps the current multi-modal state to a latent
    code z controlling the HyperNetwork.

    State = [f_pa | f_us | u_pa | u_us]
           = feat_dim*2 + 2  scalars

    Args:
        input_dim: Dimension of state vector.
        z_dim:     Latent code dimension.
        hidden_dim: MLP hidden size.
    """

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh(),   # Bounded latent code in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)


# ============================================================================
# Top-level PA+US RL-HyperNet
# ============================================================================

class PAUSRLHyperNet(nn.Module):
    """
    PA + US Multimodal Classification with RL Policy + HyperNetwork.

    Pipeline (per forward pass):
        1.  Extract PA & US embeddings from frozen pre-trained encoders.
        2.  Compute per-modality uncertainty (feature variance).
        3.  Build RL state  =  [f_pa ‖ f_us ‖ u_pa ‖ u_us].
        4.  RL policy  →  latent code  z.
        5.  HyperNetwork(z)  →  adaptive weight matrix  W.
        6.  Fuse PA & US embeddings (concat or cross-attention).
        7.  AdaptiveClassifier(fused, W)  →  logits.

    Args:
        pa_encoder:    Pre-trained PAEncoder (will be frozen).
        us_encoder:    Pre-trained USEncoder (will be frozen).
        feat_dim:      Encoder embedding dimension.
        num_classes:   Number of output classes.
        z_dim:         RL latent code dimension.
        fusion_type:   'concat' or 'cross_attention'.
        hidden_dim:    MLP hidden dim for policy and hypernet.
    """

    def __init__(
        self,
        pa_encoder: PAEncoder,
        us_encoder: USEncoder,
        feat_dim: int,
        num_classes: int,
        z_dim: int = 32,
        fusion_type: str = "cross_attention",
        hidden_dim: int = 128,
    ):
        super().__init__()

        # ── Encoders (frozen after pre-training) ─────────────────────────
        self.encoder_pair = PAUSEncoderPair(
            pa_encoder, us_encoder, freeze=True
        )

        # ── Fusion ───────────────────────────────────────────────────────
        if fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(feat_dim)
        else:
            self.fusion = ConcatFusion(feat_dim)

        # ── RL policy: state = 2*feat_dim + 2 uncertainty scalars ────────
        policy_input_dim = feat_dim * 2 + 2
        self.policy = RLPolicy(policy_input_dim, z_dim, hidden_dim)

        # ── HyperNetwork ─────────────────────────────────────────────────
        self.hypernet = HyperNetwork(z_dim, feat_dim, num_classes,
                                     hidden_dim=hidden_dim // 2)

        # ── Adaptive classifier ──────────────────────────────────────────
        self.classifier = AdaptiveClassifier()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.z_dim = z_dim

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_uncertainty(features: torch.Tensor) -> torch.Tensor:
        """
        Per-sample uncertainty = variance across embedding dimensions.

        Args:
            features: [B, feat_dim]
        Returns:
            uncertainty: [B, 1]
        """
        return torch.var(features, dim=1, keepdim=True)

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pa: PA images  [B, C, H, W]
            us: US images  [B, C, H, W]

        Returns:
            logits: [B, num_classes]
            z:      RL latent code [B, z_dim]
        """
        # 1. Encode
        f_pa, f_us = self.encoder_pair(pa, us)

        # 2. Uncertainty
        u_pa = self.compute_uncertainty(f_pa)   # (B, 1)
        u_us = self.compute_uncertainty(f_us)   # (B, 1)

        # 3. RL state
        state = torch.cat([f_pa, f_us, u_pa, u_us], dim=1)

        # 4. Policy → latent code
        z = self.policy(state)

        # 5. HyperNetwork → adaptive weights
        W = self.hypernet(z)

        # 6. Fuse
        fused = self.fusion(f_pa, f_us)

        # 7. Adaptive classification
        logits = self.classifier(fused, W)

        return logits, z

    def get_intermediate(
        self, pa: torch.Tensor, us: torch.Tensor
    ) -> dict:
        """
        Forward pass that also returns intermediate tensors for analysis.

        Returns dict with keys:
            f_pa, f_us, u_pa, u_us, z, W, fused, logits
        """
        f_pa, f_us = self.encoder_pair(pa, us)
        u_pa = self.compute_uncertainty(f_pa)
        u_us = self.compute_uncertainty(f_us)
        state = torch.cat([f_pa, f_us, u_pa, u_us], dim=1)
        z = self.policy(state)
        W = self.hypernet(z)
        fused = self.fusion(f_pa, f_us)
        logits = self.classifier(fused, W)

        return dict(
            f_pa=f_pa, f_us=f_us,
            u_pa=u_pa, u_us=u_us,
            z=z, W=W, fused=fused, logits=logits,
        )


# ============================================================================
# Reward function (compatible with original)
# ============================================================================

def compute_reward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_weight: float = 0.0,
) -> torch.Tensor:
    """
    Compute RL reward.

    Args:
        logits:             [B, num_classes]
        labels:             [B]
        confidence_weight:  Optional confidence-shaping weight.

    Returns:
        reward: scalar tensor in [-1, 1]  (+1 correct, -1 wrong).
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float()
    reward = 2.0 * correct - 1.0   # {0,1} → {-1, +1}

    if confidence_weight > 0:
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        reward = reward + confidence_weight * confidence

    return reward.mean()


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    print("=== PAUSRLHyperNet Self-Test ===\n")

    B, H, W = 4, 512, 512
    feat_dim = 256
    num_classes = 2
    z_dim = 32

    pa_enc = PAEncoder(in_channels=1, feat_dim=feat_dim, pretrained=False)
    us_enc = USEncoder(in_channels=1, feat_dim=feat_dim, pretrained=False)

    for fusion_type in ["concat", "cross_attention"]:
        model = PAUSRLHyperNet(
            pa_encoder=pa_enc,
            us_encoder=us_enc,
            feat_dim=feat_dim,
            num_classes=num_classes,
            z_dim=z_dim,
            fusion_type=fusion_type,
        )

        pa = torch.rand(B, 1, H, W)
        us = torch.rand(B, 1, H, W)
        labels = torch.randint(0, num_classes, (B,))

        logits, z = model(pa, us)
        reward = compute_reward(logits, labels)

        total  = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)

        print(f"Fusion: {fusion_type}")
        print(f"  logits: {logits.shape}  |  z: {z.shape}")
        print(f"  reward: {reward.item():.3f}")
        print(f"  total params:     {total:,}")
        print(f"  trainable params: {trainable:,}")
        print()
