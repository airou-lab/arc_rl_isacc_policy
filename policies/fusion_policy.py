"""
Fusion Features Extractor for Multi-Modal Perception (ResNet-18 backbone).

Architecture:
    Visual Stream:  ResNet-18 (standard ImageNet stem) -> AdaptiveAvgPool -> 512
                    -> Linear(512, 256) + ReLU         -> 256-dim visual features
    Physics Stream: Identity passthrough               -> vec_dim (12 by default)
    Fusion:         Concatenate + LayerNorm            -> 268-dim output

The output dim (256 + 12 = 268) is preserved so the downstream LSTM, the
planning / control heads, and the test contract all stay put. Only the
visual stack changes.

Resolution: 224 x 224 (W x H)
    Square 224x224 matches the canonical ResNet-18 ImageNet input size
    exactly, so the pretrained filter scales, BN running statistics, and
    effective receptive fields all transfer as designed. This is the
    most defensible choice for an RL vision policy that needs to argue
    its visual stream is doing real work; other end-to-end driving
    policies in the literature use the same input shape with pretrained
    backbones, which gives the architecture choice prior art to lean on.

    Spatial trace at 224x224 with the standard stem:
        Input:     (B, 3, 224, 224)
        conv1 7x7 s=2 -> (B, 64, 112, 112)
        maxpool s=2   -> (B, 64, 56, 56)
        layer1        -> (B, 64, 56, 56)
        layer2 s=2    -> (B, 128, 28, 28)
        layer3 s=2    -> (B, 256, 14, 14)
        layer4 s=2    -> (B, 512, 7, 7)
        avgpool       -> (B, 512, 1, 1) -> flatten -> 512

Aspect-ratio note:
    The D435i RGB stream is 16:9 (1920x1080) and the sim camera was
    previously rendered at 160x90 to match. Moving to a square 224x224
    changes the rendered aspect ratio. This is a deliberate choice;
    the deploy-time preprocessing on the real D435i feed must
    apply the same shape transform (e.g. center-crop to square, then
    resize to 224) so train and deploy distributions match. Camera
    intrinsics in agent/stop_line_detector.py (horizontal_aperture,
    focal_length) describe the physical D435i sensor and are not changed
    by the policy-input resolution; they will need a separate review
    when the deploy preprocessing pipeline is finalized.

Why ResNet-18:
    - Stronger inductive bias than the prior 3-conv NatureCNN.
    - ImageNet-pretrained features (edges, textures, color statistics)
      are robust to the sim-to-real photometric gap between Isaac Sim
      and the D435i RGB stream.
    - Easy to share with future heads (e.g. a YOLO-style stop-sign head
      riding on the same backbone, on the post-training-run roadmap).

RL-specific detail (BatchNorm):
    BN is brittle in RL: rollouts run at effective batch size 1 while
    training runs at larger batches, and running stats drift. We pin
    every BN module to eval mode and re-pin on every .train() call.
    Affine params (gamma, beta) still receive gradients; only running
    stats are frozen. This is the standard pattern for pretrained CNN
    backbones in RL.

PVP note:
    ResNet operates on the image only. The 12-element telemetry vector
    flows through as identity passthrough. PVP-zeroed slots (2, 8, 9, 10)
    remain zeroed by the env wrapper upstream. The resolution change
    introduces no new privileged signal.

Observation Space Contract:
    gymnasium.spaces.Dict with
        'image': Box(224, 224, 3) uint8 - RGB camera image
        'vec'  : Box(N,)         float32 - telemetry vector (N=12 standard)

    SB3 transposes (H, W, C) -> (C, H, W) and scales uint8 -> float32 [0, 1]
    before forward() is called. When pretrained=True we additionally apply
    ImageNet mean/std normalization inside this module.

Used by:
    policies/hierarchical_policy.py (HierarchicalPathPlanningPolicy)
    train_policy_ros2.py             (passed as features_extractor_class)

Author: Aaron Hamil
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ImageNet normalization constants. Applied AFTER SB3's uint8 -> float32 / 255
# step, only when running pretrained weights.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Visual feature dim after projection. Kept at 256 so fused dim stays
# 256 + 12 = 268 and downstream code (LSTM input, tests) does not move.
_VISUAL_FEATURES_DIM = 256


def _build_resnet18(pretrained: bool, cifar_stem: bool) -> nn.Module:
    """
    Build a torchvision ResNet-18 with the final FC removed.

    By default (cifar_stem=False) the network keeps the canonical 7x7 s=2
    conv1 + maxpool stem, matching the ImageNet pretraining setup. With
    cifar_stem=True we surgically replace the stem with a small-input
    variant (3x3 s=1, no early maxpool); this is preserved as a flag for
    future ablations but is NOT the default for Option A.

    Args:
        pretrained: Load IMAGENET1K_V1 weights if available.
        cifar_stem: If True, swap the stem for the small-input variant
            (conv1=3x3 s=1, maxpool=Identity) AFTER loading pretrained
            weights. The new conv1 is Kaiming-initialized; layer1-4
            still load pretrained.
    """
    # Lazy import so this file is still importable without torchvision.
    from torchvision.models import resnet18, ResNet18_Weights

    if pretrained:
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained ResNet-18 weights ({exc!r}). "
                "Falling back to random init. This is fine for offline tests "
                "but degrades sim-to-real performance for real training runs.",
                RuntimeWarning,
                stacklevel=2,
            )
            backbone = resnet18(weights=None)
    else:
        backbone = resnet18(weights=None)

    if cifar_stem:
        new_conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            new_conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        backbone.conv1 = new_conv1
        backbone.maxpool = nn.Identity()

    # Drop the 1000-class classifier; keep through avgpool + flatten.
    backbone.fc = nn.Identity()
    return backbone


class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dual-stream fusion network with a ResNet-18 visual backbone.

    Input observation (after SB3 preprocessing):
        'image': (B, 3, H, W) float32 in [0, 1]
        'vec'  : (B, vec_dim) float32

    Output:
        (B, 256 + vec_dim) LayerNorm'd fused features.
        For the standard 12-dim telemetry vector this is (B, 268).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 268,
        backbone: str = "resnet18",
        pretrained: bool = True,
        cifar_stem: bool = False,
        freeze_backbone: bool = False,
        apply_imagenet_normalization: Optional[bool] = None,
    ):
        """
        Args:
            observation_space: Dict space with 'image' (H, W, 3) uint8 and
                'vec' (N,) float32. The vec dim is read dynamically.
            features_dim: Ignored. Output dim is recomputed as 256 + vec_dim.
                Retained for SB3 features_extractor_kwargs compatibility.
            backbone: Visual backbone name. Only "resnet18" is implemented.
            pretrained: Load ImageNet-pretrained weights. Default True.
            cifar_stem: Use the small-input 3x3 s=1 stem with no early
                maxpool. Default False (Option A uses the standard
                ImageNet stem). Reserved as an ablation hook.
            freeze_backbone: Freeze ResNet params so only the projection
                and fusion LayerNorm train. Useful for cheap eval runs.
            apply_imagenet_normalization: If None (default), follows
                `pretrained`. Set explicitly to override for ablations.
        """
        if backbone != "resnet18":
            raise ValueError(
                f"Unsupported visual backbone '{backbone}'. "
                "Only 'resnet18' is implemented; add a builder for new variants."
            )

        vec_dim = observation_space["vec"].shape[0]
        total_dim = _VISUAL_FEATURES_DIM + vec_dim
        super().__init__(observation_space, features_dim=total_dim)

        if apply_imagenet_normalization is None:
            apply_imagenet_normalization = pretrained
        self._apply_imagenet_norm = apply_imagenet_normalization

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std",  std)

        # Visual stream
        self.backbone = _build_resnet18(
            pretrained=pretrained, cifar_stem=cifar_stem,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Project 512 -> 256 so fused tensor stays 268-d.
        self.visual_proj = nn.Sequential(
            nn.Linear(512, _VISUAL_FEATURES_DIM),
            nn.ReLU(),
        )

        # Physics stream
        # Identity passthrough. PVP-zeroed slots remain zeroed upstream.

        # Fusion
        # LayerNorm over [visual, physics] concat: CNN features have
        # unbounded variance, physics values are roughly bounded.
        self.fusion_norm = nn.LayerNorm(total_dim)

        # Pin BN to eval immediately and re-pin on every train() call.
        self._set_bn_eval()

    # BatchNorm management

    def _set_bn_eval(self) -> None:
        """Force every BN submodule into eval mode (frozen running stats)."""
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def train(self, mode: bool = True):
        """
        Override train() so SB3's per-update self.policy.train(True) does
        not re-enable BN running-stat updates. BN affine params (gamma,
        beta) still receive gradients normally.
        """
        super().train(mode)
        self._set_bn_eval()
        return self

    # Forward

    def forward(self, observation: dict) -> torch.Tensor:
        """
        Args:
            observation: dict with
                'image': (B, 3, H, W) float32 in [0, 1] (post-SB3 normalize)
                'vec'  : (B, vec_dim) float32

        Returns:
            (B, 256 + vec_dim) fused, LayerNorm'd feature vector.
        """
        x = observation["image"]
        if self._apply_imagenet_norm:
            x = (x - self.imagenet_mean) / self.imagenet_std

        visual_feats = self.backbone(x)                # (B, 512)
        visual_feats = self.visual_proj(visual_feats)  # (B, 256)

        physics_feats = observation["vec"]             # (B, vec_dim)
        fused = torch.cat([visual_feats, physics_feats], dim=1)
        return self.fusion_norm(fused)
