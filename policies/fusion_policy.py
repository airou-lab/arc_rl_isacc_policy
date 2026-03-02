"""
Fusion Features Extractor for Multi-Modal Perception

This module implements a dual-stream network that processes both visual (camera) and
proprioceptive (physics telemetry) inputs into a unified latent representation suitable
for recurrent policy learning.

Architecture:
    Visual Stream: 3-layer CNN (NatureCNN-style) -> 256-dim features
    Physics Stream: Identity passthrough -> 12-dim telemetry
    Fusion: Concatenate + LayerNorm -> 268-dim output

The LayerNorm is crucial for stable LSTM training, as it normalizes the heterogeneous
feature scales between high-level variance visual features and bounded physics values.

Observation Space Contract:
    This extractor expects a gymnasium.spaces.Dict with:
        'image': Box(90, 160, 3) uint8 - RGB camera image
        'vec': Box(12,) float32 - telemetry vector

    This image is processed by a 3-layer CNN:
        Conv1 (8x8, s4): 90x160 -> 21x39, 32 channels
        Conv2 (4x4, s2): 21x39 -> 9x18, 64 channels
        Conv3 (3x3, s1): 9x18 -> 7x16, 64 channels
        Flatten: 64 * 7 * 16 = 7168
        Linear: 7168 -> 256

    The vec passes through unchanged (identity).

    Output: cat(256 CNN, 12 vec) -> LayerNorm -> 268-dim tensor

Camera:
    Intel RealSense: D435i
    RGB native: 1920x1080 (16:9)
    Depth native: 1280x720 (16:9)
    Downsampled to 160x90 preserving aspect ratio.
    Depth NOT used (RGB only) - lane lines are 2D features, and simulated depth transfers
    poorly to real D435i noise profile.

Used by:
    policies/hierarchical_policy.py (HierarchicalPathPlanningPolicy)
    train_policy_ros2.py (passed as features_extractor_class)

Author: Aaron Hamil
Date: 02/17/26
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dual-Stream Fusion Network with Layer Normalization.

    Designed for 160x90 RGB input images with 12-dimensional telemetry vectors.
    The output dimension is computed dynamically as cnn_output_dim + vec_dim, which for
    our standard configuration is 256 + 12 = 268.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 268):
        """
        Initialize the fusion extractor.

        Args:
            observation_space: Dict space with 'image' and 'vec' keys.
                image must be (90, 160, 3) for the CNN math to hold.
                vec can be any 1D shape; we read the dimension dynamically.
        features_dim: Ignored - actual output dim is computed from cnn_output_dim + vec_dim. Kept for SB3 API compatibility.
        """
        # Read vector dimension from the environment's observation space.
        vec_dim = observation_space["vec"].shape[0]
        cnn_output_dim = 256
        total_dim = cnn_output_dim + vec_dim

        # Pass computed total_dim to BaseFeaturesExtractor (overrides features_dim args)
        super().__init__(observation_space, features_dim=total_dim)

        # === Visual Stream ===
        # NatureCNN-style architecture adapted for 160x90 (W X H) resolution.
        # NoteL PyTorch conv2d operates on (B, C, H, W) so input is (B, 3, 90, 160)
        #
        # Dimension trace (H X W):
        #    Input: (B, 3, 90, 160)
        #    Conv1: (B, 32, 21, 39)  kernel=8, stride=4 | floor((90-8)/4)+1=21, floor((160-8)/4)+1=39
        #    Conv2: (B, 64, 9, 18)   kernel=4, stride=2 | floor((21-4)/2)+1=9, floor((39-4)/2)+1=18
        #    Conv3: (B, 64, 7, 16)   kernel=3, stride=1 | floor((9-3)/1)+1=7, floor((18-3)/1)+1=16
        #    Flat: (B, 7168)         64 * 7 * 16 = 7168
        #    Linear: (B, 256)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7168, cnn_output_dim),
            nn.ReLU(),
        )

        # === Physics Stream ===
        # Identity passthrough - the raw telemetry vector is already
        # informative and low-dimensional. No learned transform needed.

        # === Fusion ===
        # LayerNorm over the concatenated [visual, physics] vector.
        # This is critical: CNN features have unbounded variance while
        # physics values are roughly [-1, 1]. Without normalization
        # the LSTM would be dominated by whichever stream has larger
        # magnitude, causing unstable training.
        self.fusion_norm = nn.LayerNorm(total_dim)

    def forward(self, observation: dict) -> torch.Tensor:
        """
        Forward pass: fuse visual and physics features.

        Args:
             observation: dict with 'image' (B, C, H, W) and 'vec' (B, 12)
             Note: Sb3 handles the (H,W,C) -> (C,H,W) transpose and
             uint8 -> float32 [0,1] normalization before this is called.

        Returns:
            (B, 268) fused feature vector, LayerNorm'd.
        """
        # Visual stream: CNN on camera image
        visual_feats = self.cnn(observation["image"])

        # Physics stream: raw telemetry passthrough
        physics_feats = observation["vec"]

        # Concatenate and normalize
        fused = torch.cat([visual_feats, physics_feats], dim=1)

        return self.fusion_norm(fused)
