"""
DAVE-2 CNN Architecture
========================

Faithful reproduction of NVIDIA's end-to-end driving CNN from:
    Bojarski et al., "End to End Learning for Self-Driving Cars"
    arXiv:1604.07316, 2016

Original DAVE-2 architecture (66x200 input, YUV color space):
    Normalization (hardcoded)
    Conv1: 5x5, 24 filters, stride 2  -> 31x98
    Conv2: 5x5, 36 filters, stride 2  -> 14x47
    Conv3: 5x5, 48 filters, stride 2  -> 5x22
    Conv4: 3x3, 64 filters, stride 1  -> 3x20
    Conv5: 3x3, 64 filters, stride 1  -> 1x18
    Flatten: 1152
    FC1: 100
    FC2: 50
    FC3: 10
    Output: 1 (inverse turning radius)

    ~27M connections, ~250K parameters

Our adaptation for ARCPro:
    - Input: 66x200 RGB (cropped from 90x160, then resized)
      OR 90x160 direct (configurable, with recomputed conv dims)
    - Output: [steering] or [steering, throttle] (configurable)
    - Added dropout (0.5) after FC layers for regularization
      (identified as a weakness of the original by multiple reproductions)
    - Kept batch normalization optional (original didn't use it)
    - RGB instead of YUV (simpler pipeline, negligible performance difference
      per subsequent studies)

Why this specific architecture:
    The point isn't to build the best BC model - it's to have a
    well-known, published, reproducible baseline that reviewers recognize
    immediately. DAVE-2 is THE canonical end-to-end driving CNN. When a
    reviewer sees "we compare against DAVE-2 behavioral cloning" they
    instantly know what that means.

Comparison with our hierarchical policy:
    DAVE-2: ~250K params, stateless, no planning, no memory
    Ours: ~3M params, LSTM memory, explicit waypoint planning, reward-shaped

Dependencies:
    - PyTorch
    - No SB3 dependency (this is pure supervised learning)

Author: Aaron Hamil
Date: 03/02/26
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DAVE2Net(nn.Module):
    """
    NVIDIA DAVE-2 end-to-end driving CNN.

    Maps a single RGB image directly to vehicle control commands
    using behavioral cloning (supervised learning from expert demos).

    Architecture:
        5 convolutional layers (feature extraction)
        3 fully connected layers (control)
        Optional dropout for regularization

    This is deliberately simple. No LSTM, no waypoints, no reward
    shaping, no multi-modal fusion. Pure image-to-action mapping.
    """

    def __init__(
        self,
        input_height: int = 66,
        input_width: int = 200,
        num_outputs: int = 1,
        dropout_rate: float = 0.5,
        use_batchnorm: bool = False,
    ):
        """
        Initialize DAVE-2 network.

        Args:
            input_height: Image height after preprocessing. Default 66
                matches the original paper's crop.
            input_width: Image width after preprocessing. Default 200
                matches the original paper.
            num_outputs: Number of control outputs.
                1 = steering only (original DAVE-2)
                2 = steering + throttle (our extension for comparison)
                3 = steering + throttle + brake (full Ackermann)
            dropout_rate: Dropout probability after FC layers.
                Original DAVE-2 had no dropout (noted as a weakness).
                0.5 is the standard addition from reproduction studies.
            use_batchnorm: Whether to add batch normalization after
                conv layers. Not in the original, but can help with
                training stability.
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.num_outputs = num_outputs

        # ═══════════════════════════════════════════
        #  CONVOLUTIONAL FEATURE EXTRACTION
        #  5 layers matching NVIDIA paper exactly
        # ═══════════════════════════════════════════

        conv_layers = []

        # Conv1: 5x5, 24 filters, stride 2
        conv_layers.append(nn.Conv2d(3, 24, kernel_size=5, stride=2))
        if use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(24))
        conv_layers.append(nn.ELU())

        # Conv2: 5x5, 36 filters, stride 2
        conv_layers.append(nn.Conv2d(24, 36, kernel_size=5, stride=2))
        if use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(36))
        conv_layers.append(nn.ELU())

        # Conv3: 5x5, 48 filters, stride 2
        conv_layers.append(nn.Conv2d(36, 48, kernel_size=5, stride=2))
        if use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(48))
        conv_layers.append(nn.ELU())

        # Conv4: 3x3, 64 filters, stride 1
        conv_layers.append(nn.Conv2d(48, 64, kernel_size=3, stride=1))
        if use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(64))
        conv_layers.append(nn.ELU())

        # Conv5: 3x3, 64 filters, stride 1
        conv_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(64))
        conv_layers.append(nn.ELU())

        self.conv = nn.Sequential(*conv_layers)

        # Compute flattened dimension dynamically by running a dummy
        # forward pass through the conv layers. This handles arbitrary
        # input resolutions without manual dimension math.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            conv_out = self.conv(dummy)
            self.flat_dim = conv_out.view(1, -1).shape[1]

        # ═══════════════════════════════════════════
        #  FULLY CONNECTED CONTROLLER
        #  3 FC layers: 100 -> 50 -> 10 -> output
        # ═══════════════════════════════════════════

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 100),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(50, 10),
            nn.ELU(),
        )

        # Final output layer (no activation - continuous control values)
        self.output = nn.Linear(10, num_outputs)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialization for conv and linear layers.

        The original paper didn't specify initialization, but Xavier
        uniform is standard for this type of network and prevents
        vanishing/exploding gradients with ELU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image -> control commands.

        Args:
            x: (B, 3, H, W) RGB image tensor, normalized to [0, 1].
                H, W should match input_height, input_width.

        Returns:
            (B, num_outputs) control commands.
                If num_outputs=1: [steering] in [-1, 1]
                If num_outputs=2: [steering, throttle]
                If num_outputs=3: [steering, throttle, brake]
        """
        # Convolutional feature extraction
        features = self.conv(x)

        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)

        # Fully connected controller
        features = self.fc(features)

        # Output control values
        return self.output(features)

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def architecture_summary(self) -> str:
        """Human-readable architecture summary for logging."""
        total, trainable = self.count_parameters()
        return (
            f"DAVE-2 CNN\n"
            f"  Input: ({self.input_height}, {self.input_width}, 3) RGB\n"
            f"  Conv flatten dim: {self.flat_dim}\n"
            f"  Output: {self.num_outputs} control(s)\n"
            f"  Total parameters: {total:,}\n"
            f"  Trainable parameters: {trainable:,}"
        )


class DAVE2NetWithSpeed(nn.Module):
    """
    Extended DAVE-2 that also takes current speed as input.

    This is a common modification seen in literature (e.g., the modified
    DAVE-II from Purdue's Dynamic-Weighted Simplex work). Speed input
    helps the network learn speed-dependent steering (tighter at low speed,
    gentler at high speed).

    For our ablation study, we test both:
        1. DAVE2Net (image only) - faithful to original paper
        2. DAVE2NetWithSpeed (image + speed) - fair comparison since
           our hierarchical policy also receives telemetry

    Architecture change: speed scalar is concatenated after conv features
    before the FC layers. This adds minimal parameters (~100 extra).
    """

    def __init__(
        self,
        input_height: int = 66,
        input_width: int = 200,
        num_outputs: int = 1,
        dropout_rate: float = 0.5,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.num_outputs = num_outputs

        # Same conv backbone as standard DAVE-2
        conv_layers = []
        conv_layers.extend([nn.Conv2d(3, 24, 5, stride=2), nn.ELU()])
        conv_layers.extend([nn.Conv2d(24, 36, 5, stride=2), nn.ELU()])
        conv_layers.extend([nn.Conv2d(36, 48, 5, stride=2), nn.ELU()])
        conv_layers.extend([nn.Conv2d(48, 64, 3, stride=1), nn.ELU()])
        conv_layers.extend([nn.Conv2d(64, 64, 3, stride=1), nn.ELU()])
        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            conv_out = self.conv(dummy)
            self.flat_dim = conv_out.view(1, -1).shape[1]

        # FC layers with speed concatenated (+1 dim for speed)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim + 1, 100),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(50, 10),
            nn.ELU(),
        )

        self.output = nn.Linear(10, num_outputs)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, image: torch.Tensor, speed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with speed input.

        Args:
            image: (B, 3, H, W) RGB image tensor.
            speed: (B, 1) current vehicle speed in m/s.

        Returns:
            (B, num_outputs) control commands.
        """
        features = self.conv(image)
        features = features.view(features.size(0), -1)

        # Concatenate speed before FC layers
        features = torch.cat([features, speed], dim=-1)

        features = self.fc(features)
        return self.output(features)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
