"""
Driving Dataset for Behavioral Cloning
========================================

PyTorch Dataset for loading expert driving demonstrations.

Data format (collected by collect.py):
    dataset_dir/
    ├── metadata.yaml         # Collection metadata + config
    ├── frames/
    │   ├── frame_000000.png  # RGB camera images
    │   ├── frame_000001.png
    │   └── ...
    └── labels.csv            # Columns: frame_id, steering, throttle, brake, speed

Preprocessing pipeline:
    1. Load RGB image (160x90 from Isaac Sim camera)
    2. Crop top portion (sky/horizon removal) → configurable crop
    3. Resize to DAVE-2 canonical input (66x200) or custom resolution
    4. Normalize to [0, 1] float32
    5. Apply augmentations (flip, brightness, shadow)

Augmentation strategy follows the NVIDIA paper:
    - Horizontal flip + negate steering (doubles effective dataset)
    - Random brightness adjustment (robustness to lighting)
    - Random shadow overlay (robustness to shadows on road)
    - Small random translation + steering offset (recovery training)

The augmentations are crucial for behavioral cloning because the
training distribution is heavily biased toward straight driving.
Without augmentation, the model learns to always predict near-zero
steering and can't recover from deviations.

Dependencies:
    - PyTorch
    - OpenCV (cv2)
    - NumPy
    - pandas (for CSV loading)

Author: Aaron Hamil
Date: 03/02/26
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import csv
import logging

logger = logging.getLogger(__name__)


class DrivingDataset(Dataset):
    """
    Dataset of expert driving demonstrations for behavioral cloning.

    Each sample is an (image, action) pair where:
        - image: preprocessed RGB camera frame
        - action: expert control commands [steering, ...]

    Supports train/val splitting, data augmentation, and
    configurable preprocessing to match DAVE-2 input format.
    """

    def __init__(
        self,
        data_dir: str,
        output_height: int = 66,
        output_width: int = 200,
        crop_top: int = 20,
        crop_bottom: int = 0,
        num_outputs: int = 1,
        augment: bool = False,
        augment_flip_prob: float = 0.5,
        augment_brightness_range: float = 0.2,
        augment_shadow_prob: float = 0.3,
        include_speed: bool = False,
    ):
        """
        Initialize the driving dataset.

        Args:
            data_dir: Path to dataset directory containing frames/ and labels.csv.
            output_height: Target image height after crop + resize.
            output_width: Target image width after crop + resize.
            crop_top: Pixels to crop from top of original image
                (removes sky/horizon which isn't useful for steering).
            crop_bottom: Pixels to crop from bottom (removes car hood if visible).
            num_outputs: Number of action dimensions to load.
                1: steering only (columns: steering)
                2: steering + throttle
                3: steering + throttle + brake
            augment: Whether to apply data augmentation.
            augment_flip_prob: Probability of horizontal flip.
            augment_brightness_range: Max brightness adjustment factor.
            augment_shadow_prob: Probability of adding a random shadow.
            include_speed: Whether to also return speed as a separate tensor
                (for DAVE2NetWithSpeed variant).
        """
        self.data_dir = Path(data_dir)
        self.output_height = output_height
        self.output_width = output_width
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.num_outputs = num_outputs
        self.augment = augment
        self.augment_flip_prob = augment_flip_prob
        self.augment_brightness_range = augment_brightness_range
        self.augment_shadow_prob = augment_shadow_prob
        self.include_speed = include_speed

        # Load labels
        self.samples = self._load_labels()
        logger.info(
            f"Loaded {len(self.samples)} samples from {data_dir} "
            f"(augment={augment}, outputs={num_outputs})"
        )

    def _load_labels(self) -> List[Dict]:
        """
        Load frame paths and corresponding labels from CSV.

        Expected CSV format:
            frame_id,steering,throttle,brake,speed
            frame_000000,0.12,0.3,0.0,1.5
            frame_000001,0.15,0.3,0.0,1.6
            ...

        Returns:
            List of dicts with keys: path, steering, throttle, brake, speed
        """
        labels_path = self.data_dir / "labels.csv"

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                f"Run collect.py first to generate expert demonstrations."
            )

        samples = []
        with open(labels_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_path = self.data_dir / "frames" / f"{row['frame_id']}.png"

                if not frame_path.exists():
                    logger.warning(f"Frame not found, skipping: {frame_path}")
                    continue

                sample = {
                    "path": str(frame_path),
                    "steering": float(row["steering"]),
                    "throttle": float(row.get("throttle", 0.0)),
                    "brake": float(row.get("brake", 0.0)),
                    "speed": float(row.get("speed", 0.0)),
                }
                samples.append(sample)

        if len(samples) == 0:
            raise ValueError(
                f"No valid samples found in {self.data_dir}. "
                f"Check that frames/ directory contains matching images."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Get a single preprocessed (image, action) pair.

        Returns:
            If include_speed=False:
                image: (3, H, W) float32 tensor in [0, 1]
                action: (num_outputs,) float32 tensor
            If include_speed=True:
                image: (3, H, W) float32 tensor in [0, 1]
                action: (num_outputs,) float32 tensor
                speed: (1,) float32 tensor
        """
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["path"])
        if image is None:
            raise RuntimeError(f"Failed to load image: {sample['path']}")

        # BGR -> RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build action vector
        actions = [sample["steering"]]
        if self.num_outputs >= 2:
            actions.append(sample["throttle"])
        if self.num_outputs >= 3:
            actions.append(sample["brake"])

        steering = sample["steering"]

        # Augmentation (before preprocessing to work on original resolution)
        if self.augment:
            image, steering = self._augment(image, steering)
            actions[0] = steering  # Update steering after augmentation

        # Preprocess: crop -> resize -> normalize
        image = self._preprocess(image)

        # Convert to tensors
        # image: (H, W, 3) uint8 -> (3, H, W) float32 [0, 1]
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        action_tensor = torch.tensor(actions, dtype=torch.float32)

        if self.include_speed:
            speed_tensor = torch.tensor([sample["speed"]], dtype=torch.float32)
            return image_tensor, action_tensor, speed_tensor

        return image_tensor, action_tensor

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Crop and resize image to DAVE-2 input format.

        Pipeline:
            Original (90, 160, 3) -> crop top/bottom -> resize (66, 200, 3)

        The crop removes sky/horizon (top) and car hood (bottom) which
        are not useful for steering prediction. This follows the NVIDIA
        paper's preprocessing.
        """
        h, w = image.shape[:2]

        # Crop
        top = self.crop_top
        bottom = h - self.crop_bottom if self.crop_bottom > 0 else h
        image = image[top:bottom, :, :]

        # Resize to target dimensions
        image = cv2.resize(
            image,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA,
        )

        return image

    def _augment(
        self, image: np.ndarray, steering: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply random augmentations.

        Augmentations:
            1. Horizontal flip + negate steering (most important)
            2. Random brightness (lighting robustness)
            3. Random shadow (shadow robustness)

        These are critical for BC because expert data is heavily biased
        toward straight driving. Without augmentation, the model predicts
        near-zero steering always and can't recover from deviations.
        """
        # Horizontal flip (most important augmentation for driving)
        if np.random.random() < self.augment_flip_prob:
            image = np.fliplr(image).copy()
            steering = -steering

        # Random brightness
        if self.augment_brightness_range > 0:
            factor = 1.0 + np.random.uniform(
                -self.augment_brightness_range,
                self.augment_brightness_range,
            )
            # Convert to HSV, adjust V channel, convert back
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random shadow
        if np.random.random() < self.augment_shadow_prob:
            image = self._add_random_shadow(image)

        return image, steering

    @staticmethod
    def _add_random_shadow(image: np.ndarray) -> np.ndarray:
        """
        Add a random shadow polygon to the image.

        Simulates shadows from trees, buildings, overpasses, etc.
        This is important for sim-to-real transfer since real roads
        have inconsistent lighting that simulations often miss.
        """
        h, w = image.shape[:2]

        # Random shadow boundary (vertical line at random x positions)
        x1 = np.random.randint(0, w)
        x2 = np.random.randint(0, w)

        # Create shadow mask
        pts = np.array([
            [x1, 0], [x2, h], [w, h], [w, 0]
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Apply shadow (darken one side)
        shadow_factor = np.random.uniform(0.3, 0.7)
        image = image.copy()
        image[mask == 255] = (
            image[mask == 255] * shadow_factor
        ).astype(np.uint8)

        return image

    @classmethod
    def create_splits(
        cls,
        data_dir: str,
        train_ratio: float = 0.8,
        **kwargs,
    ) -> Tuple["DrivingDataset", "DrivingDataset"]:
        """
        Create train/validation splits from a single dataset directory.

        The split is sequential (not random) to preserve temporal coherence
        and prevent data leakage from consecutive frames being in both sets.

        Args:
            data_dir: Path to dataset directory.
            train_ratio: Fraction of data for training (default 0.8).
            **kwargs: Additional args passed to DrivingDataset.

        Returns:
            (train_dataset, val_dataset)
        """
        # Load full dataset without augmentation to get sample count
        full = cls(data_dir, augment=False, **kwargs)
        n = len(full)
        split = int(n * train_ratio)

        # Create train set (with augmentation) and val set (no augmentation)
        train_ds = cls(data_dir, augment=True, **kwargs)
        val_ds = cls(data_dir, augment=False, **kwargs)

        # Apply splits by trimming the sample lists
        train_ds.samples = train_ds.samples[:split]
        val_ds.samples = val_ds.samples[split:]

        logger.info(
            f"Split: {len(train_ds.samples)} train, "
            f"{len(val_ds.samples)} val ({train_ratio:.0%} split)"
        )

        return train_ds, val_ds
