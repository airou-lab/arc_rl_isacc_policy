"""
DAVE-2 Behavioral Cloning Training
====================================

Standard supervised learning loop for training DAVE-2 on expert
demonstrations. This is the baseline training pipeline for our
ablation study.

Training procedure:
    1. Load expert dataset (image + action pairs from collect.py)
    2. Split into train/validation (80/20, sequential)
    3. Train with MSE loss on steering (+ optional throttle/brake)
    4. Evaluate on validation set each epoch
    5. Save best model + full config for reproduction

Loss function:
    MSE between predicted and expert steering/throttle values.
    This is the standard loss for behavioral cloning. More sophisticated
    alternatives (e.g., L1, Huber, weighted MSE for turns) exist but
    we use vanilla MSE to keep the baseline as canonical as possible.

Metrics tracked:
    - MSE on steering (primary metric for paper)
    - MAE on steering (more interpretable)
    - MSE on throttle (if predicting throttle)
    - Validation metrics per epoch
    - Learning curves for paper figures

Usage:
    # Train from collected data:
    python -m baselines.dave2.train \\
        --data data/expert_001 \\
        --name dave2_baseline_001 \\
        --epochs 50

    # Train with config file:
    python -m baselines.dave2.train \\
        --config experiments/dave2_baseline_001/config.yaml

Dependencies:
    - PyTorch
    - config/ (ExperimentConfig)
    - baselines/dave2/model.py (DAVE2Net)
    - baselines/dave2/dataset.py (DrivingDataset)

Author: Aaron Hamil
Date: 03/02/26
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.dave2.model import DAVE2Net, DAVE2NetWithSpeed
from baselines.dave2.dataset import DrivingDataset
from config.experiment import ExperimentConfig, BaselineConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

class DAVE2Trainer:
    """
    Trainer for DAVE-2 behavioral cloning.

    Handles the complete training lifecycle:
        - Model creation from config
        - Dataset loading and splitting
        - Training loop with validation
        - Checkpointing and logging
        - Final evaluation metrics for paper
    """

    def __init__(
        self,
        config: ExperimentConfig,
        data_dir: str,
    ):
        """
        Args:
            config: Full experiment configuration.
            data_dir: Path to expert demonstration dataset.
        """
        self.config = config
        self.bc = config.baseline
        self.data_dir = data_dir

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.training.device)

        logger.info(f"Using device: {self.device}")

        # Determine number of outputs
        self.num_outputs = 1  # steering
        if self.bc.predict_throttle:
            self.num_outputs = 2
        if self.bc.predict_brake:
            self.num_outputs = 3

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Log architecture
        total, trainable = self.model.count_parameters()
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"  Parameters: {total:,} total, {trainable:,} trainable")
        logger.info(f"  Outputs: {self.num_outputs}")

        # Create datasets
        self.train_dataset, self.val_dataset = self._create_datasets()

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.bc.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.bc.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        logger.info(
            f"Data: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val samples"
        )

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.bc.learning_rate,
            weight_decay=self.bc.weight_decay,
        )

        # Learning rate scheduler: reduce on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        self.criterion = nn.MSELoss()

        # Training state
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

    def _create_model(self) -> nn.Module:
        """Create the appropriate DAVE-2 model variant."""
        return DAVE2Net(
            input_height=self.bc.input_height,
            input_width=self.bc.input_width,
            num_outputs=self.num_outputs,
            dropout_rate=0.5,
            use_batchnorm=False,
        )

    def _create_datasets(self) -> Tuple[DrivingDataset, DrivingDataset]:
        """Create train/val dataset splits."""
        return DrivingDataset.create_splits(
            data_dir=self.data_dir,
            train_ratio=self.bc.train_val_split,
            output_height=self.bc.input_height,
            output_width=self.bc.input_width,
            num_outputs=self.num_outputs,
        )

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Dict of final metrics (best_val_loss, final_train_loss, etc.)
        """
        logger.info(f"Starting training: {self.bc.epochs} epochs")
        logger.info(f"Config: {self.config.summary()}")

        # Save config alongside model
        self.config.save(str(self.output_dir / "config.yaml"))

        start_time = time.time()

        for epoch in range(1, self.bc.epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self._validate(epoch)
            self.val_history.append(val_metrics)

            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_loss"])

            # Checkpointing
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint("best_model.pt", epoch, val_metrics)
                marker = " ★ best"
            else:
                marker = ""

            # Log progress
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d}/{self.bc.epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.6f} | "
                f"Val Loss: {val_metrics['val_loss']:.6f} | "
                f"Val MAE steer: {val_metrics['val_mae_steering']:.4f} | "
                f"LR: {lr:.2e}{marker}"
            )

        # Save final model
        self._save_checkpoint("final_model.pt", self.bc.epochs, val_metrics)

        # Save training history for paper figures
        self._save_history()

        elapsed = time.time() - start_time
        logger.info(
            f"Training complete in {elapsed:.0f}s. "
            f"Best val loss: {self.best_val_loss:.6f}"
        )

        return {
            "best_val_loss": self.best_val_loss,
            "final_train_loss": train_metrics["train_loss"],
            "final_val_loss": val_metrics["val_loss"],
            "total_time_seconds": elapsed,
            "total_epochs": self.bc.epochs,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_steer_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images, actions = batch[0].to(self.device), batch[1].to(self.device)

            # Forward
            predictions = self.model(images)
            loss = self.criterion(predictions, actions)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Track steering-specific loss
            steer_loss = nn.functional.mse_loss(
                predictions[:, 0], actions[:, 0]
            ).item()
            total_steer_loss += steer_loss

            num_batches += 1

        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_steer_mse": total_steer_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        all_steer_preds = []
        all_steer_labels = []
        num_batches = 0

        for batch in self.val_loader:
            images, actions = batch[0].to(self.device), batch[1].to(self.device)

            predictions = self.model(images)
            loss = self.criterion(predictions, actions)

            total_loss += loss.item()
            all_steer_preds.append(predictions[:, 0].cpu().numpy())
            all_steer_labels.append(actions[:, 0].cpu().numpy())
            num_batches += 1

        # Compute steering-specific metrics
        steer_preds = np.concatenate(all_steer_preds)
        steer_labels = np.concatenate(all_steer_labels)
        mae = np.mean(np.abs(steer_preds - steer_labels))
        mse = np.mean((steer_preds - steer_labels) ** 2)

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_mse_steering": float(mse),
            "val_mae_steering": float(mae),
            "val_max_error": float(np.max(np.abs(steer_preds - steer_labels))),
        }

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
            "model_class": self.model.__class__.__name__,
            "num_outputs": self.num_outputs,
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _save_history(self):
        """Save training history as CSV for paper figures."""
        import csv

        history_path = self.output_dir / "training_history.csv"
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_steer_mse",
                "val_loss", "val_mse_steering", "val_mae_steering",
            ])
            for epoch, (train, val) in enumerate(
                zip(self.train_history, self.val_history), 1
            ):
                writer.writerow([
                    epoch,
                    f"{train['train_loss']:.6f}",
                    f"{train['train_steer_mse']:.6f}",
                    f"{val['val_loss']:.6f}",
                    f"{val['val_mse_steering']:.6f}",
                    f"{val['val_mae_steering']:.6f}",
                ])

        logger.info(f"Training history saved to {history_path}")


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING (for evaluation / inference)
# ═══════════════════════════════════════════════════════════════════

def load_dave2_model(
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[nn.Module, Dict]:
    """
    Load a trained DAVE-2 model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to load model onto.

    Returns:
        (model, checkpoint_dict) where model is ready for inference.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    config = ExperimentConfig(**checkpoint["config"]) if isinstance(
        checkpoint["config"], dict
    ) else checkpoint["config"]

    bc = config.baseline
    num_outputs = checkpoint.get("num_outputs", 1)

    model = DAVE2Net(
        input_height=bc.input_height,
        input_width=bc.input_width,
        num_outputs=num_outputs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train DAVE-2 behavioral cloning baseline"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to expert demonstration dataset"
    )
    parser.add_argument(
        "--name", type=str, default="dave2_baseline",
        help="Experiment name"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (overrides other args)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda"
    )
    parser.add_argument(
        "--predict-throttle", action="store_true",
        help="Also predict throttle (2 outputs instead of 1)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig(
            name=args.name,
            method="bc",
            seed=42,
            training=__import__("config.experiment", fromlist=["TrainingConfig"]).TrainingConfig(
                device=args.device,
            ),
            baseline=BaselineConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                predict_throttle=args.predict_throttle,
            ),
        )

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Train
    trainer = DAVE2Trainer(config=config, data_dir=args.data)
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("DAVE-2 BASELINE RESULTS")
    print("=" * 60)
    print(f"  Best val loss:    {results['best_val_loss']:.6f}")
    print(f"  Final train loss: {results['final_train_loss']:.6f}")
    print(f"  Final val loss:   {results['final_val_loss']:.6f}")
    print(f"  Training time:    {results['total_time_seconds']:.0f}s")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
