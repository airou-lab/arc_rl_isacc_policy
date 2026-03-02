#!/usr/bin/env python3
"""
train_policy_ros2.py - Hierarchical PPO Training for Isaam Sim

Architecture:
    IsaacROS2Env (Gymnasium)
        -> WaypointTrackingWrapper (trajectory recording + safety backfill)
        -> Monitor (episode stats logging)
    -> DummyVecEnv
    -> RecurrentPPO with HierarchicalPathPlanningPolicy

Usage:
    # Quick test (50k steps):
    python train_policy_ros2.py --timesteps 50000 --name isaac_test_001

    # Full training:
    python train_policy_ros2.py --timesteps 500000 --name full_run

    # Resume from checkpoint:
    python train_policy_ros2.py --resume models/my_run/checkpoints/hppo_100000_steps.zip

Prerequistes:
    - Isaam Sim running with ROS2 bridge enabled
    - Camera publishing to /camera/image_raw
    - State publishing to /vehicle_state
    - Ackermann controller listening on /ackermann_cmd
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Project imports
from isaac_ros2_env import IsaacROS2Env, IsaacROS2Config
from policies.hierarchical_policy import HierarchicalPathPlanningPolicy
from policies.fusion_policy import FusionFeaturesExtractor

# Optional: waypoint tracking for self-supervised auxiliary loss
try:
    from wrappers.waypoint_tracking_wrapper import (
        WaypointTrackingWrapper,
        get_trajectory_store,
    )
    HAS_WAYPOINT_WRAPPER = True
except ImportError:
    HAS_WAYPOINT_WRAPPER = False
    print("[warn] wrappers/waypoint_tracking_wrapper.py not found - "
          "waypoint auxiliary loss disabled")

class WaypointLoggingCallback(BaseCallback):
    """Log waypoint statistics from the hierarchical policy."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        policy = self.model.policy
        if hasattr(policy, "last_waypoints") and policy.last_waypoints is not None:
            wp = policy.last_waypoints
            if isinstance(wp, torch.Tensor):
                wp_np = wp.detach().cpu().numpy()
                if wp_np.ndim == 3: # (batch, num_wp, 2)
                    self.logger.record(
                        "waypoints/mean_forward", float(np.mean(wp_np[:, :, 1]))
                    )
                    self.logger.record(
                        "waypoints/mean_lateral_abs", float(np.mean(np.abs(wp_np[:, :, 0]))),
                    )
        return True


class EpisodeStatsCallback(BaseCallback):
    """Log episode-level statistics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

                if len(self._episode_rewards) % 10 == 0:
                    recent = self._episode_rewards[-10:]
                    self.logger.record(
                        "episodes/mean_reward_10", float(np.mean(recent))
                    )
                    self.logger.record(
                        "episodes/mean_length_10", float(np.mean(self._episode_lengths[-10:])),
                    )
        return True


def make_env(config: IsaacROS2Config, use_waypoint_wrapper: bool = True):
    """
    Factory function for creating the training environment.

    Pipeline:
        IsaacROS2Env -> WaypointTrackingWrapper (optional) -> Monitor
    """

    def _init():
        env = IsaacROS2Env(config=config)

        # Wrap with trajectory tracking for waypoint auxiliary loss
        if use_waypoint_wrapper and HAS_WAYPOINT_WRAPPER:
            env = WaypointTrackingWrapper(env, env_id=0)

        # SB3 Monitor for episode stats
        env = Monitor(env)

        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical driving policy in Isaac Sim"
    )

    # Training
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="Total training timesteps"
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--save-freq", type=int, default = 10_000, help="Checkpoint frequency (steps)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Environment
    parser.add_argument(
        "--camera-topic", type=str, default="/camera/image_raw"
    )
    parser.add_argument(
        "--state-topic", type=str, default="/vehicle_state"
    )
    parser.add_argument(
        "--control-topic", type=str, default="/ackermann_cmd"
    )
    parser.add_argument(
        "--episode-timeout", type=float, default=30.0, help="Episode timeout (seconds)"
    )

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout length")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default = 0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )

    # Policy architecture
    parser.add_argument("--lstm-size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--num-waypoints", type=int, default=5, help="Number of waypoints to predict")
    parser.add_argument("-waypoint-horizon", type=float, default=2.5, help="Planning horizon (meters)")
    parser.add_argument("--no-kinematic-anchors", action="store_true", help="Disable kinematic anchors (ablation)")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")

    args = parser.parse_args()

    # Experiment directory
    if args.name is None:
        args.name = datetime.now().strftime("isaac_hppo_%Y%m%d_%H%M%S")

    run_dir = Path("models") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = run_dir / "tb"

    print(f"Experiment: {args.name}")
    print(f"  Run directory: {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"  Timesteps: {args.timesteps:,}")

    # Environment
    env_config = IsaacROS2Config(
        img_width=160,
        img_height=90,
        camera_topic=args.camera_topic,
        state_topic=args.state_topic,
        control_topic=args.control_topic,
        episode_timeout=args.episode_timeout,
    )

    vec_env = DummyVecEnv([make_env(env_config, use_waypoint_wrapper=True)])

    # Model
    if args.resume:
        print(f"Resume from: {args.resume}")
        model = RecurrentPPO.load(
            args.resume,
            env=vec_env,
            device=args.device,
            tensorboard_log=str(tb_dir),
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=FusionFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=268),
            lstm_hidden_size=args.lstm_size,
            n_lstm_layers=1,
            enable_critic_lstm=True,
            share_features_extractor=True,
            # Hierarchical planning paramaters
            num_waypoints=args.num_waypoints,
            waypoint_horizon=args.waypoint_horizon,
            use_kinematic_anchors=not args.no_kinematic_anchors,
        )

        model = RecurrentPPO(
            policy=HierarchicalPathPlanningPolicy,
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            device=args.device,
            tensorboard_log=str(tb_dir),
            policy_kwargs=policy_kwargs,
        )

    # Print architecture summary
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"Policy: {model.policy.__class__.__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LSTM hidden size: {args.lstm_size}")
    print(f"  Waypoints: {args.num_waypoints}")
    print(f"  Waypoint horizon: {args.waypoint_horizon}m")
    print(f"  Kinematic anchors: {not args.no_kinematic_anchors}")

    # Callbacks
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(ckpt_dir),
                name_prefix="hppo",
                verbose=1,
            ),
            WaypointLoggingCallback(),
            EpisodeStatsCallback(),
        ]
    )

    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}.zip")

    vec_env.close()

if __name__ == "__main__":
    main()
