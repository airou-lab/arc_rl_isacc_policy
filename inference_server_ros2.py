#!/usr/bin/env python3
"""
inference_server_ros2.py - Run Trained Policy via ROS2

Loads a trained RecurrentPPO checkpoint and runs inference against Isaac Sim (or the physical ARCPro robot) over ROS2.

Works identically for simulation and real hardware because both publish the same ROS2 topics:
    /camera/image_raw -> sensor_msgs/Image
    /vehicle_state -> custom 12-float telemetry
    /ackermann_cmd -> ackermann_msgs/AckermannDrive

Ported from inference_server_RNN.py (Unity version):
    - Replaced LiveUnityEnv with IsaacROS2Env
    - Removed TCP host/port (ROS2 handles discovery)
    - Kept LSTM state management and predict loop unchanged

Usage:
    # Run 10 episodes:
    python inference_server_ros2.py --model models/my_run/final_model.zip

    # Deterministic (no exploration noise):
    python inference_server_ros2.py --model final_model.zip --deterministic

    # Continuous (no episode limit, Ctrl+C to stop):
    python inference_server_ros2.py --model final_model.zip --episodes 0

    # Custom topics (e.g., real robot namespace):
    python inference_server_ros2.py --model final_model.zip \
        --camera-topic /robot/camera/image_raw \
        --state-topic /robot/vehicle_state \
        --control-topic /robot/ackermann_cmd
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import get_schedule_fn

from isaac_ros2_env import IsaacROS2Env, IsaacROS2Config

def run_inference(args):
    """Main inference loop."""

    # Environment
    config = IsaacROS2Config(
        img_width=160,
        img_height=90,
        camera_topic=args.camera_topic,
        state_topic=args.state_topic,
        control_topic=args.control_topic,
        episode_timeout=args.episode_timeout,
    )
    env = IsaacRos2Env(config=config)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = RecurrentPPO.load(
            args.model,
            custom_objects={
            # Schedules are training-only; provide defaults to avoid deserialization errors from checkpoints saved mid-training.
                "clip_range": get_schedule_fn(0.2),
                "lr_schedule": get_schedule_fn(3e-4),
            },
            device=args.device,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Policy: {model.policy.__class__.__name__}")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {model.device}")
    print(f"Deterministic: {args.deterministic}")
    print()

    # LSTM dimensions
    lstm_hidden_size = model.policy.lstm_hidden_size
    n_lstm_layers = model.policy.n_lstm_layers
    signle_hidden_shape = (n_lstm_layers, 1, lstm_hidden_size)

    # Inference loop
    episode = 0
    continuous = args.episodes == 0

    try:
        while continuous or episode < args.episodes:
            obs, info = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            step_count = 0

            # Reset LSTM states at episode start
            lstm_states = (
                (
                    np.zeros(single_hidden_shape, dtype=np.float32),
                    np.zeros(single_hidden_shape, dtype=float32),
                ),
                (
                    np.zeros(single_hidden_shape, dtype=np.float32),
                    np.zeros(single_hidden_shape, dtype=np.float32),
                ),
            )
            episode_starts = np.array([True], dtype=bool)

            while not (done or truncated):
                action, lstm_states = model.predict(
                    observation=obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=args.deterministic,
                )
                episode_starts = np.array([False], dtype=bool)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                step_count += 1

                # Optional: print live telemetry
                if args.verbose:
                    vec = obs.get("vec", None) if isinstance(obs, dict) else None
                    if vec is not None:
                        speed = vec[3] if len(vec) > 3 else 0.0
                        steer = action[0] if hasattr(action, "__len__") else action
                        print(
                            f"  step {step_count:4d} | "
                            f"speed={speed:.2f} steer={steer:.3f} "
                            f"reward={reward:.3f}",
                            end="\r",
                        )

            episode += 1
            print(
                f"[ep {episode:3d}] "
                f"steps={step_count:4d}  "
                f"reward={ep_reward:+8.2f}  "
                f"done={done}  trunc={truncated}"
            )

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")

    env.close()
    print("Environment closed.")


def main():
    parser = argparse.ArgumentParser(description="Run trained policy via ROS2")

    # Model
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model .zip"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions (no exploration noise)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device: cpu or cuda"
    )

    # Environment / ROS2 topics
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
        "--episode-timeout", type=float, default=30.0, help="Episode timeout in seconds"
    )

    # Run settings
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes (0 = continuous until Ctrl+C)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print live telemetry each step"
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
