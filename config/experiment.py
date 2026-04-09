"""
Experiment Configuration Dataclasses

Hierarchical config tree:

    ExperimentConfig
    |-- SimConfig          — simulator selection, camera res, physics dt
    |-- TrainingConfig     — PPO hyperparameters, timesteps, scheduling
    |-- PolicyConfig       — architecture choices (LSTM, action head)
    |-- BaselineConfig     — behavioral cloning settings (DAVE-2, etc.)

Each dataclass:
    - Has sensible defaults matching our standard ARCPro configuration
    - Serializes to/from YAML via save()/load()
    - Is immutable after creation (frozen=False but convention is don't mutate)

The 12-element telemetry vector protocol is defined here as the canonical
reference. Both the environment and policy index into this vector, so having
one definition prevents index drift bugs.

Dependencies:
    - PyYAML (pip install pyyaml)
    - Python 3.10+ (for dataclass features)

Author: Aaron Hamil
Date: 03/31/26
Updated: 04/09/26
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Any, Dict
from pathlib import Path
import yaml
import json

# Telemetry Vectors

TELEMETRY_INDICES = {
    "turn_token": 0,          # Discrete turn command {-1, 0, 1} from Worker
    "go_signal": 1,           # Go/wait {0.0, 1.0} from Scheduler
    "goal_dist": 2,           # Goal distance (masked to 0 during PVP training)
    "speed": 3,               # Vehicle speed (m/s)
    "yaw_rate": 4,            # Yaw rate (rad/s)
    "last_steer": 5,          # Previous steering command
    "last_throttle": 6,       # Previous throttle command
    "last_brake": 7,          # Previous brake command
    "lateral_offset": 8,      # Lateral offset from SimpleLaneDetector (m)
    "lane_confidence": 9,     # Lane detection confidence [0,1] - off-road termination fires when < 0.05 for > 1s
    "reserved": 10,           # Zero-padded  (PVP protocol - no geometry signal)
    "distance_traveled": 11,  # Cumulative odometry (m)
}

TELEMETRY_DIM = 12  # Length of the vec observation


# Simulator Configuration

@dataclass
class SimConfig:
    """
    Simulator and environment settings.

    These control which simulator is used, camera resolution, physics
    timestep, and parallelization. The sim_type field selects the
    environment adapter via the registry (when implemented).

    Camera resolution is fixed at 160x90 (16:9) to match the D435i
    downsampled output. Changing this requires recomputing CNN dimensions
    in FusionFeaturesExtractor.
    """

    # Simulator selection (used by future registry/factory)
    sim_type: str = "isaac"  # "isaac" | "gazebo"  (via envs.registry)

    # Rendering
    headless: bool = True
    camera_width: int = 160
    camera_height: int = 90

    # Physics
    physics_dt: float = 1.0 / 60.0        # 60 Hz physics
    render_dt: float = 1.0 / 30.0         # 30 Hz rendering
    control_hz: int = 10                  # Policy inference rate

    # Vehicle
    max_steering_angle: float = 0.5       # radians (~28.6 degrees)
    max_acceleration: float = 3.0         # m/s^2

    # Episode
    episode_timeout: float = 30.0         # seconds
    max_episode_steps: int = 300          # steps (timeout / dt)

    # Parallelization (for future vectorized envs)
    num_parallel_envs: int = 1

    # ROS2 topics (only used by isaac_ros2_env.py, not direct API envs)
    camera_topic: str = "/camera/image_raw"
    state_topic: str = "/vehicle_state"
    control_topic: str = "/ackermann_cmd"


# Training Configuration

@dataclass
class TrainingConfig:
    """
    Training hyperparameters for PPO-based reinforcement learning.

    These match the argparse defaults in train_policy_ros2.py but are
    now typed and serializable. The reward_strategy field selects which
    reward function is used (for future strategy pattern).
    """

    # Core
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 128                    # Rollout length per update
    batch_size: int = 128                 # Minibatch size
    n_epochs: int = 10                    # PPO epochs per update

    # Discounting
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO-specific
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Reward strategy (for future strategy pattern)
    reward_strategy: str = "vision_only"  # "vision_only" | "telemetry" | "hybrid"

    # Auxiliary losses
    waypoint_loss_weight: float = 0.15
    repulsion_weight: float = 2.0

    # Checkpointing
    save_freq: int = 10_000               # Steps between checkpoints
    eval_freq: int = 20_000               # Steps between evaluations

    # Device
    device: str = "auto"                  # "auto" | "cpu" | "cuda"


# Policy Configuration

@dataclass
class PolicyConfig:
    """
    Neural network architecture settings.

    Controls the hierarchical policy structure: feature extractor,
    temporal backbone, planning head, and control head.

    The temporal_backbone and action_head fields are forward-compatible
    placeholders for Phase 3 (Mamba, flow matching).
    """

    # Feature extraction
    cnn_features_dim: int = 256           # CNN output dimension
    vec_features_dim: int = TELEMETRY_DIM # Telemetry vector dimension
    fusion_features_dim: int = 268        # cnn + vec = 256 + 12

    # Temporal backbone
    temporal_backbone: str = "lstm"       # "lstm" | (future)
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    enable_critic_lstm: bool = True

    # Planning head
    num_waypoints: int = 5
    waypoint_horizon: float = 2.5         # meters
    use_kinematic_anchors: bool = True
    planning_hidden_dim: int = 128

    # Control head
    control_hidden_dim: int = 128

    # Kinematic anchor tuning
    curvature_gain: float = 0.18
    command_blend_factor: float = 0.6
    steering_blend_factor: float = 0.4
    progressive_curvature_exp: float = 1.15
    max_deviation_meters: float = 8.0

    # Action head (forward-compatible)
    action_head: str = "gaussian"         # "gaussian" | (future)

    # MLP extractor
    net_arch_pi: List[int] = field(default_factory=lambda: [64])
    net_arch_vf: List[int] = field(default_factory=lambda: [64])


# Baseline Configuration

@dataclass
class BaselineConfig:
    """
    Settings for behavioral cloning baselines (DAVE-2, etc.).

    This is separate from PolicyConfig because BC uses a completely
    different training loop (supervised MSE, not PPO), different
    data pipeline (offline dataset, not online rollouts), and a
    much simpler architecture.
    """

    # Model selection
    model_type: str = "dave2"             # "dave2" | (future)

    # DAVE-2 specific
    input_height: int = 66               # DAVE-2 canonical crop height
    input_width: int = 200               # DAVE-2 canonical crop width

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 64
    weight_decay: float = 1e-5
    train_val_split: float = 0.8         # 80% train, 20% validation

    # Data augmentation
    augment_brightness: float = 0.2
    augment_shadow: bool = True
    augment_flip: bool = True            # Horizontal flip + negate steering

    # Data collection
    collection_hz: int = 10              # Frames per second during collection
    expert_type: str = "scripted"        # "scripted" | "teleop" | "pid"

    # Output actions
    predict_throttle: bool = True        # True: predict [steer, throttle]
    predict_brake: bool = False          # True: predict [steer, throttle, brake]


# Root Experiment Configuration

@dataclass
class ExperimentConfig:
    """
    Root configuration for any experiment.

    Composes all sub-configs and provides YAML serialization.
    Every training run (RL or BC) should save this alongside the
    checkpoint for full reproducibility.

    Usage:
        config = ExperimentConfig(name="dave2_baseline_001")
        config.save("experiments/dave2_baseline_001/config.yaml")

        # Later, for reproduction:
        config = ExperimentConfig.load("experiments/dave2_baseline_001/config.yaml")
    """

    # Experiment metadata
    name: str = "unnamed_experiment"
    seed: int = 42
    description: str = ""
    method: str = "rl"                    # "rl" | "bc" | "dagger"

    # Sub-configs
    sim: SimConfig = field(default_factory=SimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    # Output paths (computed from name if not set)
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"experiments/{self.name}"

    def save(self, path: Optional[str] = None) -> Path:
        """
        Save config to YAML file.

        Args:
            path: File path. Defaults to {output_dir}/config.yaml.

        Returns:
            Path to the saved file.
        """
        if path is None:
            path = f"{self.output_dir}/config.yaml"

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(
                asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )

        return filepath

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """
        Load config from YAML file.

        Handles nested dataclass reconstruction: YAML loads as plain
        dicts, so we need to manually construct sub-dataclasses.

        Args:
            path: Path to YAML config file.

        Returns:
            Fully reconstructed ExperimentConfig.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Reconstruct nested dataclasses from dicts
        if "sim" in data and isinstance(data["sim"], dict):
            data["sim"] = SimConfig(**data["sim"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "policy" in data and isinstance(data["policy"], dict):
            data["policy"] = PolicyConfig(**data["policy"])
        if "baseline" in data and isinstance(data["baseline"], dict):
            data["baseline"] = BaselineConfig(**data["baseline"])

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (for logging, serialization)."""
        return asdict(self)

    def diff(self, other: "ExperimentConfig") -> Dict[str, Any]:
        """
        Show differences between two configs.

        Useful for comparing ablation runs:
            diff = config_a.diff(config_b)
            print(diff)  # Shows only fields that changed
        """
        d1 = asdict(self)
        d2 = asdict(other)
        return _dict_diff(d1, d2)

    def summary(self) -> str:
        """One-line summary for logging."""
        method = self.method.upper()
        if self.method == "bc":
            return (
                f"[{method}] {self.name} | "
                f"model={self.baseline.model_type} | "
                f"epochs={self.baseline.epochs} | "
                f"lr={self.baseline.learning_rate} | "
                f"seed={self.seed}"
            )
        else:
            return (
                f"[{method}] {self.name} | "
                f"sim={self.sim.sim_type} | "
                f"steps={self.training.total_timesteps:,} | "
                f"lr={self.training.learning_rate} | "
                f"seed={self.seed}"
            )


def _dict_diff(d1: dict, d2: dict, prefix: str = "") -> Dict[str, Any]:
    """Recursively find differences between two dicts."""
    diffs = {}
    all_keys = set(d1.keys()) | set(d2.keys())

    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        v1 = d1.get(key)
        v2 = d2.get(key)

        if isinstance(v1, dict) and isinstance(v2, dict):
            nested = _dict_diff(v1, v2, full_key)
            diffs.update(nested)
        elif v1 != v2:
            diffs[full_key] = {"from": v1, "to": v2}

    return diffs
