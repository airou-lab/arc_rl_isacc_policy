"""
Isaac Sim Direct API Gymnasium Environment
============================================

ROS2-free environment for training. Controls the F1Tenth vehicle and
reads camera images through Isaac Sim's native Python API, bypassing
the Action Graph and ROS2 bridge entirely.

Why this exists:
    The original IsaacROS2Env routes everything through ROS2 topics:
        Camera -> ROS2 publish -> ROS2 subscribe -> numpy
        Action -> ROS2 publish -> Action Graph -> ArticulationController

    This introduces latency, serialization jitter, and queue saturation
    that corrupt RL training signals. At 1 it/s observed throughput,
    training 3M parameters is infeasible.

    This environment calls the simulator APIs directly:
        Camera -> Replicator API -> numpy  (zero-copy when possible)
        Action -> ArticulationController   (direct joint commands)

    Expected throughput: 30-100+ steps/sec depending on GPU and headless mode.

Observation contract (identical to IsaacROS2Env):
    obs["image"] -> (90, 160, 3) uint8 RGB
    obs["vec"]   -> (12,) float32 telemetry (see config/experiment.py)

Action space:
    Box(3,) float32: [steering, throttle, brake] each in [-1, 1]
    Internally converted to Ackermann geometry for the F1Tenth joints.

Vehicle parameters (from Arika's launch_isaac_ros2.py Action Graph):
    Robot prim:     /World/F1Tenth
    Wheelbase:      0.33 m
    Track width:    0.28 m
    Wheel radius:   0.05 m
    Steering joints: Knuckle__Upright__Front_Left, Knuckle__Upright__Front_Right
    Drive joints:    Wheel__Knuckle__Front_Left,   Wheel__Knuckle__Front_Right,
                     Wheel__Upright__Rear_Left,    Wheel__Upright__Rear_Right

Camera:
    Intel RealSense D435i equivalent in sim.
    Resolution: 160x90 (16:9, downsampled from native).
    Camera prim path must be configured — it depends on how Arika
    attached the camera to the F1Tenth URDF. Common locations:
        /World/F1Tenth/chassis/Camera
        /World/F1Tenth/Camera

Dependencies:
    - isaacsim (Isaac Sim Python API)
    - omni.isaac.core
    - omni.replicator.core (for camera capture)
    - gymnasium
    - numpy
    - config/experiment.py (for TELEMETRY_INDICES)

    Does NOT require: rclpy, ros2, cv_bridge, sensor_msgs, ackermann_msgs

NOTE: This file must be run inside Isaac Sim's Python environment
      (e.g., via ./python.sh or within a SimulationApp context).
      It will not work in a vanilla Python environment.

Author: Aaron Hamil
Date: 03/05/26
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


#=====================================================================#
#                         CONFIGURATION                               #
#=====================================================================#

@dataclass
class IsaacDirectConfig:
    """
    Configuration for the direct-API Isaac Sim environment.

    Vehicle geometry is taken directly from Arika's launch_isaac_ros2.py
    Action Graph setup, which uses the F1Tenth standard dimensions.
    """

    # === Scene ===
    usd_path: str = ""  # Path to USD stage. Empty = use currently loaded stage.
    robot_prim_path: str = "/World/F1Tenth"
    camera_prim_path: str = "/World/F1Tenth/chassis/Camera"  # Adjust per actual USD

    # === Camera ===
    img_width: int = 160
    img_height: int = 90

    # === Vehicle geometry (F1Tenth standard, from launch_isaac_ros2.py) ===
    wheelbase: float = 0.33          # meters, front axle to rear axle
    track_width: float = 0.28        # meters, left wheel to right wheel
    wheel_radius: float = 0.05       # meters

    # Joint names (from Arika's Action Graph / URDF quick_inspect.py)
    steering_joints: Tuple[str, ...] = (
        "Knuckle__Upright__Front_Left",
        "Knuckle__Upright__Front_Right",
    )
    drive_joints: Tuple[str, ...] = (
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Left",
        "Wheel__Upright__Rear_Right",
    )

    # === Control limits ===
    max_steering_angle: float = 0.5  # radians (~28.6 degrees)
    max_speed: float = 3.0           # m/s

    # === Physics ===
    physics_dt: float = 1.0 / 60.0   # 60 Hz physics
    render_dt: float = 1.0 / 30.0    # 30 Hz rendering
    control_hz: int = 10             # Policy step rate (steps/sec)
    substeps: int = 6                # Physics substeps per control step (60/10)

    # === Episode ===
    episode_timeout: float = 30.0    # seconds
    max_episode_steps: int = 300     # steps before truncation

    # === Reset ===
    # Spawn position for episode reset (world frame)
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 0.05           # Slightly above ground to avoid collision
    spawn_yaw: float = 0.0          # radians

    # === Headless ===
    headless: bool = True


#=====================================================================#
#                    ACKERMANN GEOMETRY                                #
# Converts [steering_angle, speed] into individual wheel commands.    #
# Same math as the AckermannController node in the Action Graph,      #
# but computed in Python so we don't need the OmniGraph at all.       #
#=====================================================================#

class AckermannComputer:
    """
    Converts high-level Ackermann commands (speed + steering angle)
    into individual wheel angles and velocities for the F1Tenth.

    The F1Tenth has bicycle-like Ackermann steering: the inner wheel
    turns more than the outer wheel to avoid tire scrub. This class
    computes the correct angle for each front wheel and the angular
    velocity for each drive wheel.
    """

    def __init__(self, wheelbase: float, track_width: float, wheel_radius: float):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_radius = wheel_radius
        self.half_track = track_width / 2.0

    def compute(
        self, steering_angle: float, speed: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Ackermann command to per-wheel targets.

        Args:
            steering_angle: Desired steering angle in radians.
                Positive = turn right, negative = turn left.
            speed: Desired forward speed in m/s.

        Returns:
            wheel_angles: (2,) float array [left_angle, right_angle]
                for the two front steering joints (position targets).
            wheel_velocities: (4,) float array [FL, FR, RL, RR]
                angular velocities in rad/s for all four drive joints.
        """
        # Wheel angular velocity from linear speed
        base_omega = speed / self.wheel_radius

        if abs(steering_angle) < 1e-4:
            # Straight driving — all wheels same angle and speed
            wheel_angles = np.array([0.0, 0.0], dtype=np.float32)
            wheel_velocities = np.full(4, base_omega, dtype=np.float32)
            return wheel_angles, wheel_velocities

        # Ackermann geometry: compute turning radius
        turn_radius = self.wheelbase / math.tan(abs(steering_angle))

        # Inner and outer wheel angles (inner turns more)
        inner_angle = math.atan(self.wheelbase / (turn_radius - self.half_track))
        outer_angle = math.atan(self.wheelbase / (turn_radius + self.half_track))

        # Assign left/right based on turn direction
        if steering_angle > 0:
            # Turning right: right wheel is inner
            left_angle = outer_angle
            right_angle = inner_angle
        else:
            # Turning left: left wheel is inner
            left_angle = -inner_angle
            right_angle = -outer_angle

        wheel_angles = np.array([left_angle, right_angle], dtype=np.float32)

        # Wheel velocities: outer wheels travel further per revolution
        inner_radius = turn_radius - self.half_track
        outer_radius = turn_radius + self.half_track

        inner_omega = speed * (inner_radius / turn_radius) / self.wheel_radius
        outer_omega = speed * (outer_radius / turn_radius) / self.wheel_radius

        if steering_angle > 0:
            # Turning right: left wheels are outer
            wheel_velocities = np.array(
                [outer_omega, inner_omega, outer_omega, inner_omega],
                dtype=np.float32,
            )
        else:
            wheel_velocities = np.array(
                [inner_omega, outer_omega, inner_omega, outer_omega],
                dtype=np.float32,
            )

        return wheel_angles, wheel_velocities


#=====================================================================#
#                 GYMNASIUM ENVIRONMENT                                #
#=====================================================================#

class IsaacDirectEnv(gym.Env):
    """
    Gymnasium environment using Isaac Sim's direct Python API.

    This replaces IsaacROS2Env for training. It provides the same
    observation contract (Dict with 'image' and 'vec' keys) and the
    same action space (Box(3,) for [steer, throttle, brake]), but
    communicates with the simulator through native Python calls instead
    of ROS2 topics.

    Lifecycle:
        env = IsaacDirectEnv(config)
        obs, info = env.reset()
        for step in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()

    The environment handles:
        1. Loading the USD stage (if specified)
        2. Finding the robot articulation and camera
        3. Stepping physics with substeps between control steps
        4. Capturing camera images via Replicator
        5. Computing telemetry from articulation state
        6. Ackermann geometry for wheel commands
        7. Episode reset with configurable spawn position

    NOTE: Must be run inside Isaac Sim's Python environment.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[IsaacDirectConfig] = None):
        super().__init__()

        self.config = config or IsaacDirectConfig()
        self._step_count = 0
        self._episode_start_time = 0.0

        # State tracking for telemetry
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cumulative_distance = 0.0
        self._last_position = np.zeros(3, dtype=np.float64)

        # Ackermann computer
        self._ackermann = AckermannComputer(
            wheelbase=self.config.wheelbase,
            track_width=self.config.track_width,
            wheel_radius=self.config.wheel_radius,
        )

        # === Gymnasium spaces ===
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8,
            ),
            "vec": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # === Isaac Sim handles (initialized in _setup_sim) ===
        self._world = None
        self._robot_articulation = None
        self._camera = None
        self._annotator = None
        self._sim_initialized = False

        # Lane detector for reward (optional, same as IsaacROS2Env)
        self._lane_detector = None
        try:
            from lane_detector import SimpleLaneDetector
            self._lane_detector = SimpleLaneDetector()
            logger.info("Lane detector loaded for reward computation")
        except ImportError:
            logger.warning("lane_detector not found, lane rewards disabled")

    def _setup_sim(self):
        """
        Initialize Isaac Sim handles: world, robot, camera.

        Called lazily on first reset() so the SimulationApp is already
        running when we access omni APIs. This avoids import-time
        errors when the file is loaded outside Isaac Sim for syntax checks.
        """
        if self._sim_initialized:
            return

        # Isaac Sim imports (only available inside SimulationApp context)
        from isaacsim.core.api import World
        from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
        from isaacsim.core.api.robots import Robot
        import omni.usd

        # Load stage if specified
        if self.config.usd_path:
            import os
            if os.path.exists(self.config.usd_path):
                omni.usd.get_context().open_stage(self.config.usd_path)
                logger.info(f"Loaded USD stage: {self.config.usd_path}")
            else:
                logger.warning(f"USD not found: {self.config.usd_path}, using current stage")

        # Create simulation world
        self._world = World(
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.render_dt,
            stage_units_in_meters=1.0,
        )

        # Get robot articulation
        robot_path = self.config.robot_prim_path
        if not is_prim_path_valid(robot_path):
            raise RuntimeError(
                f"Robot prim not found at '{robot_path}'. "
                f"Check that the USD stage is loaded and the F1Tenth model exists. "
                f"You may need to adjust config.robot_prim_path."
            )

        self._robot_articulation = self._world.scene.add(
            Robot(prim_path=robot_path, name="f1tenth")
        )

        # Setup camera via Replicator for zero-copy image capture
        self._setup_camera()

        # Initialize world (this loads physics, creates articulation handles)
        self._world.reset()
        self._sim_initialized = True

        logger.info(
            f"Isaac Direct Env initialized: "
            f"robot={robot_path}, "
            f"camera={self.config.camera_prim_path}, "
            f"substeps={self.config.substeps}"
        )

    def _setup_camera(self):
        """
        Setup camera capture using Isaac Sim's Replicator API.

        This creates a render product from the camera prim and an
        annotator that reads RGB data. On each step, we call
        annotator.get_data() to get the latest frame as a numpy array.
        No ROS2, no serialization, no message queues.
        """
        from isaacsim.core.utils.prims import is_prim_path_valid
        import omni.replicator.core as rep

        camera_path = self.config.camera_prim_path
        if not is_prim_path_valid(camera_path):
            raise RuntimeError(
                f"Camera prim not found at '{camera_path}'. "
                f"Check the USD stage. Common paths:\n"
                f"  /World/F1Tenth/chassis/Camera\n"
                f"  /World/F1Tenth/Camera\n"
                f"  /World/F1Tenth/Cameras/front_camera\n"
                f"Arika can check with: "
                f"omni.usd.get_context().get_stage().TraverseAll()"
            )

        # Create render product at desired resolution
        render_product = rep.create.render_product(
            camera_path,
            resolution=(self.config.img_width, self.config.img_height),
        )

        # RGB annotator — reads pixel data as numpy array
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([render_product])

        logger.info(
            f"Camera attached: {camera_path} "
            f"({self.config.img_width}x{self.config.img_height})"
        )

    # ================================================================
    #  Gymnasium Interface
    # ================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment for a new episode.

        Teleports the vehicle to the spawn position, zeros velocities,
        steps physics once to settle, then returns the initial observation.
        """
        super().reset(seed=seed)

        # Lazy initialization
        self._setup_sim()

        # Reset episode state
        self._step_count = 0
        self._episode_start_time = time.time()
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cumulative_distance = 0.0

        # Teleport robot to spawn position
        self._reset_robot_pose()

        # Step physics once to settle contacts
        self._world.step(render=True)

        # Record initial position for distance tracking
        self._last_position = self._get_robot_position()

        obs = self._get_obs()
        info = {"episode_step": 0}

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one control step.

        Converts the action to Ackermann wheel commands, steps physics
        for substeps, captures a camera image, computes telemetry and
        reward.

        Args:
            action: [steering, throttle, brake] each in [-1, 1].

        Returns:
            Standard Gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._last_action = action.copy()

        # Convert to Ackermann commands
        steering_angle = action[0] * self.config.max_steering_angle
        speed = action[1] * self.config.max_speed

        # Apply brake: reduce speed proportionally
        if action[2] > 0.0:
            speed *= (1.0 - action[2])

        # Compute per-wheel targets
        wheel_angles, wheel_velocities = self._ackermann.compute(
            steering_angle, speed
        )

        # Apply to articulation
        self._apply_wheel_commands(wheel_angles, wheel_velocities)

        # Step physics for substeps (e.g., 6 physics steps per control step)
        for _ in range(self.config.substeps):
            self._world.step(render=False)

        # Render one frame for camera capture
        self._world.step(render=True)

        # Update distance tracking
        current_pos = self._get_robot_position()
        step_distance = np.linalg.norm(current_pos[:2] - self._last_position[:2])
        self._cumulative_distance += step_distance
        self._last_position = current_pos

        self._step_count += 1

        # Build observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(obs)

        # Episode termination
        elapsed = time.time() - self._episode_start_time
        terminated = self._check_termination()
        truncated = (
            self._step_count >= self.config.max_episode_steps
            or elapsed >= self.config.episode_timeout
        )

        info = {
            "episode_step": self._step_count,
            "speed": obs["vec"][3],
            "distance": self._cumulative_distance,
            "elapsed": elapsed,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up simulation resources."""
        if self._world is not None:
            self._world.stop()
            logger.info("Isaac Direct Env closed")

    # ================================================================
    #  Observation
    # ================================================================

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Build the observation dict matching IsaacROS2Env's contract.

        Returns:
            {
                "image": (90, 160, 3) uint8 RGB,
                "vec": (12,) float32 telemetry
            }
        """
        image = self._capture_camera()
        telemetry = self._compute_telemetry()

        return {
            "image": image,
            "vec": telemetry,
        }

    def _capture_camera(self) -> np.ndarray:
        """
        Read current camera frame via Replicator annotator.

        Returns:
            (H, W, 3) uint8 RGB numpy array.
        """
        data = self._annotator.get_data()

        if data is None or data.size == 0:
            logger.warning("Empty camera frame, returning black image")
            return np.zeros(
                (self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8,
            )

        # Replicator returns RGBA — drop alpha channel
        if data.ndim == 3 and data.shape[2] == 4:
            data = data[:, :, :3]

        # Ensure correct resolution (should already be right from render_product)
        if data.shape[:2] != (self.config.img_height, self.config.img_width):
            import cv2
            data = cv2.resize(
                data,
                (self.config.img_width, self.config.img_height),
                interpolation=cv2.INTER_AREA,
            )

        return data.astype(np.uint8)

    def _compute_telemetry(self) -> np.ndarray:
        """
        Build the 12-element telemetry vector from articulation state.

        Indices match config/experiment.py TELEMETRY_INDICES:
            0: turn_bias      — always 0.0 (Passive Visual Protocol)
            1: reserved        — always 0.0
            2: goal_dist       — always 0.0 (masked for PVP)
            3: speed           — current forward speed (m/s)
            4: yaw_rate        — current yaw rate (rad/s)
            5: last_steer      — previous steering command
            6: last_throttle   — previous throttle command
            7: last_brake      — previous brake command
            8: lateral_error   — from lane detector (if available)
            9: heading_error   — from lane detector (if available)
           10: curvature       — always 0.0 (not yet computed)
           11: distance_traveled — cumulative odometry
        """
        vec = np.zeros(12, dtype=np.float32)

        # === Signals from articulation state ===
        velocity = self._get_robot_velocity()
        vec[3] = float(np.linalg.norm(velocity[:2]))   # speed (m/s)
        vec[4] = float(self._get_robot_yaw_rate())      # yaw_rate (rad/s)

        # === Previous action (feedback) ===
        vec[5] = self._last_action[0]   # last_steer
        vec[6] = self._last_action[1]   # last_throttle
        vec[7] = self._last_action[2]   # last_brake

        # === Lane detector signals (if available) ===
        # Indices 8 and 9 are what the ScriptedExpert reads.
        # Without lane detector these stay 0.0, which is fine for
        # RL training (the policy learns from the image) but means
        # ScriptedExpert won't steer (it relies on these signals).
        if self._lane_detector is not None:
            try:
                # Use the last captured camera frame
                image = self._capture_camera()
                lane_info = self._lane_detector.detect(image)
                if lane_info is not None:
                    vec[8] = float(lane_info.get("lateral_error", 0.0))
                    vec[9] = float(lane_info.get("heading_error", 0.0))
                    vec[10] = float(lane_info.get("curvature", 0.0))
            except Exception as e:
                logger.debug(f"Lane detection failed: {e}")

        # === Odometry ===
        vec[11] = self._cumulative_distance

        # === PVP masked signals ===
        # Indices 0, 1, 2 stay at 0.0 — this is the Passive Visual Protocol.
        # The agent receives only image + high-level turn commands (which
        # we keep at 0 = "go straight" for now). No geometric shortcuts.

        return vec

    # ================================================================
    #  Robot Control
    # ================================================================

    def _apply_wheel_commands(
        self,
        wheel_angles: np.ndarray,
        wheel_velocities: np.ndarray,
    ):
        """
        Apply Ackermann wheel targets to the F1Tenth articulation.

        Args:
            wheel_angles: (2,) position targets for front steering joints.
            wheel_velocities: (4,) velocity targets for all drive joints.
        """
        dc = self._robot_articulation.get_articulation_controller()

        # Steering: position control on front knuckle joints
        for i, joint_name in enumerate(self.config.steering_joints):
            joint_idx = self._get_joint_index(joint_name)
            if joint_idx is not None:
                dc.apply_action(
                    joint_positions={joint_idx: float(wheel_angles[i])}
                )

        # Drive: velocity control on all four wheel joints
        for i, joint_name in enumerate(self.config.drive_joints):
            joint_idx = self._get_joint_index(joint_name)
            if joint_idx is not None:
                dc.apply_action(
                    joint_velocities={joint_idx: float(wheel_velocities[i])}
                )

    def _reset_robot_pose(self):
        """
        Teleport the robot to spawn position and zero all velocities.
        """
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        position = np.array([
            self.config.spawn_x,
            self.config.spawn_y,
            self.config.spawn_z,
        ])

        orientation = euler_angles_to_quat(
            np.array([0.0, 0.0, self.config.spawn_yaw])
        )

        self._robot_articulation.set_world_pose(
            position=position,
            orientation=orientation,
        )

        # Zero all joint velocities
        num_dof = self._robot_articulation.num_dof
        self._robot_articulation.set_joint_velocities(np.zeros(num_dof))
        self._robot_articulation.set_joint_positions(np.zeros(num_dof))

    def _get_robot_position(self) -> np.ndarray:
        """Get robot world position as (3,) float64 array."""
        position, _ = self._robot_articulation.get_world_pose()
        return np.array(position, dtype=np.float64)

    def _get_robot_velocity(self) -> np.ndarray:
        """Get robot linear velocity as (3,) float64 array."""
        linear_vel = self._robot_articulation.get_linear_velocity()
        return np.array(linear_vel, dtype=np.float64)

    def _get_robot_yaw_rate(self) -> float:
        """Get robot angular velocity around Z axis (rad/s)."""
        angular_vel = self._robot_articulation.get_angular_velocity()
        return float(angular_vel[2])  # Z component = yaw rate

    def _get_joint_index(self, joint_name: str) -> Optional[int]:
        """
        Look up joint index by name in the articulation.

        Returns None if joint not found (logs warning on first miss).
        """
        try:
            return self._robot_articulation.get_dof_index(joint_name)
        except Exception:
            logger.warning(f"Joint '{joint_name}' not found in articulation")
            return None

    # ================================================================
    #  Reward
    # ================================================================

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Compute step reward.

        Uses the same reward structure as IsaacROS2Env:
            - Forward progress reward (speed * dt)
            - Lane keeping reward (from lane detector if available)
            - Collision penalty (if detected)

        This is intentionally simple. The reward strategy pattern
        (config.training.reward_strategy) will be implemented when
        the BaseSimEnv abstract class is built.
        """
        telemetry = obs["vec"]
        speed = telemetry[3]
        lateral_error = telemetry[8]

        # Forward progress: reward for moving forward
        reward = speed * (1.0 / self.config.control_hz)

        # Lane keeping: penalize lateral offset
        if abs(lateral_error) > 0.01:
            reward -= 0.5 * abs(lateral_error)

        # Speed penalty: don't go too fast or too slow
        if speed < 0.1:
            reward -= 0.1  # Penalize being stuck
        if speed > self.config.max_speed * 0.9:
            reward -= 0.05  # Gentle penalty near max speed

        return float(reward)

    def _check_termination(self) -> bool:
        """
        Check if episode should terminate (collision, off-track, etc.).

        For now this checks if the robot has flipped or fallen below
        the ground plane, which indicates a physics failure.
        """
        position = self._get_robot_position()

        # Fell through the ground
        if position[2] < -0.5:
            logger.debug("Termination: robot fell below ground")
            return True

        # Flipped over (Z position too high for a ground vehicle)
        if position[2] > 0.5:
            logger.debug("Termination: robot flipped")
            return True

        return False

    # ================================================================
    #  Render
    # ================================================================

    def render(self) -> Optional[np.ndarray]:
        """Return current camera frame for visualization."""
        if self._annotator is not None:
            return self._capture_camera()
        return None
