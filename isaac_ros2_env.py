#!/usr/bin/env python3
"""
Isaac Sim ROS2 Gym Environment
==============================
PURPOSE: Physical ARCPro deployment and live Isaac Sim integration tests.
         NOT for training - use isaac_direct_env.py for that

         ROS2 middleware introduces latency, serialization jitter, and queue saturation that undermines sim-to-real
         transfer if used during training. This file belongs in the deployment path only.

Subscribes to:
- /camera/image_raw     (sensor_msgs/Image) - Camera RGB
- /vehicle_state        (ackermann_msgs/AckermannDriveStamped) - Current vehicle state from controller
- /imu                  (sensor_msgs/Imu) - Gyro yaw rate (angular_velocity.z)

Publishes to:
- /ackermann_cmd (ackermann_msgs/AckermannDrive) - Control commands

Camera: Intel RealSense D435i
    RGB native: 1920x1080 (16:9)
    Depth native: 1280x720 (16:9)
    Downsampled to 160x90 preserving 16:9 aspect ratio.

Observation vector (12 floats) - must match hierarchical_policy.py IDX_* constants:
    [0]  turn_bias     - High-level navigation command [-1, 1] (set via set_turn_bias())
    [1]  reserved      - Always 0.0
    [2]  goal_dist     - Always 0.0 (masked - Passive Visual Protocol)
    [3]  speed         - Vehicle speed (m/s)
    [4]  yaw_rate      - IMU angular_velocity.z (rad/s); bicycle model fallback at startup)
    [5]  last_steer    - Previous steer action
    [6]  last_throttle - Previous throttle action
    [7]  last_brake    - Previous brake action
    [8]  lat_err       - Always 0.0 (masked - PVP)
    [9]  hdg_err       - Always 0.0 (masked - PVP)
    [10] kappa         - Always 0.0 (masked - PVP)
    [11] total_dist    - Cumulative odometry (m)

Passive Visual Protocol:
    Slots [2], [8], [9], [10] are permanently zeroed. The agent must navigate using camera images and turn_bias only,
    no GPS, no lane geometry (yet), no path curvature (yet). This the the protocol's core constraint and must not be violated here
    or in isaac_direct_env.py

Author: Aaron
Date: 02/12/26
Updated: 03/10/26
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, Imu
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from cv_bridge import CvBridge
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import time
import cv2
import logging

logger = logging.getLogger(__name__)

try:
    from lane_detector import SimpleLaneDetector
except ImportError:
    logger.warning("lane_detector not found, lane rewards will be disabled")
    SimpleLaneDetector = None


@dataclass
class IsaacROS2Config:
    """Configuration for Isaac Sim ROS2 Environment"""

    # Image settings - RealSence D435i downsampled to 160x90 (16:9)
    img_width: int = 160
    img_height: int = 90

    # ROS2 topic names
    camera_topic: str = "/camera/image_raw"
    state_topic: str = "/vehicle_state" # AckermannDriveStamped from controller
    control_topic: str = "/ackermann_cmd"
    # FIX (imu): Added IMU topic. Yaw rate now sourced from gyro (angular_velocity.z) instead of the bicycle model
    # estimate. Odom is intentionally excluded - no position data is stored or given to the agent (Passive Visual Protocol).
    # EKF fusion of odom+IMU is deferred - easy to add later by replacing this subscriber with an /odometry/filtered
    # subscription without touching anything else.
    imu_topic: str = "/imu"

    # Control limits
    max_steering_angle: float = 0.5 # radians (+-28.6 degrees)
    max_acceleration: float = 3.0 # m/s^2


    # Episode settings
    episode_timeout: float = 30.0 # seconds
    observation_timeout: float = 1.0 # seconds - max wait for camera frame

    # Goal position (in world frame, for reward only, NOT given to agent)
    goal_x: float = 10.0
    goal_y: float = 0.0
    goal_z: float = 0.0

    # Vehicle parameters - ARCPro / F1TENTH
    wheelbase:float = 0.33 # meters, RACECAR wheelbase for bicycle model


class IsaacROS2Env(gym.Env):
    """
    Gymnasium environment wrapping the physical ARCPro robot via ROS2.

    Subscribes to vehicle controller outputs:
    - Camera image (RGB, 160x90) from Intel RealSense D435i
    - Vehicle state (speed, steering) from AckermannController

    Enforces Passive Visual Protocol: geometric slots [2,8,9,10] are always zero so the policy cannot access GPS,
    lateral error, heading error, or path curvature. Navigation is driven by camera images and turn_bias only.

    turn_bias must be set externally via set_turn_bias() before each episode by the route planner or training curriculum.
    """

    def __init__(self, config: Optional[IsaacROS2Config] = None):
        super().__init__()

        self.config = config or IsaacROS2Config()

        # Initialize ROS2 node
        if not rclpy.ok():
            rclpy.init()

        self.node = Node('isaac_rl_env')
        self.bridge = CvBridge()

        # Lane detector - used for reward computation only, never for observations
        if SimpleLaneDetector is not None:
            self.lane_detector = SimpleLaneDetector(
                img_width=self.config.img_width,
                img_height=self.config.img_height
            )
        else:
            self.lane_detector = None

        # Vehicle state (populated by ROS2 callbacks)
        self.latest_image: Optional[np.ndarray] = None
        self.current_speed: float = 0.0
        self.current_steering: float = 0.0
        self.current_acceleration: float = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        # FIX (imu): Real yaw rate from IMU gyro. Initialized to None so _get_observation() knows to fall back to the
        # bicycle model until the first IMU message arrives.
        self.current_yaw_rate: Optional[float] = None

        # Navigation command - must be set externally by high-level planner or randomized during
        # randomized during training curriculum to exercise kinematic anchors
        self.turn_bias: float = 0.0

        # Episode tracking
        self.episode_start_time: float = 0.0
        self.total_distance: float = 0.0
        self.last_update_time: float = 0.0
        self.stuck_timer: float = 0.0

        # Define QoS for sensor data (Best Effort + Volatile usually works for high freq)
        # But Isaac Sim reported RELIABLE. Let's try matching exactly or being permissive.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ROS2 subscribers
        self.camera_sub = self.node.create_subscription(
            Image,
            self.config.camera_topic,
            self._camera_callback,
            qos
        )

        self.state_sub = self.node.create_subscription(
            AckermannDriveStamped,
            self.config.state_topic,
            self._state_callback,
            10
        )
        # FIX(imu)
        self.imu_sub = self.node.create_subscription(
            Imu,
            self.config.imu_topic,
            self._imu_callback,
            qos,
        )

        # ROS2 publisher for control
        self.control_pub = self.node.create_publisher(
            AckermannDrive,
            self.config.control_topic,
            10
        )

        # Define observation space
        # Image: RGB (H, W, 3)
        # Vec: [turn_bias, reserved, goal_dist_masked, speed, yaw_rate, last_steer,
        #       last_throttle, last_brake, lat_err_zero, hdg_err_zero, kappa_zero, total_dist]
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8
            ),
            "vec": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32
            )
        })

        # Define action space: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.node.get_logger().info(f"Isaac ROS2 Environment initialized (ARCPro deployment)")
        self.node.get_logger().info(f"  Camera: {self.config.camera_topic}")
        self.node.get_logger().info(f"  State: {self.config.state_topic}")
        self.node.get_logger().info(f"  IMU: {self.config.imu_topic}")
        self.node.get_logger().info(f"  Control: {self.config.control_topic}")

    def set_turn_bias(self, bias: float) -> None:
        """
        Set the high-level navigation command.

        This should be called by a route planner, waypoint graph, or randomized curriculum to tell the agent which direction
        to go at intersections.

        Args:
            bias: Continuous turn command [-1, 1].
                -1 = hard left, 0 = straight, 1 = hard right.
        """
        self.turn_bias = float(np.clip(bias, -1.0, 1.0))

    def _camera_callback(self, msg: Image) -> None:
        """Receive camera frame from RealSense D435i and store as numpy RGB.."""
        try:
            # Convert ROS Image to numpy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Resize if needed
            if cv_image.shape[:2] != (self.config.img_height, self.config.img_width):
                cv_image = cv2.resize(
                    cv_image,
                    (self.config.img_width, self.config.img_height)
                )

            self.latest_image = cv_image

        except Exception as e:
            self.node.get_logger().error(f"Camera callback error: {e}")

    def _state_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Process vehicle state from AckermannController.

        This is the controller's current commanded state, which reflects the vehicle's
        actual state in simulation.
        """
        self.current_speed = abs(msg.drive.speed)
        self.current_steering = msg.drive.steering_angle
        self.current_acceleration = msg.drive.acceleration

    def _imu_callback(self, msg: Imu) -> None:
        """Receive IMU data and store yaw rate from gyro z-axis."""
        self.current_yaw_rate = float (msg.angular_velocity.z)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Builds the observation dict.

        Spins ROS2 to flush pending callbacks, then assembled the 12-float telemetry vector.
        Passive Visual Protocol slots are permanently zero.

        Returns dict with keys:
        - "image": (90, 160, 3) uint8 RGB
        - "vec": (12,) float32 telemetry
        """
        # Spin ROS2 to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.001)

        # Wait for first camera frame (up to observation_timeout)
        timeout = time.time() + self.config.observation_timeout
        while self.latest_image is None and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        if self.latest_image is None:
            self.node.get_logger().warn("Observation timeout!")
            # Return zero observation
            image = np.zeros(
                (self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8
            )
        else:
            image = self.latest_image.copy()

        # FIX(imu): Use real IMU yaw rate when available; fall back to bicycle model only until the first IMU message
        # arrives. Once self.current_yaw_rate is populated and stays populated, so the fallback is only active at startup.
        # When EKF fusion is added later, replace self.current_yaw_rate with the filtered odometry angular velocity and
        # remove the bicycle model fallback.
        if self.current_yaw_rate is not None:
            yaw_rate = self.current_yaw_rate
        elif abs(self.current_speed) > 0.01:
            # Bicycle model fallback
            yaw_rate = (self.current_speed / self.config.wheelbase) * np.tan(self.current_steering)
        else:
            yaw_rate = 0.0

        # Build vector observation
        # CRITICAL: Position and geometric info are MASKED (set to 0)
        vec = np.array([
            self.turn_bias, # IDX_TURN_BIAS - FIX: was hardcoded 0.0
            0.0, # reserved
            0.0, # goal_dist MASKED - not given to agent!
            self.current_speed,
            yaw_rate,
            self.last_action[0], # last_steer
            self.last_action[1], # last_throttle
            self.last_action[2], # last_brake
            0.0, # lat_err - NOT GIVEN, using vision!
            0.0, # hdg_err - NOT GIVEN, using vision!
            0.0, # kappa - NOT GIVEN, using vision!
            self.total_distance
        ], dtype=np.float32)

        return {'image': image, 'vec': vec}

    def _detect_lane(self, image: np.ndarray):
        """
        Run lane detection once per step and return the result.

        Called once in step() and the result is shared between _compute_reward() and _check_termination()
        to avoid duplicate CV computation.

        The result is NEVER written into the observation vector - that would violate the Passive Visual Protocol.

        Returns:
            Lane detection result, or None if lane detector is unavailable.
        """
        if self.lane_detector is not None:
            return self.lane_detector.detect(image)
        return None

    def _compute_reward(self, obs: Dict[str, np.ndarray], lane_result=None) -> float:
        """
        Compute reward using ONLY visual information and speed.

        Reward components:
        0. Speed gate - stationary penalty overrides everything else
        1. Lane staying bonus/penalty (vision-based, from lane_result)
        2. Speed reward  (encourage forward motion)
        3. Smoothness penalties (steering magnitude, yaw rate, braking)

        Args:
            obs: Current observation dict.
            lane_result: Pre-computed lane detection result from _detect_lane().

        NOTE:
            Without speed gate, a vehicle parked perfectly in-lane scores +1.0/step indefinetly. PPO will find this
            and learn to park. The gate forces movement as a prerequisite for any positive reward
        """
        speed = obs["vec"][3]

        # 0. Speed gate - must be moving to earn any reward
        if speed < 0.1:
            return -1.0

        reward = 0.0

        # 1. Lane staying reward (vision-based)
        if lane_result is not None:
            if lane_result.in_lane:
                reward += 1.0 # In lane
            else:
                reward -= abs(lane_result.lateral_offset) * 2.0 # Penalize deviation

        # 2. Speed reward
        reward += speed * 0.3

        # 3. Smoothness penalties
        reward -= abs(self.last_action[0]) * 0.1 # steering magnitude
        reward -= abs(obs["vec"][4]) * 0.2 # yaw rate
        reward -= self.last_action[2] * 0.1 # braking

        return float(reward)

    def _check_termination(self, obs: Dict[str, np.ndarray], lane_result=None) -> Tuple[bool, bool, Dict]:
        """
        Check if episode should terminate.

        Conditions:
            terminated: Off-track(vision-based)
                        Stuck (no movement for 5 seconds)
            truncated:  Episode timeout

        Args:
            obs: Current observation dict.
            lane_result: Pre-computed lane detection result from _detect_lane().

        Returns:
            (terminated, truncated, info)
        """
        info = {}
        terminated = False
        truncated = False

        # Check timeout
        elapsed = time.time() - self.episode_start_time
        if elapsed > self.config.episode_timeout:
            truncated = True
            info['termination_reason'] = 'timeout'

        # Check if off track (vision-based)
        if lane_result is not None and abs(lane_result.lateral_offset) > 0.8:
            terminated = True
            info["termination_reason"] = "off_track"

        # Check if stuck - wall-clock dt works here (deployment context)
        speed = obs['vec'][3]
        dt = time.time() - self.last_update_time
        if speed < 0.15:
            self.stuck_timer += dt
            if self.stuck_timer > 5.0:
                terminated = True
                info['termination_reason'] = 'stuck'
        else:
            self.stuck_timer = 0.0

        self.last_update_time = time.time()

        return terminated, truncated, info


    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.

        NOTE: turn_bias is intentionally NOT reset here. The route planner or training curriculum owns turn_bias and
        sets it via set_turn_bias() before calling reset() for each episode.
        """
        super().reset(seed=seed)

        # Reset state
        self.latest_image = None
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_acceleration = 0.0
        self.current_yaw_rate = None # FIX(imu): reset so fallback activates until first IMU msg
        self.last_action = np.zeros(3, dtype=np.float32)
        self.episode_start_time = time.time()
        self.total_distance = 0.0
        self.last_update_time = time.time()
        self.stuck_timer = 0.0

        # Note: turn_bias is NOT reset here - it should be set externally by the training curriculum
        #       or route planner before each episode.

        # Send zero command
        cmd = AckermannDrive()
        cmd.steering_angle = 0.0
        cmd.speed = 0.0
        self.control_pub.publish(cmd)

        # Get initial observation
        obs = self._get_observation()

        self.node.get_logger().info(f"Episode reset | turn_bias={self.turn_bias:+.2f} | Vision-based navigation")

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one control step.

        Args:
            action: [steer, throttle, brake] in [-1,1], [0,1], [0,1]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Store action
        self.last_action = action.copy()

        # Build and publish Ackermann command
        cmd = AckermannDrive()
        cmd.steering_angle = float(action[0])
        cmd.speed = (float(action[1]) - float(action[2])) * 3.0
        cmd.acceleration = (float(action[1]) - float(action[2])) * self.config.max_acceleration

        # Publish command
        self.control_pub.publish(cmd)

        # Get observation
        obs = self._get_observation()

        # Odometry - dt measured from last termination check to now
        # NOTE: includes ROS2 spin time
        dt = time.time() - self.last_update_time
        self.total_distance += self.current_speed * dt

        # Run lane deteciton ONCE and share the result
        lane_result = self._detect_lane(obs['image'])

        # Compute reward (using shared lane result)
        reward = self._compute_reward(obs, lane_result=lane_result)

        # Check termination (using shared lane result)
        terminated, truncated, info = self._check_termination(obs, lane_result=lane_result)

        # Add metrics to info
        info["speed"] = float(obs["vec"][3])
        info["total_distance"] = self.total_distance
        info["turn_bias"] = self.turn_bias
        if lane_result is not None:
            info['in_lane'] = lane_result.in_lane
            info['lateral_offset'] = lane_result.lateral_offset

        return obs, reward, terminated, truncated, info


    def close(self):
        """Cleanup."""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# Smoke Test (requires live ROS2 + ARCPro or Isaac Sim running)
if __name__ == "__main__":
    print("Testing IsaacROS2Env (requires live ROS2 environment)...")

    config = IsaacROS2Config()
    env = IsaacROS2Env(config=config)

    env.set_turn_bias(0.0)
    obs, info = env.reset()
    print(f"Reset OK | image: {obs['image'].shape} | vec: {obs['vec'].shape}")
    print(f"  turn_bias in vec[0]: {obs['vec'][0]:.2f} (should match set_turn_bias)")
    print(f"  yaw_rate in vec[4]: {obs['vec'][4]:.4f} (IMU if connected, bicycle model fallback)")
    print(f"  Passive Visual Protocol slots [2,8,9,10]: "
          f"{obs['vec'][2]:.1f}, {obs['vec'][8]:.1f}, {obs['vec'][9]:.1f}, {obs['vec'][10]:.1f}"
          f"  (all should be 0.0)")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f} | terminated={terminated} | truncated={truncated}")
        if terminated or truncated:
            print(f"  End reason: {info.get('termination_reason', 'unknown')}")
            break

    env.close()
    print("Test complete.")
