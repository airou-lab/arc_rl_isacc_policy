#!/usr/bin/env python3
"""
Isaac Sim ROS2 Gym Environment
==============================

Subscribes to:
- /camera/image_raw (sensor_msgs/Image) - Camera RGB
- /vehicle_state (ackermann_msgs/AckermannDriveStamped) - Current vehicle state from controller

Publishes to:
- /ackermann_cmd (ackermann_msgs/AckermannDrive) - Control commands

This uses the vehicle controller's actual outputs rather than full odometry,
maintaining thread safety while keeping data types simple.

Camera: Intel RealSense D435i
    RGB native: 1920x1080 (16:9)
    Depth native: 1280x720 (16:9)
    Downsampled to 160x90 preserving 16:9 aspect ratio.
Author: Aaron
Date: 02/12/26
Updated: 02/17/26
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from cv_bridge import CvBridge
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import time
import cv2

try:
    from lane_detector import SimpleLaneDetector
except ImportError:
    print("Warning: lane_detector not found, lane rewards will be disabled")
    SimpleLaneDetector = None


@dataclass
class IsaacROS2Config:
    """Configuration for Isaac Sim ROS2 Environment"""

    # Image settings
    img_width: int = 160
    img_height: int = 90

    # ROS2 topic names
    camera_topic: str = "/camera/image_raw"
    state_topic: str = "/vehicle_state" # AckermannDriveStamped from controller 
    control_topic: str = "/ackermann_cmd"

    # Control limits
    max_steering_angle: float = 0.5 # radians (+-28.6 degrees)
    max_acceleration: float = 3.0 # m/s^2


    # Episode settings
    episode_timeout: float = 30.0 # seconds
    observation_timeout: float = 1.0 # seconds

    # Goal position (in world frame, for reward only, NOT given to agent)
    goal_x: float = 10.0
    goal_y: float = 0.0
    goal_z: float = 0.0


class IsaacROS2Env(gym.Env):
    """
    Gymnasium environment for Isaac Sim via ROS2.

    Subscribes to vehicle controller outputs:
    - Camera image (RGB, 160x90)
    - Vehicle state (speed, steering from controller)

    Enforces Passive Visual Protocol by masking geometric information from agent observation.
    """

    def __init__(self, config: Optional[IsaacROS2Config] = None):
        super().__init__()

        self.config = config or IsaacROS2Config()

        # Initialize ROS2 node
        if not rclpy.ok():
            rclpy.init()

        self.node = Node('isaac_rl_env')
        self.bridge = CvBridge()

        # Lane detector for visual rewards
        if SimpleLaneDetector is not None:
            self.lane_detector = SimpleLaneDetector(
                img_width=self.config.img_width,
                img_height=self.config.img_height
            )
        else:
            self.lane_detector = None

        # State variables
        self.latest_image: Optional[np.ndarray] = None
        self.current_speed: float = 0.0
        self.current_steering: float = 0.0
        self.current_acceleration: float = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)

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

        self.node.get_logger().info(f"Isaac ROS2 Environment initialized")
        self.node.get_logger().info(f"  Camera: {self.config.camera_topic}")
        self.node.get_logger().info(f"  State: {self.config.state_topic}")
        self.node.get_logger().info(f"  Control: {self.config.control_topic}")

    def _camera_callback(self, msg: Image):
        """Process camera image."""
        print(f"DEBUG: Camera Callback Triggered. Encoding: {msg.encoding}") 
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
            print("DEBUG: Image processed successfully")

        except Exception as e:
            self.node.get_logger().error(f"Camera callback error: {e}")

    def _state_callback(self, msg: AckermannDriveStamped):
        """
        Process vehicle state from AckermannController.

        This is the controller's current commanded state, which reflects the vehicle's
        actual state in simulation.
        """
        self.current_speed = abs(msg.drive.speed)
        self.current_steering = msg.drive.steering_angle
        self.current_acceleration = msg.drive.acceleration

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.

        Returns observation dict with:
        - image: RGB camera image
        - vec: [turn_bias, reserved, goal_dist_masked, speed, yaw_rate,
                last_steer, last_throttle, last_brake, lat_err_zero, hdg_err_zero,
                kappa_zero, total_dist]
        """
        # Spin ROS2 to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.001)

        # Wait for data
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

        # Compute yaw rate from bicycle model
        # yaw_rate = (v / L) * tan(steering_angle)
        # For small angles: yaw_rate squigeq (v / L) * steering_angle
        # Assume wheelbase L squigeq 0.33m for RACECAR
        wheelbase = 0.33 # meters
        if abs(self.current_speed) > 0.01:
            yaw_rate = (self.current_speed / wheelbase) * np.tan(self.current_steering)
        else:
            yaw_rate = 0.0

        # Build vector observation
        # CRITICAL: Position and geometric info are MASKED (set to 0)
        vec = np.array([
            0.0, # turn_bias (set by high-level planner)
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

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Compute reward using ONLY visual information and speed.

        Reward components:
        1. Lane staying (from vision)
        2. Speed reward  (encourage forward motion)
        3. Smoothness penalties
        """
        reward = 0.0

        # 1. Lane staying reward (vision-based)
        if self.lane_detector is not None:
            lane_result = self.lane_detector.detect(obs['image'])
            if lane_result.in_lane:
                reward += 1 # In lane
            else:
                reward -= abs(lane_result.lateral_offset) * 2.0 # Penalize deviation

        # 2. Speed reward (encourage forward motion)
        speed = obs['vec'][3]
        reward += speed * 0.3

        # 3. Smoothness penalties
        steer = self.last_action[0]
        reward -= abs(steer) * 0.1 # Penalize large steering

        yaw_rate = obs['vec'][4]
        reward -= abs(yaw_rate) * 0.2 # Penalize spinning

        brake = self.last_action[2]
        reward -= brake * 0.1 # Prefer throttle over brake

        return reward

    def _check_termination(self, obs: Dict[str, np.ndarray]) -> Tuple[bool, bool, Dict]:
        """
        Check if episode should terminate.

        Termination conditions:
        1. Timeout
        2. Off track (vision-based)
        3. Stuck (low speed for too long)
        """
        info = {}
        done = False
        truncated = False

        # Check timeout
        elapsed = time.time() - self.episode_start_time
        if elapsed > self.config.episode_timeout:
            truncated = True
            info['termination_reason'] = 'timeout'

        # Check if off track (vision-based)
        if self.lane_detector is not None:
            lane_result = self.lane_detector.detect(obs['image'])
            if abs(lane_result.lateral_offset) > 0.8:
                done = True
                info['termination_reason'] = 'off_track'

        # Check if stuck
        speed = obs['vec'][3]
        if speed < 0.15:
            self.stuck_timer += (time.time() - self.last_update_time)
            if self.stuck_timer > 5.0:
                done = True
                info['termination_reason'] = 'stuck'
        else:
            self.stuck_timer = 0.0

        self.last_update_time = time.time()

        return done, truncated, info


    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset state
        self.latest_image = None
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_acceleration = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.episode_start_time = time.time()
        self.total_distance = 0.0
        self.last_update_time = time.time()
        self.stuck_timer = 0.0

        # Send zero command
        cmd = AckermannDrive()
        cmd.steering_angle = 0.0
        cmd.speed = 0.0
        self.control_pub.publish(cmd)

        # Get initial observation
        obs = self._get_observation()

        self.node.get_logger().info("Episode reset | Vision-based navigation")

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return observation, reward, done, truncated, info."""
        # Store action
        self.last_action = action.copy()

        # Convert to Ackermann command
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])

        # Create Ackermann message
        cmd = AckermannDrive()
        cmd.steering_angle = steer
        cmd.speed = (throttle - brake) * 3.0 # Map throttle to target speed for now
        cmd.acceleration = (throttle - brake) * self.config.max_acceleration

        # Publish command
        self.control_pub.publish(cmd)

        # Get observation
        obs = self._get_observation()

        # Update total distance
        dt = time.time() - self.last_update_time
        self.total_distance += self.current_speed * dt

        # Compute reward
        reward = self._compute_reward(obs)

        # Check termination
        done, truncated, info = self._check_termination(obs)

        # Add metrics to info
        speed = obs['vec'][3]
        info['speed'] = speed
        info['total_distance'] = self.total_distance

        if self.lane_detector is not None:
            lane_result = self.lane_detector.detect(obs['image'])
            info['in_lane'] = lane_result.in_lane
            info['lateral_offset'] = lane_result.lateral_offset

        return obs, reward, done, truncated, info


    def close(self):
        """Cleanup."""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# Test code
if __name__ == "__main__":
    print("Testing Isaac ROS2 Env...")

    config = IsaacROS2Config()
    env = IsaacROS2Env(config=config)

    print("Testing environment reset...")
    obs, info = env.reset()
    print(f"Observation space: image {obs['image'].shape}, vec {obs['vec'].shape}")

    print("\nRunning random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}, truncated={truncated}")

        if done or truncated:
            print(f"Episode end | Steps: {i} | Total dist: {info['total_distance']:.2f}m")
            break

    env.close()
    print("\nTest complete!")


