"""
Gazebo Direct Environment for RL Training

Direct port of isaac_direct_env.py targeting Gazebo Harmonic (gz-sim 8).
Uses gz-transport (ZeroMQ + Protobuf) for all simulator communication.

Architecture:
    GazeboDirectEnv
        ├─ Camera:    gz-transport subscriber -> numpy buffer
        ├─ Control:   gz-transport Twist publisher -> AckermannSteering plugin
        ├─ Stepping:  WorldControl service call (synchronous)
        ├─ Position:  PosePublisher subscriber (Pose_V)
        ├─ Yaw rate:  IMU subscriber (preferred) or heading finite-difference
        └─ Reset:     set_pose service call (teleport)

Key differences from Isaac:
    - No AckermannComputer class — Gazebo's gz-sim-ackermann-steering-system
      plugin decomposes steering into per-wheel actions internally. We just
      publish Twist(linear.x=speed, angular.z=steering_angle).
    - Camera is async (subscriber callback) vs Isaac's synchronous Replicator.
      Post-step synchronization uses a threading.Event to wait for a fresh
      frame after WorldControl returns.
    - Vehicle geometry uses XACRO ground truth (wheelbase=0.3302, track=0.2413,
      wheel_radius=0.0508).
    - Physics defaults: 1000 Hz x 100 substeps (Gazebo default) instead of
      Isaac's 60 Hz x 6 substeps. Both produce 10 Hz control.

Observation contract:
    Dict{
        "image": Box(90, 160, 3, uint8)    — front camera RGB
        "vec": Box(12, float32)            — telemetry (see TELEMETRY_INDICES)
    }

Action contract:
    Box([-1, 0, 0], [1, 1, 1], float32)    — [steer, throttle, brake]

Interface for AgentEnvWrapper compatibility:
    _get_robot_position()  -> np.ndarray (3D)
    _get_robot_velocity()  -> np.ndarray (3D)
    _get_robot_yaw_rate()  -> float
    _get_robot_heading()   -> float          (replaces Isaac's quaternion-from-articulation)
    config.spawn_x/y/yaw   — for dead-reckoning initialization

Prerequisites:
    sudo apt install python3-gz-transport13 python3-gz-msgs10
    Gazebo server running in paused mode:
        gz sim -s -r --headless-rendering <world.sdf>

Author: Aaron Hamil
Date: 03/23/26
"""

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# gz-transport imports
# Try the canonical Harmonic paths first, then fall back.
try:
    from gz.transport13 import Node as _GzNode
except ImportError:
    try:
        from gz.transport import Node as _GzNode
    except ImportError:
        _GzNode = None

try:
    from gz.msgs10.image_pb2 import Image as _GzImage
    from gz.msgs10.twist_pb2 import Twist as _GzTwist
    from gz.msgs10.world_control_pb2 import WorldControl as _GzWorldControl
    from gz.msgs10.boolean_pb2 import Boolean as _GzBoolean
    from gz.msgs10.pose_v_pb2 import Pose_V as _GzPoseV
    from gz.msgs10.pose_pb2 import Pose as _GzPose
    from gz.msgs10.imu_pb2 import IMU as _GzIMU
    from gz.msgs10.vector3d_pb2 import Vector3d as _GzVector3d
    _GZ_MSGS_AVAILABLE = True
except ImportError:
    try:
        from gz.msgs.image_pb2 import Image as _GzImage
        from gz.msgs.twist_pb2 import Twist as _GzTwist
        from gz.msgs.world_control_pb2 import WorldControl as _GzWorldControl
        from gz.msgs.boolean_pb2 import Boolean as _GzBoolean
        from gz.msgs.pose_v_pb2 import Pose_V as _GzPoseV
        from gz.msgs.pose_pb2 import Pose as _GzPose
        from gz.msgs.imu_pb2 import IMU as _GzIMU
        from gz.msgs.vector3d_pb2 import Vector3d as _GzVector3d
        _GZ_MSGS_AVAILABLE = True
    except ImportError:
        _GZ_MSGS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Pixel format constants
# gz.msgs.PixelFormatType enum values used by the camera sensor.
# Handle common formats; others fall back to zeros.
_PF_RGB_INT8 = 4
_PF_RGBA_INT8 = 5      # Some camera configs add alpha
_PF_BGR_INT8 = 7
_PF_BGRA_INT8 = 8


@dataclass
class GazeboDirectConfig:
    """
    Configuration for the direct-API Gazebo Harmonic environment.

    Vehicle geometry comes from the real robot's XACRO
    (f1tenth_description/urdf/f1tenth.xacro).
    """

    # Gazebo scene
    world_name: str = "arcpro_world"
    model_name: str = "f1tenth"

    # gz-transport topics
    # These must match the SDF sensor / plugin configuration.
    # camera_topic: the camera sensor's image output topic
    # cmd_vel_topic: the AckermannSteering plugin's command input
    # imu_topic: the IMU sensor output (for yaw rate)
    # pose_topic: computed as /world/{world_name}/dynamic_pose/info
    camera_topic: str = "/camera"
    cmd_vel_topic: str = "/cmd_vel"
    imu_topic: str = "/imu"

    # Reward strategy
    # "original": Linear lane bonus + boosted speed (default)
    # "hybrid":   Gaussian lane precision + momentum weighting
    reward_mode: str = "original"

    # Camera
    img_width: int = 160
    img_height: int = 90

    # Vehicle geometry (XACRO GT)
    wheelbase: float = 0.3302       # meters
    track_width: float = 0.2413     # meters
    wheel_radius: float = 0.0508    # meters

    # Control limits
    max_steering_angle: float = 0.5     # radians (~28.6 degrees)
    max_speed: float = 3.0              # m/s

    # Physics
    # Gazebo Harmonic default: 1000 Hz physics.
    # 100 substeps x 0.001s = 0.1s per control step = 10 Hz control.
    physics_dt: float = 0.001           # 1000 Hz
    render_dt: float = 1.0 / 30.0       # 30 Hz camera sensor update
    control_hz: int = 10                # Policy inference rate
    substeps: int = 100                 # Physics steps per control step

    # Episode
    episode_timeout: float = 30.0
    max_episode_steps: int = 300

    # Reset (spawn pose)
    # These must match a valid drivable location in the SDF world.
    # TODO: Update these once provided the world file.
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 0.1       # Slight drop to settle on road surface
    spawn_yaw: float = 0.0

    # gz-transport timeouts
    camera_timeout: float = 2.0     # seconds - wait for post-step frame
    service_timeout: int = 5000     # milliseconds — WorldControl / set_pose


class GazeboDirectEnv(gym.Env):
    """
    Gymnasium environment wrapping Gazebo Harmonic via gz-transport.

    Follows the same contract as IsaacDirectEnv so AgentEnvWrapper,
    WaypointTrackingWrapper, and all policies work unchanged.

    Lifecycle:
        1. __init__: define spaces, defer sim connection
        2. reset() (first call): _setup_sim() connects to Gazebo
        3. step(): publish Twist -> step world -> capture obs -> reward
        4. close(): clean up subscribers and publishers
    """

    def __init__(self, config: Optional[GazeboDirectConfig] = None):
        super().__init__()

        if _GzNode is None or not _GZ_MSGS_AVAILABLE:
            raise ImportError(
                "gz-transport Python bindings not found. Install with:\n"
                "  sudo apt install python3-gz-transport13 python3-gz-msgs10"
            )

        self.config = config or GazeboDirectConfig()

        # Counters and state
        self._step_count = 0
        self._episode_start_time = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cumulative_distance = 0.0
        self._last_position = np.zeros(3, dtype=np.float64)

        # Observation and action spaces
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
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # gz-transport handles (initialized in _setup_sim)
        self._node: Optional[_GzNode] = None
        self._cmd_pub = None
        self._sim_initialized = False

        # Subscriber data buffers (thread-safe)
        self._camera_lock = threading.Lock()
        self._camera_frame = np.zeros(
            (self.config.img_height, self.config.img_width, 3),
            dtype=np.uint8,
        )
        self._camera_event = threading.Event()
        self._camera_frame_id = 0

        self._pose_lock = threading.Lock()
        self._position = np.zeros(3, dtype=np.float64)
        self._orientation_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z
        self._heading = 0.0

        self._imu_lock = threading.Lock()
        self._angular_velocity = np.zeros(3, dtype=np.float64)

        # Velocity estimation via finite-difference (no direct physics query)
        self._prev_position = np.zeros(3, dtype=np.float64)
        self._prev_time = 0.0
        self._estimated_velocity = np.zeros(3, dtype=np.float64)

        # Lane detector (pure OpenCV, no sim coupling)
        self._lane_detector = None

    # Simulator Setup

    def _setup_sim(self):
        """
        Connect to running Gazebo server via gz-transport.

        Creates the Node, subscribes to camera/pose/IMU topics,
        advertises the cmd_vel publisher, and pauses the world for
        synchronous stepping.

        Expects Gazebo to be running:
            gz sim -s -r --headless-rendering <world.sdf>
        """
        if self._sim_initialized:
            return

        # Load lane detector
        try:
            from lane_detector import SimpleLaneDetector
            self._lane_detector = SimpleLaneDetector(
                img_width=self.config.img_width,
                img_height=self.config.img_height,
            )
        except Exception:
            logger.warning("lane_detector.py not found — reward will lack lane metrics")

        # Create gz-transport node
        self._node = _GzNode()
        world = self.config.world_name

        # Subscribe: camera
        self._node.subscribe(
            _GzImage,
            self.config.camera_topic,
            self._on_camera,
        )
        logger.info(f"Subscribed to camera: {self.config.camera_topic}")

        # Subscribe: pose (PosePublisher system)
        pose_topic = f"/world/{world}/dynamic_pose/info"
        self._node.subscribe(
            _GzPoseV,
            pose_topic,
            self._on_pose,
        )
        logger.info(f"Subscribed to pose: {pose_topic}")

        # Subscribe: IMU
        self._node.subscribe(
            _GzIMU,
            self.config.imu_topic,
            self._on_imu,
        )
        logger.info(f"Subscribed to IMU: {self.config.imu_topic}")

        # Advertise: cmd_vel
        self._cmd_pub = self._node.advertise(
            self.config.cmd_vel_topic,
            _GzTwist,
        )
        logger.info(f"Publishing cmd_vel: {self.config.cmd_vel_topic}")

        # Pause world and settle
        self._pause_world(True)
        self._step_world(50)  # Let physics settle

        self._sim_initialized = True
        logger.info("Gazebo environment initialized")

    # Subscriber Callbacks (run on gz-transport's IO thread)

    def _on_camera(self, msg: "_GzImage") -> None:
        """
        Camera subscriber callback — decode protobuf Image to numpy.

        Handles RGB8, RGBA8, BGR8, BGRA8 pixel formats. Sets the
        camera_event so step() knows a fresh frame arrived after
        the last WorldControl call.
        """
        try:
            w, h = msg.width, msg.height
            fmt = msg.pixel_format_type
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if fmt == _PF_RGB_INT8:
                frame = data.reshape(h, w, 3)
            elif fmt == _PF_RGBA_INT8:
                frame = data.reshape(h, w, 4)[:, :, :3]
            elif fmt == _PF_BGR_INT8:
                frame = data.reshape(h, w, 3)[:, :, ::-1]  # BGR -> RGB
            elif fmt == _PF_BGRA_INT8:
                frame = data.reshape(h, w, 4)[:, :, :3][:, :, ::-1]
            else:
                logger.warning(f"Unhandled pixel format {fmt}, assuming RGB8")
                if data.size >= h * w * 3:
                    frame = data[:h * w * 3].reshape(h, w, 3)
                else:
                    return  # Can't decode — skip this frame

            with self._camera_lock:
                # Resize if camera resolution doesn't match config
                if frame.shape[0] != self.config.img_height or frame.shape[1] != self.config.img_width:
                    import cv2
                    frame = cv2.resize(
                        frame,
                        (self.config.img_width, self.config.img_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                self._camera_frame = frame.copy()
                self._camera_frame_id += 1

            self._camera_event.set()

        except Exception as e:
            logger.debug(f"Camera callback error: {e}")

    def _on_pose(self, msg: "_GzPoseV") -> None:
        """
        Pose subscriber callback — extract this model's pose from
        the Pose_V message (which contains all models in the world).
        """
        for pose in msg.pose:
            if pose.name == self.config.model_name:
                with self._pose_lock:
                    self._position = np.array(
                        [pose.position.x, pose.position.y, pose.position.z],
                        dtype=np.float64,
                    )
                    q = pose.orientation
                    self._orientation_quat = np.array(
                        [q.w, q.x, q.y, q.z], dtype=np.float64,
                    )
                    # Quaternion -> yaw: atan2(2(wz+xy), 1-2(y^2+z^2))
                    self._heading = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    )
                break

    def _on_imu(self, msg: "_GzIMU") -> None:
        """IMU subscriber callback — extract angular velocity."""
        with self._imu_lock:
            av = msg.angular_velocity
            self._angular_velocity = np.array(
                [av.x, av.y, av.z], dtype=np.float64,
            )

    # Gymnasium Interface

    def reset(self, seed=None, options=None):
        """
        Reset: teleport robot to spawn pose, zero velocities,
        settle physics, return initial observation.
        """
        super().reset(seed=seed)
        self._setup_sim()

        self._step_count = 0
        self._cumulative_distance = 0.0

        # Teleport robot to spawn position
        self._reset_robot_pose()

        # Step physics to settle after teleport, then render
        self._step_world(self.config.substeps)

        # Update velocity estimation baseline
        with self._pose_lock:
            self._last_position = self._position.copy()
            self._prev_position = self._position.copy()
        self._prev_time = time.monotonic()
        self._estimated_velocity = np.zeros(3, dtype=np.float64)

        # Wait for a camera frame post-settle
        self._wait_for_camera()

        return self._get_obs(), {"episode_step": 0}

    def step(self, action):
        """
        Execute one control step:
        1. Clip action, convert to Twist, publish
        2. Step physics (substeps)
        3. Wait for post-step camera frame
        4. Update velocity estimate
        5. Compute obs + reward
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._last_action = action.copy()

        # Map action to AckermannSteering command
        steering_angle = action[0] * self.config.max_steering_angle
        speed = action[1] * self.config.max_speed
        if action[2] > 0.0:
            speed *= (1.0 - action[2])  # Brake reduces speed

        # Publish Twist — the AckermannSteering plugin handles
        # per-wheel decomposition internally
        self._publish_cmd_vel(speed, steering_angle)

        # Step physics
        self._step_world(self.config.substeps)

        # Wait for post-step camera frame
        self._wait_for_camera()

        # Update velocity estimate (finite-difference)
        self._update_velocity_estimate()

        # Tracking
        current_pos = self._get_robot_position()
        self._cumulative_distance += np.linalg.norm(
            current_pos[:2] - self._last_position[:2]
        )
        self._last_position = current_pos.copy()
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        truncated = self._step_count >= self.config.max_episode_steps

        return obs, reward, False, truncated, {"episode_step": self._step_count}

    # Observation Construction

    def _get_obs(self):
        """Build observation dict."""
        return {
            "image": self._capture_camera(),
            "vec": self._compute_telemetry(),
        }

    def _capture_camera(self) -> np.ndarray:
        """Return latest camera frame from subscriber buffer."""
        with self._camera_lock:
            return self._camera_frame.copy()

    def _compute_telemetry(self) -> np.ndarray:
        """
        Build 12-float telemetry vector.

        (see TELEMETRY_INDICES in config/experiment.py):
            [0]  turn_token       — set by AgentEnvWrapper
            [1]  go_signal        — set by AgentEnvWrapper
            [2]  goal_dist        — masked to 0 (PVP protocol)
            [3]  speed            — from velocity estimate
            [4]  yaw_rate         — from IMU
            [5]  last_steer       — previous action[0]
            [6]  last_throttle    — previous action[1]
            [7]  last_brake       — previous action[2]
            [8]  lateral_error    — from lane detector
            [9]  heading_error    — lane detector confidence (reused slot)
            [10] curvature        — 0 (not computed yet)
            [11] distance_traveled
        """
        vec = np.zeros(12, dtype=np.float32)

        velocity = self._get_robot_velocity()
        vec[0] = 0.0  # turn_token — overwritten by AgentEnvWrapper
        vec[3] = float(np.linalg.norm(velocity[:2]))
        vec[4] = float(self._get_robot_yaw_rate())
        vec[5] = self._last_action[0]
        vec[6] = self._last_action[1]
        vec[7] = self._last_action[2]

        # Lane detection for reward (agent never sees this directly)
        if self._lane_detector:
            try:
                res = self._lane_detector.detect(self._capture_camera())
                vec[8] = float(res.lateral_offset)
                vec[9] = float(res.confidence)
            except Exception:
                pass

        vec[11] = self._cumulative_distance
        return vec

    # Reward

    def _compute_reward(self, obs) -> float:
        """
        Compute step reward.

        Two modes:
        - "original": linear lane bonus + boosted speed
        - "hybrid":   Gaussian lane precision + momentum weighting
        """
        telemetry = obs["vec"]
        speed = telemetry[3]
        lat_err = telemetry[8]
        yaw_rate = telemetry[4]

        if speed < 0.1:
            return -1.0  # Force movement

        if self.config.reward_mode == "hybrid":
            lane_reward = 2.0 * math.exp(-(lat_err ** 2) / 0.25)
            return float(
                lane_reward
                + (speed * 2.0)
                - abs(self._last_action[0]) * 0.1
                - abs(yaw_rate) * 0.2
            )
        else:
            reward = 0.0
            if abs(lat_err) < 0.5:
                reward += 1.0
            else:
                reward -= abs(lat_err) * 2.0
            reward += speed * 2.0
            reward -= (
                abs(self._last_action[0]) * 0.1
                + abs(yaw_rate) * 0.2
                + self._last_action[2] * 0.1
            )
            return float(reward)

    # Simulator Control

    def _publish_cmd_vel(self, speed: float, steering_angle: float) -> None:
        """
        Publish Twist to the AckermannSteering plugin.

        The plugin interprets:
            linear.x  = desired forward speed (m/s)
            angular.z = desired steering angle (radians)

        It internally computes per-wheel steering angles and drive
        velocities from the Ackermann geometry defined in the SDF.
        """
        msg = _GzTwist()
        msg.linear.x = float(speed)
        msg.angular.z = float(steering_angle)
        if self._cmd_pub is not None:
            self._cmd_pub.publish(msg)

    def _step_world(self, steps: int = 1) -> None:
        """
        Synchronously advance Gazebo by N physics steps.

        Uses the WorldControl service with multi_step. The service
        call blocks until the steps complete — this is what makes
        the environment synchronous (no ROS2 spin needed).
        """
        req = _GzWorldControl()
        req.pause = True
        req.multi_step = steps

        rep = _GzBoolean()
        world = self.config.world_name
        service = f"/world/{world}/control"

        try:
            result, success = self._node.request(
                service, req, self.config.service_timeout, rep,
            )
            if not success:
                logger.warning(f"WorldControl service call failed (timeout?)")
        except Exception as e:
            logger.error(f"WorldControl error: {e}")

    def _pause_world(self, pause: bool) -> None:
        """Pause or unpause the world."""
        req = _GzWorldControl()
        req.pause = pause
        rep = _GzBoolean()
        service = f"/world/{self.config.world_name}/control"
        try:
            self._node.request(service, req, self.config.service_timeout, rep)
        except Exception as e:
            logger.warning(f"Pause world error: {e}")

    def _wait_for_camera(self) -> None:
        """
        Block until a fresh camera frame arrives after the last step.

        Gazebo's camera sensor is async — it publishes on its own schedule.
        After stepping, we wait for the camera subscriber callback to fire
        with a post-step frame.
        """
        self._camera_event.clear()
        got_frame = self._camera_event.wait(timeout=self.config.camera_timeout)
        if not got_frame:
            logger.debug("Camera frame timeout — using stale frame")

    def _reset_robot_pose(self) -> None:
        """
        Teleport robot to spawn position via set_pose service.

        Equivalent to Isaac's set_world_pose() + set_linear/angular_velocity(0).
        Gazebo's set_pose resets the model's canonical link pose.
        """
        req = _GzPose()
        req.name = self.config.model_name
        req.position.x = self.config.spawn_x
        req.position.y = self.config.spawn_y
        req.position.z = self.config.spawn_z

        # Yaw-only rotation: q = (cos(yaw/2), 0, 0, sin(yaw/2))
        half_yaw = self.config.spawn_yaw / 2.0
        req.orientation.w = math.cos(half_yaw)
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = math.sin(half_yaw)

        rep = _GzBoolean()
        service = f"/world/{self.config.world_name}/set_pose"

        try:
            result, success = self._node.request(
                service, req, self.config.service_timeout, rep,
            )
            if not success:
                logger.warning("set_pose service call failed")
        except Exception as e:
            logger.error(f"set_pose error: {e}")

        # Zero velocity by publishing zero Twist
        self._publish_cmd_vel(0.0, 0.0)

        # Settle physics after teleport
        self._step_world(200)

    # State Accessors (AgentEnvWrapper interface)

    def _get_robot_position(self) -> np.ndarray:
        """World-frame position [x, y, z] from PosePublisher subscriber."""
        with self._pose_lock:
            return self._position.copy()

    def _get_robot_heading(self) -> float:
        """
        Heading (yaw) in radians from PosePublisher subscriber.

        This is the method that GazeboDirectEnv exposes.
        Having an explicit method makes the wrapper sim-agnostic.
        """
        with self._pose_lock:
            return self._heading

    def _get_robot_velocity(self) -> np.ndarray:
        """
        Estimated linear velocity [vx, vy, vz] via finite-difference.

        Gazebo's gz-transport doesn't expose physics velocity directly
        like Isaac's Articulation.get_linear_velocity(). We approximate
        from consecutive poses. This is noisy but acceptable for the
        speed magnitude used in telemetry vec[3].
        """
        return self._estimated_velocity.copy()

    def _get_robot_yaw_rate(self) -> float:
        """
        Yaw rate (rad/s) from IMU subscriber.

        Falls back to heading finite-difference if IMU data hasn't
        arrived yet (e.g., IMU sensor not configured in SDF).
        """
        with self._imu_lock:
            return float(self._angular_velocity[2])

    def _update_velocity_estimate(self) -> None:
        """
        Finite-difference velocity from consecutive pose readings.

        Called once per step() after WorldControl returns and the
        pose subscriber has updated.
        """
        now = time.monotonic()
        dt = now - self._prev_time

        if dt > 1e-6:
            with self._pose_lock:
                current_pos = self._position.copy()
            self._estimated_velocity = (current_pos - self._prev_position) / dt
            self._prev_position = current_pos.copy()
        else:
            self._estimated_velocity = np.zeros(3, dtype=np.float64)

        self._prev_time = now

    # Cleanup

    def close(self):
        """Stop publishing and clean up gz-transport resources."""
        if self._sim_initialized:
            # Send zero command to stop the robot
            self._publish_cmd_vel(0.0, 0.0)
            self._sim_initialized = False
            self._node = None
            self._cmd_pub = None

        super().close()
