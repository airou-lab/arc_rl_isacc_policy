import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.registry import register_sim

logger = logging.getLogger(__name__)


# Joint‑name mapping (source: F1Tenth_Metric.usd in arc_rl_isacc_sim)

# Steering (position‑controlled revolute joints):
#   Joint_Steer_L  –  left front knuckle
#   Joint_Steer_R  –  right front knuckle

# Drive (velocity‑controlled continuous joints):
#   Joint_Drive_FL –  front‑left wheel
#   Joint_Drive_FR –  front‑right wheel
#   Joint_Drive_RL –  rear‑left wheel
#   Joint_Drive_RR –  rear‑right wheel

# The OLD names (Knuckle__Upright__*, Wheel__Knuckle__*) came from the
# URDF‑imported model and no longer exist in the generated USD asset.

# Drive direction:  negative velocity -> forward motion in F1Tenth_Metric.usd.
# This was verified by the sim repo's verify_metric.py (−40 rad/s = forward).
# A config flag `drive_velocity_sign` lets you flip this if the USD changes.


@dataclass
class IsaacDirectConfig:
    """
    Configuration for the direct-API Isaac Sim environment.

    Defaults are aligned with the arc_rl_isacc_sim repo (Phase 10,
    1.0× metric scale, F1Tenth_Metric.usd).
    """

    # Scene
    # Path to the scene USD that contains the track (and optionally the
    # robot).  If the robot prim is NOT already present at robot_prim_path,
    # _setup_sim() will spawn it from robot_usd_path.
    usd_path: str = ""  # Set at runtime; no hardcoded absolute path

    # USD asset for the robot itself (only used if robot_prim_path does
    # not already exist in the loaded stage).
    robot_usd_path: str = ""  # e.g. ".../assets/robot/F1Tenth_Metric.usd"

    # Prim paths – must match F1Tenth_Metric.usd hierarchy
    robot_prim_path: str = "/World/Robot"
    camera_prim_path: str = "/World/Robot/Chassis/CameraSensor"

    # Reward Strategy
    # "original": Linear lane bonus + Boosted Speed (default)
    # "hybrid":   Gaussian lane precision + Momentum weighting
    reward_mode: str = "original"

    # Camera
    img_width: int = 160
    img_height: int = 90

    # Vehicle Geometry
    # From f1tenth.xacro: wheelbase 0.3302 m, track 0.2413 m, wheel_r 0.0508 m.
    # Rounded to match project convention.
    wheelbase: float = 0.33
    track_width: float = 0.28
    wheel_radius: float = 0.05

    # Joint Names (F1Tenth_Metric.usd)
    steering_joints: Tuple[str, ...] = (
        "Joint_Steer_L",
        "Joint_Steer_R",
    )
    drive_joints: Tuple[str, ...] = (
        "Joint_Drive_FL",   # front-left
        "Joint_Drive_FR",   # front-right
        "Joint_Drive_RL",   # rear-left
        "Joint_Drive_RR",   # rear-right
    )

    # Sign applied to wheel velocity commands before sending to PhysX.
    # −1.0 because F1Tenth_Metric.usd drive joint axes point backward:
    # negative angular velocity -> forward wheel rotation -> forward motion.
    # Verified by sim repo's verify_metric.py (−40 rad/s drives forward).
    drive_velocity_sign: float = -1.0

    # Control Limits
    max_steering_angle: float = 0.5   # rad ~= 28.6 deg
    max_speed: float = 3.0            # m/s

    # Physics
    # Aligned with sim repo: 500 Hz physics, decimation 25 -> 20 Hz control.
    physics_dt: float = 0.002         # 500 Hz
    render_dt: float = 0.05           # 20 Hz (every 25th physics step)
    control_hz: int = 20
    substeps: int = 24                # 24 physics-only + 1 render = 25 total

    # Episode
    episode_timeout: float = 120.0    # seconds (matching sim repo)
    max_episode_steps: int = 2400     # 120 s x 20 Hz

    # Reset / Spawn
    # 1x metric coordinates matching sim repo's fixed spawn on the
    # yellow centerline, heading +Y (90 deg Z-up).
    spawn_x: float = -16.25375
    spawn_y: float = 5.56
    spawn_z: float = 0.05
    spawn_yaw: float = 1.5708         # pi/2 rad = +90 deg (heading +Y)

    # Termination Tuning
    warmup_grace_steps: int = 10
    stuck_speed_threshold: float = 0.1           # m/s
    stuck_timeout: float = 5.0                   # seconds
    offroad_confidence_threshold: float = 0.05   # lane confidence
    offroad_timeout: float = 1.0                 # seconds

    # Height bounds tightened to match 1x metric car (sim repo uses
    # Z in [0.02, 0.5]).
    fall_z_min: float = -0.5
    fall_z_max: float = 1.0

    headless: bool = True


class AckermannComputer:
    """
    Converts (steering_angle, speed) into per-wheel steering angles and
    per-wheel angular velocities using Ackermann geometry.

    Output ordering matches IsaacDirectConfig.drive_joints:
        [FL, FR, RL, RR]

    The caller is responsible for applying drive_velocity_sign before
    sending to PhysX.
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
        Returns:
            wheel_angles:     [left_steer, right_steer]       (float32)
            wheel_velocities: [FL, FR, RL, RR] angular vel    (float32)
        """
        base_omega = speed / self.wheel_radius

        if abs(steering_angle) < 1e-4:
            return (
                np.array([0.0, 0.0], dtype=np.float32),
                np.full(4, base_omega, dtype=np.float32),
            )

        turn_radius = self.wheelbase / math.tan(abs(steering_angle))
        inner_angle = math.atan(self.wheelbase / (turn_radius - self.half_track))
        outer_angle = math.atan(self.wheelbase / (turn_radius + self.half_track))

        if steering_angle > 0:
            # Turning right: left wheels are outer, right are inner
            left_angle, right_angle = outer_angle, inner_angle
        else:
            left_angle, right_angle = -inner_angle, -outer_angle

        wheel_angles = np.array([left_angle, right_angle], dtype=np.float32)

        inner_omega = speed * ((turn_radius - self.half_track) / turn_radius) / self.wheel_radius
        outer_omega = speed * ((turn_radius + self.half_track) / turn_radius) / self.wheel_radius

        if steering_angle > 0:
            # FL=outer, FR=inner, RL=outer, RR=inner
            wheel_velocities = np.array(
                [outer_omega, inner_omega, outer_omega, inner_omega],
                dtype=np.float32,
            )
        else:
            # FL=inner, FR=outer, RL=inner, RR=outer
            wheel_velocities = np.array(
                [inner_omega, outer_omega, inner_omega, outer_omega],
                dtype=np.float32,
            )

        return wheel_angles, wheel_velocities


@register_sim("isaac")
class IsaacDirectEnv(gym.Env):
    """
    Direct Omniverse-API gymnasium environment for F1Tenth training.

    Uses the Isaac Sim Python API (omni.isaac.core) directly — no ROS2,
    no Isaac Lab ManagerBased abstractions.  This is the training-side
    environment; ROS2 enters only at deployment.
    """

    def __init__(
        self,
        config: Optional[IsaacDirectConfig] = None,
        simulation_app=None,
    ):
        super().__init__()
        self.config = config or IsaacDirectConfig()
        self._simulation_app = simulation_app

        self._step_count = 0
        self._episode_start_time = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cumulative_distance = 0.0
        self._last_position = np.zeros(3, dtype=np.float64)

        self._ackermann = AckermannComputer(
            self.config.wheelbase,
            self.config.track_width,
            self.config.wheel_radius,
        )

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8,
            ),
            "vec": spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self._world = None
        self._robot_articulation = None
        self._annotator = None
        self._sim_initialized = False
        self._lane_detector = None

        self._stuck_timer = 0.0
        self._offroad_timer = 0.0

        # Internal lane detection state — used for reward and termination
        # only, never exposed in the observation vector (PVP compliance).
        self._lane_lateral_offset = 0.0
        self._lane_confidence = 0.0

    # Simulation Setup

    def _setup_sim(self):
        if self._sim_initialized:
            return

        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from pxr import UsdPhysics, UsdLux, Gf
        import omni.usd

        # Load Lane Detector
        try:
            from lane_detector import SimpleLaneDetector
            self._lane_detector = SimpleLaneDetector(
                img_width=self.config.img_width,
                img_height=self.config.img_height,
            )
            logger.info("Lane detector loaded successfully")
        except Exception as e:
            logger.warning(
                f"Lane detector unavailable: {e} — off-road termination disabled"
            )
            self._lane_detector = None

        # Open scene USD (track + optional pre-placed robot)
        if self.config.usd_path:
            omni.usd.get_context().open_stage(self.config.usd_path)
            for _ in range(100):
                if self._simulation_app:
                    self._simulation_app.update()

        # If the robot prim is not already in the stage, spawn it from
        # the robot USD asset.
        if not is_prim_path_valid(self.config.robot_prim_path):
            self._spawn_robot()

        self._world = World(
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.render_dt,
            stage_units_in_meters=1.0,
        )

        # Ground plane contrast fix (legacy scene compatibility)
        if is_prim_path_valid("/World/whiteGround"):
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/whiteGround")
            if prim.IsValid():
                attr = prim.GetAttribute("primvars:displayColor")
                if attr.IsValid():
                    attr.Set([Gf.Vec3f(0.2, 0.2, 0.2)])

        # Ensure lighting exists
        if not is_prim_path_valid("/World/defaultLight"):
            stage = omni.usd.get_context().get_stage()
            dome = UsdLux.DomeLight.Define(stage, "/World/defaultLight")
            dome.CreateIntensityAttr().Set(2000000)
            dome.CreateExposureAttr().Set(12.0)
            for _ in range(100):
                if self._simulation_app:
                    self._simulation_app.update()

        self._world.get_physics_context().set_gravity(-9.81)

        self._robot_articulation = Articulation(self.config.robot_prim_path)
        self._world.scene.add(self._robot_articulation)
        self._setup_camera()
        self._world.reset()
        self._robot_articulation.initialize()

        # Validate joint mapping at init — fail loud rather than silently
        # commanding None indices for an entire episode.
        self._steering_indices = self._resolve_joint_indices(
            self.config.steering_joints, "steering"
        )
        self._drive_indices = self._resolve_joint_indices(
            self.config.drive_joints, "drive"
        )

        for _ in range(50):
            self._world.step(render=False)

        self._sim_initialized = True
        logger.info(
            f"Isaac Sim initialized — robot at {self.config.robot_prim_path}, "
            f"steering joints {list(self.config.steering_joints)}, "
            f"drive joints {list(self.config.drive_joints)}"
        )

    def _spawn_robot(self):
        """
        Import the robot USD asset into the stage if it is not already
        present.  Falls back to a clear error if robot_usd_path is not
        configured.
        """
        from pxr import Usd, UsdGeom
        import omni.usd

        if not self.config.robot_usd_path:
            raise FileNotFoundError(
                f"Robot prim '{self.config.robot_prim_path}' not found in "
                f"the loaded stage, and robot_usd_path is not set. Either "
                f"use a scene USD that already contains the robot, or set "
                f"robot_usd_path to the F1Tenth_Metric.usd path."
            )

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.DefinePrim(self.config.robot_prim_path)
        robot_prim.GetReferences().AddReference(self.config.robot_usd_path)

        logger.info(
            f"Spawned robot from {self.config.robot_usd_path} "
            f"at {self.config.robot_prim_path}"
        )

        for _ in range(50):
            if self._simulation_app:
                self._simulation_app.update()

    def _resolve_joint_indices(
        self, joint_names: Tuple[str, ...], group_label: str
    ) -> list:
        """
        Look up DOF indices for the given joint names and validate that
        every one resolved.  Raises RuntimeError with a diagnostic dump
        if any are missing — prevents the silent‑None bug that plagued
        the old joint name convention.
        """
        indices = []
        missing = []
        for name in joint_names:
            idx = self._get_joint_index(name)
            if idx is None:
                missing.append(name)
            indices.append(idx)

        if missing:
            # Dump all available DOF names to help debug USD mismatches
            try:
                num_dof = self._robot_articulation.num_dof
                available = [
                    self._robot_articulation.dof_names[i]
                    for i in range(num_dof)
                ] if hasattr(self._robot_articulation, 'dof_names') else ["(introspection unavailable)"]
            except Exception:
                available = ["(introspection failed)"]

            raise RuntimeError(
                f"Joint resolution failed for {group_label} group.\n"
                f"  Missing: {missing}\n"
                f"  Available DOFs ({self._robot_articulation.num_dof}): {available}\n"
                f"  -> Check that IsaacDirectConfig joint names match the USD asset."
            )

        return indices

    def _setup_camera(self):
        import omni.replicator.core as rep

        rp = rep.create.render_product(
            self.config.camera_prim_path,
            resolution=(self.config.img_width, self.config.img_height),
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([rp])

    # Episode Lifecycle

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_sim()

        self._step_count = 0
        self._cumulative_distance = 0.0
        self._stuck_timer = 0.0
        self._offroad_timer = 0.0
        self._lane_lateral_offset = 0.0
        self._lane_confidence = 0.0

        self._reset_robot_pose()
        self._world.step(render=True)
        self._last_position = self._get_robot_position()

        return self._get_obs(), {"episode_step": 0}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._last_action = action.copy()

        # Map policy action -> Ackermann wheel commands
        steering_angle = action[0] * self.config.max_steering_angle
        speed = action[1] * self.config.max_speed
        if action[2] > 0.0:
            speed *= (1.0 - action[2])

        wheel_angles, wheel_velocities = self._ackermann.compute(
            steering_angle, speed
        )
        self._apply_wheel_commands(wheel_angles, wheel_velocities)

        # Physics substeps + render
        for _ in range(self.config.substeps):
            self._world.step(render=False)
        self._world.step(render=True)

        # Odometry
        current_pos = self._get_robot_position()
        self._cumulative_distance += np.linalg.norm(
            current_pos[:2] - self._last_position[:2]
        )
        self._last_position = current_pos
        self._step_count += 1

        obs = self._get_obs()

        # Termination
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps
        info: Dict[str, Any] = {"episode_step": self._step_count}

        # Grace period: let physics settle after reset
        if self._step_count <= self.config.warmup_grace_steps:
            reward = self._compute_reward(obs)
            return obs, reward, False, truncated, info

        # Sim-time per step: (substeps + 1) x physics_dt
        step_dt = (self.config.substeps + 1) * self.config.physics_dt

        # Stuck: speed below threshold for too long
        speed_now = obs["vec"][3]
        if speed_now < self.config.stuck_speed_threshold:
            self._stuck_timer += step_dt
        else:
            self._stuck_timer = 0.0
        if self._stuck_timer > self.config.stuck_timeout:
            terminated = True
            info["termination_reason"] = "stuck"

        # Off-road: lane confidence near zero for too long
        if self._lane_detector is not None:
            lane_conf = self._lane_confidence
            if lane_conf < self.config.offroad_confidence_threshold:
                self._offroad_timer += step_dt
            else:
                self._offroad_timer = 0.0
            if self._offroad_timer > self.config.offroad_timeout:
                terminated = True
                info["termination_reason"] = "off_road"

        # Fall: car fell through ground or launched into the air
        if current_pos[2] < self.config.fall_z_min or current_pos[2] > self.config.fall_z_max:
            terminated = True
            info["termination_reason"] = "fell"

        reward = self._compute_reward(obs)
        return obs, reward, terminated, truncated, info

    # Observation

    def _get_obs(self):
        return {
            "image": self._capture_camera(),
            "vec": self._compute_telemetry(),
        }

    def _capture_camera(self):
        data = self._annotator.get_data()
        if data is None or data.size == 0:
            return np.zeros(
                (self.config.img_height, self.config.img_width, 3),
                dtype=np.uint8,
            )
        if data.ndim == 3 and data.shape[2] == 4:
            data = data[:, :, :3]
        return data.astype(np.uint8)

    def _compute_telemetry(self):
        """
        Build the 12-element telemetry vector.

        Indices follow config/experiment.py TELEMETRY_INDICES.  Slots
        8–10 are intentionally zeroed (PVP protocol): the policy never
        sees lane geometry; lane detection is internal to reward/term.
        """
        vec = np.zeros(12, dtype=np.float32)
        velocity = self._get_robot_velocity()

        vec[0] = 0.0  # turn token — set by Worker via AgentEnvWrapper
        vec[3] = float(np.linalg.norm(velocity[:2]))
        vec[4] = float(self._get_robot_yaw_rate())
        vec[5], vec[6], vec[7] = self._last_action

        # Lane detection — stored internally for reward and termination.
        # NOT written to vec[8:10] — those stay zero (PVP protocol).
        if self._lane_detector is not None:
            try:
                res = self._lane_detector.detect(self._capture_camera())
                self._lane_lateral_offset = float(res.lateral_offset)
                self._lane_confidence = float(res.confidence)
            except Exception as e:
                logger.debug(f"Lane detection failed this step: {e}")
                self._lane_lateral_offset = 0.0
                self._lane_confidence = 0.0

        # vec[2], vec[8], vec[9], vec[10] intentionally zero-padded (PVP)
        vec[11] = self._cumulative_distance
        return vec

    # Reward

    def _compute_reward(self, obs):
        telemetry = obs["vec"]
        speed = telemetry[3]
        lat_err = self._lane_lateral_offset  # internal state, not obs (PVP)
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

    # Actuation

    def _apply_wheel_commands(self, wheel_angles, wheel_velocities):
        """
        Send Ackermann-computed steering angles and drive velocities
        to the articulation controller.

        Drive velocities are multiplied by config.drive_velocity_sign
        to match the USD joint axis convention (−1 = forward for
        F1Tenth_Metric.usd).
        """
        from omni.isaac.core.utils.types import ArticulationAction

        dc = self._robot_articulation.get_articulation_controller()

        # Steering -> position targets
        dc.apply_action(ArticulationAction(
            joint_positions=wheel_angles,
            joint_indices=self._steering_indices,
        ))

        # Drive -> velocity targets (with sign correction)
        signed_velocities = wheel_velocities * self.config.drive_velocity_sign
        dc.apply_action(ArticulationAction(
            joint_velocities=signed_velocities,
            joint_indices=self._drive_indices,
        ))

    # Reset

    def _reset_robot_pose(self):
        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        position = np.array([
            self.config.spawn_x,
            self.config.spawn_y,
            self.config.spawn_z,
        ])
        orientation = euler_angles_to_quat(
            np.array([0.0, 0.0, self.config.spawn_yaw])
        )

        self._robot_articulation.set_world_pose(
            position=position, orientation=orientation
        )
        self._robot_articulation.set_linear_velocity(np.zeros(3))
        self._robot_articulation.set_angular_velocity(np.zeros(3))
        self._robot_articulation.set_joint_velocities(
            np.zeros(self._robot_articulation.num_dof)
        )

        for _ in range(20):
            self._world.step(render=False)

    # State Accessors

    def _get_robot_position(self):
        return np.array(
            self._robot_articulation.get_world_pose()[0], dtype=np.float64
        )

    def _get_robot_velocity(self):
        return np.array(
            self._robot_articulation.get_linear_velocity(), dtype=np.float64
        )

    def _get_robot_yaw_rate(self):
        return float(self._robot_articulation.get_angular_velocity()[2])

    def _get_joint_index(self, joint_name):
        try:
            return self._robot_articulation.get_dof_index(joint_name)
        except Exception:
            return None

    # Cleanup

    def close(self):
        if self._sim_initialized:
            self._world.stop()
            self._sim_initialized = False
        super().close()
