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

@dataclass
class IsaacDirectConfig:
    """
    Configuration for the direct-API Isaac Sim environment.
    """
    # Scene
    usd_path: str = "/home/arika/Documents/arcpro/arcpro_system/src/examples/ARCPro_RL/arc_rl_isacc_sim/openStreetUSD/no_graph_sim.usd"
    robot_prim_path: str = "/World/F1Tenth"
    camera_prim_path: str = "/World/F1Tenth/Rigid_Bodies/Chassis/Camera_Left"

    # Reward Strategy
    # "original": Linear lane bonus + Boosted Speed (default)
    # "hybrid": Gaussian lane precision + Momentum weighting
    reward_mode: str = "original"

    # Camera
    img_width: int = 160
    img_height: int = 90

    # Vehicle geometry
    # NOTE: URDF xacro values are 0.3302 / 0.2413 / 0.0508 - discrepancy is an open issue
    wheelbase: float = 0.33
    track_width: float = 0.28
    wheel_radius: float = 0.05

    steering_joints: Tuple[str, ...] = ("Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right")
    drive_joints: Tuple[str, ...] = ("Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right", "Wheel__Upright__Rear_Left", "Wheel__Upright__Rear_Right")

    # Control limits
    max_steering_angle: float = 0.5
    max_speed: float = 3.0

    # Physics
    physics_dt: float = 1.0 / 60.0
    render_dt: float = 1.0 / 30.0
    control_hz: int = 10
    substeps: int = 6

    # Episode
    episode_timeout: float = 30.0
    max_episode_steps: int = 300

    # Reset
    spawn_x: float = -125.0
    spawn_y: float = 62.0
    spawn_z: float = 0.5
    spawn_yaw: float = 0.0

    # Termination turning
    warmup_grace_steps: int = 10                 # Skip termination checks for first N steps after reset
    stuck_speed_threshold: float = 0.1           # m/s - below this count as stuck
    stuck_timeout: float = 5.0                   # seconds stuck before termination
    offroad_confidence_threshold: float = 0.05   # lane confidence below this counts as off-road
    offroad_timeout: float = 1.0                 # seconds off-road before termination

    headless: bool = True

class AckermannComputer:
    def __init__(self, wheelbase: float, track_width: float, wheel_radius: float):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_radius = wheel_radius
        self.half_track = track_width / 2.0

    def compute(self, steering_angle: float, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        base_omega = speed / self.wheel_radius
        if abs(steering_angle) < 1e-4:
            return np.array([0.0, 0.0], dtype=np.float32), np.full(4, base_omega, dtype=np.float32)
        turn_radius = self.wheelbase / math.tan(abs(steering_angle))
        inner_angle = math.atan(self.wheelbase / (turn_radius - self.half_track))
        outer_angle = math.atan(self.wheelbase / (turn_radius + self.half_track))
        left_angle, right_angle = (outer_angle, inner_angle) if steering_angle > 0 else (-inner_angle, -outer_angle)
        wheel_angles = np.array([left_angle, right_angle], dtype=np.float32)
        inner_omega = speed * ((turn_radius - self.half_track) / turn_radius) / self.wheel_radius
        outer_omega = speed * ((turn_radius + self.half_track) / turn_radius) / self.wheel_radius
        wheel_velocities = np.array([outer_omega, inner_omega, outer_omega, inner_omega], dtype=np.float32) if steering_angle > 0 else np.array([inner_omega, outer_omega, inner_omega, outer_omega], dtype=np.float32)
        return wheel_angles, wheel_velocities

@register_sim("isaac")
class IsaacDirectEnv(gym.Env):
    def __init__(self, config: Optional[IsaacDirectConfig] = None, simulation_app=None):
        super().__init__()
        self.config = config or IsaacDirectConfig()
        self._simulation_app = simulation_app
        self._step_count = 0
        self._episode_start_time = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cumulative_distance = 0.0
        self._last_position = np.zeros(3, dtype=np.float64)
        self._ackermann = AckermannComputer(self.config.wheelbase, self.config.track_width, self.config.wheel_radius)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.config.img_height, self.config.img_width, 3), dtype=np.uint8),
            "vec": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self._world = None
        self._robot_articulation = None
        self._annotator = None
        self._sim_initialized = False
        self._lane_detector = None
        self._stuck_timer = 0.0
        self._offroad_timer = 0.0

    def _setup_sim(self):
        if self._sim_initialized: return
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
            logger.warning(f"Lane detector unavailable: {e} - off-road termination will be disabled")
            self._lane_detector = None

        if self.config.usd_path:
            omni.usd.get_context().open_stage(self.config.usd_path)
            for _ in range(100):
                if self._simulation_app: self._simulation_app.update()
        self._world = World(physics_dt=self.config.physics_dt, rendering_dt=self.config.render_dt, stage_units_in_meters=1.0)

        # Ground plane contrast fix
        if is_prim_path_valid("/World/whiteGround"):
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/whiteGround")
            if prim.IsValid():
                attr = prim.GetAttribute("primvars:displayColor")
                if attr.IsValid(): attr.Set([Gf.Vec3f(0.2, 0.2, 0.2)])

        if not is_prim_path_valid("/World/defaultLight"):
            stage = omni.usd.get_context().get_stage()
            dome = UsdLux.DomeLight.Define(stage, "/World/defaultLight")
            dome.CreateIntensityAttr().Set(2000000)
            dome.CreateExposureAttr().Set(12.0)
            for _ in range(100):
                if self._simulation_app: self._simulation_app.update()

        self._world.get_physics_context().set_gravity(-9.81)
        self._robot_articulation = Articulation(self.config.robot_prim_path)
        self._world.scene.add(self._robot_articulation)
        self._setup_camera()
        self._world.reset()
        self._robot_articulation.initialize()
        for _ in range(50): self._world.step(render=False)
        self._sim_initialized = True

    def _setup_camera(self):
        import omni.replicator.core as rep
        rp = rep.create.render_product(self.config.camera_prim_path, resolution=(self.config.img_width, self.config.img_height))
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([rp])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_sim()
        self._step_count = 0
        self._cumulative_distance = 0.0
        self._stuck_timer = 0.0
        self._offroad_timer = 0.0
        self._reset_robot_pose()
        self._world.step(render=True)
        self._last_position = self._get_robot_position()
        return self._get_obs(), {"episode_step": 0}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._last_action = action.copy()
        steering_angle = action[0] * self.config.max_steering_angle
        speed = action[1] * self.config.max_speed
        if action[2] > 0.0: speed *= (1.0 - action[2])
        wheel_angles, wheel_velocities = self._ackermann.compute(steering_angle, speed)
        self._apply_wheel_commands(wheel_angles, wheel_velocities)
        for _ in range(self.config.substeps): self._world.step(render=False)
        self._world.step(render=True)
        current_pos = self._get_robot_position()
        self._cumulative_distance += np.linalg.norm(current_pos[:2] - self._last_position[:2])
        self._last_position = current_pos
        self._step_count += 1
        obs = self._get_obs()

        # Termination
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps
        info = {"episode_step": self._step_count}

        # Grace period: let physics settle after reset before checking termination
        if self._step_count <= self.config.warmup_grace_steps:
            reward = self._compute_reward(obs)
            return obs, reward, False, truncated, info

        # Sim-time per step: substeps * physics_dt + render_dt
        step_dt = self.config.substeps * self.config.physics_dt + self.config.render_dt

        # Stuck: speed below threshold for too long
        speed_now = obs["vec"][3]
        if speed_now < self.config.stuck_speed_threshold:
            self._stuck_timer += step_dt
        else:
            self._stuck_timer = 0.0
        if self._stuck_timer > self.config.stuck_timeout:
            terminated = True
            info["termination_reason"] = "stuck"

        # Off-road: lane confidence near zero for too long (only if lane detector is available)
        if self._lane_detector is not None:
            lane_conf = obs["vec"][9]
            if lane_conf < self.config.offroad_confidence_threshold:
                self._offroad_timer += step_dt
            else:
                self._offroad_timer = 0.0
            if self._offroad_timer > self.config.offroad_timeout:
                terminated = True
                info["termination_reason"] = "off_road"

        # Fall: car fell through ground or flipped
        if current_pos[2] < -1.0 or current_pos[2] > 5.0:
            terminated = True
            info["termination_reason"] = "fell"

        reward = self._compute_reward(obs)
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return {"image": self._capture_camera(), "vec": self._compute_telemetry()}

    def _capture_camera(self):
        data = self._annotator.get_data()
        if data is None or data.size == 0: return np.zeros((self.config.img_height, self.config.img_width, 3), dtype=np.uint8)
        if data.ndim == 3 and data.shape[2] == 4: data = data[:, :, :3]
        return data.astype(np.uint8)

    def _compute_telemetry(self):
        vec = np.zeros(12, dtype=np.float32)
        velocity = self._get_robot_velocity()
        vec[0] = 0.0 # turn token (set by Worker node at deployment)
        vec[3] = float(np.linalg.norm(velocity[:2]))
        vec[4] = float(self._get_robot_yaw_rate())
        vec[5], vec[6], vec[7] = self._last_action

        # Lane detection for reward computation (agent never sees this directly)
        if self._lane_detector is not None:
            try:
                res = self._lane_detector.detect(self._capture_camera())
                vec[8] = float(res.lateral_offset)
                vec[9] = float(res.confidence)
            except Exception as e:
                logger.debug(f"Lane detection failed this step: {e}")
                # vec[8] and vec[9] stay 0.0

        # vec [10] intentionally zero-padded (PVP protocol)
        vec[11] = self._cumulative_distance
        return vec

    def _compute_reward(self, obs):
        telemetry = obs["vec"]
        speed = telemetry[3]
        lat_err = telemetry[8]
        yaw_rate = telemetry[4]

        if speed < 0.1: return -1.0 # Force movement

        if self.config.reward_mode == "hybrid":
            lane_reward = 2.0 * math.exp(-(lat_err**2) / 0.25)
            return float(lane_reward + (speed * 2.0) - abs(self._last_action[0])*0.1 - abs(yaw_rate)*0.2)
        else:
            reward = 0.0
            if abs(lat_err) < 0.5: reward += 1.0
            else: reward -= abs(lat_err) * 2.0
            reward += speed * 2.0
            reward -= abs(self._last_action[0])*0.1 + abs(yaw_rate)*0.2 + self._last_action[2]*0.1
            return float(reward)

    def _apply_wheel_commands(self, wheel_angles, wheel_velocities):
        from omni.isaac.core.utils.types import ArticulationAction
        dc = self._robot_articulation.get_articulation_controller()
        steering_indices = [self._get_joint_index(name) for name in self.config.steering_joints]
        drive_indices = [self._get_joint_index(name) for name in self.config.drive_joints]
        dc.apply_action(ArticulationAction(joint_positions=wheel_angles, joint_indices=steering_indices))
        dc.apply_action(ArticulationAction(joint_velocities=wheel_velocities, joint_indices=drive_indices))

    def _reset_robot_pose(self):
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        position = np.array([self.config.spawn_x, self.config.spawn_y, self.config.spawn_z])
        orientation = euler_angles_to_quat(np.array([0.0, 0.0, self.config.spawn_yaw]))
        self._robot_articulation.set_world_pose(position=position, orientation=orientation)
        self._robot_articulation.set_linear_velocity(np.zeros(3))
        self._robot_articulation.set_angular_velocity(np.zeros(3))
        self._robot_articulation.set_joint_velocities(np.zeros(self._robot_articulation.num_dof))
        for _ in range(20): self._world.step(render=False)

    def _get_robot_position(self): return np.array(self._robot_articulation.get_world_pose()[0], dtype=np.float64)
    def _get_robot_velocity(self): return np.array(self._robot_articulation.get_linear_velocity(), dtype=np.float64)
    def _get_robot_yaw_rate(self): return float(self._robot_articulation.get_angular_velocity()[2])
    def _get_joint_index(self, joint_name):
        try: return self._robot_articulation.get_dof_index(joint_name)
        except: return None

    def close(self):
        if self._sim_initialized:
            self._world.stop()
            self._sim_initialized = False
        super().close()
