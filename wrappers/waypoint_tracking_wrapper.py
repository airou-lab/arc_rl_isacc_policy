"""
Waypoint Tracking Wrapper (Isaac Sim Adaptation)

Tracks the vehicle's trajectory via dead reckoning for self-supervised waypoint prediction learning.
The planning head in the hierarchical policy predicts future waypoints, and this wrapper records where the
vehicle actually went, providing supervision targets without any human labels.

Key features:
    1. Safety backfill: Marks last N steps as unsafe when crash occurs, so the repulsions loss can teach the
       planner to avoid crash paths.
    2. Global TrajectoryStore singleton: Allows the training loop to access trajectory data that would otherwise
       be lost in SB3's rollout buffer.
    3. Dead-reckoning position estimation from speed + yaw_rate.
    4. Proper episode boundary handling.

Adaptation from Unity version:
    * Removed Unity ground-truth position/yaw feedback (info['car_position']).
    * Dead reckoning uses speed * dt instead of per-step ds delta, because isaac_ros2_env[11] is cumulative
      total_distance, not ds.
    * Removed Unity visualization forwarding (set_waypoints on env).

Used by:
    train_policy_ros2.py (wraps IsaacROS2Env in make_env())
    losses/waypoint_losses.py (reads from TrajectoryStore for aux loss)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
import threading

class TrajectoryStore:
    """
    Thread-safe global store for trajectory data.

    Problem: SB3's rollout buffer discards episode info dicts after each rollout. But the waypoint auxiliary
             loss needs full trajectory from the just-completed episode to compute supervision targets.

    Solution: This singleton stores trajectory data per-environment, accessible from both the wrapper (which writes)
              and the custom PPO training loop (which reads for loss computation).

    Thread safety: Required because SB3 may reset environments from different threads during async collection.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._trajectories: Dict[int, Dict] = {}
        self._episode_safety: Dict[int, np.ndarray] = {}
        self._data_lock = threading.Lock()

    def store_trajectory(self, env_id: int, trajectory: Dict, safety_mask: np.ndarray):
        """Store trajectory data for an environment."""
        with self._data_lock:
            self._trajectories[env_id] = {
                "positions": trajectory["positions"].copy(),
                "yaws": trajectory["yaws"].copy(),
                "speeds": trajectory["speeds"].copy(),
            }
            self._episode_safety[env_id] = safety_mask.copy()

    def get_trajectory(self, env_id: int) -> Optional[Dict]:
        """Get stored trajectory for an environment."""
        with self._data_lock:
            return self._trajectories.get(env_id)

    def get_safety_mask(self, env_id: int) -> Optional[np.ndarray]:
        """Get safety mask for an environment."""
        with self._data_lock:
            return _episode_safety.get(env_id)

    def clear(self, env_id: Optional[int] = None):
        """Clear stored data."""
        with self._data_lock:
            if env_id is not None:
                self._trajectories.pop(env_id, None)
                self._episode_safety.pop(env_id, None)
            else:
                self._trajectories.clear()
                self._episode_safety.clear()

# Global instance
_trajectory_store = TrajectoryStore()


def get_trajectory_store() -> TrajectoryStore:
    """Get the global trajectory store instance."""
    return _trajectory_store


class WaypointTrackingWrapper(gym.Wrapper):
    """
    Wrapper that tracks:
        1. Predicted waypoints from policy (set externally via set_predicted_waypoints)
        2. Actual trajectory via dead reckoning (for self-supervised learning)
        3. Safety flags with backfill for crash trajectories

    Dead reckoning:
         Uses speed and yaw_rate from the telemetry vector to integrate position over time. This is the ONLY
         position source in Isaac Sim (unlike Unity which could provide ground-truth car_position).
    """

    # Steps to backfill as unsafe when crash occurs (~0.5s at 50Hz)
    SAFETY_BACKFILL_STEPS = 25

    # Physics timestep (must match isaac_ros2_env.py)
    DT = 0.02 # 50 Hz

    # Telemetry indices
    IDX_SPEED = 3
    IDX_YAW_RATE = 4
    IDX_DS = 11

    def __init__(self, env: gym.Env, env_id: int = 0):
        super().__init__(env)
        self.env_id = env_id

        # Trajectory buffers
        self.position_history: list = []
        self.yaw_history: list = []
        self.speed_history: list = []
        self.reward_history: list = []

        # Safety tracking
        self.safety_history: list = []

        # Waypoint tracking
        self.last_predicted_waypoints: Optional[np.ndarray] = None
        self.waypoint_prediction_history: list = []

        # Dead-reckoning state
        self._estimated_pos = np.zeros(3, dtype=np.float32)
        self._estimated_yaw = 0.0

        # Episode tracking
        self._step_count = 0

        # Global store
        self._store = get_trajectory_store()

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and clear trajectory buffers."""
        self.position_history = []
        self.yaw_history = []
        self.speed_history = []
        self.reward_history = []
        self.safety_history = []
        self.waypoint_prediction_history = []
        self.last_predicted_waypoints = None

        self._estimated_pos = np.zeros(3, dtype=np.float32)
        self._estimated_yaw = 0.0
        self._step_count = 0

        self._store.clear(self.env_id)

        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            return ret
        return ret, {}

    def step(self, action):
        """Step environment and record trajectory data."""
        obs, reward, done, truncated, info = self.env.step(action)
        self._step_count += 1

        # Dead-reckoning position estimate
        # (Isaac Sim does not provide ground-truth position in info dict)
        position = self._update_position_estimate(obs)
        yaw = self._estimated_yaw

        # Extract speed from observation
        if isinstance(obs, dict) and "vec" in obs:
            speed = float(obs["vec"][self.IDX_SPEED])
        else:
            speed = 0.0

        # Store trajectory data
        self.position_history.append(position.copy())
        self.yaw_history.append(yaw)
        self.speed_history.append(speed)
        self.reward_history.append(float(reward))

        # Initially mark all steps as safe (1.0)
        # Backfill will mark unsafe steps when episode ends
        self.safety_history.append(1.0)

        # Store waypoint prediction if available
        if self.last_predicted_waypoints is not None:
            self.waypoint_prediction_history.append(
                {
                    "step": self._step_count,
                    "waypoints": self.last_predicted_waypoints.copy(),
                    "position": position.copy(),
                    "yaw": way,
                }
            )

        # Handle episode end. Apply safety backfill
        if done or truncated:
            self._handle_episode_end(done, truncated, reward)

        # Add trajectory info to info dict (useful for debugging)
        info["trajectory"] = {
            "positions": np.array(self.position_history),
            "yaws": np.array(self.yaw_history),
            "speeds": np.array(self.speed_history),
            "safety": np.array(self.safety_history),
        }

        if self.last_predicted_waypoints is not None:
            info["predicted_waypoints"] = self.last_predicted_waypoints.copy()

        return obs, reward, done, truncated, info

    def _update_position_estimate(self, obs) -> np.ndarray:
        """
        Dead-reckoning position update using speed and yaw_rate.

        Integrates velocity over the physics timestep to estimate position.
        Uses speed * dt for distance (NOT vec[11] which is cumulative total distance in Isaac, not a per-step deltat like Unity's ds).

        Coordinate frame:
            pos[0] = X (lateral, positive = right)
            pos[2] = Z (forwardm positive = ahead)
            yaw    = heading angle (radians)
        """

        if not isinstance(obs, dict) or "vec" not in obs:
            return self._estimated_pos.copy()

        vec = obs["vec"]
        speed = float(vec[self.IDX_SPEED])
        yaw_rate = float(vec[self.IDX_YAW_RATE])

        # Update yaw
        self._estimated_yaw += yaw_rate * self.DT

        # Normalize to [-pi, pi]
        while self._estimated_yaw > np.pi:
            self._estimated_yaw -= 2 * np.pi
        while self._estimated_yaw < -np.pi:
            self._estimated_yaw += 2 * np.pi

        # Update position using speed * dt
        ds = speed * self.DT
        if abs(ds) > 0.0001:
            self._estimated_pos[0] += ds * np.sin(self._estimated_yaw) # X
            self._estimated_pos[2] += ds * np.cos(self._estimated_yaw) # Z

        return self._estimated_pos.copy()

    def _handle_episode_end(self, done: bool, truncated: bool, final_reward: float):
        """
        Handle episode end, apply safety backfill if crash occured.

        Safety logic:
        * done=True, truncated=False, reward < 0: CRASH (backfill unsafe)
        * done=True, truncated=False, reward > 0: SUCCESS (all safe)
        * truncated=True: TIMEOUT (all safe, just ran out of time)
        """
        is_crash = done and not truncated and final_reward < 0

        if is_crash:
            num_steps = len(self.safety_history)
            backfill_start = max(0, num_steps - self.SAFETY_BACKFILL_STEPS)

            for i in range(backfill_start, num_steps):
                self.safety_history[i] = 0.0

        # Store in global trajectory store for training loop access
        trajectory_data = {
            "positions": np.array(self.position_history),
            "yaws": np.array(self.yaw_history),
            "speeds": np.array(self.speed_history),
        }
        safety_mask = np.array(self.safety_history)

        self._store.store_trajectory(self.env_id, trajectory_data, safety_mask)

    def set_predicted_waypoints(self, waypoints: np.ndarray):
        """Called by callback/policy to set the current predicted waypoints."""
        self.last_predicted_waypoints = waypoints.copy()

    def get_waypoint_history(self) -> list:
        """Get history of waypoint predictions for analysis."""
        return self.waypoint_prediction_history.copy()

    def get_current_trajectory_for_loss(self) -> Optional[Dict]:
        """
        Get trajectory data fromatted for WaypointLoss computation.

        Return dict with:
        * positions: (N, 3) array of dead-reckoned world positions
        * yaws: (N,) array of yaw angles
        * safety: (N,) array of safety flags (1.0=safe, 0.0=unsafe)
        * current_idx: int, index of the most recent step
        """
        if len(self.position_history) < 2:
            return None

        return {
            "positions": np.array(self.position_history),
            "yaws": np.array(self.yaw_history),
            "safety": np.array(self.safety_history),
            "current_idx": len(self.position_history - 1),
        }
