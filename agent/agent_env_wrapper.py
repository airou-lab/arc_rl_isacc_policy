"""
Agent Environment Wrapper
=========================

Gymnasium wrapper that integrates AgentNode (Worker + Main) into any
environment that uses the 12-float telemetry vector protocol.

Replaces the old set_turn_bias() pattern entirely. Instead of the
training script manually calling env.set_turn_bias(random_value),
the Worker node inside the Agent queries the intersection graph
every step and injects a principled turn_token based on the agent's
position and route plan.

From SB3's perspective, nothing changes:
    - Observation space: still Dict{"image": Box, "vec": Box(12,)}
    - Action space: still Box([steer, throttle, brake])
    - The wrapper is transparent to RecurrentPPO

What changes under the hood:
    - vec[0] is now a discrete turn_token from the Worker (was: manual turn_bias)
    - vec[1] is now a go_signal from the Scheduler (was: always 0.0)
    - Actions are gated: if go_signal == 0, throttle is zeroed and brake applied

Integration:
    # Old way (hardcoded):
    env = IsaacDirectEnv(config)
    env.set_turn_bias(0.0)  # or random

    # New way (graph-based):
    env = IsaacDirectEnv(config)
    graph = IntersectionGraph.from_json("config/intersection_graph.json")
    env = AgentEnvWrapper(env, graph=graph, agent_config=AgentConfig())

    # SB3 training is unchanged:
    model = RecurrentPPO(policy=HierarchicalPathPlanningPolicy, env=env, ...)
    model.learn(total_timesteps=500_000)

Position Source:
    The wrapper needs the agent's world-frame position and heading every
    step so the Worker can query the intersection graph. There are two
    modes:

    1. Direct API (IsaacDirectEnv): position comes from PhysX via
       env._get_robot_position() and env._get_robot_yaw_rate().
       This is ground truth — fine for training.

    2. ROS2 deployment (IsaacROS2Env): position comes from EKF
       (robot_localization). Not yet wired — need to provide
       the /odometry/filtered topic. For now, the wrapper falls back
       to dead-reckoning from speed + yaw_rate, which is sufficient
       for the intersection trigger radius check.

Dependencies:
    - agent/agent_node.py (AgentNode, AgentConfig)
    - agent/intersection_graph.py (IntersectionGraph)
    - agent/worker_scheduler.py (WorkerScheduler) [optional]
    - gymnasium
    - numpy

Author: Aaron Hamil
Date: 03/12/26
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from agent.agent_node import AgentNode, AgentConfig, IDX_SPEED, IDX_YAW_RATE
from agent.intersection_graph import IntersectionGraph
from agent.worker_scheduler import WorkerScheduler

logger = logging.getLogger(__name__)


class AgentEnvWrapper(gym.Wrapper):
    """
    Wraps a driving environment with AgentNode (Worker + Main).

    Intercepts reset() and step() to:
    1. Run the Worker node (route planning + scheduler coordination)
    2. Inject turn_token and go_signal into observations
    3. Gate actions with go/brake safety override

    The inner environment is unmodified. Observation and action spaces
    pass through unchanged (Worker's output replaces vec[0:2] which
    were already allocated in the telemetry protocol).
    """

    def __init__(
        self,
        env: gym.Env,
        graph: IntersectionGraph,
        agent_config: Optional[AgentConfig] = None,
        scheduler: Optional[WorkerScheduler] = None,
        control_dt: float = 0.1,
    ):
        """
        Args:
            env: Inner environment (IsaacDirectEnv or IsaacROS2Env).
            graph: Intersection graph for route planning.
            agent_config: Configuration for the Agent's Worker and Main.
            scheduler: Optional multi-agent scheduler. If None, single-agent
                       mode — Worker always gets go_signal = 1.0.
            control_dt: Time between steps (seconds). Used for dead-reckoning
                        fallback when ground-truth position isn't available.
        """
        super().__init__(env)
        self.graph = graph
        self.scheduler = scheduler
        self.control_dt = control_dt

        self.agent = AgentNode(
            graph=graph,
            config=agent_config,
            scheduler=scheduler,
        )

        # Dead-reckoning state (fallback when no PhysX position)
        self._dr_x: float = 0.0
        self._dr_y: float = 0.0
        self._dr_heading: float = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment and Agent.

        No more set_turn_bias() — the Worker handles navigation from
        the first step based on position and the intersection graph.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        self.agent.reset()

        # Initialize dead-reckoning from spawn position
        self._init_position_from_env()

        # Run Worker on initial observation to set turn_token
        pos, heading, speed = self._get_agent_state(obs)
        self.agent.worker_step(pos, heading, speed, dt=self.control_dt)

        # Inject Worker's commands into observation
        obs = self.agent.prepare_obs(obs)

        # Add agent info to info dict
        info.update(self.agent.info)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step with Agent orchestration.

        Flow:
        1. Gate action with go/brake override (from last step's go_signal)
        2. Send gated action to inner env
        3. Get new observation
        4. Extract position from env (PhysX) or dead-reckoning
        5. Run Worker on new position -> new turn_token + go_signal
        6. Inject into observation for next policy forward pass
        """
        # 1. Apply go/brake safety gate on the action
        gated_action = self.agent.apply_action_gate(action)

        # 2. Step the inner environment
        obs, reward, terminated, truncated, info = self.env.step(gated_action)

        # 3. Scheduler housekeeping
        if self.scheduler is not None:
            self.scheduler.tick()

        # 4. Get agent's world-frame state
        pos, heading, speed = self._get_agent_state(obs)

        # 5. Worker step: check intersection, plan route, coordinate
        self.agent.worker_step(pos, heading, speed, dt=self.control_dt)

        # 6. Inject Worker's commands into observation
        obs = self.agent.prepare_obs(obs)

        # 7. Clear scheduler intent if agent left intersection
        if self.scheduler is not None and self.agent.worker.state == "cruising":
            self.scheduler.clear_agent(self.agent.agent_id)

        # Enrich info
        info.update(self.agent.info)
        info["action_gated"] = not np.array_equal(action, gated_action)

        return obs, reward, terminated, truncated, info

    # Position Extraction

    def _get_agent_state(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[Tuple[float, float], float, float]:
        """
        Extract agent position, heading, speed from environment.

        Tries PhysX ground truth first (IsaacDirectEnv), falls back
        to dead-reckoning from telemetry (IsaacROS2Env).

        Returns:
            ((x, y), heading_rad, speed_mps)
        """
        speed = float(obs["vec"][IDX_SPEED])
        yaw_rate = float(obs["vec"][IDX_YAW_RATE])

        # Try ground truth from IsaacDirectEnv
        inner = self.env
        # Unwrap through any other wrappers (Monitor, WaypointTracking)
        while hasattr(inner, 'env'):
            if hasattr(inner, '_get_robot_position'):
                break
            inner = inner.env

        if hasattr(inner, '_get_robot_position') and hasattr(inner, '_robot_articulation'):
            try:
                pos_3d = inner._get_robot_position()
                x, y = float(pos_3d[0]), float(pos_3d[1])

                # Get heading from angular velocity integration
                # or from the articulation's orientation
                if hasattr(inner, '_robot_articulation') and inner._robot_articulation is not None:
                    try:
                        quat = inner._robot_articulation.get_world_pose()[1]
                        # Quaternion to yaw: atan2(2(wz+xy), 1-2(yy+zz))
                        w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                        heading = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                    except Exception:
                        heading = self._dr_heading
                else:
                    heading = self._dr_heading

                # Update dead-reckoning state for consistency
                self._dr_x = x
                self._dr_y = y
                self._dr_heading = heading

                return (x, y), heading, speed

            except Exception:
                pass  # Fall through to dead-reckoning

        # Dead-reckoning fallback (IsaacROS2Env or error)
        self._dr_heading += yaw_rate * self.control_dt
        self._dr_x += speed * math.cos(self._dr_heading) * self.control_dt
        self._dr_y += speed * math.sin(self._dr_heading) * self.control_dt

        return (self._dr_x, self._dr_y), self._dr_heading, speed

    def _init_position_from_env(self) -> None:
        """Initialize dead-reckoning from env's spawn position."""
        inner = self.env
        while hasattr(inner, 'env'):
            if hasattr(inner, 'config'):
                break
            inner = inner.env

        if hasattr(inner, 'config'):
            config = inner.config
            self._dr_x = getattr(config, 'spawn_x', 0.0)
            self._dr_y = getattr(config, 'spawn_y', 0.0)
            self._dr_heading = getattr(config, 'spawn_yaw', 0.0)
        else:
            self._dr_x = 0.0
            self._dr_y = 0.0
            self._dr_heading = 0.0
