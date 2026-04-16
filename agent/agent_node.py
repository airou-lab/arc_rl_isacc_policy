"""
Agent Node - Hierarchical Driver-Worker Architecture

Each vehicle in the simulation is managed by one AgentNode, which contains two internal nodes that
run at different levels of abstraction.

AgentNode
|-- Worker (Route Planner - WHERE to go)
|   |-- Reads position from PhysX (training) / EKF (deployment)
|   |-- Consults IntersectionGraph for available exits
|   |-- Selects turn command (LEFT / STRAIGHT / RIGHT)
|   |-- Registers intent with external WorkerScheduler
|   |-- Receives go/wait signal from Scheduler
|
|-- Main (Vehicle Driver - HOW to get there
    |-- Receives camera image + telemetry vector
    | (Worker's turn_token in vec[0], go_signal in vec[1])
    |-- OUTER loop: direction decision (from turn_token)
    |-- INNER loop: go/brake decision (visual scene + go_signal)
    |-- Outputs [steer, throttle, brake] to vehicle actuators

Execution flow (every env.step()):
    1. Worker.step(position, heading, speed)
       -> queries graph, picks turn, talks to scheduler
       -> returns (turn_token, go_signal)

    2. turn_token and go_signal are injected into obs["vec"][0:2]

    3. Main.step(obs, policy)
       -> policy forward pass with enriched observation
       -> OUTER: turn_token conditions kinematic anchors
       -> INNER: go_signal gates throttle/brake
       -> returns action [steer, throttle, brake]

    4. action applied to vehicle

The Worker and Scheduler are classical (non-learned) planners. The Main uses the learned CNN+LSTM+PPO policy.
This seperation means that the policy never has to learn route planning, it only learns visual-motor control
conditioned on discrete navigation intent.

Telemetry Vector COntract (updated from experiment.py):
    vec[0] = turn_token {-1, 0, 1} from Worker (was turn_bias)
    vec[1] = go_signal {0.0, 1.0} from Scheduler (was reserved/0)
    vec [2..11] unchanged - speed, yaw_rate, last action, PVP zeros

Training Modes for Worker:
    "route": Follow pre-defined route (list of turn commands)
    "random": Sample uniformely from available exits
    "curriculum": Weighted sampling, straight-biased early, uniform late

Dependencies:
    agent/intersection_graph (IntersectionGraph, TurnCommand)
    agent/worker_scheduler.py
    numpy

Author: Aaron Hamil
Date: 03/12/26
Updated: 04/16/26
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
    ExitOption,
)

logger = logging.getLogger(__name__)


# Telemetry Indices (matches experiment.py)

IDX_TURN_TOKEN = 0     # Discrete turn command from Worker
IDX_GO_SIGNAL = 1      # Go/wait from Scheduler
IDX_GOAL_DIST = 2      # Zero-padded (PVP)
IDX_SPEED = 3
IDX_YAW_RATE = 4
IDX_LAST_STEER = 5
IDX_LAST_THROTTLE = 6
IDX_LAST_BRAKE = 7
IDX_LAT_ERR = 8        # Lateral error (m) — populated in Isaac Lab env, zeroed in PVP direct env
IDX_HDG_ERR = 9        # Heading error (rad) — populated in Isaac Lab env, zeroed in PVP direct env
IDX_KAPPA = 10         # Zero-padded (PVP)
IDX_DIST = 11


# Worker Node

@dataclass
class WorkerConfig:
    """Configuration for the Worker node inside an Agent."""
    mode: str = "random"                            # "route" | "random" | "curriculum"
    route: List[int] = field(default_factory=list)  # Pre-planned turns
    curriculum_straight_bias: float = 0.6           # Initial bias toward STRAIGHT
    curriculum_decay_steps: int = 100_000           # Steps to reach uniform sampling
    intersection_cooldown: float = 3.0              # Seconds before re-triggering
                                                    # same intersection


class WorkerNode:
    """
    Route planning node inside an Agent.

    Decides WHICH direction to go at intersections. Does NOT control
    the vehicle, that's Main's job. The Worker emits a discrete
    turn_token that gets injected into the observation vector for
    the policy to read.

    State machine:
        CRUISING -> (enter intersection radius) -> DECIDING
        DECIDING -> (pick turn, register with scheduler) -> COMMITTED
        COMMITTED -> (leave intersection radius) -> CRUISING

    The Worker holds its last committed turn_token until the agent
    leaves the intersection, so the policy has a stable signal
    throughout the maneuver.
    """

    # Worker states
    CRUISING = "cruising"       # On open road, no decision needed
    DECIDING = "deciding"       # At intersection, picking a turn
    COMMITTED = "committed"     # Turn chosen, waiting to clear intersection

    def __init__(
        self,
        agent_id: str,
        graph: IntersectionGraph,
        config: Optional[WorkerConfig] = None,
    ):
        self.agent_id = agent_id
        self.graph = graph
        self.config = config or WorkerConfig()

        # State
        self._state = self.CRUISING
        self._current_intersection: Optional[str] = None
        self._turn_token: int = TurnCommand.STRAIGHT
        self._go_signal: float = 1.0   # Default: go (no scheduler block)
        self._route_index: int = 0
        self._total_steps: int = 0

        # Cooldown: prevent re-triggering the same intersection
        # immediately after leaving it
        self._last_intersection_id: Optional[str] = None
        self._cooldown_remaining: float = 0.0

    @property
    def turn_token(self) -> int:
        """Current turn command for the policy."""
        return self._turn_token

    @property
    def go_signal(self) -> float:
        """Current go/wait signal from Scheduler."""
        return self._go_signal

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_intersection_id(self) -> Optional[str]:
        return self._current_intersection

    def step(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float = 1.0 / 10.0,
        scheduler=None,
    ) -> Tuple[int, float]:
        """
        Worker step using GLOBAL position (PhysX ground truth).

        Called every environment step by AgentNode.step().
        For topological EKF position, use step_topological() instead.

        Args:
            position: (x, y) in world frame from PhysX or EKF global estimate.
            heading: Agent heading in radians.
            speed: Agent speed in m/s.
            dt: Time since last step (for cooldown decay).
            scheduler: Optional WorkerScheduler for multi-agent.

        Returns:
            (turn_token, go_signal) to inject into vec[0:2].
        """
        self._total_steps += 1

        # Decay cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)

        x, y = position
        intersection = self.graph.nearest_intersection(x, y)

        # State transitions
        if self._state == self.CRUISING:
            if intersection is not None:
                # Check cooldown, don't re-trigger the one we just left
                if (
                    intersection.node_id == self._last_intersection_id
                    and self._cooldown_remaining > 0
                ):
                    pass  # Still in cooldown, stay CRUISING
                else:
                    self._state = self.DECIDING
                    self._current_intersection = intersection.node_id
                    self._decide_turn(intersection, heading, speed, position, scheduler)
                    self._state = self.COMMITTED

        elif self._state == self.COMMITTED:
            if intersection is None:
                # Left the intersection -> back to cruising
                self._last_intersection_id = self._current_intersection
                self._cooldown_remaining = self.config.intersection_cooldown
                self._current_intersection = None
                self._state = self.CRUISING
                # Keep turn_token stable, Main is still executing
            else:
                # Still in intersection, update scheduler coordination
                if scheduler is not None:
                    self._go_signal = scheduler.query_go_signal(
                        self.agent_id,
                        self._current_intersection,
                        self._turn_token,
                        position,
                        heading,
                        speed,
                    )

        # On open road, go_signal is always 1.0 (no intersection conflict)
        # NOTE: Subject to change given brakecheck
        if self._state == self.CRUISING:
            self._go_signal = 1.0

        return self._turn_token, self._go_signal

    def step_topological(
        self,
        edge_id: str,
        at_intersection: bool,
        downstream_node_id: Optional[str],
        heading: float,
        speed: float,
        position: Tuple[float, float] = (0.0, 0.0),
        dt: float = 1.0 / 10.0,
        scheduler=None,
    ) -> Tuple[int, float]:
        """
        Worker step using TOPOLOGICAL position from EKF.

        Instead of converting back to global coords and doing a radius
        check, this directly uses the EKF's knowledge of which edge
        we're on and whether we've reached the downstream node.

        This eliminates the failure mode where global drift causes
        missed intersection triggers.

        Args:
            edge_id: Current road segment from EKF.
            at_intersection: True if EKF says s >= (edge_length - threshold).
            downstream_node_id: The intersection at the end of this edge.
            heading: Edge heading + theta_err from EKF.
            speed: Current speed from EKF.
            position: Global position estimate (for scheduler only).
            dt: Time since last step.
            scheduler: Optional WorkerScheduler.

        Returns:
            (turn_token, go_signal) to inject into vec[0:2].
        """
        self._total_steps += 1

        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)

        # State transitions (topological)

        if self._state == self.CRUISING:
            if at_intersection and downstream_node_id is not None:
                if (
                    downstream_node_id == self._last_intersection_id
                    and self._cooldown_remaining > 0
                ):
                    pass  # Cooldown active
                else:
                    intersection = self.graph.get_intersection(downstream_node_id)
                    if intersection is not None:
                        self._state = self.DECIDING
                        self._current_intersection = downstream_node_id
                        self._decide_turn(
                            intersection, heading, speed, position, scheduler
                        )
                        self._state = self.COMMITTED

        elif self._state == self.COMMITTED:
            if not at_intersection:
                # Left intersection (EKF transitioned to new edge, s is small)
                self._last_intersection_id = self._current_intersection
                self._cooldown_remaining = self.config.intersection_cooldown
                self._current_intersection = None
                self._state = self.CRUISING
            else:
                if scheduler is not None:
                    self._go_signal = scheduler.query_go_signal(
                        self.agent_id,
                        self._current_intersection,
                        self._turn_token,
                        position,
                        heading,
                        speed,
                    )

        if self._state == self.CRUISING:
            self._go_signal = 1.0

        return self._turn_token, self._go_signal

    def _decide_turn(
        self,
        intersection: IntersectionNode,
        heading: float,
        speed: float,
        position: Tuple[float, float],
        scheduler=None,
    ) -> None:
        """
        Pick a turn command based on the Worker's mode.

        Modifies self._turn_token and self._go_signal in place.
        """
        exits = self.graph.get_exit_options(intersection.node_id, heading)

        if not exits:
            # No recognized approach — default to straight
            logger.debug(
                f"[{self.agent_id}] No exits at {intersection.node_id} "
                f"for heading {math.degrees(heading):.0f}deg, defaulting STRAIGHT"
            )
            self._turn_token = TurnCommand.STRAIGHT
            return

        available_commands = [e.turn_command for e in exits]

        if self.config.mode == "route":
            self._turn_token = self._pick_from_route(available_commands)
        elif self.config.mode == "curriculum":
            self._turn_token = self._pick_curriculum(available_commands)
        else:  # "random"
            self._turn_token = self._pick_random(available_commands)

        # Register with scheduler for multi-agent coordination
        if scheduler is not None:
            self._go_signal = scheduler.register_intent(
                agent_id=self.agent_id,
                intersection_id=intersection.node_id,
                turn_command=self._turn_token,
                position=position,
                heading=heading,
                speed=speed,
            )
        else:
            self._go_signal = 1.0  # No scheduler = always go

        logger.debug(
            f"[{self.agent_id}] At {intersection.node_id}: "
            f"chose {TurnCommand.name(self._turn_token)} "
            f"(available: {[TurnCommand.name(c) for c in available_commands]}) "
            f"go={self._go_signal}"
        )

    def _pick_from_route(self, available: List[int]) -> int:
        """Pick next turn from pre-defined route sequence."""
        if not self.config.route:
            return TurnCommand.STRAIGHT

        cmd = self.config.route[self._route_index % len(self.config.route)]
        self._route_index += 1

        # If planned turn isn't available, fall back to straight or first available
        if cmd not in available:
            if TurnCommand.STRAIGHT in available:
                return TurnCommand.STRAIGHT
            return available[0]
        return cmd

    def _pick_random(self, available: List[int]) -> int:
        """Uniform random from available exits."""
        return int(np.random.choice(available))

    def _pick_curriculum(self, available: List[int]) -> int:
        """
        Weighted sampling: straight-biased early, uniform late

        Early training: agent mostly goes straight, learning basic
        lane following before tackling turns. As training progresses,
        the distribution shifts toward uniform sampling over all exits.
        """
        progress = min(1.0, self._total_steps / max(1, self.config.curriculum_decay_steps))
        straight_weight = self.config.curriculum_straight_bias * (1.0 - progress)

        weights = []
        for cmd in available:
            if cmd == TurnCommand.STRAIGHT:
                weights.append(1.0 + straight_weight)
            else:
                weights.append(1.0)

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        return int(np.random.choice(available, p=weights))

    def reset(self) -> None:
        """Reset Worker state for new episode (not training progress)."""
        self._state = self.CRUISING
        self._current_intersection = None
        self._turn_token = TurnCommand.STRAIGHT
        self._go_signal = 1.0
        self._last_intersection_id = None
        self._cooldown_remaining = 0.0
        # NOTE: _route_index and _total_steps persist across episodes


# Main Node

class MainNode:
    """
    Vehicle driver node inside an Agent.

    The Main node doesn't own the policy, the policy lives in SB3's
    RecurrentPPO and is called by the training loop. What Main does is:

    1. Inject Worker's commands into the observation before the policy
       sees it (turn_token -> vec[0], go_signal -> vec[1])

    2. Post-process the policy's raw action output with the go/brake
       inner loop: if go_signal == 0 (Scheduler says wait), override
       throttle to 0 and apply brake regardless of what the policy
       wants. This is a safety layer, not learned behavior.

    The nested loop structure is:
        OUTER: Worker decides direction -> turn_token conditions anchors
        INNER: Scheduler + visual scene -> go or brake
            - Scheduler go_signal == 0 -> hard brake (safety override)
            - Scheduler go_signal == 1 -> policy controls throttle/brake
              (policy still sees go_signal in vec[1] so it can learn
               to anticipate stops, but the override catches failures)

    This means:
        - The policy LEARNS to stop when go_signal is 0 (from experience)
        - The safety override GUARANTEES it stops (even if policy ignores)
        - Over training, the policy's behavior converges with the override
          and the override triggers less often
    """

    def __init__(self, agent_id: str, brake_decel: float = 0.8):
        """
        Args:
            agent_id: Parent agent identifier
            brake_decel: How hard to brake when go_signal is 0
                         Range [0, 1] where 1.0 = full brake
        """
        self.agent_id = agent_id
        self.brake_decel = brake_decel

    def prepare_observation(
        self,
        obs: Dict[str, np.ndarray],
        turn_token: int,
        go_signal: float,
    ) -> Dict[str, np.ndarray]:
        """
        Inject Worker's commands into the observation vector

        This is called BEFORE the policy forward pass so the policy
        can read the turn_token and go_signal as part of its input

        Args:
            obs: Raw observation from environment
            turn_token: Worker's discrete turn command {-1, 0, 1}
            go_signal: Scheduler's go/wait {0.0, 1.0}

        Returns:
            Modified observation dict (vec[0:2] overwritten)
        """
        obs = dict(obs)  # Shallow copy to avoid mutating env's obs
        vec = obs["vec"].copy()
        vec[IDX_TURN_TOKEN] = float(turn_token)
        vec[IDX_GO_SIGNAL] = float(go_signal)
        obs["vec"] = vec
        return obs

    def apply_go_brake_gate(
        self,
        action: np.ndarray,
        go_signal: float,
    ) -> np.ndarray:
        """
        Inner-loop go/brake safety override.

        If the Scheduler says WAIT (go_signal == 0), we override the
        policy's throttle/brake output to force a stop. This is the
        hard safety layer, the policy also sees go_signal and should
        learn to stop on its own, but this catches policy failures.

        If go_signal == 1 (GO), the policy's action passes through
        unchanged, the policy owns throttle/brake decisions during
        normal driving.

        Args:
            action: [steer, throttle, brake] from policy.
            go_signal: 0.0 = WAIT, 1.0 = GO.

        Returns:
            Potentially modified action array.
        """
        if go_signal < 0.5:
            # WAIT: override to stop
            action = action.copy()
            action[1] = 0.0                  # Zero throttle
            action[2] = self.brake_decel     # Apply brake
        return action


# Agent Node (Top-Level)

@dataclass
class AgentConfig:
    """Configuration for one Agent."""
    agent_id: str = "agent_0"
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    brake_decel: float = 0.8


class AgentNode:
    """
    Top-level agent containing Worker and Main nodes.

    One AgentNode per vehicle. The environment creates AgentNodes and
    calls agent.step() each timestep, threading position data to the
    Worker and observation data to the Main.

    Integration with IsaacDirectEnv:
        The env calls:
            1. agent.worker_step(pos, heading, speed, dt, scheduler)
               -> gets (turn_token, go_signal)
            2. agent.prepare_obs(raw_obs)
               -> injects token + signal into obs
            3. Policy runs on prepared obs -> raw_action
            4. agent.apply_action_gate(raw_action)
               -> applies go/brake safety override
            5. env applies gated action to vehicle

    Integration with training loop (train_policy_ros2.py):
        The AgentNode doesn't interfere with SB3's training loop.
        It wraps the observation before SB3 sees it and gates the
        action after SB3 produces it. From SB3's perspective, the
        observation space is unchanged (still Dict with image + vec),
        and the action space is unchanged (still Box [steer, thr, brk]).
        The Worker's decisions appear as part of the observation.
    """

    def __init__(
        self,
        graph: IntersectionGraph,
        config: Optional[AgentConfig] = None,
        scheduler=None,
    ):
        """
        Args:
            graph: Shared IntersectionGraph (read-only).
            config: Agent configuration.
            scheduler: Optional WorkerScheduler for multi-agent.
        """
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id
        self.scheduler = scheduler

        # Internal nodes
        self.worker = WorkerNode(
            agent_id=self.agent_id,
            graph=graph,
            config=self.config.worker,
        )
        self.main = MainNode(
            agent_id=self.agent_id,
            brake_decel=self.config.brake_decel,
        )

        # Latest state for logging / debugging
        self._last_turn_token: int = TurnCommand.STRAIGHT
        self._last_go_signal: float = 1.0

    def worker_step(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float = 0.1,
    ) -> Tuple[int, float]:
        """
        Run the Worker node for this timestep.

        Queries the intersection graph, picks a turn if at intersection,
        coordinates with Scheduler if present.

        Args:
            position: (x, y) from PhysX ground truth or EKF.
            heading: Current heading in radians.
            speed: Current speed in m/s.
            dt: Time since last call.

        Returns:
            (turn_token, go_signal) to be injected into obs.
        """
        token, go = self.worker.step(
            position=position,
            heading=heading,
            speed=speed,
            dt=dt,
            scheduler=self.scheduler,
        )
        self._last_turn_token = token
        self._last_go_signal = go
        return token, go

    def worker_step_topological(
        self,
        ekf_state,
        downstream_node_id: Optional[str],
        position: Tuple[float, float] = (0.0, 0.0),
        dt: float = 0.1,
    ) -> Tuple[int, float]:
        """
        Run the Worker using topological EKF state.

        Uses the EKF's edge-based position directly instead of
        converting back to global coordinates. This is the deployment
        path — robust to odometry drift because intersection detection
        is based on arc-length along the edge, not global distance.

        Args:
            ekf_state: TopologicalState from the EKF.
            downstream_node_id: Intersection at end of current edge.
            position: Global estimate (for scheduler RVO only).
            dt: Time since last call.

        Returns:
            (turn_token, go_signal) to be injected into obs.
        """
        token, go = self.worker.step_topological(
            edge_id=ekf_state.edge_id,
            at_intersection=ekf_state.at_intersection,
            downstream_node_id=downstream_node_id,
            heading=ekf_state.edge_heading + ekf_state.theta_err,
            speed=ekf_state.speed,
            position=position,
            dt=dt,
            scheduler=self.scheduler,
        )
        self._last_turn_token = token
        self._last_go_signal = go
        return token, go

    def prepare_obs(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Inject Worker's commands into observation for the policy.

        Call this AFTER worker_step() and BEFORE policy forward pass.
        """
        return self.main.prepare_observation(
            obs,
            turn_token=self._last_turn_token,
            go_signal=self._last_go_signal,
        )

    def apply_action_gate(self, action: np.ndarray) -> np.ndarray:
        """
        Apply go/brake safety override on policy's raw action.

        Call this AFTER policy produces action, BEFORE sending to vehicle.
        """
        return self.main.apply_go_brake_gate(action, self._last_go_signal)

    def reset(self) -> None:
        """Reset for new episode."""
        self.worker.reset()
        self._last_turn_token = TurnCommand.STRAIGHT
        self._last_go_signal = 1.0

    @property
    def info(self) -> Dict:
        """Current agent state for logging."""
        return {
            "agent_id": self.agent_id,
            "worker_state": self.worker.state,
            "turn_token": self._last_turn_token,
            "turn_name": TurnCommand.name(self._last_turn_token),
            "go_signal": self._last_go_signal,
            "intersection": self.worker.current_intersection_id,
        }
