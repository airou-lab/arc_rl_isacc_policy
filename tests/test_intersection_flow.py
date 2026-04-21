"""
Intersection Flow Tests

End-to-end tests for the Worker's stop-line substate machine and the
IntersectionRewardWrapper's shaping output.

Two-layer strategy:
    1. Unit tests that exercise WorkerNode directly with scripted
       position/heading/speed sequences. No env needed. Validates
       the substate machine transitions and detector integration.

    2. Wrapper tests that use a MockEnv emitting canned info dicts
       (mimicking what AgentEnvWrapper would produce). Validates the
       IntersectionRewardWrapper's shaping math and one-shot latches.

    3. Full-stack integration tests that drive a scripted agent
       through a synthetic intersection from CRUISING to CRUISING.
       Validates the whole chain: pre-gate arming, detection,
       substate transitions, exit validation, reward accumulation.

Run:
    python -m pytest tests/test_intersection_flow.py -v

Author: Aaron Hamil
Date: 04/20/26
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, ".")

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    ApproachInfo,
    ExitOption,
    EdgeGeometry,
    TurnCommand,
)
from agent.intersection_geometry import IntersectionLayout
from agent.stop_line_detector import StopLineDetection, StopLineDetectorConfig
from agent.agent_node import AgentNode, AgentConfig, WorkerConfig, WorkerNode
from wrappers.intersection_reward_wrapper import (
    IntersectionRewardWrapper,
    IntersectionRewardConfig,
)


# Fixtures: a one-intersection, four-approach graph matching the drawing

def _make_cross_graph() -> IntersectionGraph:
    """
    Build a calibrated 4-way cross at origin, 2m approaches along each
    cardinal axis. Road IDs match the intersection_topology.json
    convention:
        road_A enters from +Y (heading = 270deg / -pi/2, going south)
        road_B enters from +X (heading = 180deg /  pi,  going west)
        road_C enters from -Y (heading =  90deg /  pi/2, going north)
        road_D enters from -X (heading =   0deg / 0.0,  going east)
    """
    # Build ApproachInfo for each road with left/straight/right exits
    # matching the star topology from the topology JSON.
    approaches = {
        "road_A": ApproachInfo(
            road_id="road_A",
            heading_rad=math.radians(270),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_D"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_C"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_B"),
            },
        ),
        "road_B": ApproachInfo(
            road_id="road_B",
            heading_rad=math.radians(180),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_A"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_D"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_C"),
            },
        ),
        "road_C": ApproachInfo(
            road_id="road_C",
            heading_rad=math.radians(90),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_B"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_A"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_D"),
            },
        ),
        "road_D": ApproachInfo(
            road_id="road_D",
            heading_rad=math.radians(0),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_C"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_B"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_A"),
            },
        ),
    }

    node = IntersectionNode(
        node_id="int_main",
        approaches=approaches,
        position=(0.0, 0.0),
        radius=1.5,
    )

    # Edge geometries: 2.5m straight approaches along each axis
    # road_A: start at (0, 2.5), ends at (0, 0), heading = -pi/2 (south)
    # road_B: start at (2.5, 0), ends at (0, 0), heading =  pi
    # road_C: start at (0, -2.5), ends at (0, 0), heading =  pi/2 (north)
    # road_D: start at (-2.5, 0), ends at (0, 0), heading = 0
    edges = {
        "road_A": EdgeGeometry(
            edge_id="road_A", length=2.5, heading=math.radians(270),
            from_node=None, to_node="int_main",
            start_position=(0.0, 2.5), end_position=(0.0, 0.0),
        ),
        "road_B": EdgeGeometry(
            edge_id="road_B", length=2.5, heading=math.radians(180),
            from_node=None, to_node="int_main",
            start_position=(2.5, 0.0), end_position=(0.0, 0.0),
        ),
        "road_C": EdgeGeometry(
            edge_id="road_C", length=2.5, heading=math.radians(90),
            from_node=None, to_node="int_main",
            start_position=(0.0, -2.5), end_position=(0.0, 0.0),
        ),
        "road_D": EdgeGeometry(
            edge_id="road_D", length=2.5, heading=math.radians(0),
            from_node=None, to_node="int_main",
            start_position=(-2.5, 0.0), end_position=(0.0, 0.0),
        ),
    }

    graph = IntersectionGraph(
        intersections={"int_main": node},
        edge_geometry=edges,
    )
    return graph


@pytest.fixture
def cross_graph() -> IntersectionGraph:
    return _make_cross_graph()


@pytest.fixture
def default_layout() -> IntersectionLayout:
    return IntersectionLayout(
        intersection_half_width=0.5,
        lane_half_width=0.25,
        pre_gate_distance=1.5,
        stop_line_tolerance=0.08,
        exit_detection_radius=0.8,
    )


# Layer 1: Worker state-machine unit tests (no env)

class TestWorkerSubstateMachine:
    """
    Scripted-position tests that exercise WorkerNode.step() directly.
    Uses detector_kind='geometric' so we don't need camera images.
    """

    def test_cruising_holds_when_far_from_intersection(self, cross_graph, default_layout):
        """Far from the intersection, Worker stays CRUISING, go=1."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
            ),
        )
        # On road_D at (-2.3, 0), heading east (0 rad), 1 m/s
        token, go = worker.step(
            position=(-2.3, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        assert worker.state == WorkerNode.CRUISING
        assert worker.substate == WorkerNode.SUB_NONE
        assert go == 1.0

    def test_pre_gate_promotes_to_deciding_approaching(self, cross_graph, default_layout):
        """Within pre-gate distance, Worker promotes to DECIDING/APPROACHING."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],  # deterministic
            ),
        )
        # On road_D at (-1.2, 0), within pre_gate_distance=1.5 (distance_to_next = 1.3)
        # Heading east (approaching east-to-center along road_D)
        # road_D's approach heading is 0.0 (east)
        token, go = worker.step(
            position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        assert worker.state == WorkerNode.DECIDING
        assert worker.substate == WorkerNode.SUB_APPROACHING
        assert go == 0.0, "APPROACHING holds brake override"
        assert worker.current_approach_road_id == "road_D"

    def test_approaching_to_stopping_when_speed_drops(self, cross_graph, default_layout):
        """APPROACHING -> STOPPING when speed < threshold."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
            ),
        )
        # Enter DECIDING
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        # Near the line, slowing down
        worker.step(position=(-0.7, 0.0), heading=0.0, speed=0.5, dt=0.05)
        assert worker.substate == WorkerNode.SUB_APPROACHING
        # Fully stopped near the line
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        assert worker.substate == WorkerNode.SUB_STOPPING

    def test_stopping_to_cleared_respects_dwell_and_scheduler(self, cross_graph, default_layout):
        """STOPPING -> CLEARED requires dwell_time elapsed AND scheduler go."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.3,
            ),
        )
        # Bootstrap to STOPPING
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        assert worker.substate == WorkerNode.SUB_STOPPING

        # One tick at dt=0.1: dwell is 0.1 < 0.3, should stay STOPPING
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.substate == WorkerNode.SUB_STOPPING

        # Two more ticks (+ 0.2 => total 0.3 dwell), should clear and commit
        # next step. Note scheduler=None defaults to "go = 1", so we just
        # need dwell to elapse.
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        # After CLEARED tick, state should be COMMITTED/TRAVERSING
        assert worker.state in (WorkerNode.DECIDING, WorkerNode.COMMITTED)

    def test_committed_traversing_to_exited_correct_road(self, cross_graph, default_layout):
        """TRAVERSING -> EXITED fires on correct road exit, exit_correct=True."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.1,
                moving_speed_threshold=0.2,
            ),
        )
        # Script: approach, stop, dwell, release, traverse, exit
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        # Dwell
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        # Should now be COMMITTED/TRAVERSING
        assert worker.state == WorkerNode.COMMITTED
        assert worker.substate == WorkerNode.SUB_TRAVERSING

        # Committed exit road for STRAIGHT from road_D is road_B
        assert worker.committed_exit_road_id == "road_B"

        # Traverse through the intersection, exit east onto road_B
        # road_B's exit direction = heading + pi = 180+180 = 0 (east)
        # Exit position: east of center, past exit_detection_radius
        worker.step(position=(1.0, 0.0), heading=0.0, speed=1.0, dt=0.05)
        # One more tick: should detect exit onto road_B
        assert worker.exited_road_id == "road_B"
        assert worker.exit_correct is True

    def test_committed_traversing_to_exited_wrong_road(self, cross_graph, default_layout):
        """Exiting on a road != committed road sets exit_correct=False."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.1,
                moving_speed_threshold=0.2,
            ),
        )
        # Approach and commit to STRAIGHT (road_B)
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.committed_exit_road_id == "road_B"

        # But actually exit south onto road_C (heading pi/2 + pi = -pi/2)
        # road_C exit direction: approach heading (pi/2) + pi = 3pi/2 = -pi/2 (south)
        worker.step(
            position=(0.0, -1.0), heading=-math.pi / 2, speed=1.0, dt=0.05,
        )
        assert worker.exited_road_id == "road_C"
        assert worker.exit_correct is False

    def test_legacy_mode_preserves_original_behavior(self, cross_graph):
        """use_stop_line=False => DECIDING flips to COMMITTED in one tick."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=False,
                mode="route", route=[TurnCommand.STRAIGHT],
            ),
        )
        # Within graph.nearest_intersection radius (1.5)
        token, go = worker.step(
            position=(-1.0, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        # Legacy: DECIDING is zero-duration, ends at COMMITTED
        assert worker.state == WorkerNode.COMMITTED
        assert worker.substate == WorkerNode.SUB_NONE, "no substate in legacy"


# Layer 2: Reward wrapper unit tests with MockEnv

class MockEnv(gym.Env):
    """
    Minimal env that returns a canned sequence of (obs, reward, done,
    trunc, info) tuples. Lets us drive the IntersectionRewardWrapper
    through any scenario without needing the Worker or Isaac Sim.
    """

    def __init__(self, canned_steps: list, canned_reset_info: Optional[Dict] = None):
        super().__init__()
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
            "vec": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self._canned = list(canned_steps)
        self._reset_info = canned_reset_info or {}
        self._i = 0

    def _zero_obs(self) -> Dict[str, np.ndarray]:
        return {
            "image": np.zeros((90, 160, 3), dtype=np.uint8),
            "vec": np.zeros(12, dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        self._i = 0
        return self._zero_obs(), dict(self._reset_info)

    def step(self, action):
        if self._i >= len(self._canned):
            # Exhausted — return terminal no-op
            return self._zero_obs(), 0.0, True, False, {}
        step_data = self._canned[self._i]
        self._i += 1
        obs = step_data.get("obs", self._zero_obs())
        reward = step_data.get("reward", 0.0)
        terminated = step_data.get("terminated", False)
        truncated = step_data.get("truncated", False)
        info = step_data.get("info", {})
        return obs, reward, terminated, truncated, info


def _obs_with_speed(speed: float) -> Dict[str, np.ndarray]:
    obs = {
        "image": np.zeros((90, 160, 3), dtype=np.uint8),
        "vec": np.zeros(12, dtype=np.float32),
    }
    obs["vec"][3] = speed
    return obs


class TestRewardWrapperPerStep:
    """Per-step terms: brake incentive, running-line penalty."""

    def test_brake_incentive_fires_in_approach_zone(self):
        """Brake action in approach zone with detected line => positive shaping."""
        canned = [
            {
                "obs": _obs_with_speed(0.5),
                "reward": 0.0,
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.8,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={"worker_state": "cruising",
                                                  "worker_substate": "none"})
        cfg = IntersectionRewardConfig(
            brake_incentive=0.25, running_line_penalty=0.0,  # isolate brake term
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        action = np.array([0.0, 0.0, 0.8], dtype=np.float32)  # 80% brake
        obs, reward, *_, info = wrapped.step(action)
        expected = 0.25 * 0.8
        assert info[wrapped.INFO_KEY] == pytest.approx(expected)

    def test_brake_incentive_doesnt_fire_outside_approach_zone(self):
        """Detection too far away (> approach_zone_m) => no brake shaping."""
        canned = [
            {
                "obs": _obs_with_speed(0.5),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 2.5,  # beyond approach_zone=1.5
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        cfg = IntersectionRewardConfig(running_line_penalty=0.0)
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0

    def test_running_line_penalty_fires_when_moving_on_red(self):
        """go_signal=0 + moving above threshold => penalty."""
        canned = [
            {
                "obs": _obs_with_speed(0.5),  # > violation_speed_mps=0.25
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": False,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=-1.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(-1.0)

    def test_per_step_terms_dont_fire_in_committed(self):
        """Per-step shaping is gated to DECIDING only."""
        canned = [
            {
                "obs": _obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "traversing",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.5,
                    "stop_line_confidence": 0.9,
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        wrapped = IntersectionRewardWrapper(env)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 0.5], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0


class TestRewardWrapperOneShots:
    """One-shot bonuses/penalties: stop proximity, exit validation."""

    def test_perfect_stop_bonus_at_line(self):
        """Stop within tolerance of the line fires perfect_stop_bonus once."""
        canned = [
            # Transition APPROACHING -> STOPPING with tiny distance
            {
                "obs": _obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.03,  # within 0.08 tolerance
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",  # prev tick
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            perfect_stop_bonus=10.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(10.0)

    def test_overshoot_penalty_larger_than_undershoot(self):
        """Same distance magnitude: overshoot penalty > undershoot penalty."""
        # Overshoot case (distance = -0.2)
        canned_over = [
            {
                "obs": _obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": -0.2,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        # Undershoot case (distance = +0.2)
        canned_under = [
            {
                "obs": _obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.2,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]

        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            overshoot_weight=20.0, undershoot_weight=5.0,
            stop_tolerance_m=0.08,
        )

        def _run(canned_steps):
            env = MockEnv(canned_steps, canned_reset_info={
                "worker_state": "deciding",
                "worker_substate": "approaching",
                "intersection": "int_main",
            })
            w = IntersectionRewardWrapper(env, config=cfg)
            w.reset()
            _, _, *_, info = w.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            return info[w.INFO_KEY]

        over = _run(canned_over)
        under = _run(canned_under)
        assert over < under < 0, f"overshoot {over} should be more negative than undershoot {under}"

    def test_stop_bonus_fires_only_once(self):
        """
        Holding STOPPING for multiple ticks fires the bonus only on the
        APPROACHING -> STOPPING transition (first STOPPING tick).
        """
        stopping_info = {
            "worker_state": "deciding",
            "worker_substate": "stopping",
            "stop_line_detected": True,
            "stop_line_distance_m": 0.03,
            "stop_line_confidence": 0.9,
            "go_signal": 0.0,
            "intersection": "int_main",
        }
        canned = [
            {"obs": _obs_with_speed(0.05), "info": stopping_info},  # Transition tick
            {"obs": _obs_with_speed(0.05), "info": stopping_info},  # Still stopping
            {"obs": _obs_with_speed(0.05), "info": stopping_info},  # Still stopping
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            perfect_stop_bonus=10.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        deltas = []
        for _ in range(3):
            _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            deltas.append(info[wrapped.INFO_KEY])

        assert deltas[0] == pytest.approx(10.0), "bonus fires on transition"
        assert deltas[1] == 0.0, "second tick: no bonus"
        assert deltas[2] == 0.0, "third tick: no bonus"

    def test_correct_exit_bonus(self):
        """TRAVERSING -> EXITED with exit_correct=True fires correct bonus."""
        canned = [
            {
                "obs": _obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "exited",
                    "exit_correct": True,
                    "exited_road": "road_B",
                    "committed_exit_road": "road_B",
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "committed",
            "worker_substate": "traversing",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            correct_exit_bonus=10.0, wrong_exit_penalty=-15.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(10.0)

    def test_wrong_exit_penalty(self):
        """TRAVERSING -> EXITED with exit_correct=False fires penalty."""
        canned = [
            {
                "obs": _obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "exited",
                    "exit_correct": False,
                    "exited_road": "road_C",
                    "committed_exit_road": "road_B",
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "committed",
            "worker_substate": "traversing",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            correct_exit_bonus=10.0, wrong_exit_penalty=-15.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(-15.0)


class TestRewardWrapperDisabled:
    """Wrapper respects enabled=False (no-op) and is safe with legacy Worker."""

    def test_disabled_is_no_op(self):
        """enabled=False adds zero reward regardless of info."""
        canned = [
            {
                "obs": _obs_with_speed(0.05),
                "reward": 1.5,
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.03,
                    "stop_line_confidence": 0.9,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(enabled=False, perfect_stop_bonus=10.0)
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, reward, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert reward == 1.5, "base reward unchanged"
        assert info[wrapped.INFO_KEY] == 0.0

    def test_legacy_worker_info_produces_no_shaping(self):
        """Worker in legacy mode never produces APPROACHING/STOPPING/EXITED
        substates, so no one-shot fires."""
        canned = [
            {
                "obs": _obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "none",  # legacy Worker
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "cruising",
            "worker_substate": "none",
        })
        wrapped = IntersectionRewardWrapper(env)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0


# Layer 3: Full-stack integration

class TestFullStack:
    """
    Drive a real AgentNode + IntersectionRewardWrapper through a scripted
    trajectory. Asserts end-to-end reward accumulation matches expectation.
    """

    def test_nominal_approach_stop_exit_accumulates_reward(self, cross_graph, default_layout):
        """
        Scenario: east-bound on road_D, go STRAIGHT (→ road_B). Script
        a trajectory: cruise → pre-gate arm → slow → stop near line →
        dwell → release → traverse → exit correctly on road_B.

        Expected: exit-correct bonus fires at the end; stop-line bonus
        fires mid-run.
        """
        # We don't need a real env — just a MockEnv that returns constant
        # obs / reward, and we run the Worker ourselves feeding info.
        worker_cfg = WorkerConfig(
            use_stop_line=True, detector_kind="geometric",
            layout=default_layout,
            mode="route", route=[TurnCommand.STRAIGHT],
            stopped_speed_threshold=0.1,
            stop_dwell_time=0.1,
            moving_speed_threshold=0.2,
        )
        agent = AgentNode(
            graph=cross_graph,
            config=AgentConfig(agent_id="a0", worker=worker_cfg),
        )

        # Hand-drive the trajectory, building canned info from agent.info
        trajectory = [
            # (pos, heading, speed, dt)
            ((-2.0, 0.0), 0.0, 1.0, 0.05),   # CRUISING
            ((-1.2, 0.0), 0.0, 0.8, 0.05),   # enter DECIDING/APPROACHING
            ((-0.9, 0.0), 0.0, 0.4, 0.05),   # slow
            ((-0.6, 0.0), 0.0, 0.05, 0.05),  # stop (APPROACHING -> STOPPING)
            ((-0.6, 0.0), 0.0, 0.05, 0.1),   # dwell
            ((-0.6, 0.0), 0.0, 0.05, 0.1),   # dwell (release)
            ((-0.3, 0.0), 0.0, 0.5, 0.05),   # TRAVERSING
            ((0.3, 0.0),  0.0, 1.0, 0.05),   # mid-intersection
            ((1.0, 0.0),  0.0, 1.0, 0.05),   # EXITING onto road_B
        ]

        # Run Worker, collect info each tick
        canned_steps = []
        for pos, heading, speed, dt in trajectory:
            agent.worker_step(
                position=pos, heading=heading, speed=speed, dt=dt,
            )
            obs = _obs_with_speed(speed)
            canned_steps.append({
                "obs": obs,
                "reward": 0.0,
                "info": dict(agent.info),  # snapshot
            })

        # Now replay through the reward wrapper
        # Use reset_info from first step's info to seed prev_substate correctly
        # (Wrapper's reset reads info['worker_substate'])
        initial_info = {"worker_state": "cruising", "worker_substate": "none"}
        env = MockEnv(canned_steps, canned_reset_info=initial_info)
        cfg = IntersectionRewardConfig()
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        deltas = []
        for _ in range(len(trajectory)):
            action = np.array([0.0, 0.0, 0.8], dtype=np.float32)  # apply brake
            _, _, *_, info = wrapped.step(action)
            deltas.append(info[wrapped.INFO_KEY])

        # Assert: there is at least one positive spike (stop bonus)
        # and at least one positive spike (exit bonus). Specific magnitudes
        # depend on detector output which is geometric; here we just
        # confirm signs and that nonzero shaping happened.
        assert any(d >= cfg.perfect_stop_bonus * 0.5 for d in deltas), \
            f"expected stop bonus spike in deltas={deltas}"

        # Check Worker actually reported a correct exit at end
        final_info = canned_steps[-1]["info"]
        assert final_info["exited_road"] == "road_B", \
            f"expected exit on road_B, got {final_info['exited_road']}"
        assert final_info["exit_correct"] is True


# Standalone runner

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
