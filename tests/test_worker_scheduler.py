"""
Worker Scheduler Tests

Pytest coverage for agent/worker_scheduler.py. Mirrors the four legacy
cases that live in test_all_so_far.py and adds the graph-calibrated
paths that were uncovered before the time-gap rewrite.

Two layers:
    1. No-graph cases — exercise the conservative fallback used by
       legacy single-process fixtures. These match what the previous
       (buggy) implementation produced for the same inputs and are the
       behavior every existing caller expects.

    2. Graph-calibrated cases — exercise the rewritten time-gap
       arbitration with a real IntersectionGraph. These are the tests
       that would have caught the agent-to-agent-distance bug.

Run:
    python -m pytest tests/test_worker_scheduler.py -v

Author: Aaron Hamil
Date: 04/25/26
"""

from __future__ import annotations

import math
import sys

import pytest

sys.path.insert(0, ".")

from agent.intersection_graph import (
    ApproachInfo,
    ExitOption,
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
)
from agent.worker_scheduler import (
    IntentPhase,
    SchedulerConfig,
    WorkerScheduler,
    _paths_conflict,
)


# Fixtures

@pytest.fixture
def calibrated_graph() -> IntersectionGraph:
    """A 4-way cross at the origin with calibrated center for time-gap math."""
    approaches = {
        "road_W": ApproachInfo(
            road_id="road_W", heading_rad=0.0,
            exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_E")},
        ),
        "road_S": ApproachInfo(
            road_id="road_S", heading_rad=math.radians(90),
            exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_N")},
        ),
        "road_E": ApproachInfo(
            road_id="road_E", heading_rad=math.radians(180),
            exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_W")},
        ),
        "road_N": ApproachInfo(
            road_id="road_N", heading_rad=math.radians(270),
            exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_S")},
        ),
    }
    node = IntersectionNode(
        node_id="int_main",
        approaches=approaches,
        position=(0.0, 0.0),
        radius=1.5,
    )
    return IntersectionGraph(intersections={"int_main": node})


# Layer 1 — Path-conflict heuristic (no scheduler instance needed)

class TestPathsConflict:
    """Geometry-free conflict matrix used as the first-pass filter."""

    def test_same_direction_no_conflict(self):
        # Two agents from the same approach following each other
        assert not _paths_conflict(
            TurnCommand.STRAIGHT, 0.0,
            TurnCommand.STRAIGHT, math.radians(10),
        )

    def test_opposite_straight_no_conflict(self):
        # Two agents going straight from opposite directions pass each other
        assert not _paths_conflict(
            TurnCommand.STRAIGHT, 0.0,
            TurnCommand.STRAIGHT, math.radians(180),
        )

    def test_opposite_with_left_conflicts(self):
        # Opposite directions, one turning left -> conflict
        assert _paths_conflict(
            TurnCommand.LEFT, 0.0,
            TurnCommand.STRAIGHT, math.radians(180),
        )

    def test_perpendicular_default_conflict(self):
        assert _paths_conflict(
            TurnCommand.STRAIGHT, 0.0,
            TurnCommand.STRAIGHT, math.radians(90),
        )

    def test_both_right_no_conflict(self):
        # Right-hand traffic: two right turns don't cross
        assert not _paths_conflict(
            TurnCommand.RIGHT, 0.0,
            TurnCommand.RIGHT, math.radians(90),
        )


# Layer 2 — Scheduler with no graph (legacy fallback path)

class TestSchedulerNoGraph:
    """
    Replicates the four cases from test_all_so_far.py and pins the
    "conservative block on conflict, no graph" semantics.
    """

    def test_single_agent_gets_go(self):
        sched = WorkerScheduler()
        go = sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        assert go == 1.0

    def test_perpendicular_conflict_blocks(self):
        sched = WorkerScheduler(SchedulerConfig(time_gap_seconds=10.0))
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, speed=1.0, position=(1.0, 0.0),
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.LEFT,
            heading=math.radians(90), speed=1.0, position=(0.0, 1.0),
        )
        assert go == 0.0

    def test_both_right_no_conflict(self):
        sched = WorkerScheduler()
        sched.register_intent(
            "a0", "int_main", TurnCommand.RIGHT,
            heading=0.0, speed=1.0, position=(1.0, 0.0),
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.RIGHT,
            heading=math.radians(90), speed=1.0, position=(0.0, 1.0),
        )
        assert go == 1.0

    def test_clear_removes_intent(self):
        sched = WorkerScheduler()
        sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        sched.clear_agent("a0")
        assert "a0" not in sched.active_intents


# Layer 3 — Phase transitions

class TestIntentPhase:
    """The DECIDING -> COMMITTED transition fires on the GO tick."""

    def test_first_in_queue_promotes_to_committed(self):
        sched = WorkerScheduler()
        sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        assert sched.active_intents["a0"].phase == IntentPhase.COMMITTED

    def test_blocked_agent_stays_deciding(self):
        sched = WorkerScheduler(SchedulerConfig(time_gap_seconds=10.0))
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, speed=1.0, position=(1.0, 0.0),
        )
        sched.register_intent(
            "a1", "int_main", TurnCommand.LEFT,
            heading=math.radians(90), speed=1.0, position=(0.0, 1.0),
        )
        # a1 is blocked -> stays DECIDING; a0 is head of queue -> COMMITTED
        assert sched.active_intents["a0"].phase == IntentPhase.COMMITTED
        assert sched.active_intents["a1"].phase == IntentPhase.DECIDING


# Layer 4 — Scheduler with calibrated graph (the new TTI path)

class TestSchedulerWithGraph:
    """
    These tests are the ones that would have caught the
    agent-to-agent-distance bug. They exercise the rewritten
    occupancy-interval math against a calibrated intersection.
    """

    def test_close_perpendicular_blocks(self, calibrated_graph):
        """Two perpendicular straights, both 1 m from center at 1 m/s.
        Time-gap exceeds nominal clearance window -> second arrival waits."""
        sched = WorkerScheduler(graph=calibrated_graph)
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-1.0, 0.0), speed=1.0,
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -1.0), speed=1.0,
        )
        assert go == 0.0

    def test_far_slow_second_arrival_clears(self, calibrated_graph):
        """Second agent so far away that the head will have cleared
        long before they arrive -> they get GO."""
        sched = WorkerScheduler(graph=calibrated_graph)
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-0.3, 0.0), speed=2.0,  # close, fast
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -10.0), speed=0.5,
        )
        assert go == 1.0

    def test_committed_agent_extends_window(self, calibrated_graph):
        """A COMMITTED head agent has its window stretched to "from now",
        so a perpendicular newcomer at moderate distance still blocks."""
        sched = WorkerScheduler(graph=calibrated_graph)
        # a0 registers and is immediately COMMITTED
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-0.4, 0.0), speed=0.5,
        )
        assert sched.active_intents["a0"].phase == IntentPhase.COMMITTED
        # a1 perpendicular, 1.5 m away at 1 m/s -> arrives in ~1 s
        # a0's exit time at center radius = (0.4 + 0.5) / 0.5 = 1.8 s
        # Margin 1.5 s -> block until 3.3 s; 1 s < 3.3 s -> WAIT
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -1.5), speed=1.0,
        )
        assert go == 0.0

    def test_uncalibrated_node_falls_back_to_block(self):
        """A graph whose node has no calibrated position falls back to
        the conservative no-graph behavior."""
        # Build a graph without a position
        approaches = {
            "road_W": ApproachInfo(
                road_id="road_W", heading_rad=0.0,
                exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_E")},
            ),
            "road_S": ApproachInfo(
                road_id="road_S", heading_rad=math.radians(90),
                exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_N")},
            ),
        }
        node = IntersectionNode(
            node_id="int_main", approaches=approaches, position=None,
        )
        graph = IntersectionGraph(intersections={"int_main": node})

        sched = WorkerScheduler(
            config=SchedulerConfig(time_gap_seconds=10.0),
            graph=graph,
        )
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, speed=1.0, position=(1.0, 0.0),
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), speed=1.0, position=(0.0, 1.0),
        )
        assert go == 0.0


# Layer 5 — Stale-intent expiry

class TestSchedulerTick:
    """Stale intents are GC'd by tick() after intent_timeout."""

    def test_tick_does_not_expire_fresh_intents(self):
        sched = WorkerScheduler(SchedulerConfig(intent_timeout=10.0))
        sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        sched.tick()
        assert "a0" in sched.active_intents

    def test_tick_expires_stale_intents(self, monkeypatch):
        """Patch time.monotonic to force expiry."""
        import agent.worker_scheduler as ws_mod

        clock = [0.0]
        monkeypatch.setattr(ws_mod.time, "monotonic", lambda: clock[0])

        sched = WorkerScheduler(SchedulerConfig(intent_timeout=5.0))
        sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        clock[0] = 10.0  # advance past timeout
        sched.tick()
        assert "a0" not in sched.active_intents


# Standalone runner

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
