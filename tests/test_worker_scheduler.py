"""
Worker Scheduler Tests

Pytest coverage for the WorkerScheduler facade and the SchedulerCore
it delegates to. Mirrors the four legacy cases that live in
test_all_so_far.py and adds the graph-calibrated paths that were
uncovered before the time-gap rewrite.

Layers:
    1. Path-conflict heuristic — geometry-free first-pass filter.
    2. No-graph cases — exercise the conservative fallback used by
       the legacy single-process fixtures.
    3. Phase transitions — DECIDING -> COMMITTED on the GO tick.
    4. Graph-calibrated cases — exercise the rewritten time-gap math.
    5. Stale-intent expiry — tick() GC.
    6. Facade pluggability — confirm the WorkerScheduler facade
       accepts a custom transport and routes through it.

Run:
    python -m pytest tests/test_worker_scheduler.py -v

Author: Aaron Hamil
Date: 04/28/26
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
from agent.scheduler_core import SchedulerCore
from agent.scheduler_transport import (
    ClearanceReply,
    IntentMessage,
    LocalTransport,
    SchedulerTransport,
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
        assert not _paths_conflict(
            TurnCommand.STRAIGHT, 0.0,
            TurnCommand.STRAIGHT, math.radians(10),
        )

    def test_opposite_straight_no_conflict(self):
        assert not _paths_conflict(
            TurnCommand.STRAIGHT, 0.0,
            TurnCommand.STRAIGHT, math.radians(180),
        )

    def test_opposite_with_left_conflicts(self):
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
        assert not _paths_conflict(
            TurnCommand.RIGHT, 0.0,
            TurnCommand.RIGHT, math.radians(90),
        )


# Layer 2 — Scheduler with no graph (legacy fallback path)

class TestSchedulerNoGraph:
    """
    Replicates the four cases from test_all_so_far.py and pins the
    "conservative block on conflict, no graph" semantics. The facade
    builds a default LocalTransport(SchedulerCore()) — these tests
    therefore exercise both layers end-to-end.
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
        assert sched.active_intents["a0"].phase == IntentPhase.COMMITTED
        assert sched.active_intents["a1"].phase == IntentPhase.DECIDING


# Layer 4 — Scheduler with calibrated graph (the new TTI path)

class TestSchedulerWithGraph:
    """
    These tests would have caught the agent-to-agent-distance bug.
    They exercise the rewritten occupancy-interval math against a
    calibrated intersection.
    """

    def test_close_perpendicular_blocks(self, calibrated_graph):
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
        sched = WorkerScheduler(graph=calibrated_graph)
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-0.3, 0.0), speed=2.0,
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -10.0), speed=0.5,
        )
        assert go == 1.0

    def test_committed_agent_extends_window(self, calibrated_graph):
        sched = WorkerScheduler(graph=calibrated_graph)
        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-0.4, 0.0), speed=0.5,
        )
        assert sched.active_intents["a0"].phase == IntentPhase.COMMITTED
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -1.5), speed=1.0,
        )
        assert go == 0.0

    def test_uncalibrated_node_falls_back_to_block(self):
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
        """
        Patch time.monotonic on the scheduler_core module — that's
        where the post-refactor arbitration core calls it from.
        """
        import agent.scheduler_core as core_mod

        clock = [0.0]
        monkeypatch.setattr(core_mod.time, "monotonic", lambda: clock[0])

        sched = WorkerScheduler(SchedulerConfig(intent_timeout=5.0))
        sched.register_intent("a0", "int_main", TurnCommand.STRAIGHT)
        clock[0] = 10.0
        sched.tick()
        assert "a0" not in sched.active_intents


# Layer 6 — Facade pluggability

class _RecordingTransport(SchedulerTransport):
    """
    Minimal transport that records every call. Used to confirm that
    the facade routes through the supplied transport rather than
    silently building a default.
    """

    def __init__(self):
        self.intents_received: list[IntentMessage] = []
        self.cleared: list[str] = []
        self.ticks: int = 0

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        self.intents_received.append(msg)
        return ClearanceReply(go_signal=1.0)

    def clear(self, agent_id: str) -> None:
        self.cleared.append(agent_id)

    def tick(self) -> None:
        self.ticks += 1


class TestFacadePluggability:
    """
    Confirm the WorkerScheduler facade accepts an injected transport
    and forwards every public-API call through it. This is the only
    test that exercises the facade-vs-core indirection directly.
    """

    def test_default_transport_is_local(self):
        sched = WorkerScheduler()
        assert isinstance(sched.transport, LocalTransport)

    def test_injected_transport_is_used(self):
        recorder = _RecordingTransport()
        sched = WorkerScheduler(transport=recorder)
        assert sched.transport is recorder

        go = sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            position=(1.0, 2.0), heading=3.0, speed=4.0,
        )
        assert go == 1.0
        assert len(recorder.intents_received) == 1
        msg = recorder.intents_received[0]
        assert msg.agent_id == "a0"
        assert msg.intersection_id == "int_main"
        assert msg.turn_command == TurnCommand.STRAIGHT
        assert msg.position == (1.0, 2.0)
        assert msg.heading == 3.0
        assert msg.speed == 4.0
        assert msg.phase == IntentPhase.DECIDING

        sched.clear_agent("a0")
        assert recorder.cleared == ["a0"]

        sched.tick()
        assert recorder.ticks == 1

    def test_active_intents_unavailable_for_non_local_transport(self):
        recorder = _RecordingTransport()
        sched = WorkerScheduler(transport=recorder)
        with pytest.raises(AttributeError):
            _ = sched.active_intents

    def test_local_transport_round_trip_via_core(self, calibrated_graph):
        """
        Build a SchedulerCore + LocalTransport explicitly and pass to
        the facade. Behavior should match the default-construction
        path bit-for-bit.
        """
        core = SchedulerCore(graph=calibrated_graph)
        transport = LocalTransport(core)
        sched = WorkerScheduler(transport=transport)

        sched.register_intent(
            "a0", "int_main", TurnCommand.STRAIGHT,
            heading=0.0, position=(-1.0, 0.0), speed=1.0,
        )
        go = sched.register_intent(
            "a1", "int_main", TurnCommand.STRAIGHT,
            heading=math.radians(90), position=(0.0, -1.0), speed=1.0,
        )
        assert go == 0.0  # same as TestSchedulerWithGraph::test_close_perpendicular_blocks
        # active_intents returns a fresh dict copy each call, so compare
        # contents (==), not identity (is).
        assert sched.active_intents == core.active_intents


# Standalone runner

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
