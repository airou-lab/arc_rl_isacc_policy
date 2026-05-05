"""
Tests for IntersectionNodeServer (stage 3 of the multi-agent rollout).

Three groups:
    1. Conflict matrix construction — sizes, expected entries.
    2. Approach resolution — on-axis, off-axis, unknown intersection.
    3. Parity with SchedulerCore — single-agent behavior must be
       byte-equivalent so the first clean training run is not affected
       by the existence of the server class.

Plus targeted multi-agent scenarios that exercise the matrix path the
single-process design relies on.
"""

from __future__ import annotations

import math
import random
from typing import Dict

import pytest

from agent.intersection_graph import (
    ApproachInfo,
    ExitOption,
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
)
from agent.scheduler_core import (
    IntentPhase,
    SchedulerConfig,
    SchedulerCore,
)
from agent.intersection_node_server import (
    APPROACH_TOLERANCE_RAD,
    IntersectionNodeServer,
)


# fixtures

# Heading convention from agent/planar_planner.py: world frame +X right,
# +Y up, heading 0 = +X. Approach heading is the direction the car
# FACES when driving toward the intersection center.

HEADING_FACING_EAST  = 0.0
HEADING_FACING_NORTH = math.pi / 2.0
HEADING_FACING_WEST  = math.pi
HEADING_FACING_SOUTH = -math.pi / 2.0


def _exit(turn: int, road: str) -> ExitOption:
    return ExitOption(turn_command=turn, exit_road_id=road)


def make_4way_graph(calibrated: bool = True) -> IntersectionGraph:
    """4-way intersection at origin. Roads 'north','south','east','west'."""
    approaches: Dict[str, ApproachInfo] = {
        # Coming from the north going south -> facing south
        "north": ApproachInfo(
            road_id="north",
            heading_rad=HEADING_FACING_SOUTH,
            exits={
                TurnCommand.LEFT:     _exit(TurnCommand.LEFT,     "east"),
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "south"),
                TurnCommand.RIGHT:    _exit(TurnCommand.RIGHT,    "west"),
            },
        ),
        # Coming from the south going north -> facing north
        "south": ApproachInfo(
            road_id="south",
            heading_rad=HEADING_FACING_NORTH,
            exits={
                TurnCommand.LEFT:     _exit(TurnCommand.LEFT,     "west"),
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "north"),
                TurnCommand.RIGHT:    _exit(TurnCommand.RIGHT,    "east"),
            },
        ),
        # Coming from the east going west -> facing west
        "east": ApproachInfo(
            road_id="east",
            heading_rad=HEADING_FACING_WEST,
            exits={
                TurnCommand.LEFT:     _exit(TurnCommand.LEFT,     "south"),
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "west"),
                TurnCommand.RIGHT:    _exit(TurnCommand.RIGHT,    "north"),
            },
        ),
        # Coming from the west going east -> facing east
        "west": ApproachInfo(
            road_id="west",
            heading_rad=HEADING_FACING_EAST,
            exits={
                TurnCommand.LEFT:     _exit(TurnCommand.LEFT,     "north"),
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "east"),
                TurnCommand.RIGHT:    _exit(TurnCommand.RIGHT,    "south"),
            },
        ),
    }
    pos = (0.0, 0.0) if calibrated else None
    node = IntersectionNode(
        node_id="A",
        approaches=approaches,
        position=pos,
        radius=3.0,
    )
    return IntersectionGraph({"A": node})


def make_3way_graph() -> IntersectionGraph:
    """3-way 'T' intersection at origin. Roads 'north','east','west'."""
    approaches: Dict[str, ApproachInfo] = {
        "north": ApproachInfo(
            road_id="north",
            heading_rad=HEADING_FACING_SOUTH,
            exits={
                TurnCommand.LEFT:  _exit(TurnCommand.LEFT,  "east"),
                TurnCommand.RIGHT: _exit(TurnCommand.RIGHT, "west"),
            },
        ),
        "east": ApproachInfo(
            road_id="east",
            heading_rad=HEADING_FACING_WEST,
            exits={
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "west"),
                TurnCommand.RIGHT:    _exit(TurnCommand.RIGHT,    "north"),
            },
        ),
        "west": ApproachInfo(
            road_id="west",
            heading_rad=HEADING_FACING_EAST,
            exits={
                TurnCommand.STRAIGHT: _exit(TurnCommand.STRAIGHT, "east"),
                TurnCommand.LEFT:     _exit(TurnCommand.LEFT,     "north"),
            },
        ),
    }
    node = IntersectionNode(
        node_id="T",
        approaches=approaches,
        position=(0.0, 0.0),
        radius=3.0,
    )
    return IntersectionGraph({"T": node})


@pytest.fixture
def graph_4way():
    return make_4way_graph()


@pytest.fixture
def graph_3way():
    return make_3way_graph()


@pytest.fixture
def server_4way(graph_4way):
    return IntersectionNodeServer(graph=graph_4way)


# 1. matrix construction

class TestConflictMatrix:

    def test_matrix_size_4way(self, server_4way):
        # 4 approaches x 4 approaches x 3 turns x 3 turns = 144
        assert server_4way.conflict_matrix_size == 144

    def test_matrix_size_3way(self, graph_3way):
        server = IntersectionNodeServer(graph=graph_3way)
        # 3 approaches x 3 approaches x 3 turns x 3 turns = 81
        assert server.conflict_matrix_size == 81

    def test_same_approach_does_not_conflict(self, server_4way):
        # Agents on the same approach are following each other.
        for turn_a in TurnCommand.all():
            for turn_b in TurnCommand.all():
                key = ("A", "north", turn_a, "north", turn_b)
                assert server_4way._conflict_matrix[key] is False, (
                    f"same-approach pair should not conflict: {key}"
                )

    def test_opposite_straight_no_conflict(self, server_4way):
        # north <-> south, both straight: classic through-traffic.
        assert server_4way._conflict_matrix[
            ("A", "north", TurnCommand.STRAIGHT, "south", TurnCommand.STRAIGHT)
        ] is False
        assert server_4way._conflict_matrix[
            ("A", "east", TurnCommand.STRAIGHT, "west", TurnCommand.STRAIGHT)
        ] is False

    def test_opposite_left_conflicts(self, server_4way):
        # Opposite-direction left turns cross paths.
        assert server_4way._conflict_matrix[
            ("A", "north", TurnCommand.LEFT, "south", TurnCommand.LEFT)
        ] is True

    def test_perpendicular_straight_conflicts(self, server_4way):
        # north (facing south) vs east (facing west): perpendicular,
        # both straight: classic broadside conflict.
        assert server_4way._conflict_matrix[
            ("A", "north", TurnCommand.STRAIGHT, "east", TurnCommand.STRAIGHT)
        ] is True

    def test_both_right_no_conflict(self, server_4way):
        # Right-on-right never conflicts in right-hand traffic.
        for road_a in ("north", "south", "east", "west"):
            for road_b in ("north", "south", "east", "west"):
                key = ("A", road_a, TurnCommand.RIGHT, road_b, TurnCommand.RIGHT)
                assert server_4way._conflict_matrix[key] is False, (
                    f"both-right should not conflict: {key}"
                )

    def test_no_graph_raises(self):
        with pytest.raises(ValueError):
            IntersectionNodeServer(graph=None)


#  2. approach resolution

class TestApproachResolution:

    def test_on_axis_resolves(self, server_4way):
        # Heading exactly along an approach axis: must resolve.
        assert server_4way._resolve_approach("A", HEADING_FACING_SOUTH) == "north"
        assert server_4way._resolve_approach("A", HEADING_FACING_NORTH) == "south"
        assert server_4way._resolve_approach("A", HEADING_FACING_WEST)  == "east"
        assert server_4way._resolve_approach("A", HEADING_FACING_EAST)  == "west"

    def test_within_tolerance_resolves(self, server_4way):
        # 20deg off axis is inside the 30deg tolerance: should resolve.
        eps = math.radians(20.0)
        assert server_4way._resolve_approach("A", HEADING_FACING_SOUTH + eps) == "north"

    def test_outside_tolerance_does_not_resolve(self, server_4way):
        # Halfway between two approaches: 45deg off either axis.
        between = (HEADING_FACING_SOUTH + HEADING_FACING_WEST) / 2.0
        assert server_4way._resolve_approach("A", between) is None

    def test_unknown_intersection_returns_none(self, server_4way):
        assert server_4way._resolve_approach("DOES_NOT_EXIST", 0.0) is None


# 3. parity with SchedulerCore

class TestParityWithSchedulerCore:
    """
    Parity guarantee: for single-agent rollouts the server's go_signal
    sequence must match a SchedulerCore's exactly. This is the
    "first clean training run is unaffected" check from the design doc.
    """

    @pytest.mark.parametrize("seed", list(range(8)))
    def test_single_agent_random_walk_parity(self, graph_4way, seed):
        rng = random.Random(seed)
        config = SchedulerConfig()
        core = SchedulerCore(config=config, graph=graph_4way)
        server = IntersectionNodeServer(config=config, graph=graph_4way)

        # Random sequence of register / query / clear / tick on a single
        # agent. The queue head always returns GO regardless of conflict
        # logic, so this is the byte-equivalence test.
        approaches = [
            HEADING_FACING_SOUTH, HEADING_FACING_NORTH,
            HEADING_FACING_WEST, HEADING_FACING_EAST,
        ]
        turns = TurnCommand.all()
        registered = False

        for _ in range(40):
            op = rng.choice(["register", "query", "tick", "clear"])
            heading = rng.choice(approaches) + rng.uniform(-0.1, 0.1)
            turn = rng.choice(turns)
            pos = (rng.uniform(-5, 5), rng.uniform(-5, 5))
            speed = rng.uniform(0.0, 2.0)

            if op == "register" or (op == "query" and not registered):
                go_a = core.register_intent(
                    "agent_a", "A", turn, pos, heading, speed,
                )
                go_b = server.register_intent(
                    "agent_a", "A", turn, pos, heading, speed,
                )
                assert go_a == go_b
                registered = True
            elif op == "query":
                go_a = core.query_go_signal(
                    "agent_a", "A", turn, pos, heading, speed,
                )
                go_b = server.query_go_signal(
                    "agent_a", "A", turn, pos, heading, speed,
                )
                assert go_a == go_b
            elif op == "clear":
                core.clear_agent("agent_a")
                server.clear_agent("agent_a")
                registered = False
            elif op == "tick":
                core.tick()
                server.tick()

    def test_solo_agent_always_gets_go(self, server_4way):
        go = server_4way.register_intent(
            "agent_a", "A", TurnCommand.STRAIGHT,
            position=(5.0, 0.0), heading=HEADING_FACING_WEST, speed=1.0,
        )
        assert go == 1.0
        # Phase advances to COMMITTED on the GO tick.
        assert server_4way.active_intents["agent_a"].phase == IntentPhase.COMMITTED


# 4. multi-agent scenarios

class TestMultiAgent:

    def _far_pose(self, heading: float, dist: float = 5.0):
        # Place the agent `dist` meters back along its approach axis
        # from the origin, facing the intersection.
        x = -dist * math.cos(heading)
        y = -dist * math.sin(heading)
        return (x, y)

    def test_perpendicular_blocks_second(self, server_4way):
        # Agent A registers from north going straight (S-bound).
        go_a = server_4way.register_intent(
            "A", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_SOUTH),
            heading=HEADING_FACING_SOUTH, speed=1.0,
        )
        assert go_a == 1.0

        # Agent B registers from east going straight (W-bound).
        # Perpendicular, both straight => conflict.
        go_b = server_4way.register_intent(
            "B", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_WEST),
            heading=HEADING_FACING_WEST, speed=1.0,
        )
        assert go_b == 0.0

    def test_clear_releases_blocked(self, server_4way):
        server_4way.register_intent(
            "A", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_SOUTH),
            heading=HEADING_FACING_SOUTH, speed=1.0,
        )
        server_4way.register_intent(
            "B", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_WEST),
            heading=HEADING_FACING_WEST, speed=1.0,
        )
        # B blocked while A holds the queue head.
        go_b = server_4way.query_go_signal(
            "B", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_WEST),
            heading=HEADING_FACING_WEST, speed=1.0,
        )
        assert go_b == 0.0
        # A clears the intersection.
        server_4way.clear_agent("A")
        # B now at the head of the queue, gets GO.
        go_b = server_4way.query_go_signal(
            "B", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_WEST),
            heading=HEADING_FACING_WEST, speed=1.0,
        )
        assert go_b == 1.0

    def test_both_right_pass_simultaneously(self, server_4way):
        server_4way.register_intent(
            "A", "A", TurnCommand.RIGHT,
            position=self._far_pose(HEADING_FACING_SOUTH),
            heading=HEADING_FACING_SOUTH, speed=1.0,
        )
        # B from a perpendicular approach also turning right: matrix
        # says no conflict, so B should not be blocked even though A
        # is at the queue head.
        go_b = server_4way.register_intent(
            "B", "A", TurnCommand.RIGHT,
            position=self._far_pose(HEADING_FACING_WEST),
            heading=HEADING_FACING_WEST, speed=1.0,
        )
        assert go_b == 1.0

    def test_off_axis_falls_back_to_heuristic(self, server_4way):
        # A on a normal approach; B with a heading that doesn't match
        # any approach axis (45deg off everything). The matrix lookup
        # for B's pair will find no key and fall back to the heuristic,
        # which still computes the same conflict result from the raw
        # heading. Result: server matches SchedulerCore behavior.
        server_4way.register_intent(
            "A", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_SOUTH),
            heading=HEADING_FACING_SOUTH, speed=1.0,
        )
        weird_heading = (HEADING_FACING_SOUTH + HEADING_FACING_WEST) / 2.0
        # The heuristic at this heading vs HEADING_FACING_SOUTH:
        # diff = 45 deg, perpendicular, both straight => conflict.
        # Server should produce the same blocked result as a base
        # SchedulerCore would.
        go_b_server = server_4way.register_intent(
            "B", "A", TurnCommand.STRAIGHT,
            position=(2.0, 2.0), heading=weird_heading, speed=1.0,
        )

        core = SchedulerCore(graph=make_4way_graph())
        core.register_intent(
            "A", "A", TurnCommand.STRAIGHT,
            position=self._far_pose(HEADING_FACING_SOUTH),
            heading=HEADING_FACING_SOUTH, speed=1.0,
        )
        go_b_core = core.register_intent(
            "B", "A", TurnCommand.STRAIGHT,
            position=(2.0, 2.0), heading=weird_heading, speed=1.0,
        )
        assert go_b_server == go_b_core
