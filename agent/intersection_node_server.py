"""
Intersection Node Server — Shared In-Process Multi-Agent Arbiter

Stage 3 of the multi-agent rollout. See
.planning/INTERSECTION_NODE_DESIGN.md.

Role
A single shared object that several agents' WorkerSchedulers point at
so they observe one consistent registry, queue, and time-gap arbitration
at each intersection. Subclasses SchedulerCore (the single source of
truth for arbitration semantics) and adds:

    - Precomputed (intersection, approach, turn) conflict matrix for
      O(1) lookup, replacing the per-call heading-diff math in the
      base class's _paths_conflict heuristic.
    - Approach resolution from agent heading using the same
      APPROACH_TOLERANCE_RAD the IntersectionGraph already uses for
      get_exit_options.
    - Graceful fallback to _paths_conflict when an approach can't be
      resolved (uncalibrated topology, off-axis heading, unknown
      approach). Failure modes match SchedulerCore exactly.

Public API is inherited from SchedulerCore one-to-one so the same
LocalTransport can wrap either a SchedulerCore or an
IntersectionNodeServer:

    register_intent(agent_id, intersection_id, turn_command, ...)  -> float
    query_go_signal(agent_id, intersection_id, ...)                -> float
    clear_agent(agent_id)                                          -> None
    tick()                                                         -> None
    active_intents                                                 -> Dict

Single-process training is the only training environment per
INTERSECTION_NODE_DESIGN.md. There is no networking. Each agent's
LocalTransport(server) is a plain Python function call.

Parity guarantee
For single-agent rollouts an IntersectionNodeServer produces the same
go_signal sequence as a SchedulerCore with the same config and graph.
tests/test_intersection_node_server.py asserts this.

Deployment note
At deployment each physical intersection runs its own
IntersectionNodeServer hosted by a ROS2 node. Ros2Transport (Stage 5)
will translate IntentMessage / ClearanceReply over rclpy. The server
class itself is unchanged across training and deployment.

Author: Aaron Hamil
Date: 05/04/26
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from agent.intersection_graph import TurnCommand
from agent.scheduler_core import (
    IntentPhase,
    IntentRecord,
    SchedulerConfig,
    SchedulerCore,
    _angle_diff,
    _paths_conflict,
)

if TYPE_CHECKING:
    from agent.intersection_graph import IntersectionGraph

logger = logging.getLogger(__name__)


# Match IntersectionGraph.APPROACH_TOLERANCE_RAD exactly. Kept as a
# module constant rather than re-imported so the server has no graph
# dependency at lookup time.
APPROACH_TOLERANCE_RAD = math.radians(30.0)


# Conflict matrix key:
#   (intersection_id, approach_a_road_id, turn_a,
#                     approach_b_road_id, turn_b) -> bool
ConflictKey = Tuple[str, str, int, str, int]


class IntersectionNodeServer(SchedulerCore):
    """
    Shared in-process arbiter for multi-agent intersection coordination.

    Subclasses SchedulerCore so a LocalTransport built around either is
    interchangeable. Adds a precomputed conflict matrix and per-agent
    approach tracking; everything else (registry, queue, time-gap math,
    stale-intent GC) is inherited unchanged.
    """

    def __init__(
        self,
        graph: "IntersectionGraph",
        config: Optional[SchedulerConfig] = None,
    ):
        if graph is None:
            raise ValueError(
                "IntersectionNodeServer requires a graph. Use "
                "SchedulerCore directly for graph-less arbitration."
            )
        super().__init__(config=config, graph=graph)

        # Per-agent cached approach_id, populated at register / query
        # time so the conflict-check path doesn't re-resolve every step.
        self._approach_for_agent: Dict[str, str] = {}

        # Precomputed conflict table. Built once at construction.
        self._conflict_matrix: Dict[ConflictKey, bool] = {}
        self._build_conflict_matrix(graph)

    # Public API overrides

    def register_intent(
        self,
        agent_id: str,
        intersection_id: str,
        turn_command: int,
        position: Tuple[float, float] = (0.0, 0.0),
        heading: float = 0.0,
        speed: float = 0.0,
    ) -> float:
        self._refresh_approach_cache(agent_id, intersection_id, heading)
        return super().register_intent(
            agent_id=agent_id,
            intersection_id=intersection_id,
            turn_command=turn_command,
            position=position,
            heading=heading,
            speed=speed,
        )

    def query_go_signal(
        self,
        agent_id: str,
        intersection_id: str,
        turn_command: int,
        position: Tuple[float, float],
        heading: float,
        speed: float,
    ) -> float:
        self._refresh_approach_cache(agent_id, intersection_id, heading)
        return super().query_go_signal(
            agent_id=agent_id,
            intersection_id=intersection_id,
            turn_command=turn_command,
            position=position,
            heading=heading,
            speed=speed,
        )

    def clear_agent(self, agent_id: str) -> None:
        self._approach_for_agent.pop(agent_id, None)
        super().clear_agent(agent_id)

    def tick(self) -> None:
        before = set(self._intents.keys())
        super().tick()
        for gone in before - set(self._intents.keys()):
            self._approach_for_agent.pop(gone, None)

    @property
    def conflict_matrix_size(self) -> int:
        """Number of entries in the precomputed conflict matrix."""
        return len(self._conflict_matrix)

    # Arbitration override

    def _compute_go_signal(
        self, agent_id: str, intersection_id: str
    ) -> float:
        """
        Matrix-aware version of SchedulerCore._compute_go_signal.

        Mirrors the structure of the base implementation so that
        single-agent behavior is byte-equivalent: queue head returns
        GO immediately, phase advances on the GO tick, lower-priority
        agents are blocked only if a conflicting higher-priority intent
        violates the time gap.

        Conflict check goes through the precomputed matrix when both
        approaches resolve; otherwise it falls back to the inherited
        _paths_conflict heuristic (same code path SchedulerCore uses).
        """
        my_intent = self._intents.get(agent_id)
        if my_intent is None:
            return 1.0

        queue = self._priority_order.get(intersection_id, [])
        if not queue or queue[0] == agent_id:
            my_intent.phase = IntentPhase.COMMITTED
            return 1.0

        my_approach = self._approach_for_agent.get(agent_id)

        for higher_id in queue:
            if higher_id == agent_id:
                break

            other_intent = self._intents.get(higher_id)
            if other_intent is None:
                continue
            if other_intent.intersection_id != intersection_id:
                continue
            if other_intent.cleared:
                continue

            other_approach = self._approach_for_agent.get(higher_id)
            if self._lookup_conflict(
                intersection_id,
                my_intent, my_approach,
                other_intent, other_approach,
            ) and self._time_gap_violated(my_intent, other_intent):
                return 0.0

        my_intent.phase = IntentPhase.COMMITTED
        return 1.0

    # Matrix construction

    def _build_conflict_matrix(self, graph: "IntersectionGraph") -> None:
        """
        Precompute conflicts for every (approach, turn) pair at every
        intersection. Reuses _paths_conflict so the matrix is exactly
        consistent with the heuristic fallback path.

        For an N-approach intersection this is N x N x 9 entries. A
        4-way intersection produces 144 entries.
        """
        turns = TurnCommand.all()
        for iid, node in graph.all_intersections.items():
            approach_ids = list(node.approaches.keys())
            for road_a in approach_ids:
                h_a = node.approaches[road_a].heading_rad
                for road_b in approach_ids:
                    h_b = node.approaches[road_b].heading_rad
                    for turn_a in turns:
                        for turn_b in turns:
                            self._conflict_matrix[
                                (iid, road_a, turn_a, road_b, turn_b)
                            ] = _paths_conflict(turn_a, h_a, turn_b, h_b)
        logger.debug(
            "IntersectionNodeServer: precomputed %d conflict entries "
            "across %d intersections",
            len(self._conflict_matrix), len(graph.all_intersections),
        )

    # Approach resolution

    def _refresh_approach_cache(
        self, agent_id: str, intersection_id: str, heading: float
    ) -> None:
        approach = self._resolve_approach(intersection_id, heading)
        if approach is not None:
            self._approach_for_agent[agent_id] = approach
        elif agent_id in self._approach_for_agent:
            # Heading drifted off-axis; drop cache so we fall back to
            # the heuristic on the next conflict check.
            del self._approach_for_agent[agent_id]

    def _resolve_approach(
        self, intersection_id: str, agent_heading: float
    ) -> Optional[str]:
        """
        Return the road_id of the closest approach within
        APPROACH_TOLERANCE_RAD, or None if no approach matches.

        Mirrors the matching logic in IntersectionGraph.get_exit_options
        but returns the approach road_id instead of its exits.
        """
        if self._graph is None:
            return None
        node = self._graph.get_intersection(intersection_id)
        if node is None:
            return None
        best_road: Optional[str] = None
        best_diff = float("inf")
        for road_id, approach in node.approaches.items():
            diff = abs(_angle_diff(agent_heading, approach.heading_rad))
            if diff <= APPROACH_TOLERANCE_RAD and diff < best_diff:
                best_road = road_id
                best_diff = diff
        return best_road

    # Conflict lookup

    def _lookup_conflict(
        self,
        intersection_id: str,
        my_intent: IntentRecord,
        my_approach: Optional[str],
        other_intent: IntentRecord,
        other_approach: Optional[str],
    ) -> bool:
        """
        Matrix lookup with heuristic fallback when either approach is
        unresolved. Failure mode: identical to SchedulerCore.
        """
        if my_approach is not None and other_approach is not None:
            key = (
                intersection_id,
                my_approach, my_intent.turn_command,
                other_approach, other_intent.turn_command,
            )
            cached = self._conflict_matrix.get(key)
            if cached is not None:
                return cached
            logger.debug(
                "IntersectionNodeServer: matrix miss for %s, "
                "falling back to heuristic", key,
            )
        return _paths_conflict(
            my_intent.turn_command, my_intent.heading,
            other_intent.turn_command, other_intent.heading,
        )

    def __repr__(self) -> str:
        n_intents = len(self._intents)
        n_intersections = (
            len(self._graph.all_intersections) if self._graph else 0
        )
        return (
            f"IntersectionNodeServer(active={n_intents}, "
            f"intersections={n_intersections}, "
            f"matrix={len(self._conflict_matrix)} entries)"
        )
