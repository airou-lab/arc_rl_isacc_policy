"""
Worker Scheduler — Multi-Agent Intersection Coordination

Sits outside all AgentNodes and coordinates their Workers at shared
intersections. Each Worker registers its intended turn with the
Scheduler before executing, and the Scheduler returns a go/wait
signal based on conflict analysis and time-gap reservation.

Architecture:
    WorkerScheduler (one per simulation, in-process for now)
    ├── Maintains an intent registry: {agent_id -> IntentRecord}
    ├── Detects conflicts: overlapping paths through intersection
    ├── Computes occupancy intervals at the crossing center
    └── Returns go_signal in {0.0, 1.0} per agent per step

Conflict Detection:
    Two agents conflict if they're at the same intersection AND their
    intended paths cross. Path-pair conflict is determined from
    relative approach headings and turn commands (_paths_conflict).

Time-Gap Reservation (replaces the previous buggy RVO sketch):
    Each agent's distance to the calibrated intersection CENTER is
    used to compute an occupancy interval [enter, exit] at the
    crossing zone. The crossing zone is a disk of radius
    `crossing_radius_m` around the intersection center.

    A lower-priority agent B is blocked by a higher-priority agent A
    when A and B's paths conflict AND
        agent_enter_B < other_exit_A + time_gap_seconds

    If A is already COMMITTED (mid-traversal), A's window is treated
    as starting at now, so B waits for A to fully clear plus margin.

    When the graph passed in at construction has no calibrated
    position for the intersection, the scheduler falls back to a
    conservative block on any path conflict — same effective behavior
    as the legacy single-agent fixture tests.

Phase Field on IntentRecord:
    DECIDING  — registered, awaiting clearance
    COMMITTED — released by scheduler, traversing
    CLEARING  — about to leave (currently unused; kept for future
                hand-off semantics with a remote IntersectionNodeServer)

    Phase advances inside _compute_go_signal the moment the agent
    is granted go=1.0. The Worker does not read or write phase.

Single-process default:
    The Worker still calls register_intent / query_go_signal /
    clear_agent directly on this object. The class-level interface is
    what a future SchedulerTransport (gz-transport / ROS2) will adapt.

Dependencies:
    - numpy
    - agent/intersection_graph.py (TurnCommand, IntersectionGraph)

Author: Aaron Hamil
Date: 03/12/26
Updated: 04/25/26  — TTI uses crossing-center distance; IntentPhase added
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from agent.intersection_graph import TurnCommand

if TYPE_CHECKING:
    # Avoid a hard import cycle at runtime; only used for type hints.
    from agent.intersection_graph import IntersectionGraph

logger = logging.getLogger(__name__)


# Phase Constants

class IntentPhase:
    """
    Lifecycle of a registered intent inside the scheduler.

    Mirrors the string-style state constants used elsewhere in the
    codebase (WorkerNode.CRUISING, etc.) rather than enum.Enum, so
    serialization to a future remote IntersectionNodeServer is just
    plain JSON strings.
    """
    DECIDING = "deciding"     # Registered, awaiting clearance
    COMMITTED = "committed"   # Released by scheduler, traversing
    CLEARING = "clearing"     # About to leave (reserved for remote node)


# Conflict Rules

# Which turn pairs conflict at a standard 4-way intersection?
# Two agents conflict if their paths through the intersection cross.
# This is conservative: if in doubt, mark as conflicting.
#
# Rule:
# Two agents at the same intersection conflict UNLESS they are:
#   1. Following each other from the same direction (no crossing)
#   2. Both going straight from opposite directions (no crossing)
#   3. Both turning right (paths don't cross in right-hand traffic)
#
# Everything else is conservatively treated as conflicting.

def _paths_conflict(
    turn_a: int, heading_a: float,
    turn_b: int, heading_b: float,
) -> bool:
    """
    Check if two agents' intended paths through an intersection conflict.

    Uses heading difference to determine relative approach direction,
    then applies conflict rules.

    Args:
        turn_a, turn_b: TurnCommand values.
        heading_a, heading_b: Approach headings in radians.

    Returns:
        True if paths are potentially conflicting (conservative).
    """
    hdg_diff = abs(_angle_diff(heading_a, heading_b))

    # Same direction (< 45deg): following each other, no conflict
    if hdg_diff < np.radians(45):
        return False

    # Opposite directions (135deg-180deg)
    if hdg_diff > np.radians(135):
        # Both straight from opposite = no conflict (pass each other)
        if turn_a == TurnCommand.STRAIGHT and turn_b == TurnCommand.STRAIGHT:
            return False
        # Both right from opposite = no conflict (paths don't cross)
        if turn_a == TurnCommand.RIGHT and turn_b == TurnCommand.RIGHT:
            return False
        return True

    # Perpendicular approaches (45deg-135deg): almost always conflict
    # Exception: both turning right
    if turn_a == TurnCommand.RIGHT and turn_b == TurnCommand.RIGHT:
        return False

    return True


# Data Types

@dataclass
class IntentRecord:
    """One agent's registered intent at an intersection."""
    agent_id: str
    intersection_id: str
    turn_command: int
    heading: float                          # Approach heading (radians)
    position: Tuple[float, float]
    speed: float
    registered_at: float                    # Timestamp of registration
    phase: str = IntentPhase.DECIDING       # Lifecycle phase (see IntentPhase)
    cleared: bool = False                   # True once agent leaves intersection


@dataclass
class RVOConstraint:
    """
    Velocity constraint from time-gap analysis.

    Kept for forward compatibility with a future continuous-velocity
    extension; not consumed by the current go/wait API.
    """
    agent_id: str
    max_speed: float
    wait: bool


# Scheduler Configuration

@dataclass
class SchedulerConfig:
    """Configuration for the WorkerScheduler."""
    time_gap_seconds: float = 1.5        # Minimum gap between crossings
    intent_timeout: float = 15.0         # Stale intent auto-expires (seconds)
    vehicle_length: float = 0.33         # F1Tenth wheelbase (meters)
    safety_margin: float = 0.5           # Extra distance margin (meters)

    # Time-gap arbitration parameters
    min_speed_for_tti: float = 0.1
    """Floor on speed used in TTI calculation (m/s). Prevents division
    by zero when an agent is stopped at the line. With v_min = 0.1
    m/s and a 0.5 m crossing radius, a stopped agent's "exit time"
    becomes 5 s — which the time_gap_seconds margin then dominates."""

    crossing_radius_m: float = 0.5
    """Radius around the intersection center used to define the
    occupancy zone. An agent is considered to "occupy" the crossing
    while it's within this radius. F1TENTH crossings on the current
    USD scene are roughly 0.5 m on each side, so 0.5 m is a sensible
    starting value."""


# Scheduler

class WorkerScheduler:
    """
    Multi-agent intersection coordinator.

    Tracks all agents' intended intersection maneuvers and determines
    which agents can proceed (GO) and which must wait (WAIT) to
    prevent path conflicts.

    Usage:
        scheduler = WorkerScheduler(graph=graph)

        # Workers call these during their step:
        go = scheduler.register_intent(agent_id, intersection_id, turn_cmd)
        go = scheduler.query_go_signal(agent_id, intersection_id, ...)

        # Env calls this when an agent leaves an intersection:
        scheduler.clear_agent(agent_id)

        # Env calls this every step to expire stale intents:
        scheduler.tick()
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        graph: Optional["IntersectionGraph"] = None,
    ):
        """
        Args:
            config: Scheduler configuration. Defaults to SchedulerConfig().
            graph: Optional IntersectionGraph for crossing-center lookups.
                If None, the scheduler conservatively blocks on any
                path conflict (preserves legacy single-process behavior
                used in the existing test fixtures).
        """
        self.config = config or SchedulerConfig()
        self._graph = graph
        self._intents: Dict[str, IntentRecord] = {}
        self._priority_order: Dict[str, List[str]] = {}

    def register_intent(
        self,
        agent_id: str,
        intersection_id: str,
        turn_command: int,
        position: Tuple[float, float] = (0.0, 0.0),
        heading: float = 0.0,
        speed: float = 0.0,
    ) -> float:
        """
        Register an agent's intended turn at an intersection.

        Called by WorkerNode._decide_turn() when the agent enters
        an intersection's trigger radius.

        Returns go_signal: 1.0 if safe to proceed, 0.0 if must wait.
        """
        now = time.monotonic()

        self._intents[agent_id] = IntentRecord(
            agent_id=agent_id,
            intersection_id=intersection_id,
            turn_command=turn_command,
            heading=heading,
            position=position,
            speed=speed,
            registered_at=now,
            phase=IntentPhase.DECIDING,
        )

        # Add to priority queue (arrival order)
        if intersection_id not in self._priority_order:
            self._priority_order[intersection_id] = []

        queue = self._priority_order[intersection_id]
        if agent_id not in queue:
            queue.append(agent_id)

        return self._compute_go_signal(agent_id, intersection_id)

    def query_go_signal(
        self,
        agent_id: str,
        intersection_id: str,
        turn_command: int,
        position: Tuple[float, float],
        heading: float,
        speed: float,
    ) -> float:
        """
        Query whether an agent can proceed.

        Called every step while the agent is inside or approaching an
        intersection. Updates the agent's position/speed for the
        time-gap calculation.

        Returns:
            1.0 = GO, 0.0 = WAIT.
        """
        intent = self._intents.get(agent_id)
        if intent is not None:
            intent.position = position
            intent.heading = heading
            intent.speed = speed

        return self._compute_go_signal(agent_id, intersection_id)

    def clear_agent(self, agent_id: str) -> None:
        """
        Remove an agent's intent (called when agent leaves intersection).
        """
        intent = self._intents.pop(agent_id, None)
        if intent is not None:
            iid = intent.intersection_id
            if iid in self._priority_order:
                queue = self._priority_order[iid]
                if agent_id in queue:
                    queue.remove(agent_id)
                if not queue:
                    del self._priority_order[iid]

    def tick(self) -> None:
        """
        Housekeeping: expire stale intents.

        Call this once per environment step.
        """
        now = time.monotonic()
        expired = [
            aid for aid, intent in self._intents.items()
            if (now - intent.registered_at) > self.config.intent_timeout
        ]
        for aid in expired:
            logger.debug("Scheduler: expired stale intent for %s", aid)
            self.clear_agent(aid)

    def _compute_go_signal(
        self, agent_id: str, intersection_id: str
    ) -> float:
        """
        Determine if agent can proceed based on conflict analysis.

        Priority by arrival order. The agent at the head of the queue
        always gets GO. For agents behind the head, a path-conflict
        check followed by a time-gap check decides GO vs WAIT against
        every higher-priority agent at the same intersection.

        Phase advances to COMMITTED on the tick that returns GO.
        """
        my_intent = self._intents.get(agent_id)
        if my_intent is None:
            return 1.0  # No registered intent = go

        queue = self._priority_order.get(intersection_id, [])
        if not queue or queue[0] == agent_id:
            # First in line (or no queue) — release immediately
            my_intent.phase = IntentPhase.COMMITTED
            return 1.0

        # Check every higher-priority agent for a conflict
        for higher_id in queue:
            if higher_id == agent_id:
                break  # Reached ourselves in priority order

            other_intent = self._intents.get(higher_id)
            if other_intent is None:
                continue
            if other_intent.intersection_id != intersection_id:
                continue
            if other_intent.cleared:
                continue

            if _paths_conflict(
                my_intent.turn_command, my_intent.heading,
                other_intent.turn_command, other_intent.heading,
            ):
                if self._time_gap_violated(my_intent, other_intent):
                    return 0.0  # WAIT — conflict with higher priority

        # No blocking conflicts — release and advance phase
        my_intent.phase = IntentPhase.COMMITTED
        return 1.0

    def _time_gap_violated(
        self, agent: IntentRecord, other: IntentRecord
    ) -> bool:
        """
        Check whether `agent` should wait for `other` to clear.

        Computes both agents' occupancy intervals at the calibrated
        intersection crossing. Blocks `agent` if its arrival would
        precede `other`'s clearance plus a configurable safety margin.

        If `other` is already in the COMMITTED phase, its window is
        treated as starting at now (it is already inside the
        crossing), so `agent` waits for `other` to fully clear.

        Without a calibrated graph, falls back to a conservative
        "always block" — this preserves the behavior the previous
        (buggy) implementation produced for the existing perpendicular
        path-conflict test fixture.
        """
        center = self._intersection_center(agent.intersection_id)
        if center is None:
            # No calibrated geometry available. Caller has already
            # confirmed a path conflict; without timing we conservatively
            # block. Matches the legacy single-process test expectation.
            return True

        d_agent = _euclid(agent.position, center)
        d_other = _euclid(other.position, center)

        v_agent = max(self.config.min_speed_for_tti, agent.speed)
        v_other = max(self.config.min_speed_for_tti, other.speed)

        cr = self.config.crossing_radius_m

        # Other's occupancy interval at the crossing
        if other.phase == IntentPhase.COMMITTED:
            # Already inside the crossing — assume occupying now
            other_enter = 0.0
            other_exit = max(0.0, (d_other + cr) / v_other)
        else:
            other_enter = max(0.0, (d_other - cr) / v_other)
            other_exit = (d_other + cr) / v_other

        # When `agent` would enter the crossing
        agent_enter = max(0.0, (d_agent - cr) / v_agent)

        # Block if `agent` would arrive before `other` clears (plus margin)
        return agent_enter < (other_exit + self.config.time_gap_seconds)

    def _intersection_center(
        self, intersection_id: str
    ) -> Optional[Tuple[float, float]]:
        """
        Return the calibrated (x, y) center of an intersection, or None
        if no graph is configured or the node is uncalibrated.
        """
        if self._graph is None:
            return None
        node = self._graph.get_intersection(intersection_id)
        if node is None or not node.is_calibrated:
            return None
        return node.position

    @property
    def active_intents(self) -> Dict[str, IntentRecord]:
        """Read-only access to all active intents."""
        return dict(self._intents)

    def __repr__(self) -> str:
        n = len(self._intents)
        intersections = set(i.intersection_id for i in self._intents.values())
        cal = "calibrated" if self._graph is not None else "no-graph"
        return f"WorkerScheduler(active={n}, intersections={intersections}, {cal})"


# Helpers

def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference in [-pi, pi]."""
    d = (a - b) % (2 * np.pi)
    if d > np.pi:
        d -= 2 * np.pi
    return d


def _euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Planar Euclidean distance between two (x, y) tuples."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return float(np.sqrt(dx * dx + dy * dy))
