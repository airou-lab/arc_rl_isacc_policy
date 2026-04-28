"""
Scheduler Core — Pure-Python Arbitration Logic

Extracts the arbitration math from agent/worker_scheduler.py so it
can be reached through any SchedulerTransport (in-process via
LocalTransport, gz-transport via GzTransport, ROS2 via Ros2Transport).

The class here owns:
    - the intent registry  ({agent_id -> IntentRecord})
    - the priority queue   ({intersection_id -> [agent_ids by arrival]})
    - the path-conflict heuristic
    - the time-gap arbitration math
    - stale-intent garbage collection

WorkerScheduler is now a thin facade in worker_scheduler.py that
delegates to a SchedulerTransport, which (for in-process operation)
wraps an instance of SchedulerCore.

This module is the single source of truth for arbitration semantics.
The standalone IntersectionNodeServer (stage 3) imports from here
rather than from worker_scheduler.

Author: Aaron Hamil
Date: 04/28/26
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from agent.intersection_graph import TurnCommand

if TYPE_CHECKING:
    from agent.intersection_graph import IntersectionGraph

logger = logging.getLogger(__name__)


# Phase Constants

class IntentPhase:
    """
    Lifecycle of a registered intent inside the arbitration core.

    String-style constants on purpose: serializes as plain JSON when
    the same record traverses a network transport.
    """
    DECIDING = "deciding"     # Registered, awaiting clearance
    COMMITTED = "committed"   # Released, traversing
    CLEARING = "clearing"     # About to leave (reserved for remote node)


# Path-Conflict Heuristic

# Two agents at the same intersection conflict UNLESS:
#   1. They follow each other from the same direction
#   2. Both go straight from opposite directions
#   3. Both turn right (paths don't cross in right-hand traffic)
#
# Everything else is conservatively conflicting.

def _paths_conflict(
    turn_a: int, heading_a: float,
    turn_b: int, heading_b: float,
) -> bool:
    """Return True if two intended maneuvers' paths cross."""
    hdg_diff = abs(_angle_diff(heading_a, heading_b))

    # Same direction (< 45deg): following each other
    if hdg_diff < np.radians(45):
        return False

    # Opposite directions (135-180deg)
    if hdg_diff > np.radians(135):
        if turn_a == TurnCommand.STRAIGHT and turn_b == TurnCommand.STRAIGHT:
            return False
        if turn_a == TurnCommand.RIGHT and turn_b == TurnCommand.RIGHT:
            return False
        return True

    # Perpendicular (45-135deg): conflict unless both right
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
    registered_at: float                    # time.monotonic() at registration
    phase: str = IntentPhase.DECIDING
    cleared: bool = False


@dataclass
class RVOConstraint:
    """
    Velocity constraint output (forward-compatibility placeholder).

    Reserved for a future continuous-velocity extension; not consumed
    by the current go/wait API.
    """
    agent_id: str
    max_speed: float
    wait: bool


@dataclass
class SchedulerConfig:
    """Configuration shared by SchedulerCore and the WorkerScheduler facade."""
    time_gap_seconds: float = 1.5        # Minimum gap between crossings
    intent_timeout: float = 15.0         # Stale intent auto-expires (seconds)
    vehicle_length: float = 0.33         # F1Tenth wheelbase (meters)
    safety_margin: float = 0.5           # Extra distance margin (meters)

    min_speed_for_tti: float = 0.1
    """Floor on speed used in TTI calculation (m/s). Prevents division
    by zero when an agent is stopped at the line."""

    crossing_radius_m: float = 0.5
    """Radius around the intersection center used to define the
    occupancy zone. F1TENTH crossings on the current USD scene are
    roughly 0.5 m on each side, so 0.5 m is a sensible default."""


# Arbitration Core

class SchedulerCore:
    """
    Pure-compute arbitration core.

    Maintains the intent registry and priority queue, runs path-conflict
    and time-gap checks, advances phase. No I/O, no networking, no
    transport awareness — those live in SchedulerTransport
    implementations that wrap (or remote) this core.

    Methods mirror the legacy WorkerScheduler API one-to-one so the
    facade in agent/worker_scheduler.py can delegate without translation.
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        graph: Optional["IntersectionGraph"] = None,
    ):
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
        """Upsert an intent and return go_signal (1.0 = GO, 0.0 = WAIT)."""
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
        """Recompute go_signal with updated kinematic state."""
        intent = self._intents.get(agent_id)
        if intent is not None:
            intent.position = position
            intent.heading = heading
            intent.speed = speed

        return self._compute_go_signal(agent_id, intersection_id)

    def clear_agent(self, agent_id: str) -> None:
        """Remove an agent's intent (called when agent leaves intersection)."""
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
        """Housekeeping: expire stale intents."""
        now = time.monotonic()
        expired = [
            aid for aid, intent in self._intents.items()
            if (now - intent.registered_at) > self.config.intent_timeout
        ]
        for aid in expired:
            logger.debug("SchedulerCore: expired stale intent for %s", aid)
            self.clear_agent(aid)

    @property
    def active_intents(self) -> Dict[str, IntentRecord]:
        """Read-only access to all active intents."""
        return dict(self._intents)

    # Arbitration

    def _compute_go_signal(
        self, agent_id: str, intersection_id: str
    ) -> float:
        """
        Determine if `agent_id` can proceed.

        FCFS by registration order. The agent at the head of the queue
        always gets GO. For agents behind the head, every higher-priority
        intent at the same intersection is checked for path conflict
        followed by time-gap. Phase advances to COMMITTED on the tick
       that grants GO.
        """
        my_intent = self._intents.get(agent_id)
        if my_intent is None:
            return 1.0

        queue = self._priority_order.get(intersection_id, [])
        if not queue or queue[0] == agent_id:
            my_intent.phase = IntentPhase.COMMITTED
            return 1.0

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

            if _paths_conflict(
                my_intent.turn_command, my_intent.heading,
                other_intent.turn_command, other_intent.heading,
            ):
                if self._time_gap_violated(my_intent, other_intent):
                    return 0.0

        my_intent.phase = IntentPhase.COMMITTED
        return 1.0

    def _time_gap_violated(
        self, agent: IntentRecord, other: IntentRecord
    ) -> bool:
        """
        Check whether `agent` should wait for `other` to clear.

        Computes both agents' occupancy intervals at the calibrated
        intersection center. Blocks `agent` if its arrival precedes
        `other`'s clearance plus a safety margin.

        If `other` is COMMITTED, its window starts at now (it is
        already inside the crossing), so `agent` waits for full
        clearance.

        Without a calibrated graph, falls back to a conservative
        "always block" — preserves legacy single-process behavior.
        """
        center = self._intersection_center(agent.intersection_id)
        if center is None:
            return True

        d_agent = _euclid(agent.position, center)
        d_other = _euclid(other.position, center)

        v_agent = max(self.config.min_speed_for_tti, agent.speed)
        v_other = max(self.config.min_speed_for_tti, other.speed)

        cr = self.config.crossing_radius_m

        if other.phase == IntentPhase.COMMITTED:
            other_enter = 0.0
            other_exit = max(0.0, (d_other + cr) / v_other)
        else:
            other_enter = max(0.0, (d_other - cr) / v_other)
            other_exit = (d_other + cr) / v_other

        agent_enter = max(0.0, (d_agent - cr) / v_agent)

        return agent_enter < (other_exit + self.config.time_gap_seconds)

    def _intersection_center(
        self, intersection_id: str
    ) -> Optional[Tuple[float, float]]:
        """Calibrated (x, y) center, or None if unknown."""
        if self._graph is None:
            return None
        node = self._graph.get_intersection(intersection_id)
        if node is None or not node.is_calibrated:
            return None
        return node.position

    def __repr__(self) -> str:
        n = len(self._intents)
        intersections = set(i.intersection_id for i in self._intents.values())
        cal = "calibrated" if self._graph is not None else "no-graph"
        return f"SchedulerCore(active={n}, intersections={intersections}, {cal})"


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
