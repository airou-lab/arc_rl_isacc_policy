"""
Worker Scheduler — Multi-Agent Intersection Coordination
========================================================

Sits outside all AgentNodes and coordinates their Workers at shared
intersections. Each Worker registers its intended turn with the
Scheduler before executing, and the Scheduler returns a go/wait
signal based on conflict analysis and RVO velocity constraints.

Architecture:
    WorkerScheduler (one per simulation)
    ├── Maintains an intent registry: {agent_id -> IntentRecord}
    ├── Detects conflicts: overlapping paths through intersection
    ├── Computes RVO velocity constraints for trajectory spacing
    └── Returns go_signal in {0.0, 1.0} per agent per step

Conflict Detection:
    Two agents conflict if they're at the same intersection AND their
    intended paths cross. The conflict matrix is defined per intersection
    in the graph JSON (or computed from exit headings).

    Simple version (Phase 1): Priority by arrival order. First agent to
    register at an intersection gets GO, later agents get WAIT until
    the first clears.

    RVO version (Phase 2): Agents approaching with conflicting paths
    receive velocity constraints that keep their trajectories separated
    by a minimum time gap.

Reciprocal Velocity Obstacles (RVO):
    Standard RVO computes a velocity-space constraint for each agent
    pair such that, if both agents respect their half of the avoidance,
    collision is guaranteed impossible. For intersection coordination:

    - We compute the time-to-intersection (TTI) for each agent
    - If two agents' TTIs overlap within a safety margin, the later
      one receives go_signal = 0 (WAIT)
    - The safety margin accounts for vehicle length, braking distance,
      and sim-to-real timing jitter

    This is simpler than full continuous RVO because intersections have
    discrete conflict points (the crossing zone), not continuous
    obstacle boundaries. We only need to ensure temporal separation
    at the crossing, not spatial avoidance along arbitrary paths.

Dependencies:
    - numpy
    - agent/intersection_graph.py (TurnCommand)

Author: Aaron Hamil
Date: 03/12/26
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agent.intersection_graph import TurnCommand

logger = logging.getLogger(__name__)


# Conflict Rules

# Which turn pairs conflict at a standard 4-way intersection?
# Two agents conflict if their paths through the intersection cross.
# This is conservative: if in doubt, mark as conflicting.
#
# Key: (approach_A_relative, turn_A, approach_B_relative, turn_B) -> bool
# "approach_X_relative" is the relative direction the other agent comes
# from (opposite, left, right). Same-direction approaches never conflict
# because agents are in the same lane.
#
# Rather than enumerate all combinations, we use a simpler rule:
# Two agents at the same intersection conflict UNLESS they are:
#   1. Both going straight from opposite directions (no crossing)
#   2. Both turning right (paths don't cross in right-hand traffic)
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
    # Heading difference determines relative approach
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
        # All other opposite combos conflict
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
    heading: float           # Approach heading (radians)
    position: Tuple[float, float]
    speed: float
    registered_at: float     # Timestamp of registration
    cleared: bool = False    # True once agent leaves intersection


@dataclass
class RVOConstraint:
    """Velocity constraint from RVO computation."""
    agent_id: str
    max_speed: float         # Speed limit to maintain time separation
    wait: bool               # If True, agent should stop entirely


# Scheduler

@dataclass
class SchedulerConfig:
    """Configuration for the WorkerScheduler."""
    time_gap_seconds: float = 1.5        # Minimum time between agents at crossing
    intent_timeout: float = 15.0         # Stale intent auto-expires (seconds)
    vehicle_length: float = 0.33         # F1Tenth wheelbase (meters)
    safety_margin: float = 0.5           # Extra distance margin (meters)


class WorkerScheduler:
    """
    Multi-agent intersection coordinator.

    Tracks all agents' intended intersection maneuvers and determines
    which agents can proceed (GO) and which must wait (WAIT) to
    prevent path conflicts.

    Usage:
        scheduler = WorkerScheduler()

        # Workers call these during their step:
        go = scheduler.register_intent(agent_id, intersection_id, turn_cmd)
        go = scheduler.query_go_signal(agent_id, intersection_id, ...)

        # Env calls this when an agent leaves an intersection:
        scheduler.clear_agent(agent_id)

        # Env calls this every step to expire stale intents:
        scheduler.tick()
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._intents: Dict[str, IntentRecord] = {}  # agent_id -> IntentRecord
        self._priority_order: Dict[str, List[str]] = {}  # intersection_id -> [agent_ids by arrival]

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

        Called every step while the agent is inside an intersection.
        Updates the agent's position/speed for RVO computation.

        Returns:
            1.0 = GO, 0.0 = WAIT.
        """
        # Update position for RVO
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
            logger.debug(f"Scheduler: expired stale intent for {aid}")
            self.clear_agent(aid)

    def _compute_go_signal(
        self, agent_id: str, intersection_id: str
    ) -> float:
        """
        Determine if agent can proceed based on conflict analysis.

        Phase 1: Priority by arrival order with conflict checking.
        If the agent has the highest priority among conflicting agents
        at this intersection, it gets GO. Otherwise WAIT.

        Phase 2 (RVO): Additionally check time-to-intersection overlap.
        """
        my_intent = self._intents.get(agent_id)
        if my_intent is None:
            return 1.0  # No registered intent = go

        queue = self._priority_order.get(intersection_id, [])
        if not queue or queue[0] == agent_id:
            return 1.0  # First in line or no queue = go

        # Check if any higher-priority agent has a conflicting path
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

            # Check path conflict
            if _paths_conflict(
                my_intent.turn_command, my_intent.heading,
                other_intent.turn_command, other_intent.heading,
            ):
                # RVO time-gap check
                if self._rvo_time_gap_violated(my_intent, other_intent):
                    return 0.0  # WAIT — conflict with higher priority

        return 1.0  # No conflicts = GO

    def _rvo_time_gap_violated(
        self, agent: IntentRecord, other: IntentRecord
    ) -> bool:
        """
        Check if the time gap between two agents is too small.

        Simple RVO: estimate time-to-crossing for both agents.
        If the gap is less than config.time_gap_seconds, the later
        agent should wait.

        For F1Tenth scale (0.33m wheelbase, ~2m/s max), the crossing
        zone is roughly 0.5m x 0.5m. Time to cross ~= 0.25s at speed.
        We want at least time_gap_seconds between exits.
        """
        # Distance from agent to intersection center (approximation)
        # A more precise version would use the actual crossing point
        # but for Phase 1 this suffices
        dist_agent = max(0.1, np.sqrt(
            (agent.position[0] - other.position[0]) ** 2 +
            (agent.position[1] - other.position[1]) ** 2
        ))

        speed_agent = max(0.1, agent.speed)
        speed_other = max(0.1, other.speed)

        # Rough time-to-crossing estimates
        tti_agent = dist_agent / speed_agent
        tti_other = dist_agent / speed_other  # Same distance approximation

        time_gap = abs(tti_agent - tti_other)

        return time_gap < self.config.time_gap_seconds

    @property
    def active_intents(self) -> Dict[str, IntentRecord]:
        """Read-only access to all active intents."""
        return dict(self._intents)

    def __repr__(self) -> str:
        n = len(self._intents)
        intersections = set(i.intersection_id for i in self._intents.values())
        return f"WorkerScheduler(active={n}, intersections={intersections})"


# Helpers

def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference in [-pi, pi]."""
    d = (a - b) % (2 * np.pi)
    if d > np.pi:
        d -= 2 * np.pi
    return d
