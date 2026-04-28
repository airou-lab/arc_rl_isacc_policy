"""
Scheduler Transport — Pluggable Backend for WorkerScheduler

Lives on `feature/intersection-node` until the multi-agent rollout
ships. See .planning/INTERSECTION_NODE_DESIGN.md.

Updated 04/28/26 (stage 2 refactor):
    LocalTransport now wraps SchedulerCore (not WorkerScheduler).
    The SchedulerCore is the single source of truth for arbitration
    semantics; WorkerScheduler is now a facade that delegates here.

Purpose
The Worker calls `register_intent` / `query_go_signal` / `clear_agent`
on a WorkerScheduler facade. The facade translates each call into an
IntentMessage and dispatches it through a SchedulerTransport.
Implementations:

    LocalTransport      — wraps a SchedulerCore in-process (default; tests)
    GzTransport         — gz-transport13 RPC (stub; stage 3)
    Ros2Transport       — rclpy service / topic (stub; stage 5)

Wire format
IntentMessage and ClearanceReply are JSON-serializable dataclasses
that match the SchedulerCore's IntentRecord / return type. The same
payloads flow through every transport.

Author: Aaron Hamil
Date: 04/28/26
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from agent.scheduler_core import SchedulerCore


# Wire format

@dataclass
class IntentMessage:
    """
    JSON-serializable intent payload exchanged between agents and the
    intersection node server.

    Mirrors the fields of SchedulerCore.IntentRecord exactly so the
    server can deserialize directly into its registry.
    """
    agent_id: str
    intersection_id: str
    turn_command: int                         # TurnCommand.{LEFT,STRAIGHT,RIGHT}
    position: Tuple[float, float]
    heading: float                            # radians
    speed: float                              # m/s
    phase: str = "deciding"                   # IntentPhase string
    sent_at_monotonic: float = 0.0


@dataclass
class ClearanceReply:
    """
    Server's response to an IntentMessage.

    `suggested_retry_s` is informational; the Worker still polls every
    tick. Provided so future versions can implement smarter back-off.
    """
    go_signal: float = 1.0
    suggested_retry_s: float = 0.0
    server_time: float = 0.0


# Abstract transport

class SchedulerTransport(ABC):
    """
    Pluggable transport between the Worker-side scheduler facade and
    the arbiter (in-process SchedulerCore or a remote server).

    Implementations are expected to be **synchronous** from the
    Worker's perspective. Async backends must wait for a reply (with
    a configurable timeout) before returning.
    """

    @abstractmethod
    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        """Submit an intent and return the arbiter's clearance reply."""

    @abstractmethod
    def clear(self, agent_id: str) -> None:
        """Tell the arbiter an agent has left the intersection."""

    @abstractmethod
    def tick(self) -> None:
        """Per-step housekeeping. Drains pending replies for net transports."""


# In-process adapter

class LocalTransport(SchedulerTransport):
    """
    Same-process transport. Wraps a SchedulerCore directly.

    Used by:
        - The default WorkerScheduler() construction (single-process
          training, every existing test).
        - Unit tests that want to exercise the wire format without a
          network round-trip.

    Exposes the wrapped core as `self.core` so the facade can offer
    `active_intents` without going through a query message.
    """

    def __init__(self, core: "SchedulerCore"):
        self.core = core

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        # Delegate to the in-process core. The core upserts the intent
        # and recomputes go_signal in one call (register_intent and
        # query_go_signal share the same arbitration path; the only
        # difference is whether the record exists yet).
        go = self.core.register_intent(
            agent_id=msg.agent_id,
            intersection_id=msg.intersection_id,
            turn_command=msg.turn_command,
            position=msg.position,
            heading=msg.heading,
            speed=msg.speed,
        )
        return ClearanceReply(go_signal=float(go))

    def clear(self, agent_id: str) -> None:
        self.core.clear_agent(agent_id)

    def tick(self) -> None:
        self.core.tick()


# Gazebo / gz-transport13 (stage 3)

class GzTransport(SchedulerTransport):
    """
    gz-transport13 client transport. Stub.

    Topic schema (per intersection_id `iid`):
        /arcpro/intersection/<iid>/intent              (pub from agent)
        /arcpro/intersection/<iid>/clearance/<agent>   (pub from server)
        /arcpro/intersection/<iid>/clear               (pub from agent on exit)

    Implementation deferred to stage 3. Constructing this transport
    raises NotImplementedError so accidental wiring fails loudly.
    """

    def __init__(
        self,
        intersection_ids: List[str],
        timeout_ms: int = 50,
        retry_count: int = 1,
    ):
        raise NotImplementedError(
            "GzTransport: implement at stage 3 of the rollout. See "
            ".planning/INTERSECTION_NODE_DESIGN.md."
        )

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        raise NotImplementedError

    def clear(self, agent_id: str) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        raise NotImplementedError


# ROS2 / rclpy (stage 5)

class Ros2Transport(SchedulerTransport):
    """
    ROS2 client transport for hardware deployment. Stub.

    Same topic schema as GzTransport, backed by rclpy. Implemented at
    stage 5 once GzTransport is stable in simulation.
    """

    def __init__(
        self,
        node_name: str = "arcpro_intersection_client",
        timeout_ms: int = 100,
    ):
        raise NotImplementedError(
            "Ros2Transport: implement at stage 5 of the rollout."
        )

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        raise NotImplementedError

    def clear(self, agent_id: str) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        raise NotImplementedError
