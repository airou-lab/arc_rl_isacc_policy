"""
Scheduler Transport — Pluggable Backend for WorkerScheduler

NOT FOR `dev` BRANCH. This file is part of the multi-agent
intersection-node work (see .planning/INTERSECTION_NODE_DESIGN.md)
and lands on `feature/intersection-node` until the first clean
single-agent training run completes.

Purpose
-------
The Worker calls `register_intent` / `query_go_signal` / `clear_agent`
on a scheduler-shaped object. Today that object is the in-process
`WorkerScheduler`. After stage 2 of the rollout, agents instead call
those methods on a `WorkerScheduler` facade that delegates to a
`SchedulerTransport`, which moves the call to wherever the actual
arbiter lives:

    LocalTransport      — same-process function call (default; tests)
    GzTransport         — gz-transport13 RPC over a Gazebo network
    Ros2Transport       — rclpy service / topic over a ROS2 graph

The Worker's API does not change. Only the wiring changes.

Wire format
-----------
`IntentMessage` and `ClearanceReply` are JSON-serializable dataclasses
that match the existing `IntentRecord` / scheduler return type. The
same payloads flow through every transport.

Status
------
- LocalTransport: implemented, wraps `WorkerScheduler`.
- GzTransport: stub, raises NotImplementedError. Implement at stage 3.
- Ros2Transport: stub, raises NotImplementedError. Implement at stage 5.

Author: Aaron Hamil
Date: 04/25/26
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from agent.worker_scheduler import WorkerScheduler


# Wire format

@dataclass
class IntentMessage:
    """
    JSON-serializable intent payload exchanged between agents and the
    intersection node server.

    Mirrors the fields of WorkerScheduler.IntentRecord exactly so
    the server can deserialize directly into its registry.
    """
    agent_id: str
    intersection_id: str
    turn_command: int                         # TurnCommand.{LEFT,STRAIGHT,RIGHT}
    position: Tuple[float, float]
    heading: float                            # radians
    speed: float                              # m/s
    phase: str = "deciding"                   # IntentPhase string
    sent_at_monotonic: float = 0.0            # client clock at send


@dataclass
class ClearanceReply:
    """
    Server's response to an IntentMessage.

    `suggested_retry_s` is informational; the Worker still polls every
    tick. Provided so future versions can implement smarter back-off.
    """
    go_signal: float = 1.0                    # 1.0 = GO, 0.0 = WAIT
    suggested_retry_s: float = 0.0
    server_time: float = 0.0


# Abstract transport

class SchedulerTransport(ABC):
    """
    Pluggable transport between the Worker-side scheduler facade and
    the IntersectionNodeServer.

    Implementations are expected to be **synchronous** from the
    Worker's perspective. If the underlying mechanism is async,
    the implementation must wait for the reply (with a configurable
    timeout) before returning.
    """

    @abstractmethod
    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        """Submit an intent and return the server's clearance reply."""

    @abstractmethod
    def clear(self, agent_id: str) -> None:
        """Tell the server an agent has left the intersection."""

    @abstractmethod
    def tick(self) -> None:
        """
        Per-step housekeeping. For network transports, drain any
        pending replies and prune stale state.
        """


# In-process adapter (current behavior)

class LocalTransport(SchedulerTransport):
    """
    Same-process transport. Used when there is no need for a network
    boundary, e.g. single-process training and unit tests.

    Wraps an existing `WorkerScheduler` instance and translates each
    transport call into the corresponding scheduler method. The
    scheduler does its own arbitration and the reply is constructed
    from its return value.
    """

    def __init__(self, scheduler: "WorkerScheduler"):
        self._sched = scheduler

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        # Delegate to the in-process scheduler. The scheduler upserts
        # the intent and recomputes go_signal in one call.
        go = self._sched.register_intent(
            agent_id=msg.agent_id,
            intersection_id=msg.intersection_id,
            turn_command=msg.turn_command,
            position=msg.position,
            heading=msg.heading,
            speed=msg.speed,
        )
        return ClearanceReply(go_signal=float(go))

    def clear(self, agent_id: str) -> None:
        self._sched.clear_agent(agent_id)

    def tick(self) -> None:
        self._sched.tick()


# Gazebo / gz-transport13 (multi-agent training)

class GzTransport(SchedulerTransport):
    """
    gz-transport13 client transport.

    Topic schema (per intersection_id `iid`):
        /arcpro/intersection/<iid>/intent              (pub from agent)
        /arcpro/intersection/<iid>/clearance/<agent>   (pub from server)
        /arcpro/intersection/<iid>/clear               (pub from agent on exit)

    Implementation deferred to stage 3 of the rollout. Constructing
    this transport raises NotImplementedError so accidental wiring
    fails loudly rather than degrading silently.
    """

    def __init__(
        self,
        intersection_ids: list[str],
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


# ROS2 / rclpy (deployment)

class Ros2Transport(SchedulerTransport):
    """
    ROS2 client transport for hardware deployment.

    Same topic schema as GzTransport, but backed by rclpy. Brought
    online once GzTransport works in simulation and the first
    multi-NUC physical run is queued.
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
