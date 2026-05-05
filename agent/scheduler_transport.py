"""
Scheduler Transport — Pluggable Backend for WorkerScheduler

See .planning/INTERSECTION_NODE_DESIGN.md.

Updated 05/04/26 (stage 3 prep):
    GzTransport removed. The training loop runs in a single Python
    process inside Isaac Sim, so multi-agent coordination during
    training is in-process function calls through LocalTransport
    against one shared IntersectionNodeServer (or a SchedulerCore
    for single-agent runs). gz-transport13 was never required.

Updated 04/28/26 (stage 2 refactor):
    LocalTransport now wraps SchedulerCore (not WorkerScheduler).
    The SchedulerCore is the single source of truth for arbitration
    semantics; WorkerScheduler is now a facade that delegates here.

Purpose
The Worker calls `register_intent` / `query_go_signal` / `clear_agent`
on a WorkerScheduler facade. The facade translates each call into an
IntentMessage and dispatches it through a SchedulerTransport.
Implementations:

    LocalTransport      — wraps a SchedulerCore-compatible arbiter
                          in-process (default; tests; multi-agent
                          training when wrapping a shared
                          IntersectionNodeServer)
    Ros2Transport       — rclpy service / topic (stub; stage 5,
                          deployment only)

Wire format
IntentMessage and ClearanceReply are JSON-serializable dataclasses
that match the SchedulerCore's IntentRecord / return type. In
training (LocalTransport) the same dataclasses are passed by
reference; no serialization actually happens, but constructing an
IntentMessage is the unified call path so deployment via
Ros2Transport reuses it without translation.

Author: Aaron Hamil
Date: 04/28/26 (stage 2), updated 05/04/26 (stage 3 prep)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    the arbiter (in-process SchedulerCore / IntersectionNodeServer or
    a remote ROS2 server).

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
    Same-process transport. Wraps a SchedulerCore-compatible arbiter
    directly. Accepts either:

        - SchedulerCore: single-agent training, every existing test.
        - IntersectionNodeServer: multi-agent training, one shared
          server instance hands clearance to every agent's
          WorkerScheduler.

    Both expose the same public surface (register_intent /
    query_go_signal / clear_agent / tick / active_intents), so the
    transport doesn't care which it's holding.

    Exposes the wrapped arbiter as `self.core` so the facade can offer
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


# ROS2 / rclpy (stage 5, deployment only)

class Ros2Transport(SchedulerTransport):
    """
    ROS2 client transport for hardware deployment. Stub.

    Topic schema (per intersection_id `iid`):
        /arcpro/intersection/<iid>/intent              (pub from agent)
        /arcpro/intersection/<iid>/clearance/<agent>   (pub from server)
        /arcpro/intersection/<iid>/clear               (pub from agent on exit)

    Implementation deferred to stage 5 once stages 3-4 are stable in
    simulation. Constructing this transport raises NotImplementedError
    so accidental wiring fails loudly.
    """

    def __init__(
        self,
        node_name: str = "arcpro_intersection_client",
        timeout_ms: int = 100,
    ):
        raise NotImplementedError(
            "Ros2Transport: implement at stage 5 of the rollout. "
            "See .planning/INTERSECTION_NODE_DESIGN.md."
        )

    def send_intent(self, msg: IntentMessage) -> ClearanceReply:
        raise NotImplementedError

    def clear(self, agent_id: str) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        raise NotImplementedError
