"""
Worker Scheduler — Facade Over SchedulerTransport

Stage 2 refactor (04/28/26):
    The arbitration logic moved to agent/scheduler_core.py
    (SchedulerCore). This file is now a thin facade that delegates
    to a SchedulerTransport. The Worker's call surface is unchanged:

        register_intent(agent_id, intersection_id, turn_command, ...)
        query_go_signal(agent_id, intersection_id, ...)
        clear_agent(agent_id)
        tick()
        active_intents             # only available with LocalTransport

    By default, WorkerScheduler() builds a LocalTransport wrapping
    a fresh SchedulerCore. Existing call sites continue to work
    without modification.

Why a facade
A single entry point for the Worker that can target either an
in-process arbiter (LocalTransport, today) or a remote
IntersectionNodeServer (GzTransport / Ros2Transport, stages 3-5).
The Worker doesn't need to know which.

Backward compatibility
- IntentPhase, IntentRecord, SchedulerConfig, _paths_conflict,
  _angle_diff, _euclid are re-exported from this module so existing
  imports like `from agent.worker_scheduler import IntentPhase`
  continue to work.
- WorkerScheduler(config=..., graph=...) keeps its old signature.
- register_intent / query_go_signal / clear_agent / tick keep their
  old signatures and return types.

Author: Aaron Hamil
Updated: 04/28/26 — refactored into facade over SchedulerTransport
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

# Backward-compat re-exports — keep `from agent.worker_scheduler import X`
# working for every X that was exported by the pre-refactor module.
from agent.scheduler_core import (
    IntentPhase,
    IntentRecord,
    RVOConstraint,
    SchedulerCore,
    SchedulerConfig,
    _angle_diff,
    _euclid,
    _paths_conflict,
)

if TYPE_CHECKING:
    from agent.intersection_graph import IntersectionGraph
    from agent.scheduler_transport import SchedulerTransport

logger = logging.getLogger(__name__)


class WorkerScheduler:
    """
    Facade between the Worker and the active SchedulerTransport.

    Translates the Worker's method-call API into IntentMessage /
    ClearanceReply exchanges with the transport. For in-process
    runs (the default), the transport is a LocalTransport wrapping
    a SchedulerCore — making the facade an essentially zero-cost
    indirection.

    Usage (unchanged from the pre-refactor API):
        sched = WorkerScheduler(graph=graph)
        go = sched.register_intent(agent_id, iid, turn_cmd, ...)
        sched.tick()
        sched.clear_agent(agent_id)

    Plugging a custom transport:
        from agent.scheduler_transport import LocalTransport
        from agent.scheduler_core import SchedulerCore
        core = SchedulerCore(config=cfg, graph=graph)
        transport = LocalTransport(core)
        sched = WorkerScheduler(transport=transport)
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        graph: Optional["IntersectionGraph"] = None,
        transport: Optional["SchedulerTransport"] = None,
    ):
        """
        Args:
            config: Scheduler config. Used only if `transport` is
                None (to build the default LocalTransport).
            graph: Calibrated IntersectionGraph. Same caveat as config.
            transport: A SchedulerTransport instance. If provided,
                `config` and `graph` are ignored at this layer (they
                are properties of the transport's backing core or
                remote server). If None, a LocalTransport wrapping a
                fresh SchedulerCore(config, graph) is built.
        """
        if transport is None:
            # Lazy import to avoid circular dependency at module load.
            from agent.scheduler_transport import LocalTransport
            self._core: Optional[SchedulerCore] = SchedulerCore(
                config=config, graph=graph,
            )
            self._transport = LocalTransport(self._core)
        else:
            if config is not None or graph is not None:
                logger.warning(
                    "WorkerScheduler: config/graph passed alongside an "
                    "explicit transport; ignored at the facade layer "
                    "(set them on the transport's backing core instead)."
                )
            self._transport = transport
            # Preserve a direct ref to the core when the transport
            # exposes one (LocalTransport does). This lets active_intents
            # work without an extra round-trip.
            self._core = getattr(transport, "core", None)

    # Public API — mirrors the pre-refactor scheduler

    def register_intent(
        self,
        agent_id: str,
        intersection_id: str,
        turn_command: int,
        position: Tuple[float, float] = (0.0, 0.0),
        heading: float = 0.0,
        speed: float = 0.0,
    ) -> float:
        """Submit an intent. Returns go_signal (1.0 = GO, 0.0 = WAIT)."""
        from agent.scheduler_transport import IntentMessage  # avoid cycle
        msg = IntentMessage(
            agent_id=agent_id,
            intersection_id=intersection_id,
            turn_command=turn_command,
            position=position,
            heading=heading,
            speed=speed,
            phase=IntentPhase.DECIDING,
        )
        reply = self._transport.send_intent(msg)
        return float(reply.go_signal)

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
        Re-query go_signal with updated kinematic state.

        For LocalTransport this is identical to register_intent
        (the underlying core's two methods are also identical apart
        from the initial-registration upsert). For network transports
        the transport may choose to use a different message kind, but
        the semantics are the same.
        """
        from agent.scheduler_transport import IntentMessage
        msg = IntentMessage(
            agent_id=agent_id,
            intersection_id=intersection_id,
            turn_command=turn_command,
            position=position,
            heading=heading,
            speed=speed,
            phase=IntentPhase.DECIDING,
        )
        reply = self._transport.send_intent(msg)
        return float(reply.go_signal)

    def clear_agent(self, agent_id: str) -> None:
        """Tell the transport an agent has left the intersection."""
        self._transport.clear(agent_id)

    def tick(self) -> None:
        """Per-step housekeeping (delegates to transport)."""
        self._transport.tick()

    @property
    def active_intents(self) -> Dict[str, IntentRecord]:
        """
        Read-only access to active intents.

        Only available when the transport exposes a backing core
        (LocalTransport). Network transports raise — for those, query
        the remote server directly if you need registry inspection.
        """
        if self._core is not None:
            return self._core.active_intents
        raise AttributeError(
            "active_intents is only available with LocalTransport. "
            "Network transports do not expose registry state through "
            "the facade; query the remote server directly."
        )

    # Convenience accessor

    @property
    def transport(self) -> "SchedulerTransport":
        """The underlying transport. Useful for tests and diagnostics."""
        return self._transport

    def __repr__(self) -> str:
        return f"WorkerScheduler(transport={self._transport.__class__.__name__})"


__all__ = [
    # Re-exports from scheduler_core (backward compat)
    "IntentPhase",
    "IntentRecord",
    "RVOConstraint",
    "SchedulerCore",
    "SchedulerConfig",
    # Facade
    "WorkerScheduler",
    # Helpers (used by tests and a future IntersectionNodeServer)
    "_paths_conflict",
    "_angle_diff",
    "_euclid",
]
