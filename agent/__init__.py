
"""
Agent Module — Hierarchical Driver-Worker Architecture

Each vehicle is managed by an AgentNode containing:
    - Worker node: route planning via intersection graph
    - Main node: vehicle control via learned policy

External coordination via WorkerScheduler.

Replaces the old hardcoded set_turn_bias() pattern with graph-based
route planning and multi-agent intersection coordination.

Author: Aaron Hamil
Date: 03/12/26
Updated: 04/25/26
"""

from agent.agent_node import (
    AgentNode,
    AgentConfig,
    WorkerNode,
    WorkerConfig,
    MainNode,
    IDX_TURN_TOKEN,
    IDX_GO_SIGNAL,
)
from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
    ExitOption,
    EdgeGeometry,
)
from agent.worker_scheduler import (
    WorkerScheduler,
    SchedulerConfig,
    IntentPhase,
    IntentRecord,
)
from agent.agent_env_wrapper import AgentEnvWrapper
# TopologicalEKF / TopologicalState removed — see branch
# `legacy/frenet-topological` for the shelved deployment-side path.
from agent.geometry_calibrator import (
    GeometryCalibrator,
    CalibrationConfig,
)
from agent.planar_planner import (
    PlanarPathPlanner,
    PlanarPath,
    PlanarWaypoint,
)

__all__ = [
    "AgentNode",
    "AgentConfig",
    "WorkerNode",
    "WorkerConfig",
    "MainNode",
    "IntersectionGraph",
    "IntersectionNode",
    "TurnCommand",
    "ExitOption",
    "EdgeGeometry",
    "WorkerScheduler",
    "SchedulerConfig",
    "IntentPhase",
    "IntentRecord",
    "AgentEnvWrapper",
    "GeometryCalibrator",
    "CalibrationConfig",
    "PlanarPathPlanner",
    "PlanarPath",
    "PlanarWaypoint",
    "IDX_TURN_TOKEN",
    "IDX_GO_SIGNAL",
]
