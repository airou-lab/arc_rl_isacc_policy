"""
Intersection Graph — Topology + Geometry Separation
=====================================================

Two layers:
    1. TOPOLOGY (from JSON, human-authored, stable):
       Which roads exist, which turns connect them, approximate headings.
       Changes only when roads are added/removed from the map.

    2. GEOMETRY (from calibration or PhysX, auto-discovered):
       Intersection positions, edge lengths, exact headings.
       Changes whenever the USD scene is modified.

The JSON defines topology only; no positions, no lengths required.
Geometry is populated at runtime by GeometryCalibrator or loaded from
a saved calibration file.

JSON Schema (topology only):
    {
      "intersections": {
        "<id>": {
          "approaches": {
            "<road_id>": {
              "heading": <degrees>,
              "exits": {
                "left":     { "road": "<road_id>" },
                "straight": { "road": "<road_id>" },
                "right":    { "road": "<road_id>" }
              }
            }
          }
        }
      }
    }

Author: Aaron Hamil
Date: 03/12/26
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Turn Commands

class TurnCommand:
    """Discrete navigation commands replacing continuous turn_bias."""
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    _NAMES = {-1: "LEFT", 0: "STRAIGHT", 1: "RIGHT"}

    @classmethod
    def name(cls, value: int) -> str:
        return cls._NAMES.get(value, f"UNKNOWN({value})")

    @classmethod
    def all(cls) -> List[int]:
        return [cls.LEFT, cls.STRAIGHT, cls.RIGHT]


# Topology Data Types

@dataclass(frozen=True)
class ExitOption:
    """One possible exit from an intersection."""
    turn_command: int
    exit_road_id: str


@dataclass
class ApproachInfo:
    """One approach direction into an intersection (topology)."""
    road_id: str
    heading_rad: float              # Approximate, refined by calibration
    exits: Dict[int, ExitOption]    # TurnCommand -> ExitOption


@dataclass
class IntersectionNode:
    """A single intersection in the road network."""
    node_id: str
    approaches: Dict[str, ApproachInfo]

    # Geometry (None until calibrated)
    position: Optional[Tuple[float, float]] = None
    radius: float = 3.0

    @property
    def is_calibrated(self) -> bool:
        return self.position is not None

    def distance_to(self, x: float, y: float) -> float:
        if self.position is None:
            return float("inf")
        dx = self.position[0] - x
        dy = self.position[1] - y
        return math.sqrt(dx * dx + dy * dy)


# Edge Geometry

@dataclass
class EdgeGeometry:
    """
    Measured geometry for one road segment.
    Populated by calibration, NOT from JSON.
    """
    edge_id: str
    length: float                                          # meters
    heading: float                                         # radians
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    curvature: float = 0.0
    lane_width: float = 0.3
    start_position: Optional[Tuple[float, float]] = None   # world frame
    end_position: Optional[Tuple[float, float]] = None


# Main Graph

class IntersectionGraph:
    """
    Road network: topology from JSON, geometry from calibration.

    The graph is functional with topology alone - the Worker can pick
    turns. Geometry is needed for spatial queries (nearest_intersection),
    EKF arc-length tracking, and Scheduler RVO.
    """

    APPROACH_TOLERANCE_RAD = math.radians(30.0)

    def __init__(
        self,
        intersections: Dict[str, IntersectionNode],
        edge_geometry: Optional[Dict[str, EdgeGeometry]] = None,
    ):
        self._intersections = intersections
        self._edge_geometry = edge_geometry or {}
        self._road_to_node: Dict[str, str] = {}
        for node_id, node in intersections.items():
            for road_id in node.approaches:
                self._road_to_node[road_id] = node_id

    @classmethod
    def from_json(cls, path: str | Path) -> IntersectionGraph:
        """Load topology from JSON. Call load_geometry() or calibrate
        to add positions and lengths."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        intersections: Dict[str, IntersectionNode] = {}

        for node_id, node_data in data["intersections"].items():
            approaches: Dict[str, ApproachInfo] = {}

            for road_id, approach_data in node_data["approaches"].items():
                heading_rad = math.radians(float(approach_data["heading"]))
                exits: Dict[int, ExitOption] = {}

                for turn_name, exit_info in approach_data.get("exits", {}).items():
                    turn_cmd = {"left": TurnCommand.LEFT,
                                "straight": TurnCommand.STRAIGHT,
                                "right": TurnCommand.RIGHT}.get(turn_name)
                    if turn_cmd is None or exit_info is None:
                        continue

                    exit_road = exit_info["road"] if isinstance(exit_info, dict) else exit_info
                    exits[turn_cmd] = ExitOption(turn_command=turn_cmd, exit_road_id=exit_road)

                approaches[road_id] = ApproachInfo(
                    road_id=road_id, heading_rad=heading_rad, exits=exits,
                )

            # Position is optional in JSON (backward compatible)
            pos = tuple(node_data["position"]) if "position" in node_data else None
            radius = float(node_data.get("radius", 3.0))

            intersections[node_id] = IntersectionNode(
                node_id=node_id, approaches=approaches,
                position=pos, radius=radius,
            )

        return cls(intersections)

    # Geometry Management

    def set_geometry(self, edge_geometry: Dict[str, EdgeGeometry]) -> None:
        """Inject calibrated edge geometry."""
        self._edge_geometry = edge_geometry

    def set_intersection_position(
        self, node_id: str, position: Tuple[float, float], radius: float = 3.0
    ) -> None:
        """Set a calibrated intersection position."""
        node = self._intersections.get(node_id)
        if node:
            node.position = position
            node.radius = radius

    def get_edge_geometry(self, edge_id: str) -> Optional[EdgeGeometry]:
        return self._edge_geometry.get(edge_id)

    @property
    def is_calibrated(self) -> bool:
        """True if all intersections have positions and all roads have geometry."""
        nodes_ok = all(n.is_calibrated for n in self._intersections.values())
        roads = set(self._road_to_node.keys())
        edges_ok = roads.issubset(set(self._edge_geometry.keys()))
        return nodes_ok and edges_ok

    def save_geometry(self, path: str | Path) -> None:
        """Save calibrated geometry to JSON for reuse."""
        path = Path(path)
        data = {
            "intersections": {},
            "edges": {},
        }
        for nid, node in self._intersections.items():
            data["intersections"][nid] = {
                "position": list(node.position) if node.position else None,
                "radius": node.radius,
            }
        for eid, edge in self._edge_geometry.items():
            data["edges"][eid] = {
                "length": edge.length,
                "heading": math.degrees(edge.heading),
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "curvature": edge.curvature,
                "start_position": list(edge.start_position) if edge.start_position else None,
                "end_position": list(edge.end_position) if edge.end_position else None,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_geometry(self, path: str | Path) -> None:
        """Load previously calibrated geometry."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        for nid, geo in data.get("intersections", {}).items():
            pos = tuple(geo["position"]) if geo.get("position") else None
            self.set_intersection_position(nid, pos, geo.get("radius", 3.0))

        edges = {}
        for eid, geo in data.get("edges", {}).items():
            sp = tuple(geo["start_position"]) if geo.get("start_position") else None
            ep = tuple(geo["end_position"]) if geo.get("end_position") else None
            edges[eid] = EdgeGeometry(
                edge_id=eid,
                length=geo["length"],
                heading=math.radians(geo["heading"]),
                from_node=geo.get("from_node"),
                to_node=geo.get("to_node"),
                curvature=geo.get("curvature", 0.0),
                start_position=sp,
                end_position=ep,
            )
        self.set_geometry(edges)

    # Spatial Queries (require geometry)

    def nearest_intersection(
        self, x: float, y: float
    ) -> Optional[IntersectionNode]:
        """Find intersection within trigger radius. Requires calibrated positions."""
        best_node = None
        best_dist = float("inf")
        for node in self._intersections.values():
            if not node.is_calibrated:
                continue
            dist = node.distance_to(x, y)
            if dist <= node.radius and dist < best_dist:
                best_node = node
                best_dist = dist
        return best_node

    # Topology Queries (always available)

    def get_exit_options(
        self, node_id: str, agent_heading: float
    ) -> List[ExitOption]:
        """Get available exits for an approach heading."""
        node = self._intersections.get(node_id)
        if node is None:
            return []

        best_approach = None
        best_diff = float("inf")
        for approach in node.approaches.values():
            diff = abs(_angle_diff(agent_heading, approach.heading_rad))
            if diff <= self.APPROACH_TOLERANCE_RAD and diff < best_diff:
                best_approach = approach
                best_diff = diff

        return list(best_approach.exits.values()) if best_approach else []

    def get_intersection(self, node_id: str) -> Optional[IntersectionNode]:
        return self._intersections.get(node_id)

    def get_node_for_road(self, road_id: str) -> Optional[str]:
        """Which intersection does this road approach?"""
        return self._road_to_node.get(road_id)

    def get_all_road_ids(self) -> List[str]:
        return list(self._road_to_node.keys())

    @property
    def all_intersections(self) -> Dict[str, IntersectionNode]:
        return dict(self._intersections)

    @property
    def all_edge_geometry(self) -> Dict[str, EdgeGeometry]:
        return dict(self._edge_geometry)

    def __len__(self) -> int:
        return len(self._intersections)

    def __repr__(self) -> str:
        cal = "calibrated" if self.is_calibrated else "uncalibrated"
        return f"IntersectionGraph(n={len(self._intersections)}, {cal})"


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d
