"""
Planar Path Planner — Worker-side 2D Trajectory

One classical planner per Worker, producing a single Cartesian
reference path for one intersection traversal. Replaces the previous
"Planning Head does all planning" pattern for intersection-level
intent — the Driver's Planning Head still generates dense vehicle-frame
waypoints for control, but the Worker now publishes a coarse world-frame
path representing "here is where this robot intends to go through the
intersection".

Scope:
    - Generate a 5-waypoint linear polyline per intersection traversal:
        (current_xy, approach_stop_line, in_lane_midpoint,
         exit_entry, exit_plan_end)
    - Straight turns: all five waypoints collinear in the right lane.
    - Left/right turns: entry and exit waypoints lie on different
      approach axes; the polyline passes through an interior midpoint
      of the intersection box forming the turn shape.
    - Store the plan on the Worker (which hangs off AgentNode). MARL
      coordination can later read plans from every agent to detect
      trajectory overlap at the traffic director.

Design choices (after conversing with Arika):
    - Cartesian absolute (x, y) from PhysX ground truth. No Frenet.
    - Linear polyline, no arc/Bezier/clothoid smoothing. Upgradable to
      dense + spline when we want cross-track reward shaping (phase 2).
    - One path per Worker, unique per robot. Not shared across agents.
    - Not in the observation vector. Privileged under PVP.
    - Plan persists through the full intersection maneuver and is
      cleared on return to CRUISING, matching exited_road_id lifetime.

Frame conventions:
    World frame (+X right, +Y up). Heading in radians, 0 = +X.
    Approach heading = direction the car FACES when driving toward
    the intersection center. Exit heading = approach_heading + pi.

PVP:
    This module consumes privileged world coordinates and the topology
    graph. Outputs live on the Worker/AgentNode and feed info dicts,
    scheduler coordination, and future reward shaping. They do NOT
    enter the Driver's observation vector.

Author: Aaron Hamil
Date: 04/23/26
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.intersection_graph import (
    IntersectionNode,
    TurnCommand,
)
from agent.intersection_geometry import (
    IntersectionLayout,
    approach_axes,
    stop_line_center_world,
)


# Data types

@dataclass(frozen=True)
class PlanarWaypoint:
    """
    One point on a planar reference path.

    Fields:
        x, y: World frame position (meters).
        heading: Tangent direction at this waypoint (radians). For
            linear polyline segments this equals the segment's
            direction; at polyline vertices it is the direction of
            the OUTGOING segment (so the final waypoint's heading
            matches the exit road direction).
        s: Cumulative arc length from the first waypoint (meters).
    """
    x: float
    y: float
    heading: float
    s: float


@dataclass
class PlanarPath:
    """
    A sequence of planar waypoints representing one intersection
    traversal plan.

    Invariants:
        - waypoints is non-empty and in order of increasing s.
        - waypoints[0].s == 0.0.
        - intersection_node_id matches the intersection this plan
          routes through.
        - entry_road_id and exit_road_id are set and match the plan.
    """
    waypoints: List[PlanarWaypoint]
    intersection_node_id: str
    entry_road_id: str
    exit_road_id: str
    turn_command: int

    @property
    def length(self) -> float:
        """Total plan length in meters."""
        if not self.waypoints:
            return 0.0
        return self.waypoints[-1].s

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def closest_waypoint_index(self, xy: Tuple[float, float]) -> int:
        """
        Index of the plan waypoint closest to (x, y).

        Linear scan — fine for 5 waypoints and also fine if we densify
        later to ~50.
        """
        best_i = 0
        best_d2 = float("inf")
        for i, wp in enumerate(self.waypoints):
            dx = wp.x - xy[0]
            dy = wp.y - xy[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def cross_track_distance(self, xy: Tuple[float, float]) -> float:
        """
        Unsigned perpendicular distance from (x, y) to the nearest
        polyline segment.

        Used for logging today and as the basis for cross-track reward
        shaping in phase 2. Returns the Euclidean distance to the
        closest waypoint when the path has only one point.
        """
        if len(self.waypoints) == 0:
            return float("inf")
        if len(self.waypoints) == 1:
            wp = self.waypoints[0]
            dx = wp.x - xy[0]
            dy = wp.y - xy[1]
            return math.sqrt(dx * dx + dy * dy)

        best = float("inf")
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            d = _point_segment_distance(xy, (a.x, a.y), (b.x, b.y))
            if d < best:
                best = d
        return best

    def progress(self, xy: Tuple[float, float]) -> float:
        """
        Fractional progress along the plan based on closest waypoint.
        Returns a value in [0, 1]. Useful for logging; more precise
        arc-length-projection progress is a phase-2 item.
        """
        if self.length <= 0:
            return 0.0
        i = self.closest_waypoint_index(xy)
        return max(0.0, min(1.0, self.waypoints[i].s / self.length))


# Planner

class PlanarPathPlanner:
    """
    Generates a 5-waypoint PlanarPath for one intersection traversal.

    Usage pattern (from WorkerNode._handle_cruising, post-_decide_turn):

        plan = planner.plan(
            current_xy=position,
            current_heading=heading,
            intersection=intersection_node,
            entry_road_id=road_id,
            exit_road_id=committed_exit_road_id,
            turn_command=turn_token,
            layout=self.config.layout,
        )
        if plan is not None:
            self._current_plan = plan

    The planner returns None if it cannot construct a valid plan
    (missing intersection center, missing exit road in the approach
    dict, etc.). The Worker treats a None plan as "no plan this
    traversal" — the substate machine does not depend on plan
    presence; the plan is a side-channel signal.
    """

    def __init__(self, exit_plan_ahead_m: float = 1.5):
        """
        Args:
            exit_plan_ahead_m: How far past the exit road's entry
                point to extend the final plan waypoint, in meters.
                Roughly 1-2 car lengths at F1TENTH scale. Gives the
                controller/reward shaping a bit of runway on the
                exit road before the plan ends.
        """
        if exit_plan_ahead_m <= 0:
            raise ValueError("exit_plan_ahead_m must be positive")
        self._exit_plan_ahead_m = exit_plan_ahead_m

    def plan(
        self,
        current_xy: Tuple[float, float],
        current_heading: float,
        intersection: IntersectionNode,
        entry_road_id: str,
        exit_road_id: Optional[str],
        turn_command: int,
        layout: IntersectionLayout,
    ) -> Optional[PlanarPath]:
        """
        Build the 5-waypoint plan, or return None if inputs are invalid.

        Waypoint layout:
            wp0: current vehicle pose
            wp1: approach stop line (in-lane, one lane_half_width
                 offset to driver's right from road centerline)
            wp2: in-lane midpoint between wp1 and wp3 (interior to
                 the intersection box)
            wp3: exit entry (mirror of wp1 on the exit road, in-lane
                 for the exit driving direction)
            wp4: wp3 extended by exit_plan_ahead_m along the exit
                 driving direction

        All heading fields at waypoints are set to the direction of
        the OUTGOING segment, so wp4's heading matches the exit
        driving direction.

        Args:
            current_xy: Vehicle world (x, y).
            current_heading: Vehicle heading in radians (used only
                for wp0.heading and for a sanity check; the plan
                geometry itself is fully determined by the road graph).
            intersection: IntersectionNode for the traversal.
            entry_road_id: Road the agent is approaching on.
            exit_road_id: Committed exit road; None aborts planning.
            turn_command: TurnCommand enum value. Stored on the plan
                but does not alter the geometry (that's encoded by
                entry/exit road axes).
            layout: Physical intersection dimensions.

        Returns:
            PlanarPath with 5 waypoints, or None if invalid.
        """
        if intersection.position is None:
            return None
        if exit_road_id is None:
            return None

        entry_approach = intersection.approaches.get(entry_road_id)
        exit_approach = intersection.approaches.get(exit_road_id)
        if entry_approach is None or exit_approach is None:
            return None

        center = intersection.position
        entry_heading = entry_approach.heading_rad
        # Exit DRIVING direction is opposite the exit approach heading:
        # approach heading points TOWARD the center, so leaving along
        # that road means driving AWAY from the center, i.e.
        # exit_approach.heading_rad + pi (wrapped).
        exit_drive_heading = _wrap_angle(exit_approach.heading_rad + math.pi)

        # wp1: in-lane stop line on the entry approach.
        stop_line_xy = stop_line_center_world(center, entry_heading, layout)

        # wp3: in-lane exit entry. This is the "stop line" geometry on
        # the exit road, but from the exit side — meaning we want a
        # point at intersection_half_width OUT from center along the
        # exit driving direction, offset to the driver's right of the
        # exit road's centerline.
        exit_entry_xy = _exit_lane_entry_world(
            intersection_center=center,
            exit_drive_heading=exit_drive_heading,
            layout=layout,
        )

        # wp2: in-lane mid-intersection point. The geometric center
        # would drag STRAIGHT polylines off the right lane
        # (stop_line -> center -> exit_entry = right-lane -> centerline
        # -> right-lane zigzag). Using the midpoint of wp1 and wp3
        # keeps the polyline inside the lane envelope for STRAIGHT
        # traversals and still passes through an interior point of
        # the intersection box for LEFT/RIGHT turns.
        mid_xy = (
            0.5 * (stop_line_xy[0] + exit_entry_xy[0]),
            0.5 * (stop_line_xy[1] + exit_entry_xy[1]),
        )

        # wp4: extend wp3 along the exit driving direction.
        dx = math.cos(exit_drive_heading) * self._exit_plan_ahead_m
        dy = math.sin(exit_drive_heading) * self._exit_plan_ahead_m
        exit_plan_end_xy = (exit_entry_xy[0] + dx, exit_entry_xy[1] + dy)

        # Sanity check: require vehicle to actually be on the approach
        # side. If the dot of (vehicle -> center) with the entry along
        # direction is non-positive, the vehicle is already past or
        # beside the intersection and a new plan from here would be
        # ill-defined. Abort and let the caller decide.
        along_entry, _ = approach_axes(entry_heading)
        to_center = (center[0] - current_xy[0], center[1] - current_xy[1])
        if (to_center[0] * along_entry[0] + to_center[1] * along_entry[1]) <= 0:
            return None

        # Assemble with headings set to each waypoint's OUTGOING segment.
        raw_points: List[Tuple[float, float]] = [
            current_xy,
            stop_line_xy,
            mid_xy,
            exit_entry_xy,
            exit_plan_end_xy,
        ]
        waypoints = _polyline_to_waypoints(raw_points, final_heading=exit_drive_heading)

        # Replace wp0.heading with the vehicle's actual current heading
        # so logging/debug can compare planned-vs-measured heading at
        # the start. Every other waypoint keeps its segment tangent.
        waypoints[0] = PlanarWaypoint(
            x=waypoints[0].x,
            y=waypoints[0].y,
            heading=current_heading,
            s=waypoints[0].s,
        )

        return PlanarPath(
            waypoints=waypoints,
            intersection_node_id=intersection.node_id,
            entry_road_id=entry_road_id,
            exit_road_id=exit_road_id,
            turn_command=turn_command,
        )


# Internal helpers

def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _exit_lane_entry_world(
    intersection_center: Tuple[float, float],
    exit_drive_heading: float,
    layout: IntersectionLayout,
) -> Tuple[float, float]:
    """
    World (x, y) of the in-lane entry point on the exit road.

    Sits at intersection_half_width meters OUT from the center along
    the exit driving direction, offset lane_half_width to the driver's
    right of that direction's centerline.
    """
    # Axes for the EXIT DRIVING direction (driver on exit road faces
    # away from the center).
    cx, cy = intersection_center
    c = math.cos(exit_drive_heading)
    s = math.sin(exit_drive_heading)
    along = (c, s)
    # Right-perpendicular of the exit driving direction
    right_perp = (s, -c)

    # Out from center along exit driving direction, then into the
    # right-side lane.
    x = cx + layout.intersection_half_width * along[0] + layout.lane_half_width * right_perp[0]
    y = cy + layout.intersection_half_width * along[1] + layout.lane_half_width * right_perp[1]
    return (x, y)


def _polyline_to_waypoints(
    points: List[Tuple[float, float]],
    final_heading: float,
) -> List[PlanarWaypoint]:
    """
    Convert a list of raw (x, y) polyline vertices into PlanarWaypoints
    with cumulative arc length and per-vertex headings.

    Heading at vertex i equals the direction of segment (i, i+1) when
    that segment exists, otherwise final_heading for the last vertex.
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [PlanarWaypoint(points[0][0], points[0][1], final_heading, 0.0)]

    # Segment directions
    seg_headings: List[float] = []
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        seg_headings.append(math.atan2(dy, dx))

    # Cumulative arc length
    cum_s = [0.0] * n
    for i in range(1, n):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cum_s[i] = cum_s[i - 1] + math.sqrt(dx * dx + dy * dy)

    result: List[PlanarWaypoint] = []
    for i in range(n):
        if i < n - 1:
            hdg = seg_headings[i]
        else:
            hdg = final_heading
        result.append(PlanarWaypoint(
            x=points[i][0],
            y=points[i][1],
            heading=hdg,
            s=cum_s[i],
        ))
    return result


def _point_segment_distance(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """
    Unsigned distance from p to line segment ab.

    Handles degenerate (a == b) segments by returning |p - a|.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        qx, qy = px - ax, py - ay
        return math.sqrt(qx * qx + qy * qy)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    ex = px - cx
    ey = py - cy
    return math.sqrt(ex * ex + ey * ey)
