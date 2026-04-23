"""
Intersection Geometry — Planar Pre-Gate + Approach Helpers

Stateless geometric helpers used by the Worker's intersection logic.

Purpose:
    1. Compute the map-based pre-gate: "am I close enough to an
       intersection along my approach axis that the stop-line detector
       should be armed?" Planar Cartesian check, no Frenet, no EKF.

    2. Compute per-approach geometric quantities (stop-line center,
       distance-to-stop-line, exit-road detection) used by the Worker's
       substate machine and the GeometricStopLineDetector.

Scope: simulator-agnostic. Pure Python + math module. Scalar inputs,
scalar outputs.

Frame conventions (matching the rest of the repo):
    - World frame: +X right, +Y up, Z up (Isaac Sim convention).
    - Heading: radians, 0 = +X, pi/2 = +Y, counter-clockwise.
    - An approach heading is the direction the car faces when driving
      TOWARD the intersection center (not away from it).

History:
    This module previously contained Frenet projection utilities
    (project_to_approach_frenet, the old within_pre_gate) that fed
    a TopologicalEKF-shaped signal to the Worker. That path is shelved
    on branch `legacy/frenet-topological`. The planar pre-gate is the
    minimal 2D replacement — the Worker only ever needed "is the agent
    within pre_gate_distance of the intersection along its approach
    axis?" which does not require arc-length tracking.

PVP note:
    Everything in this module operates on world-frame ground-truth
    position. Outputs feed the Worker's classical planner, never the
    Driver's observation vector. The Worker's outputs to the Driver
    (turn_token, go_signal) remain discrete/bounded.

Author: Aaron Hamil
Date: 04/20/26
Updated: 04/23/26 — Frenet path shelved, planar pre-gate added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    ApproachInfo,
)


# Intersection Layout

@dataclass(frozen=True)
class IntersectionLayout:
    """
    Physical dimensions of a single intersection, in meters.

    All values are at 1.0x metric scale (F1TENTH-native).

    Attributes:
        intersection_half_width: Distance from intersection center to
            the edge of the crossing box, measured along the approach
            axis. The stop line sits at this distance from center.
        lane_half_width: Half the width of a single travel lane.
            For right-hand-drive, the approach lane centerline is
            offset by +lane_half_width from the road centerline.
        pre_gate_distance: Signed along-approach distance from the
            intersection center below which the stop-line detector is
            armed. The Worker's planar pre-gate fires when the agent
            is between 0 and pre_gate_distance meters OUT from center
            along the approach axis.
        stop_line_tolerance: Longitudinal tolerance for "stopped at
            the stop line" when evaluating final-stop proximity.
        exit_detection_radius: Along-edge distance past the intersection
            center at which the agent is considered "committed to an
            exit road". Used for completion logic.
    """
    intersection_half_width: float = 0.5
    lane_half_width: float = 0.25
    pre_gate_distance: float = 1.5
    stop_line_tolerance: float = 0.08
    exit_detection_radius: float = 0.8


# Angle Helpers

def _wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to (-pi, pi]."""
    return _wrap_angle(a - b)


# Approach Axes

def approach_axes(
    approach_heading_rad: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the along-approach and right-perpendicular unit vectors.

    The along vector points from the approach road toward the
    intersection center (the direction the approaching car faces).
    The right-perpendicular vector points to the driver's right,
    which for North American right-hand-drive is also the direction
    offset from road-centerline to approach-lane-centerline.

    Args:
        approach_heading_rad: Approach heading in radians.

    Returns:
        (along_unit, right_perp_unit), each a 2-tuple (x, y).
    """
    c = math.cos(approach_heading_rad)
    s = math.sin(approach_heading_rad)
    along = (c, s)
    # Right-perpendicular (90deg clockwise from heading):
    # rotate (c, s) by -pi/2 -> (s, -c)
    right_perp = (s, -c)
    return along, right_perp


# Stop Line Center (Privileged)

def stop_line_center_world(
    intersection_center: Tuple[float, float],
    approach_heading_rad: float,
    layout: IntersectionLayout,
) -> Tuple[float, float]:
    """
    World-frame (x, y) of the center of the stop line for one approach.

    The stop line sits at the edge of the intersection box along the
    approach axis (distance = intersection_half_width back from center),
    and in the driver's-right approach lane (lateral offset =
    +lane_half_width from road centerline).

    Args:
        intersection_center: World (x, y) of the intersection.
        approach_heading_rad: Approach heading (direction TOWARD the
            intersection).
        layout: Intersection geometric layout.

    Returns:
        World (x, y) of the stop-line center.
    """
    cx, cy = intersection_center
    along, right_perp = approach_axes(approach_heading_rad)

    # Start from intersection center
    # Move BACK along -along by intersection_half_width (to edge of box)
    # Move RIGHT by lane_half_width (into approach lane)
    x = cx - layout.intersection_half_width * along[0] + layout.lane_half_width * right_perp[0]
    y = cy - layout.intersection_half_width * along[1] + layout.lane_half_width * right_perp[1]
    return (x, y)


# Planar Pre-Gate (replaces the Frenet pre-gate)

def signed_distance_along_approach(
    agent_xy: Tuple[float, float],
    intersection_center: Tuple[float, float],
    approach_heading_rad: float,
) -> float:
    """
    Signed along-approach distance from the intersection center to the
    agent.

    Positive while the agent is still approaching (i.e. upstream of
    the intersection along the approach road).
    Negative once the agent has crossed the intersection center.

    This is the planar replacement for the old Frenet distance_to_next
    signal. Derivation:
        along points TOWARD the center (approach direction).
        (agent - center) dotted with -along gives the distance from the
        center outward along the approach road.

    Args:
        agent_xy: World (x, y) of the agent.
        intersection_center: World (x, y) of the intersection.
        approach_heading_rad: Approach heading (direction TOWARD the
            intersection center).

    Returns:
        Signed meters. Positive = agent is still approaching.
    """
    along, _ = approach_axes(approach_heading_rad)
    rx = agent_xy[0] - intersection_center[0]
    ry = agent_xy[1] - intersection_center[1]
    return -(rx * along[0] + ry * along[1])


def within_pre_gate_planar(
    agent_xy: Tuple[float, float],
    intersection_center: Tuple[float, float],
    approach_heading_rad: float,
    layout: IntersectionLayout,
) -> bool:
    """
    Planar Cartesian pre-gate: should the stop-line detector arm?

    True iff the agent is on the approach side of the intersection
    (signed distance >= 0) AND within layout.pre_gate_distance meters
    of the center along the approach axis.

    This is the collaboration point between map knowledge and vision:
    the map says "an intersection is coming up"; the detector then
    runs and says "here is the white stripe, N meters ahead".

    Replaces the Frenet-based within_pre_gate from the shelved
    agent.topological_ekf path. Behavior on straight approach edges
    is identical — signed distance along approach equals edge_length - s.

    Args:
        agent_xy: World (x, y) of the agent.
        intersection_center: World (x, y) of the intersection.
        approach_heading_rad: Approach heading (direction TOWARD the
            intersection center).
        layout: Intersection layout config.

    Returns:
        True if the detector should run this frame.
    """
    d = signed_distance_along_approach(
        agent_xy, intersection_center, approach_heading_rad
    )
    return 0.0 <= d <= layout.pre_gate_distance


# Infer current approach / exit road

def infer_current_approach(
    agent_xy: Tuple[float, float],
    agent_heading_rad: float,
    node: IntersectionNode,
    graph: IntersectionGraph,
    heading_tolerance_rad: float = math.radians(35.0),
) -> Optional[Tuple[str, ApproachInfo]]:
    """
    Identify which approach the agent is currently on at a given node.

    Uses the same best-heading-match rule as IntersectionGraph but
    additionally returns the matched road_id so callers can reference
    per-approach data downstream.

    Unlike the previous version, this does NOT require edge geometry
    to be present — the planar pre-gate only needs the intersection
    center (on IntersectionNode) and the approach heading (on
    ApproachInfo), both of which are populated from the topology JSON.

    Args:
        agent_xy: World (x, y) of the agent (reserved — currently only
            used for tie-breaking if multiple approaches match, which
            shouldn't happen in well-formed 4-way topologies).
        agent_heading_rad: Agent heading in world frame.
        node: The IntersectionNode in question.
        graph: The containing graph (unused in the planar path; kept
            for signature compatibility with callers that expected the
            Frenet version's edge lookup).
        heading_tolerance_rad: Max allowed heading mismatch for a
            match. Default 35deg; generous enough to tolerate turn
            wobble without matching a perpendicular approach.

    Returns:
        (road_id, ApproachInfo) if a match is found; None otherwise.
    """
    del agent_xy, graph  # reserved / signature-compatible

    best_road_id: Optional[str] = None
    best_approach: Optional[ApproachInfo] = None
    best_diff = float("inf")

    for road_id, approach in node.approaches.items():
        diff = abs(_angle_diff(agent_heading_rad, approach.heading_rad))
        if diff <= heading_tolerance_rad and diff < best_diff:
            best_road_id = road_id
            best_approach = approach
            best_diff = diff

    if best_road_id is None or best_approach is None:
        return None

    return (best_road_id, best_approach)


def detect_exited_road(
    agent_xy: Tuple[float, float],
    agent_heading_rad: float,
    node: IntersectionNode,
    layout: IntersectionLayout,
    heading_tolerance_rad: float = math.radians(35.0),
) -> Optional[str]:
    """
    Identify which road the agent has EXITED onto, if any.

    An agent has "exited onto road A" iff:
        1. Agent is outside the intersection box along road A's axis
           (dot(agent - center, -along_A) > exit_detection_radius);
        2. Agent's heading matches road A's EXIT direction
           (= approach_heading + pi, wrapped) within tolerance;
        3. Agent's lateral offset from road A's centerline is
           within road-half-width — i.e. actually on the road, not
           in an adjacent field.

    Used by the Worker's COMMITTED -> CRUISING transition to validate
    whether the agent ended up on the road its turn_token specified.

    Args:
        agent_xy: World (x, y) of the agent.
        agent_heading_rad: Agent heading.
        node: Intersection the agent was committed at.
        layout: Geometric layout config.
        heading_tolerance_rad: Heading match tolerance.

    Returns:
        road_id of the road the agent has exited onto, or None if the
        agent is still inside the intersection or hasn't cleanly
        committed to an exit road.
    """
    if node.position is None:
        return None
    cx, cy = node.position

    # The maximum lateral offset we'll allow for "on this road".
    # Full road width = 2 * lane_half_width (two lanes, no shoulders).
    road_half_width = 2.0 * layout.lane_half_width

    best_road: Optional[str] = None
    best_along: float = layout.exit_detection_radius  # must exceed this

    for road_id, approach in node.approaches.items():
        exit_heading = _wrap_angle(approach.heading_rad + math.pi)
        along, right_perp = approach_axes(approach.heading_rad)

        # "Out" direction for this road = -along (away from intersection
        # center, toward the dead end of road A)
        out_x, out_y = -along[0], -along[1]

        # Project agent - center onto out direction
        rx = agent_xy[0] - cx
        ry = agent_xy[1] - cy
        along_out = rx * out_x + ry * out_y

        # Lateral offset (perp to road axis)
        lateral = abs(rx * right_perp[0] + ry * right_perp[1])

        # Heading match
        hdg_diff = abs(_angle_diff(agent_heading_rad, exit_heading))

        if (
            along_out > best_along
            and lateral < road_half_width
            and hdg_diff < heading_tolerance_rad
        ):
            best_along = along_out
            best_road = road_id

    return best_road


# Geometric distance to stop line (bootstrap / fallback)

def distance_to_stop_line_world(
    agent_xy: Tuple[float, float],
    intersection_center: Tuple[float, float],
    approach_heading_rad: float,
    layout: IntersectionLayout,
) -> float:
    """
    Signed along-approach distance from agent to the stop line.

    Positive while the agent is still approaching (line is ahead).
    Negative once the agent has crossed the line.

    This is the SAME signal the visual detector returns (as
    StopLineDetection.distance_m), but computed from privileged
    world-frame geometry. The GeometricStopLineDetector uses this;
    the VisualStopLineDetector recovers the same quantity from
    camera pixels.

    Args:
        agent_xy: World (x, y) of the agent.
        intersection_center: World (x, y) of the intersection.
        approach_heading_rad: Approach heading.
        layout: Geometric layout config.

    Returns:
        Signed distance in meters. Positive = line ahead, negative =
        line behind.
    """
    stop_x, stop_y = stop_line_center_world(
        intersection_center, approach_heading_rad, layout
    )
    along, _ = approach_axes(approach_heading_rad)
    rx = agent_xy[0] - stop_x
    ry = agent_xy[1] - stop_y
    # dot(agent - stop, along) is positive when agent is PAST the line.
    # Negate to get "distance remaining".
    return -(rx * along[0] + ry * along[1])
