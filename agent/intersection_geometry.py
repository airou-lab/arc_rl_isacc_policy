"""
Intersection Geometry — Frenet Projection + Map Pre-Gate

Stateless geometric helpers used by the Worker's intersection logic.

Purpose:
    1. Synthesize a TopologicalState from global (x, y, heading) at
       TRAINING time so the Worker's substate machine runs on the same
       data shape the EKF emits at DEPLOYMENT (see agent/topological_ekf.py).
       Training path and deployment path feed the Worker identically.

    2. Compute the map-based pre-gate: "am I close enough to an
       intersection that the visual stop-line detector should be armed?"
       This is cheap (scalar arithmetic) and stable. It does NOT tell us
       where the stop line IS — that's the detector's job. It only tells
       us whether looking for one is worthwhile.

    3. Compute per-approach geometric quantities (stop-line center,
       distance-along-approach) used by the GeometricStopLineDetector
       as a bootstrap fallback when the scene lacks visible stop-line
       meshes or the CV pipeline hasn't been validated yet.

Scope: this module is simulator-agnostic. Pure Python + math module,
no numpy / torch / cv2. Scalar inputs, scalar outputs. It drops into
Arika's torch-based Isaac Lab managers without a rewrite.

Frame conventions (matching the rest of the repo):
    - World frame: +X right, +Y up, Z up (Isaac Sim convention).
    - Heading: radians, 0 = +X, pi/2 = +Y, counter-clockwise.
    - An approach heading is the direction the car faces when driving
      TOWARD the intersection center (not away from it).
    - Frenet s: arc-length along an edge, measured from the upstream
      end. s increases in the direction of travel toward the downstream
      intersection. This matches TopologicalState.s.
    - Frenet d: lateral offset from the road centerline. Positive d is
      to the DRIVER'S RIGHT (matches TopologicalEKF convention).
      For right-hand-drive North American roads, an agent correctly
      placed in its approach lane has d ~= +lane_half_width.

PVP note:
    Everything in this module is privileged at training (uses world-frame
    ground-truth position). The OUTPUTS (TopologicalState + pre-gate bool)
    are deployment-realizable because the EKF produces the same shape
    from real sensors. The privileged geometry never enters the policy
    observation vector, and the reward wrapper does not consume meters
    from this module — it consumes Worker state and detector output,
    both of which are one layer removed.

Author: Aaron Hamil
Date: 04/20/26
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    ApproachInfo,
    EdgeGeometry,
)
from agent.topological_ekf import TopologicalState


# Intersection Layout

@dataclass(frozen=True)
class IntersectionLayout:
    """
    Physical dimensions of a single intersection, in meters.

    All values are at 1.0x metric scale (F1TENTH-native). For Arika's
    current scene scaled from 8x -> 1x, these defaults target a
    ~1.0m-wide road with a 1.0m-square intersection box.

    Attributes:
        intersection_half_width: Distance from intersection center to
            the edge of the crossing box, measured along the approach
            axis. The stop line sits at this distance from center.
        lane_half_width: Half the width of a single travel lane.
            For right-hand-drive, the approach lane centerline is
            offset by +lane_half_width from the road centerline.
        pre_gate_distance: Arc-length from the downstream intersection
            below which the stop-line detector is armed. This is the
            map-based pre-gate radius in Frenet s-space. Distinct from
            IntersectionNode.radius (the 2D trigger used by the legacy
            WorkerNode.step path).
        stop_line_tolerance: Longitudinal tolerance for "stopped at
            the stop line" when evaluating final-stop proximity. Used
            by IntersectionRewardWrapper; not consumed here.
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


# Frenet Projection (Training-side synthesis of TopologicalState)

def project_to_approach_frenet(
    agent_xy: Tuple[float, float],
    agent_heading_rad: float,
    agent_speed: float,
    edge: EdgeGeometry,
) -> TopologicalState:
    """
    Project a world-frame agent pose onto an approach edge in Frenet.

    At training time, we don't have the EKF running — but the Worker's
    substate machine expects TopologicalState-shaped input. This function
    synthesizes the same shape from privileged world-frame data.

    At deployment, TopologicalEKF.state produces the identical object
    from real sensors. The Worker cannot distinguish the two paths.

    Assumes the edge is a straight segment (1D approach). This is true
    for all current intersection approaches per project convention;
    curved edges will need a spline-based version later.

    Args:
        agent_xy: World (x, y) of the agent.
        agent_heading_rad: Agent heading (radians, world frame).
        agent_speed: Agent speed (m/s).
        edge: EdgeGeometry for the approach the agent is on. Must have
            start_position, end_position, heading, length populated
            (i.e. the graph must be calibrated).

    Returns:
        TopologicalState with edge_id, s, d, theta_err, speed,
        edge_length, edge_heading filled in. start_position and
        end_position define the 1D axis.

    Raises:
        ValueError: if edge lacks calibrated start/end positions.
    """
    if edge.start_position is None or edge.end_position is None:
        raise ValueError(
            f"Cannot project onto edge '{edge.edge_id}': missing "
            f"calibrated start_position or end_position. Run the "
            f"GeometryCalibrator first."
        )

    ex = edge.end_position[0] - edge.start_position[0]
    ey = edge.end_position[1] - edge.start_position[1]
    edge_len_sq = ex * ex + ey * ey
    if edge_len_sq < 1e-9:
        raise ValueError(f"Degenerate edge '{edge.edge_id}' (zero length)")

    # Agent relative to edge start
    rx = agent_xy[0] - edge.start_position[0]
    ry = agent_xy[1] - edge.start_position[1]

    # s = projection onto edge direction (already length-normalized dot)
    s = (rx * ex + ry * ey) / math.sqrt(edge_len_sq)

    # d = signed lateral offset (cross product sign convention:
    # positive d on the RIGHT of travel direction, matching TopologicalEKF)
    # 2D cross product (ex, ey) x (rx, ry) = ex*ry - ey*rx
    # That gives left-positive. We want right-positive, so negate.
    cross = ex * ry - ey * rx
    d = -cross / math.sqrt(edge_len_sq)

    # Heading error = agent heading - edge heading
    theta_err = _angle_diff(agent_heading_rad, edge.heading)

    return TopologicalState(
        edge_id=edge.edge_id,
        s=float(s),
        d=float(d),
        theta_err=float(theta_err),
        speed=float(agent_speed),
        edge_length=float(edge.length),
        edge_heading=float(edge.heading),
    )


# Map Pre-Gate (for arming the stop-line detector)

def within_pre_gate(
    frenet: TopologicalState,
    layout: IntersectionLayout,
) -> bool:
    """
    Map-based pre-gate: should the visual stop-line detector be armed?

    True iff the agent is within `pre_gate_distance` of the downstream
    intersection along the current edge. Uses arc-length (s-space),
    not 2D Euclidean, so long straight approaches don't false-trigger
    at 3m out while short approaches still arm in time.

    This is the collaboration point between map knowledge and vision:
    the map says "an intersection is coming up"; the detector then
    runs and says "here is the white stripe, N meters ahead".

    Args:
        frenet: Current Frenet state (from EKF or projection).
        layout: Intersection layout config.

    Returns:
        True if the detector should run this frame.
    """
    if frenet.edge_length <= 0:
        return False
    return frenet.distance_to_next <= layout.pre_gate_distance


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
    additionally returns the matched road_id so callers can look up
    edge geometry. Also requires the matched approach's edge geometry
    to be available (for Frenet projection downstream).

    Args:
        agent_xy: World (x, y) of the agent (reserved — currently only
            used for tie-breaking if multiple approaches match, which
            shouldn't happen in well-formed 4-way topologies).
        agent_heading_rad: Agent heading in world frame.
        node: The IntersectionNode in question.
        graph: The containing graph (for edge geometry lookup).
        heading_tolerance_rad: Max allowed heading mismatch for a
            match. Default 35deg; generous enough to tolerate turn
            wobble without matching a perpendicular approach.

    Returns:
        (road_id, ApproachInfo) if a match is found; None otherwise.
    """
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

    # Require edge geometry to exist (so downstream projection works)
    if graph.get_edge_geometry(best_road_id) is None:
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
