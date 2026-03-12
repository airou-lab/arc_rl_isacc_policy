"""
Topological EKF — Graph-Embedded Odometry + IMU Fusion
=======================================================

Fuses wheel odometry and IMU data into the intersection graph's
topological coordinate frame rather than a global Cartesian frame.

Why this matters:
    Global EKF on the F1Tenth will drift. The D435i IMU has gyro bias,
    wheel odometry has slip, and there's no GPS indoors. After a few
    laps, the global (x, y) estimate drifts far enough that the Worker's
    nearest_intersection() check fails — the robot thinks it's 3 meters
    from where it actually is, misses intersection triggers, and makes
    wrong turns or no turns.

    Topological EKF solves this by tracking position WITHIN the graph:
    "I'm on road_A, 2.3m from intersection INT_A, 0.04m left of center."
    This is a 1D localization problem (distance along edge) with periodic
    hard resets at intersection nodes. Drift between intersections is
    bounded by the edge length, and every intersection crossing resets
    the accumulated error to near zero.

State Vector (Frenet frame along current edge):
    s           — arc-length along current edge (meters from upstream node)
    d           — lateral offset from road centerline (meters, + = right)
    theta_err   — heading error relative to edge direction (radians)
    v           — longitudinal speed (m/s)

    Edge identity (current_edge_id) is tracked discretely outside the
    EKF state, it transitions when s exceeds the edge length.

Prediction Model (from IMU + wheel odometry):
    s-dot     = v · cos(theta_err)
    d-dot     = v · sin(theta_err)
    theta-dot_err = omega_imu − kappa · v         (yaw rate minus road curvature effect)
    v-dot     = a_odom                        (from wheel odometry or IMU accel)

    where kappa is the road curvature (from graph edge metadata, usually 0
    for straight segments, nonzero for curves).

Update / Correction Events:
    1. INTERSECTION ARRIVAL (s >= edge_length):
       Hard reset — agent is at the downstream intersection node.
       s -> theta on the new outgoing edge, d -> 0, theta_err -> 0.
       Covariance collapses to near-zero. This kills diminishes drift.

    2. LANE DETECTION (visual):
       Soft update — lateral_offset from lane_detector.py provides
       a measurement of d. Heading error can be inferred from
       consecutive lateral offset readings.

    3. INTERSECTION DETECTION (visual):
       Soft update — if the camera detects intersection features
       (widening, markings) before the distance estimate says we're
       there, it provides an early correction to s.

Graph Edge Metadata (extends intersection_graph.json):
    Each road segment needs:
        "length": <meters>,          // distance between connected intersections
        "heading": <degrees>,        // direction of travel along this edge
        "curvature": <1/meters>,     // average curvature (0 for straight)

    These are added to the JSON without changing the IntersectionGraph
    class, the EKF reads them directly.

Integration with Worker:
    The Worker currently calls:
        graph.nearest_intersection(x, y)

    With topological EKF, it instead calls:
        ekf.is_at_intersection() -> bool
        ekf.current_edge         -> str
        ekf.distance_to_next     -> float

    No global coordinates needed. The Worker never sees (x, y).

Training vs Deployment:
    Training (Isaac Sim): PhysX ground truth -> initialize EKF, but the
    EKF still runs so the policy trains with realistic localization noise.
    Optionally, inject noise into the EKF prediction to simulate real
    sensor characteristics.

    Deployment (F1Tenth): Real IMU + wheel odometry -> EKF prediction.
    Real camera -> lane detection updates. No PhysX available.
    The topological frame makes this work without GPS.

Dependencies:
    - numpy
    - agent/intersection_graph.py

Author: Aaron Hamil
Date: 03/12/26
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
    _angle_diff,
)

logger = logging.getLogger(__name__)


# Edge Metadata

@dataclass
class EdgeInfo:
    """
    Metadata for one road segment (edge) in the topological graph.

    An edge connects two intersection nodes (or an intersection to a
    dead end). The EKF tracks position along this edge in the Frenet
    frame: s = arc-length, d = lateral offset.
    """
    edge_id: str                    # Same as road_id in intersection graph
    from_node: Optional[str]        # Upstream intersection ID (or None)
    to_node: Optional[str]          # Downstream intersection ID (or None)
    length: float                   # Edge length in meters
    heading: float                  # Direction of travel (radians)
    curvature: float = 0.0          # Average curvature (1/m), 0 = straight
    lane_width: float = 0.3         # Lane width (meters) for F1Tenth scale


@dataclass
class TopologicalState:
    """
    Agent's position in the topological frame.

    This is what the Worker reads instead of global (x, y).
    """
    edge_id: str                    # Which road segment we're on
    s: float                        # Distance along edge (meters)
    d: float                        # Lateral offset from centerline (meters)
    theta_err: float                # Heading error from edge direction (radians)
    speed: float                    # Longitudinal speed (m/s)

    # Metadata (from edge)
    edge_length: float = 0.0       # Total length of current edge
    edge_heading: float = 0.0      # Direction of current edge (radians)

    @property
    def distance_to_next(self) -> float:
        """Distance remaining to downstream intersection."""
        return max(0.0, self.edge_length - self.s)

    @property
    def progress(self) -> float:
        """Fraction of edge traversed [0, 1]."""
        if self.edge_length <= 0:
            return 0.0
        return min(1.0, self.s / self.edge_length)

    @property
    def at_intersection(self) -> bool:
        """True if within arrival threshold of downstream node."""
        return self.distance_to_next < 0.5  # 0.5m threshold


# EKF Configuration

@dataclass
class TopologicalEKFConfig:
    """Configuration for the topological EKF."""

    # Process noise (how much we distrust the prediction model)
    # These are standard deviations per sqrt(second)
    sigma_s: float = 0.05            # Arc-length noise (m/sqrt(s)) — wheel slip
    sigma_d: float = 0.02            # Lateral noise (m/sqrt(s)) — steering play
    sigma_theta: float = 0.03        # Heading noise (rad/sqrt(s)) — gyro drift
    sigma_v: float = 0.1             # Speed noise (m/s/sqrt(s)) — encoder noise

    # Measurement noise
    sigma_lane_d: float = 0.05       # Lane detector lateral offset noise (m)
    sigma_lane_theta: float = 0.1    # Lane detector heading noise (rad)
    sigma_odom_v: float = 0.05       # Wheel odometry speed noise (m/s)

    # Intersection arrival threshold
    arrival_threshold: float = 0.5   # Meters from node center to trigger transition

    # Covariance after intersection reset
    reset_sigma_s: float = 0.1       # Post-reset arc-length uncertainty
    reset_sigma_d: float = 0.05      # Post-reset lateral uncertainty
    reset_sigma_theta: float = 0.05  # Post-reset heading uncertainty

    # Training noise injection (simulates real sensor characteristics)
    inject_noise: bool = False
    noise_scale: float = 1.0         # Multiplier on process noise


# Topological EKF

class TopologicalEKF:
    """
    Extended Kalman Filter in the intersection graph's topological frame.

    State: x = [s, d, theta_err, v]  (4-dimensional)

    The EKF tracks position along a road segment (edge) in Frenet
    coordinates. When the agent reaches the end of an edge (s >= length),
    it transitions to the next edge and the covariance resets — this is
    what makes the topological frame robust to drift.

    Usage:
        ekf = TopologicalEKF(graph, edge_info_map)
        ekf.initialize("road_A", s=0.0)

        # Every step:
        ekf.predict(imu_yaw_rate, odom_speed, dt)
        ekf.update_lane(lateral_offset, confidence)

        # Worker reads:
        state = ekf.state
        if state.at_intersection:
            exits = graph.get_exit_options(...)
    """

    # State indices
    S = 0       # Arc-length along edge
    D = 1       # Lateral offset
    THETA = 2   # Heading error
    V = 3       # Speed
    STATE_DIM = 4

    def __init__(
        self,
        graph: IntersectionGraph,
        edge_info: Dict[str, EdgeInfo],
        config: Optional[TopologicalEKFConfig] = None,
    ):
        """
        Args:
            graph: The intersection topology (for node positions, exits).
            edge_info: Map of edge_id -> EdgeInfo with lengths, headings.
            config: EKF tuning parameters.
        """
        self.graph = graph
        self.edge_info = edge_info
        self.config = config or TopologicalEKFConfig()

        # EKF state
        self._x = np.zeros(self.STATE_DIM, dtype=np.float64)  # [s, d, theta, v]
        self._P = np.eye(self.STATE_DIM, dtype=np.float64)    # Covariance

        # Current edge tracking (discrete, outside EKF state)
        self._current_edge_id: Optional[str] = None
        self._current_edge: Optional[EdgeInfo] = None

        # State for external consumption
        self._topo_state = TopologicalState(
            edge_id="", s=0.0, d=0.0, theta_err=0.0, speed=0.0
        )

        # Transition tracking
        self._at_intersection: bool = False
        self._pending_node_id: Optional[str] = None

    @property
    def state(self) -> TopologicalState:
        """Current topological state for the Worker."""
        return self._topo_state

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance (4x4)."""
        return self._P.copy()

    @property
    def position_uncertainty(self) -> float:
        """Scalar uncertainty in arc-length position (meters, 1-sigma)."""
        return float(np.sqrt(self._P[self.S, self.S]))

    def initialize(
        self,
        edge_id: str,
        s: float = 0.0,
        d: float = 0.0,
        theta_err: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        """
        Initialize EKF on a specific edge.

        Called at episode start. During training, use PhysX ground truth
        to determine the starting edge and position. During deployment,
        the robot starts at a known location.

        Args:
            edge_id: Starting road segment.
            s: Initial arc-length along edge.
            d: Initial lateral offset.
            theta_err: Initial heading error.
            speed: Initial speed.
        """
        if edge_id not in self.edge_info:
            raise ValueError(
                f"Unknown edge '{edge_id}'. "
                f"Available: {list(self.edge_info.keys())}"
            )

        self._current_edge_id = edge_id
        self._current_edge = self.edge_info[edge_id]

        self._x = np.array([s, d, theta_err, speed], dtype=np.float64)

        # Initial covariance — tight if starting position is known
        self._P = np.diag([0.1**2, 0.05**2, 0.05**2, 0.1**2])

        self._at_intersection = False
        self._pending_node_id = None
        self._update_topo_state()

    def initialize_from_global(
        self,
        x: float, y: float, heading: float, speed: float = 0.0
    ) -> None:
        """
        Initialize from global coordinates by finding nearest edge.

        Used during training when PhysX provides global position.
        Projects the global position onto the nearest graph edge.

        Args:
            x, y: Global position.
            heading: Global heading (radians).
            speed: Initial speed.
        """
        best_edge_id = None
        best_s = 0.0
        best_d = 0.0
        best_theta_err = 0.0
        best_dist = float("inf")

        for eid, edge in self.edge_info.items():
            # Project point onto edge line
            s_proj, d_proj = self._project_onto_edge(x, y, edge)
            dist = abs(d_proj)

            if dist < best_dist and 0 <= s_proj <= edge.length:
                best_edge_id = eid
                best_s = s_proj
                best_d = d_proj
                best_theta_err = _angle_diff(heading, edge.heading)
                best_dist = dist

        if best_edge_id is None:
            # Fallback: pick first edge, zero position
            best_edge_id = next(iter(self.edge_info))
            logger.warning(
                f"Could not project ({x:.1f}, {y:.1f}) onto any edge. "
                f"Defaulting to {best_edge_id}"
            )

        self.initialize(
            edge_id=best_edge_id,
            s=best_s,
            d=best_d,
            theta_err=best_theta_err,
            speed=speed,
        )

    # Prediction Step

    def predict(
        self, imu_yaw_rate: float, odom_speed: float, dt: float
    ) -> None:
        """
        EKF prediction using IMU yaw rate and wheel odometry speed.

        Propagates the state forward using the kinematic model in
        Frenet coordinates along the current edge.

        Args:
            imu_yaw_rate: Angular velocity from IMU (rad/s).
            odom_speed: Speed from wheel odometry (m/s).
            dt: Time step (seconds).
        """
        if self._current_edge is None:
            return

        s, d, theta, v = self._x
        kappa = self._current_edge.curvature

        # State prediction (Euler integration)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        s_new = s + v * cos_theta * dt
        d_new = d + v * sin_theta * dt
        theta_new = theta + (imu_yaw_rate - kappa * v) * dt
        v_new = odom_speed  # Direct measurement, not integrated

        # Wrap heading error to [-pi, pi]
        theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi

        self._x = np.array([s_new, d_new, theta_new, v_new])

        # Jacobian of state transition (F matrix)
        F = np.eye(self.STATE_DIM)
        F[self.S, self.THETA] = -v * sin_theta * dt
        F[self.S, self.V] = cos_theta * dt
        F[self.D, self.THETA] = v * cos_theta * dt
        F[self.D, self.V] = sin_theta * dt
        F[self.THETA, self.V] = -kappa * dt

        # Process noise
        cfg = self.config
        noise_scale = cfg.noise_scale if cfg.inject_noise else 1.0
        Q = np.diag([
            (cfg.sigma_s * noise_scale) ** 2 * dt,
            (cfg.sigma_d * noise_scale) ** 2 * dt,
            (cfg.sigma_theta * noise_scale) ** 2 * dt,
            (cfg.sigma_v * noise_scale) ** 2 * dt,
        ])

        # Covariance prediction
        self._P = F @ self._P @ F.T + Q

        # Check for edge transition
        self._check_edge_transition()

        self._update_topo_state()

    # Measurement Updates

    def update_lane(
        self, lateral_offset: float, confidence: float = 1.0
    ) -> None:
        """
        Update EKF with lane detection measurement.

        The lane detector provides lateral offset from lane center,
        which directly measures state d. Higher confidence -> smaller
        measurement noise -> stronger correction.

        Args:
            lateral_offset: Lateral offset from lane_detector.py.
                            Sign convention: negative = left, positive = right.
            confidence: Detection confidence [0, 1]. Low confidence
                        inflates measurement noise.
        """
        if confidence < 0.1:
            return  # Too uncertain, skip update

        # Measurement: z = lateral_offset, measures state d
        z = np.array([lateral_offset])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.D] = 1.0

        # Measurement noise (inflated by low confidence)
        r = (self.config.sigma_lane_d / max(confidence, 0.1)) ** 2
        R = np.array([[r]])

        self._ekf_update(z, H, R)

    def update_speed(self, measured_speed: float) -> None:
        """
        Update EKF with wheel odometry speed measurement.

        Provides a direct measurement of state v.

        Args:
            measured_speed: Speed from wheel encoders (m/s).
        """
        z = np.array([measured_speed])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.V] = 1.0

        R = np.array([[self.config.sigma_odom_v ** 2]])

        self._ekf_update(z, H, R)

    def update_intersection_visual(self, detected: bool) -> None:
        """
        Update EKF when camera detects intersection features.

        If the camera sees intersection characteristics (road widening,
        lane marking changes, etc.) and the EKF thinks we're near the
        end of the edge, this provides a strong correction to s.

        If the EKF thinks we're mid-edge but camera detects intersection
        features, it's either a false positive or the edge length is
        wrong — we log but don't correct aggressively.

        Args:
            detected: True if visual intersection features detected.
        """
        if not detected or self._current_edge is None:
            return

        edge_length = self._current_edge.length
        s = self._x[self.S]
        remaining = edge_length - s

        # Only apply correction if we're in the approach zone (last 30%)
        if remaining < edge_length * 0.3:
            # Soft correction: pull s toward edge_length
            z = np.array([edge_length])
            H = np.zeros((1, self.STATE_DIM))
            H[0, self.S] = 1.0

            # Noise depends on how close we already think we are
            r = max(0.1, remaining) ** 2
            R = np.array([[r]])

            self._ekf_update(z, H, R)
        else:
            logger.debug(
                f"Visual intersection detected but s={s:.1f} is far "
                f"from edge end ({edge_length:.1f}m). Ignoring."
            )

    # Edge Transition Logic

    def transition_to_edge(self, new_edge_id: str) -> None:
        """
        Manually transition to a new edge (called after Worker decides turn).

        When the Worker picks a turn at an intersection, the EKF
        transitions to the corresponding outgoing edge. This resets
        the arc-length to 0 and snaps the covariance.

        Args:
            new_edge_id: The outgoing edge to transition to.
        """
        if new_edge_id not in self.edge_info:
            logger.error(f"Cannot transition to unknown edge '{new_edge_id}'")
            return

        old_edge = self._current_edge_id
        self._current_edge_id = new_edge_id
        self._current_edge = self.edge_info[new_edge_id]

        # Reset state: at start of new edge
        speed = self._x[self.V]
        self._x = np.array([0.0, 0.0, 0.0, speed])

        # Reset covariance — tight, because we know we're at the node
        cfg = self.config
        self._P = np.diag([
            cfg.reset_sigma_s ** 2,
            cfg.reset_sigma_d ** 2,
            cfg.reset_sigma_theta ** 2,
            cfg.sigma_v ** 2,
        ])

        self._at_intersection = False
        self._pending_node_id = None
        self._update_topo_state()

        logger.debug(
            f"EKF transition: {old_edge} -> {new_edge_id} "
            f"(covariance reset, s=0)"
        )

    def _check_edge_transition(self) -> None:
        """
        Check if the agent has reached the downstream intersection.

        If s >= (edge_length - arrival_threshold), flag that we're at
        an intersection. The actual transition to a new edge happens
        when the Worker picks a turn and calls transition_to_edge().
        """
        if self._current_edge is None:
            return

        s = self._x[self.S]
        remaining = self._current_edge.length - s

        if remaining <= self.config.arrival_threshold:
            if not self._at_intersection:
                self._at_intersection = True
                self._pending_node_id = self._current_edge.to_node
                logger.debug(
                    f"EKF: approaching intersection "
                    f"{self._pending_node_id} on {self._current_edge_id} "
                    f"(remaining={remaining:.2f}m, sigma_s={self.position_uncertainty:.3f}m)"
                )

    # Standard EKF Math

    def _ekf_update(
        self, z: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> None:
        """Standard EKF measurement update."""
        y = z - H @ self._x                          # Innovation
        S = H @ self._P @ H.T + R                    # Innovation covariance
        K = self._P @ H.T @ np.linalg.inv(S)         # Kalman gain
        self._x = self._x + K @ y                    # State update
        I = np.eye(self.STATE_DIM)
        self._P = (I - K @ H) @ self._P              # Covariance update

        # Wrap heading error
        self._x[self.THETA] = (
            (self._x[self.THETA] + math.pi) % (2 * math.pi) - math.pi
        )

    # Helpers

    def _project_onto_edge(
        self, x: float, y: float, edge: EdgeInfo
    ) -> Tuple[float, float]:
        """
        Project a global point onto an edge's Frenet frame.

        Returns (s, d) where s is distance along the edge from the
        upstream node, and d is lateral offset (+ = right of heading).
        """
        # Get upstream node position
        from_node = self.graph.get_intersection(edge.from_node) if edge.from_node else None
        if from_node is None:
            return 0.0, 0.0

        ox, oy = from_node.position

        # Edge direction vector
        dx = math.cos(edge.heading)
        dy = math.sin(edge.heading)

        # Vector from upstream node to point
        px = x - ox
        py = y - oy

        # Project: s = dot product, d = cross product
        s = px * dx + py * dy
        d = px * (-dy) + py * dx  # Perpendicular (right-hand rule)

        return float(s), float(d)

    def _update_topo_state(self) -> None:
        """Sync the public TopologicalState from internal EKF state."""
        self._topo_state = TopologicalState(
            edge_id=self._current_edge_id or "",
            s=float(self._x[self.S]),
            d=float(self._x[self.D]),
            theta_err=float(self._x[self.THETA]),
            speed=float(self._x[self.V]),
            edge_length=self._current_edge.length if self._current_edge else 0.0,
            edge_heading=self._current_edge.heading if self._current_edge else 0.0,
        )

    def get_global_position_estimate(self) -> Tuple[float, float, float]:
        """
        Convert topological state back to approximate global (x, y, heading).

        This is useful for visualization and for the transition period
        where some code still expects global coordinates. NOT for
        localization — use the topological state directly.

        Returns:
            (x, y, heading) in global frame. Accuracy degrades with
            distance from last intersection reset.
        """
        if self._current_edge is None:
            return 0.0, 0.0, 0.0

        edge = self._current_edge
        from_node = self.graph.get_intersection(edge.from_node) if edge.from_node else None

        if from_node is None:
            return 0.0, 0.0, edge.heading

        ox, oy = from_node.position
        s = self._x[self.S]
        d = self._x[self.D]
        theta_err = self._x[self.THETA]

        # Position along edge + lateral offset
        dx = math.cos(edge.heading)
        dy = math.sin(edge.heading)
        # Perpendicular (right of heading)
        nx = -dy
        ny = dx

        x = ox + s * dx + d * nx
        y = oy + s * dy + d * ny
        heading = edge.heading + theta_err

        return float(x), float(y), float(heading)


# Edge Info Builder

def build_edge_info_from_graph(
    graph: IntersectionGraph,
    default_length: float = 20.0,
    default_lane_width: float = 0.3,
) -> Dict[str, EdgeInfo]:
    """
    Build EdgeInfo map from an IntersectionGraph.

    Computes edge lengths from intersection positions where possible.
    Falls back to default_length for dead-end roads or disconnected
    segments.

    This is a convenience function for bootstrapping. For production,
    edge lengths should be measured in the USD scene and stored in
    the JSON directly.

    Args:
        graph: Loaded IntersectionGraph.
        default_length: Fallback edge length in meters.
        default_lane_width: Lane width in meters (F1Tenth scale).

    Returns:
        Dict mapping edge_id -> EdgeInfo.
    """
    edges: Dict[str, EdgeInfo] = {}

    # First pass: collect which intersections each road connects
    # A road_id appears as an approach at one intersection (inbound)
    # and as an exit at potentially another intersection (outbound from there)
    road_to_node: Dict[str, str] = {}   # road_id -> intersection it approaches
    road_from_node: Dict[str, str] = {} # road_id -> intersection that exits onto it

    for node_id, node in graph.all_intersections.items():
        # This road approaches (leads TO) this intersection
        for road_id in node.approaches:
            road_to_node[road_id] = node_id

        # Check exits: if an exit road leads_to another intersection,
        # this intersection is the FROM node for that exit road
        for approach in node.approaches.values():
            for exit_opt in approach.exits.values():
                road_from_node[exit_opt.exit_road_id] = node_id

    # Second pass: build EdgeInfo for each road
    for node_id, node in graph.all_intersections.items():
        for road_id, approach in node.approaches.items():
            if road_id in edges:
                continue

            # This road approaches node_id (to_node = node_id)
            to_node_id = node_id
            from_node_id = road_from_node.get(road_id)

            # Compute length from node positions
            length = default_length
            if from_node_id is not None and from_node_id != to_node_id:
                other = graph.get_intersection(from_node_id)
                if other is not None:
                    dx = node.position[0] - other.position[0]
                    dy = node.position[1] - other.position[1]
                    computed = math.sqrt(dx * dx + dy * dy)
                    if computed > 1.0:  # Sanity check: ignore degenerate distances
                        length = computed

            edges[road_id] = EdgeInfo(
                edge_id=road_id,
                from_node=from_node_id,
                to_node=to_node_id,
                length=length,
                heading=approach.heading_rad,
                curvature=0.0,
                lane_width=default_lane_width,
            )

    return edges
