"""
Geometry Calibrator — Auto-Discover Road Geometry from PhysX
=============================================================

Runs during environment initialization in Isaac Sim. Uses PhysX
ground truth position to automatically measure:
    - Intersection center positions (x, y)
    - Edge lengths (distance from dead-end to intersection)
    - Edge headings (refined from approximate JSON values)

Two modes:
    1. LIVE CALIBRATION (first run on a new map):
       A scripted agent drives each road segment while recording
       PhysX position. Edge length = distance traveled. Intersection
       position = where the agent enters the trigger zone.
       Results are saved to config/geometry_cache.json.

    2. CACHED (subsequent runs):
       Load previously calibrated geometry from the cache file.
       Skip calibration entirely. Only re-calibrate if the cache
       is missing or the user requests it.

Why not just read USD prim positions?
    - USD prims don't have "road length" attributes; we'd need to
      trace spline curves or measure between prims
    - PhysX measurement captures the ACTUAL drivable distance
      including any road curvature, not just straight-line distance
    - This approach works identically in Isaac Sim, CARLA, or any
      simulator that provides ground truth position

Live Calibration Protocol:
    1. For each road in the topology:
       a. Teleport agent to a known start position (far end of road)
       b. Drive straight toward the intersection at constant speed
       c. Record (x, y) at each step
       d. When agent enters intersection trigger zone, record:
          - Edge length = cumulative distance driven
          - Intersection position = current (x, y) [first road only,
            subsequent roads verify against the established position]
          - Edge heading = atan2 of the travel direction vector

    For a single intersection with 4 roads, this takes ~4 drives
    of a few seconds each. Total calibration time: ~30 seconds.

Dependencies:
    - agent/intersection_graph.py
    - numpy

Author: Aaron Hamil
Date: 03/12/26
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from agent.intersection_graph import (
    IntersectionGraph,
    EdgeGeometry,
    TurnCommand,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for geometry calibration."""
    cache_path: str = "config/geometry_cache.json"
    trigger_radius: float = 3.0         # meters — how close to intersection center
    min_edge_length: float = 2.0        # meters — reject implausibly short edges
    max_edge_length: float = 200.0      # meters — reject implausibly long edges
    force_recalibrate: bool = False      # If True, ignore cache


class GeometryCalibrator:
    """
    Auto-discovers road geometry from PhysX ground truth.

    Usage (in env init):
        calibrator = GeometryCalibrator(graph, config)

        # Try loading cached geometry first
        if calibrator.try_load_cache():
            print("Loaded cached geometry")
        else:
            # No cache — need live calibration
            # This requires driving each road, so it happens in a
            # calibration episode before training starts
            calibrator.calibrate_from_position_log(position_logs)
            calibrator.save_cache()
    """

    def __init__(
        self,
        graph: IntersectionGraph,
        config: Optional[CalibrationConfig] = None,
    ):
        self.graph = graph
        self.config = config or CalibrationConfig()

    def try_load_cache(self) -> bool:
        """
        Try to load geometry from cache file.

        Returns True if cache exists and was loaded successfully.
        Returns False if cache is missing or force_recalibrate is set.
        """
        if self.config.force_recalibrate:
            return False

        cache_path = Path(self.config.cache_path)
        if not cache_path.exists():
            return False

        try:
            self.graph.load_geometry(cache_path)
            if self.graph.is_calibrated:
                logger.info(f"Loaded cached geometry from {cache_path}")
                return True
            else:
                logger.warning("Cache loaded but graph not fully calibrated")
                return False
        except Exception as e:
            logger.warning(f"Failed to load geometry cache: {e}")
            return False

    def save_cache(self) -> None:
        """Save current geometry to cache file."""
        cache_path = Path(self.config.cache_path)
        self.graph.save_geometry(cache_path)
        logger.info(f"Saved geometry cache to {cache_path}")

    def calibrate_from_drives(
        self,
        drive_logs: Dict[str, List[Tuple[float, float]]],
    ) -> None:
        """
        Calibrate geometry from recorded drive data.

        Each drive_log is a list of (x, y) positions recorded while
        driving one road segment from its dead end toward the
        intersection.

        Args:
            drive_logs: Dict mapping road_id -> list of (x, y) positions.
                        Positions should be in chronological order
                        (start of road -> intersection).
        """
        edge_geometries: Dict[str, EdgeGeometry] = {}
        intersection_positions: Dict[str, List[Tuple[float, float]]] = {}

        for road_id, positions in drive_logs.items():
            if len(positions) < 2:
                logger.warning(f"Road {road_id}: too few positions ({len(positions)})")
                continue

            # Compute cumulative distance
            total_dist = 0.0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i - 1][0]
                dy = positions[i][1] - positions[i - 1][1]
                total_dist += math.sqrt(dx * dx + dy * dy)

            # Heading: average direction of travel
            start = np.array(positions[0])
            end = np.array(positions[-1])
            direction = end - start
            heading = float(math.atan2(direction[1], direction[0]))

            # Validate length
            if total_dist < self.config.min_edge_length:
                logger.warning(
                    f"Road {road_id}: length {total_dist:.1f}m below minimum "
                    f"({self.config.min_edge_length}m), skipping"
                )
                continue
            if total_dist > self.config.max_edge_length:
                logger.warning(
                    f"Road {road_id}: length {total_dist:.1f}m above maximum "
                    f"({self.config.max_edge_length}m), skipping"
                )
                continue

            # Which intersection does this road approach?
            to_node = self.graph.get_node_for_road(road_id)

            # End position is approximately the intersection center
            end_pos = tuple(positions[-1])
            if to_node:
                if to_node not in intersection_positions:
                    intersection_positions[to_node] = []
                intersection_positions[to_node].append(end_pos)

            edge_geometries[road_id] = EdgeGeometry(
                edge_id=road_id,
                length=total_dist,
                heading=heading,
                from_node=None,     # Dead end (star topology)
                to_node=to_node,
                start_position=tuple(positions[0]),
                end_position=end_pos,
            )

            logger.info(
                f"Road {road_id}: length={total_dist:.1f}m, "
                f"heading={math.degrees(heading):.0f}deg"
            )

        # Compute intersection positions as centroid of all end positions
        for node_id, end_positions in intersection_positions.items():
            xs = [p[0] for p in end_positions]
            ys = [p[1] for p in end_positions]
            center = (float(np.mean(xs)), float(np.mean(ys)))
            self.graph.set_intersection_position(
                node_id, center, self.config.trigger_radius
            )
            logger.info(
                f"Intersection {node_id}: position=({center[0]:.1f}, {center[1]:.1f}) "
                f"from {len(end_positions)} road endpoints"
            )

        self.graph.set_geometry(edge_geometries)

    def calibrate_from_position_fn(
        self,
        get_position: Callable[[], Tuple[float, float]],
        get_heading: Callable[[], float],
        teleport: Callable[[float, float, float], None],
        drive_straight: Callable[[float, float], None],
        road_starts: Dict[str, Tuple[float, float, float]],
        drive_distance: float = 50.0,
        step_distance: float = 0.1,
    ) -> None:
        """
        Live calibration using simulator callbacks.

        This is the fully automated path: provide functions to read
        position, teleport the agent, and drive forward. The calibrator
        drives each road and measures everything.

        Args:
            get_position: Returns current (x, y).
            get_heading: Returns current heading (radians).
            teleport: Moves agent to (x, y, heading).
            drive_straight: Drives forward for (distance, speed).
            road_starts: Dict of road_id -> (x, y, heading) spawn points.
            drive_distance: Max distance to drive per road.
            step_distance: Distance between position recordings.
        """
        drive_logs: Dict[str, List[Tuple[float, float]]] = {}

        for road_id, (sx, sy, sh) in road_starts.items():
            logger.info(f"Calibrating road {road_id}...")
            teleport(sx, sy, sh)

            positions = [get_position()]
            remaining = drive_distance

            while remaining > 0:
                step = min(step_distance, remaining)
                drive_straight(step, 1.5)
                pos = get_position()
                positions.append(pos)
                remaining -= step

                # Early exit: reached an intersection
                node = self.graph.nearest_intersection(pos[0], pos[1])
                if node is not None:
                    logger.debug(
                        f"  Reached intersection {node.node_id} at "
                        f"({pos[0]:.1f}, {pos[1]:.1f})"
                    )
                    break

            drive_logs[road_id] = positions

        self.calibrate_from_drives(drive_logs)

    def calibrate_from_episode_observations(
        self,
        road_id: str,
        positions: List[Tuple[float, float]],
    ) -> None:
        """
        Incremental calibration: add one road from an observed episode.

        Call this during normal training episodes. As the agent drives
        each road for the first time, record its positions and feed
        them here. After all roads have been observed at least once,
        the graph is fully calibrated.

        This is the zero-cost calibration path — no special calibration
        episode needed. Geometry is discovered as a side effect of
        normal training.

        Args:
            road_id: Which road was driven.
            positions: List of (x, y) from episode (start -> intersection).
        """
        existing = self.graph.get_edge_geometry(road_id)
        if existing is not None:
            return  # Already calibrated

        self.calibrate_from_drives({road_id: positions})

        if self.graph.is_calibrated:
            logger.info("All roads calibrated — saving cache")
            self.save_cache()
