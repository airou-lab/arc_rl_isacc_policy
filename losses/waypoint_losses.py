"""
Waypoint Auxiliary Loss for Self-Supervised Planning

Provides a learning signal for the planning head in HierarchicalPathPlanningPolicy without requiring human-labeled waypoints.
Instead, the vehicle's actual trajectory (recorded by WaypointTrackingWrapper) serves as the supervision source.

Three loss components:
    1. Imitation Loss: on SAFE trajectory segments, predicted waypoints should match where the vehicle actually drove. This teaches
       the planner to produce realistic, driveable paths.

    2. Repulsion Loss: on UNSAFE trajectory segments (pre-crash), predicted waypoints should be pushed AWAY from where the vehicle went.
       This teaches the planner to steer away from crash-inducing paths.

    3. Goal-Directed Loss: Predicted waypoints should generally point the goal direction (derived from turn_bias). This provides a weak
       navigational signaml that helps early training before the planner has learned from enough trajectory data.

Integration:
    This loss is computed OUTSIDE the standard PPO loss and added as an auxiliary term. The training script (train_policy_ros2.py) calls
    compute_waypoint_loss() after each PPO update step and adds it to the total loss with weight policy.waypoint_loss_weight.

Dependencies:
    - wrappers/waypoint_tracking_wrapper.py (TrajectoryStore for trajectory data)
    - policies/hierarchical_policy.py (provides predicted waypoints via get_waypoints())

Used by:
    - train_policy_ros2.py (custom training loop with auxiliary loss)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

from wrappers.waypoint_tracking_wrapper import get_trajectory_store

class WaypointLoss:
    """
    Computes auxiliary waypoint loss from trajectory data.

    The loss operates in the vehicle's local frame at the time each waypoint was predicted. Trajectory positions are transformed from
    world frame to vehicle frame at prediciton time using the recorded yaw angle.
    """
    def __init__(
        self,
        num_waypoints: int = 5,
        waypoint_spacing: float = 0.5,
        imitation_weight: float = 1.0,
        repulsion_weight: float = 2.0,
        goal_weight: float = 0.3,
        repulsion_margin: float = 1.0,
    ):
        """
        Args:
            num_waypoints: Number of waypoints the policy predicts.
            waypoint_spacing: Distance between consecutive waypoints (meters).
            imitation_weight: Weight for imitation loss on safe trajectories.
            repulsion_weight: Weight for repulsion loss on unsafe trajectories.
            goal_weight: Weight for goal-directed loss.
            repulsion_margin: Minimum distance (meters) waypoints should be from unsafe trajectory points. Loss is zero beyond this margin.
        """

        self.num_waypoints = num_waypoints
        self.waypoint_spacing = waypoint_spacing
        self.imitation_weight = imitation_weight
        self.repulsion_weight = repulsion_weight
        self.goal_weight = goal_weight
        self.repulsion_margin = repulsion_margin

        self._store = get_trajectory_store()

    def compute(self,
                predictated_waypoints: torch.Tensor,
                obs_vec: torch.Tensor,
                env_id: int = 0,
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined waypoint auxiliary loss.

        Args:
            predicted_waypoints: (batch, num_waypoints, 2) from policy.get_waypoints().
                In vehicle frame [X=lateral, Y=forward].
            obs_vec: (batch, 12) telemetry vector for goal-directed component.
            env_id: Environment ID to look up trajectory in store.

        Returns:
            loss: predicted_waypoints.device
            batch_size = predicted_waypoints.shape[0]
        """

        device = predicted_waypoints.device
        batch_size = predicted_waypoints.shape[0]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        info = {
            "waypoint_loss/imitation": 0.0,
            "waypoint_loss/repulsion": 0.0,
            "waypoint_loss/goal": 0.0,
            "waypoint_loss/total": 0.0,
        }

        # Get trajectory from store
        trajectory = self._store.get_trajectory(env_id)
        safety_mask = self._store.get_safety_mask(env_id)

        has_trajectory = trajectory is not None and safety_mask is not None

        # 1. Imitation Loss (safe segments)
        imitation_loss = torch.tensor(0.0, device=device)
        if has_trajectory:
            imitation_loss = self._imitation_loss(predicted_waypoints, trajectory, safety_mask, device)

        # 2. Repulsion Loss (unsafe segments)
        repulsion_loss = torch.tensor(0.0, device=device)
        if has_trajectory:
            repulsion_loss = self._repulsion_loss(predicted_waypoints, trajectory, safety_mask, device)

        # 3. Goal-Directed Loss
        goal_loss = self._goal_directed_loss(predicted_waypoints, obs_vec)

        # Combined Loss
        total_loss = (
            self.imitation_weight * imitation_loss
            + self.repulsion_weight * repulsion_loss
            + self.goal_weight * goal_loss
        )

        info["waypoint_loss/imitation"] = imitation_loss.item()
        info["waypoint_loss/repulsion"] = repulsion_loss.item()
        info["waypoint_loss/goal"] = goal_loss.item()
        info["waypoint_loss/total"] = total_loss.item()

        return total_loss, info

    def _imitation_loss(
        self,
        predicted: torch.Tensor,
        trajectory: Dict,
        safety_mask: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        MSE loss between predicted waypoints and actual safe trajectory.

        For each predicted waypoint at distance d ahead, find the trajectory point that was approximately d meters ahead of the vehicle's position
        at the end of the episode. Transform to vehicle-local frame.
        """
        positions = trajectory["positions"] # (N, 3)
        yaws = trajectory["yaws"] # (N,)

        # Only use safe trajectory segments
        safe_indices = np.where(safety_mask > 0.5)[0]
        if len(safe_indices) < self.num_waypoints + 1:
            return torch.tensor(0.0, device=device)

        # Use the last safe position as reference point
        ref_idx = safe_indices[-1]
        ref_pos = positions[ref_idx]
        ref_yaw = yaws[ref_idx]

        # Find trajectory points at each waypoint distance
        target_waypoints = []
        for wp_idx in range(self.num_waypoints):
            target_dist = (wp_idx + 1) * self.waypoint_spacing

            # Find the trajectory point closest to target_dist ahead
            best_idx = self._find_point_at_distance(positions, safe_indices, ref_idx, target_dist)

            if best_idx is not None:
                # Transform to vehicle-local frame
                world_pos = positions[best_idx]
                local = self._world_to_local(world_pos, ref_pos, ref_yaw)
                target_waypoints.append(local)
            else:
                # Extrapolate straight ahead if no trajectory point found
                target_waypoints.append(np.array([0.0, target_dist], dtype=np.float32))

        targets = torch.tensor(np.array(target_waypoints), dtype=torch.float32, device=device) # (num_waypoints, 2)

        # Expand targets to match batch dimension
        targets = targets.unsqueeze(0).expand(predicted.shape[0], -1, -1)

        return F.mse_loss(predicted, targets)

    def _repulsion_loss(self, predicted: torch.Tensor, trajectory: Dict, safety_mask: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Hinge loss pushing predicted waypoints away from unsafe trajectory.

        For unsafe trajectory points, compute distance to each predicted waypoint. If distance < repulsion_margin, apply penalty.
        """
        positions = trajectory["positions"] # (N, 3)
        yaws = trajectory["yaws"] # (N,)

        # Only use unsafe trajectory segments
        unsafe_indices = np.where(safety_mask < 0.5)[0]
        if len(unsafe_indices) < 2:
            return torch.tensor(0.0, device=device)

        # use the last safe position before the crash as reference
        safe_indices = np.where(safety_mask > 0.5)[0]
        if len(safe_indices) == 0:
            return torch.tensor(0.0, device=device)

        ref_idx = safe_indices[-1]
        ref_pos = positions[ref_idx]
        ref_yaw = yaws[ref_idx]

        # Transform unsafe positions to vehicle-local frame
        unsafe_local = []
        for idx in unsafe_indices:
            local = safe._world_to_local(positions[idx], ref_pos, ref_yaw)
            unsafe_local.append(local)

        unsafe_pts = torch.tensor(np.array(unsafe_local), dtype=torch.float32, device=device) # (U, 2)

        # Compute pairwise distances between predicted waypoints and unsafe points.
        # predicted: (B, W, 2), unsafe_pts: (U, 2)
        # Expand for broadcasting
        pred_expanded = predicted.unsqueeze(2) # (B, W, 1, 2)
        unsafe_expanded = unsafe_pts.unsqueeze(0).unsqueeze(0) # (1, 1, U, 2)

        distances = torch.norm(pred_expanded - unsafe_expanded, dim=-1) # (B, W, U)

        # Hinge loss: penalty when distance < margin
        violations = torch.clamp(self.repulsion_margin - distance, min=0.0)
        repulsion = violations.mean()

        return repulsion

    def _goal_directed_loss(self, predicted: torch.Tensor, obs_vec: torch.Tensor) -> torch.Tensor:
        """
        Soft loss encouraging waypoints to curve in the turn_bias direction.

        If turn_bias > 0 (turn right), waypoints should have positive X.
        If turn_bias < 0 (turn left), waypoints should have negative X.
        If turn_bias = 0 (straight), waypoints should have X near 0.

        This is a weak signal that helps training bootstrap before enough crash/safe trajectory is available.
        """
        turn_bias = obs_vec[:, 0] # (B,) turn command

        # Average lateral position of predicted waypoints
        avg_lateral = predicted[:, :, 0].mean(dim=1) # (B,)

        # Loss: penalize when lateral direction disagrees with turn_bias
        # Using negative cosine similarity style:
        # If both point in same direction -> low loss
        # If they disagree -> high loss
        alignment = avg_lateral * turn_bias # positive when aligned
        goal_loss = F.relu(-alignment).mean() # penalty when misaligned

        return goal_loss

    @staticmethod
    def _world_to_local(world_pos: np.ndarray, ref_pos:np.ndarray, ref_yaw: float,) -> np.ndarray:
        """
        Transform a world position to vehicle-local frame.

        Args:
            world_pos: (3,) world position [x, y, z].
            ref_pos: (3,) reference (vehicle) world position.
            ref_yaw: Vehicle heading angle (radians)

        Returns:
            local: (2,) local position [x_lateral, y_lateral].
        """
        dx = world_pos[0] - ref_pos[0]
        dz = world_pos[2] - ref_pos[2]

        cos_yaw = np.cos(ref_yaw)
        sin_yaw = np.sin(ref_yaw)

        # Rotate into vehicle frame
        local_x = dx * cos_yaw - dz * sin_yaw # Lateral
        local_z = dx * sin_yaw + dz * cos_yaw # Forward

        return np.array([local_x, local_y], dtype=np.float32)

    @staticmethod
    def _find_point_at_distance(positions: np.ndarray, valid_indices: np.ndarray, ref_idx: int, target_dist: float,) -> Optional[int]:
        """
        Find the trajectory index approximately target_dist ahead of ref_idx.

        Walks forward through the trajectory, accumulating distance, and returns the index closest to the target distance.
        """
        # Only consider indices after ref_idx
        future_indices = valid_indices[valid_indices > ref_idx]
        if len(future_indices) == 0:
            return None

        accumulated = 0.0
        prev_pos = positions[ref_idx]

        best_idx = None
        best_diff = float("inf")

        for idx in future_indices:
            curr_pos = positions[idx]
            step_dist = np.linalg.norm(curr_pos - prev_pos)
            accumulated += step_dist
            prev_pos = curr_pos

            diff = abs(accumulated - target_dist)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

            # Early exit if we've passed the target
            if accumulated > target_dist * 1.5:
                break

        return best_idx
