"""
Lane Detection for Visual Reward Computation
===
Simple vision-based lane detection to compute rewards WITHOUT giving priviliged information to the agent.

This module provides lane staying rewards based purely on visual cues

Two detection modes:
1. Semantic Segementation (if Isaac Sim provides lane labels)
2. Classical CV (edge detection + clustering for any camera)

The agent never sees the output of this detector, it's only for reward
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LaneDetectionResult:
    """Results from lane detection."""

    # Primary metrics
    in_lane: bool            # Is vehicle in valid lane?
    lateral_offset: float    # Offset from lane center (-1=left, +1=right)
    confidence: float        # Detection confidence [0, 1]

    # Optional detailed results
    left_edge: Optional[np.ndarray] = None     # Left lane boundary points
    right_edge: Optional[np.ndarray] = None    # Right lane boundary points
    lane_center: Optional[float] = None        # Detected lane center x-coordinate
    lane_width: Optional[float] = None         # Detected lane width in pixels


class SimpleLaneDetector:
    """
    Simple lane detector using classical CV methods.

    Works on any RGB camera image without semantic labels.
    Uses edge detection + vertical clustering to find lane boundaries.

    This is designed for structured environments like:
    - Roads with visable lane markings
    - Tracks with clear boundaries
    - Hallways with distinct edges
    """

    def __init__(
        self,
        img_width: int = 160,
        img_height: int = 90,
        roi_top_ratio: float = 0.4,       # Focus on the lower 60% of image
        edge_threshold: int = 50,         # Canny edge threshold
        min_lane_width_px: int = 25,      # Minimum lane width
        max_lane_width_px: int = 125,     # Maximum lane width
        center_tolerance_px: int = 19,    # Tolerance for "in lane"
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.roi_top_ratio = roi_top_ratio
        self.edge_threshold = edge_threshold
        self.min_lane_width = min_lane_width_px
        self.max_lane_width = max_lane_width_px
        self.center_tolerance = center_tolerance_px

        # Define region of interest (lower portion of the image)
        self.roi_top = int(img_height * roi_top_ratio)

    def detect(self, image: np.ndarray) -> LaneDetectionResult:
        """
        Detect lane boundaries from RGB image.

        Args:
            image: (H, W, 3) RGB image, 128x128 for case but any scale fine

        Returns:
            LaneDetectionResult with lane staying metrics
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply ROI mask (focus on road ahead, ignore sky)
        roi = gray.copy()
        roi[:self.roi_top, :] = 0

        # Edge detection
        edges = cv2.Canny(roi, self.edge_threshold, self.edge_threshold * 2)

        # Find verticle edges (lane boundaries are typically vertical in view)
        left_edge, right_edge = self._find_lane_edges(edges)

        if left_edge is None or right_edge is None:
            # Failed to detect lanes
            return LaneDetectionResult(
                in_lane=False,
                lateral_offset=0.0,
                confidence=0.0
            )

        # Compue lane center and width
        lane_center = (left_edge + right_edge) / 2.0
        lane_width = right_edge - left_edge

        # Image center (where the vehicle should be)
        image_center = self.img_width / 2

        # Compute lateral offset normalized to [-1, 1]
        # Negative = left of center, Positive = right of center
        max_deviation = lane_width / 2.0
        lateral_offset = (image_center - lane_center) / max_deviation
        lateral_offset = np.clip(lateral_offset, -1.0, 1.0)

        # Check if vehicle is within lane boundaries
        deviation_px = abs(image_center - lane_center)
        in_lane = deviation_px < self.center_tolerance

        # Confidence based on lane width validity
        width_valid = self.min_lane_width <= lane_width <= self.max_lane_width
        confidence = 1.0 if width_valid else 0.5

        return LaneDetectionResult(
            in_lane=in_lane,
            lateral_offset=lateral_offset,
            confidence=confidence,
            left_edge=np.array([left_edge]),
            right_edge=np.array([right_edge]),
            lane_center=lane_center,
            lane_width=lane_width
        )

    def _find_lane_edges(self, edges: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Find left and right lane boundaries from edge image.

        Strategy: Look for strong vertical edge clusters in lower half of image.
        Left boundary = strongest edge cluster in left half
        Right boundary = strongest edge cluster in right half

        Args:
            edges: Binary edge map

        Returns:
            (left_edge_x, right_edge_x) or (None, None) if not found
        """
        # Sum edges vertically to get horizontal edge strength profile
        vertical_projection = np.sum(edges[self.roi_top:, :], axis=0)

        if np.max(vertical_projection) == 0:
            return None, None

        # Split into left and right halves
        mid = self.img_width // 2
        left_half = vertical_projection[:mid]
        right_half = vertical_projection[mid:]

        # Find strongest edge in each half
        left_edge_x = None
        right_edge_x = None

        if np.max(left_half) > 0:
            # Find leftmost strong edge (left lane boundary)
            threshold = np.max(left_half) * 0.3
            left_candidates = np.where(left_half > threshold)[0]
            if len(left_candidates) > 0:
                # Take median of strong edges for robustness
                left_edge_x = float(np.median(left_candidates))

        if np.max(right_half) > 0:
            # Find rightmost strong edge (lane right boundary)
            threshold = np.max(right_half) * 0.3
            right_candidates = np.where(right_half > threshold)[0]
            if len(right_candidates) > 0:
                right_edge_x = float(mid + np.median(right_candidates))

        # Validate lane width
        if left_edge_x is not None and right_edge_x is not None:
            width = right_edge_x - left_edge_x
            if self.min_lane_width <= width <= self.max_lane_width:
                return left_edge_x, right_edge_x

        return None, None

class SemanticLaneDetector:
    """
    Lane detector using semantic segmentation labels.

    If Isaac Sim provides semantic segmentation with lane labels, this is more accurate than classical CV methods.

    Expects semantic image where lane pixels have a specific label.
    """

    def __init__(
        self,
        img_width: int = 160,
        img_height: int = 90,
        lane_label: int = 1,              # Semantic label for lane
        center_tolerance_px: int = 19,    # Tolerance for "in lane"
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.lane_label = lane_label
        self.center_tolerance = center_tolerance_px

    def detect(self, semantic_image: np.ndarray) -> LaneDetectionResult:
        """
        Detect lane from semantic segmentation.

        Args:
            semantic_image: (H, W) semantic labels or (H, W, 3) RGB where lane pixels are marked with specific color

        Returns:
            LaneDetectionResult
        """
        # Extract lane pixels
        if len(semantic_image.shape) == 3:
            # Assuming lane is marked with specific RGB color
            # Adjust based on our semantic labeling scheme
            lane_mask = (semantic_image[:, :, 0] == self.lane_label)
        else:
            lane_mask = (semantic_image == self.lane_label)

        if np.sum(lane_mask) == 0:
            # No lane detected
            return LaneDetectionResult(
                in_lane=False,
                lateral_offset=0.0,
                confidence=0.0
            )

        # Find lane boundaries
        # For each row, find leftmost and rightmost lane pixels
        rows, cols = np.where(lane_mask)

        if len(rows) == 0:
            return LaneDetectionResult(
                in_lane=False,
                lateral_offset=0.0,
                confidence=0.0
            )

        # Focus on lower half of image (road ahead)
        roi_mask = rows >= (self.img_height // 2)
        roi_cols = cols[roi_mask]

        if len(roi_cols) == 0:
            return LaneDetectionResult(
                in_lane=False,
                lateral_offset=0.0,
                confidence=0.0
            )

        # Lane boundaries
        left_edge = float(np.min(roi_cols))
        right_edge = float(np.max(roi_cols))
        lane_center = (left_edge + right_edge) / 2.0
        lane_width = right_edge - left_edge

        # Image center
        image_center = self.img_width / 2.0

        # Lateral offset
        max_deviation = lane_width / 2.0
        if max_deviation > 0:
            lateral_offset = (image_center - lane_center) / max_deviation
            lateral_offset = np.clip(lateral_offset, -1.0, 1.0)
        else:
            lateral_offset = 0.0

        # In lane check
        deviation_px = abs(image_center - lane_center)
        in_lane = deviation_px < self.center_tolerance

        # Confidence based on lane pixel count
        lane_coverage = np.sum(lane_mask) / (self.img_width * self.img_height)
        confidence = min(lane_coverage * 5.0, 1.0) # Scale appropriately

        return LaneDetectionResult(
            in_lane=in_lane,
            lateral_offset=lateral_offset,
            confidence=confidence,
            left_edge=np.array([left_edge]),
            right_edge=np.array([right_edge]),
            lane_center=lane_center,
            lane_width=lane_width
        )


def visualize_lane_detection(
    image: np.ndarray,
    result: LaneDetectionResult,
    show: bool = False
) -> np.ndarray:
    """
    Visualize lane detection results on image.

    Args:
        image: RGB image
        result: LaneDetectionResult
        show: If True, display using cv2.imshow

    Returns:
        Annotated iamge
    """

    vis = image.copy()

    if result.confidence > 0:
        # Draw lane boundaries
        if result.left_edge is not None:
            x = int(result.left_edge[0])
            cv2.line(vis, (x, 0), (x, vis.shape[0]), (255, 0, 0), 2)

        if result.right_edge is not None:
            x = int(result.right_edge[0])
            cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 255, 0), 2)

        # Draw lane center
        if result.lane_center is not None:
            x = int(result.lane_center)
            cv2.line(vis, (x, 0), (x, vis.shape[0]), (255, 255, 0), 1)

        # Draw vehicle center
        x = vis.shape[1] // 2
        color = (0, 255, 0) if result.in_lane else (255, 0, 0)
        cv2.line(vis, (x, 0), (x, vis.shape[0]), color, 2)

        # Add text
        status = "IN LANE" if result.in_lane else "OUT OF LANE"
        color = (0, 255, 0) if result.in_lane else (255, 0, 0)
        cv2.putText(
            vis, status, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        offset_text = f"Offset: {result.lateral_offset:.2f}"
        cv2.putText(
            vis, offset_text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

    if show:
        cv2.imshow("Lane Detection", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return vis


# Example usage and testing
if __name__ == "__main__":
    """Test lane detector with synthetic images."""

    print("Testing SimpleLaneDetector...")

    # Creating synthetic test image (road with lane markings)
    test_img = np.zeros((90, 160, 3), dtype=np.uint8)

    # Draw road (gray)
    test_img[36:, :] = 60

    # Draw lane markings (white)
    test_img[36:, 37:42] = 255 # Left boundary
    test_img[36:, 116:121] = 255 # Right boundary
    test_img[36:, 77:82] = 200 # Center line

    # Create detector
    detector = SimpleLaneDetector(img_width=160, img_height=90)

    # Test detection
    result = detector.detect(test_img)

    print(f"\nDetection Results:")
    print(f"  In Lane: {result.in_lane}")
    print(f"  Lateral Offset: {result.lateral_offset:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")

    if result.lane_center is not None:
        print(f"  Lane Center: {result.lane_center:.1f}")
    if result.lane_width is not None:
        print(f"  Lane Width: {result.lane_width:.1f}")

    # Visualize
    vis = visualize_lane_detection(test_img, result, show=False)

    # Test with offset vehicle (simulate steering right)
    test_img_offset = np.zeros((90, 160, 3), dtype=np.uint8)
    test_img_offset[36:, :] = 60
    test_img_offset[36:, 50:55] = 255 # Left boundary (shifted)
    test_img_offset[36:, 129:134] = 255 # Right boundary (shifted)

    result_offset = detector.detect(test_img_offset)
    print(f"\nOffset Vehicle Detection:")
    print(f"  In Lane: {result_offset.in_lane}")
    print(f"  Lateral Offset: {result_offset.lateral_offset:.3f}")

    print("\n Lane detector test complete!")

