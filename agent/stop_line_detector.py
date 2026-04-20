"""
Stop Line Detector — Visual and Geometric Implementations

Two pluggable detectors that return identical dataclass output:

    VisualStopLineDetector    — classical CV on the forward camera
                                image. Deployment-realizable: the same
                                pipeline runs on the real F1TENTH's
                                RealSense stream.

    GeometricStopLineDetector — synthesizes detections from privileged
                                world-frame geometry. Training-only
                                bootstrap for use BEFORE the visual
                                pipeline is validated on Arika's scene
                                (or when the scene lacks visible
                                stop-line meshes).

Both accept a StopLineDetectionContext and return a StopLineDetection.
The AgentEnvWrapper configures which detector to use per experiment.

Following the existing lane_detector.py conventions:
    - Dataclass return type with confidence in [0, 1]
    - Class with __init__(config) and detect(ctx) -> Result
    - cv2 for the visual variant (already a project dependency)
    - Stateless per-frame (no tracker across frames)

Pixel-to-ground projection (visual variant):
    Uses pinhole + level-ground assumption. Given a detected image row
    below the horizon, projects to ahead-of-camera distance via:
        d = camera_height / tan(pitch_angle)
    where pitch_angle is derived from pixel position and camera intrinsics.

    Camera intrinsics come from IsaacDirectEnv / ARCProSceneCfg:
        horizontal_aperture = 2.65
        focal_length        = 1.93
        image size          = 160 x 90
        height above ground ~ 0.20 m (chassis + mount)
        pitch               = 0 (level forward)
    All configurable via StopLineDetectorConfig.

Author: Aaron Hamil
Date: 04/20/26
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from agent.intersection_geometry import (
    IntersectionLayout,
    distance_to_stop_line_world,
)

logger = logging.getLogger(__name__)


# Detection Output (shared by both detectors)

@dataclass
class StopLineDetection:
    """
    Result of one stop-line detection attempt.

    Contract: same structure regardless of detector source. A null /
    no-detection result has detected=False, distance_m=0.0, confidence=0.0.

    Fields:
        detected: True if a stop line was found with acceptable
            confidence. False implies the other numeric fields are
            meaningless (still zeroed for predictability).
        distance_m: Ahead-of-camera distance to the line along the
            approach axis, meters. Positive = line is in front of the
            agent. Negative = line is behind (agent has crossed it).
        confidence: [0, 1]. For VisualStopLineDetector this is derived
            from the peak's pixel-row fraction + geometric plausibility.
            For GeometricStopLineDetector this is always 1.0 when the
            detection is gated as active (since we have ground truth),
            0.0 otherwise.
        image_row: Pixel row where the line was detected (visual only).
            None for geometric detector. Useful for overlay debugging.
        source: "visual" | "geometric" | "none". Indicates which
            pipeline produced this result.
    """
    detected: bool = False
    distance_m: float = 0.0
    confidence: float = 0.0
    image_row: Optional[int] = None
    source: str = "none"


@dataclass
class StopLineDetectionContext:
    """
    Per-frame inputs to a StopLineDetector.

    Carries every signal either detector might need. Each detector picks
    out what it uses and ignores the rest. The AgentEnvWrapper builds
    this context once per step.

    Fields:
        image: Forward RGB image (H, W, 3), uint8. Required for visual.
        agent_xy: World (x, y) of the agent. Required for geometric.
        intersection_center: World (x, y) of the arming intersection.
            Required for geometric.
        approach_heading_rad: Approach heading (direction toward the
            intersection). Required for geometric.
        active: Whether the detector should run at all. When False, the
            detector returns a null detection without processing.
            This is set by the Worker's map-based pre-gate.
    """
    image: Optional[np.ndarray] = None
    agent_xy: Optional[Tuple[float, float]] = None
    intersection_center: Optional[Tuple[float, float]] = None
    approach_heading_rad: Optional[float] = None
    active: bool = False


# Config

@dataclass
class StopLineDetectorConfig:
    """
    Configuration for VisualStopLineDetector.

    Defaults align with IsaacDirectEnv / ARCProSceneCfg camera. Verify
    camera_height_m and camera_pitch_rad against the actual USD mount
    before first training; a mismatch here biases the entire pixel-to-
    distance projection.

    Attributes (image geometry):
        img_width / img_height: Camera resolution (pixels).
        horizontal_aperture: Isaac Sim camera attr (stage units).
        focal_length: Isaac Sim camera attr (stage units).
        camera_height_m: Camera optical center height above ground,
            meters. Chassis_base (~0.05m) + mount (~0.15m) = ~0.20m.
        camera_pitch_rad: Pitch of optical axis from horizontal,
            radians. Positive = looking down. Default 0 = level forward.

    Attributes (detection thresholds):
        roi_top_ratio: Upper bound of the region searched for a stop
            line, as a fraction of image height. Stop lines don't
            appear above the horizon, so we mask the top. Default 0.5
            searches the lower half.
        white_threshold: Grayscale value above which pixels are
            considered white. Default 200 matches the lane_detector
            convention for bright road paint.
        min_line_width_px: Minimum horizontal span (pixels) for a
            detected bright row to count as a line. Rejects small
            bright speckles.
        max_line_thickness_px: Maximum vertical extent (pixels) of the
            detected bright region. Rejects large bright patches
            (e.g. buildings, signs) that span many rows.
        min_fraction: Minimum fraction of the peak row that must be
            white to accept the detection. Default 0.35.

    Attributes (output):
        min_confidence: Below this confidence, reports detected=False.
    """
    img_width: int = 160
    img_height: int = 90
    horizontal_aperture: float = 2.65
    focal_length: float = 1.93
    camera_height_m: float = 0.20
    camera_pitch_rad: float = 0.0

    roi_top_ratio: float = 0.5
    white_threshold: int = 200
    min_line_width_px: int = 40
    max_line_thickness_px: int = 8
    min_fraction: float = 0.35

    min_confidence: float = 0.3

    @property
    def vertical_aperture(self) -> float:
        """Derived vertical aperture assuming square pixels."""
        return self.horizontal_aperture * self.img_height / self.img_width


# Base

class StopLineDetectorBase:
    """
    Interface contract for stop-line detectors.

    Subclasses implement detect(ctx) -> StopLineDetection. The context
    carries every signal any variant might need; subclasses use what
    they need and ignore the rest.
    """

    source: str = "none"

    def detect(self, ctx: StopLineDetectionContext) -> StopLineDetection:
        raise NotImplementedError


# Visual

class VisualStopLineDetector(StopLineDetectorBase):
    """
    Classical CV stop-line detector.

    Pipeline:
        1. If not active, return null.
        2. Extract lower-half ROI (mask out above-horizon region).
        3. Convert to grayscale, threshold for bright pixels.
        4. Compute row-wise white-pixel counts.
        5. Find peak row with strongest horizontal white run.
        6. Reject if peak fraction below threshold (not a real line).
        7. Reject if bright region spans too many rows (not a stripe).
        8. Project peak row to ahead-of-camera ground distance.
        9. Confidence = peak_fraction capped at 1.0.

    Deliberately uses the same conventions as SimpleLaneDetector
    (lane_detector.py): RGB input uint8, cv2 grayscale, integer
    threshold values, dataclass output.
    """

    source = "visual"

    def __init__(self, config: Optional[StopLineDetectorConfig] = None):
        if cv2 is None:
            raise ImportError(
                "VisualStopLineDetector requires opencv-python (cv2). "
                "Install with `pip install opencv-python`."
            )
        self.config = config or StopLineDetectorConfig()
        self._roi_top = int(self.config.img_height * self.config.roi_top_ratio)

    def detect(self, ctx: StopLineDetectionContext) -> StopLineDetection:
        if not ctx.active or ctx.image is None:
            return StopLineDetection(source=self.source)

        image = ctx.image
        h, w = self.config.img_height, self.config.img_width
        if image.shape[0] != h or image.shape[1] != w:
            logger.debug(
                "VisualStopLineDetector: image shape %s != expected (%d, %d)",
                image.shape, h, w,
            )
            return StopLineDetection(source=self.source)

        # Grayscale
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Mask upper region (above horizon): stop lines only appear
        # in the lower half of the image for a level-forward camera
        mask = gray.copy()
        mask[: self._roi_top, :] = 0

        # Threshold for white paint
        _, white = cv2.threshold(
            mask, self.config.white_threshold, 255, cv2.THRESH_BINARY
        )

        # Horizontal whiteness per row
        row_counts = np.sum(white > 0, axis=1)  # (H,) int
        if row_counts.max() == 0:
            return StopLineDetection(source=self.source)

        # Pick peak row as candidate line location
        peak_row = int(np.argmax(row_counts))
        peak_count = int(row_counts[peak_row])

        # Line-width gate: reject speckles
        if peak_count < self.config.min_line_width_px:
            return StopLineDetection(source=self.source)

        # Thickness gate: reject large bright regions (signs, buildings)
        # Thick region = many neighboring rows also near peak count
        neighborhood_threshold = max(
            int(peak_count * 0.6), self.config.min_line_width_px
        )
        above_thresh = row_counts >= neighborhood_threshold
        # Count contiguous run of rows around peak that exceed threshold
        run_len = 1
        # expand up
        r = peak_row - 1
        while r >= 0 and above_thresh[r]:
            run_len += 1
            r -= 1
        # expand down
        r = peak_row + 1
        while r < h and above_thresh[r]:
            run_len += 1
            r += 1
        if run_len > self.config.max_line_thickness_px:
            return StopLineDetection(source=self.source)

        # Confidence: fraction of row that's white, clipped to [0, 1]
        fraction = peak_count / float(w)
        if fraction < self.config.min_fraction:
            return StopLineDetection(source=self.source)
        confidence = min(fraction / max(self.config.min_fraction, 1e-6), 1.0)
        if confidence < self.config.min_confidence:
            return StopLineDetection(source=self.source)

        # Pixel row -> ground distance ahead via pinhole + level-ground
        distance = self._row_to_distance(peak_row)
        if distance is None:
            return StopLineDetection(source=self.source)

        return StopLineDetection(
            detected=True,
            distance_m=float(distance),
            confidence=float(confidence),
            image_row=peak_row,
            source=self.source,
        )

    def _row_to_distance(self, row: int) -> Optional[float]:
        """
        Project a pixel row to ahead-of-camera ground distance.

        Assumes level-forward optical axis (pitch = 0) and flat ground.
        Rows above the horizon return None (line cannot be on ground).

        For a standard forward-looking pinhole camera with the image
        convention row=0 at top, row=H-1 at bottom:
            - The horizon is at row = H/2 when pitch = 0.
            - Rows below the horizon look at the ground; the angle
              below horizontal is atan(y_img / focal_length) where
              y_img is the signed vertical distance from the optical
              center on the image plane.
            - Distance along the ground = camera_height / tan(angle).
        """
        cfg = self.config
        H = cfg.img_height
        # Signed position on image plane (in stage units). Bottom row has
        # maximum positive y_img; center row has y_img = 0.
        pixel_pitch = cfg.vertical_aperture / H
        y_img = (row - H / 2.0) * pixel_pitch

        pitch_angle = cfg.camera_pitch_rad + math.atan2(y_img, cfg.focal_length)
        # Must be looking DOWN to hit the ground
        if pitch_angle <= 1e-4:
            return None

        return cfg.camera_height_m / math.tan(pitch_angle)


# Geometric (privileged bootstrap)

class GeometricStopLineDetector(StopLineDetectorBase):
    """
    Privileged ground-truth stop-line "detector" for training bootstrap.

    Uses world-frame agent position + intersection center + approach
    heading to compute the exact distance to the stop line (via
    intersection_geometry.distance_to_stop_line_world). Returns a
    detection with confidence=1.0 whenever active.

    Rationale: lets the Worker substate machine and reward wrapper be
    validated end-to-end BEFORE the visual detector is tuned against
    Arika's scene. Once the visual detector is empirically acceptable,
    config flip swaps this out for VisualStopLineDetector; no Worker
    or reward code changes.

    PVP sanity: this detector consumes privileged signals, so it must
    NEVER be used at deployment. It exists purely for training
    bootstrap. The experiment config gates it explicitly.
    """

    source = "geometric"

    def __init__(
        self,
        layout: IntersectionLayout,
        max_effective_distance: float = 3.0,
    ):
        """
        Args:
            layout: Intersection layout (provides the offset from
                center at which the stop line sits).
            max_effective_distance: Clamp the signed distance to
                [-max, +max] so the synthesized detection behaves
                like a real one with finite range. Default 3.0 m.
        """
        self.layout = layout
        self.max_effective_distance = max_effective_distance

    def detect(self, ctx: StopLineDetectionContext) -> StopLineDetection:
        if not ctx.active:
            return StopLineDetection(source=self.source)
        if (
            ctx.agent_xy is None
            or ctx.intersection_center is None
            or ctx.approach_heading_rad is None
        ):
            return StopLineDetection(source=self.source)

        distance = distance_to_stop_line_world(
            ctx.agent_xy,
            ctx.intersection_center,
            ctx.approach_heading_rad,
            self.layout,
        )

        # Range clamp (a real detector won't see arbitrarily far lines)
        if abs(distance) > self.max_effective_distance:
            return StopLineDetection(source=self.source)

        return StopLineDetection(
            detected=True,
            distance_m=float(distance),
            confidence=1.0,
            image_row=None,
            source=self.source,
        )


# Factory

def make_stop_line_detector(
    kind: str,
    config: Optional[StopLineDetectorConfig] = None,
    layout: Optional[IntersectionLayout] = None,
) -> StopLineDetectorBase:
    """
    Construct the configured detector by name.

    Args:
        kind: "visual" or "geometric".
        config: Required if kind == "visual".
        layout: Required if kind == "geometric".

    Returns:
        A StopLineDetectorBase subclass instance.

    Raises:
        ValueError: unknown kind or missing required config.
    """
    if kind == "visual":
        return VisualStopLineDetector(config=config)
    elif kind == "geometric":
        if layout is None:
            raise ValueError("GeometricStopLineDetector requires IntersectionLayout")
        return GeometricStopLineDetector(layout=layout)
    else:
        raise ValueError(f"Unknown detector kind: {kind!r}")


# Overlay helper (parallels visualize_lane_detection)

def visualize_stop_line_detection(
    image: np.ndarray,
    result: StopLineDetection,
) -> np.ndarray:
    """
    Overlay a detection onto an image for debugging.

    Draws a horizontal line across the detected row, colored green
    for detected / red for not detected, and annotates distance and
    confidence. Requires cv2.

    Args:
        image: RGB image.
        result: StopLineDetection.

    Returns:
        Annotated RGB image.
    """
    if cv2 is None:
        return image
    vis = image.copy()
    h, w = vis.shape[:2]

    if result.detected and result.image_row is not None:
        color = (0, 255, 0)
        cv2.line(vis, (0, result.image_row), (w - 1, result.image_row), color, 2)
        cv2.putText(
            vis, f"{result.distance_m:.2f}m", (5, max(12, result.image_row - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
    cv2.putText(
        vis, f"src={result.source} conf={result.confidence:.2f}",
        (5, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
    )
    return vis
