"""
Intersection Reward Wrapper

Additive reward shaping for intersection approach, stopping, and exit
validation. Reads the Worker's substate machine output from info and
adds per-step + one-shot bonuses/penalties to the inner environment's
reward.

PVP transfer-realizability audit:
    Every signal consumed here is deployment-realizable. The wrapper
    uses only:
        - info["worker_state"], info["worker_substate"]
        - info["stop_line_detected"], info["stop_line_distance_m"],
          info["stop_line_confidence"]
        - info["go_signal"]
        - info["exit_correct"]
        - action[2] (brake output from policy)
        - obs["vec"][3] (speed from odometry)

    None of these are privileged at deployment:
        - Worker substate = Worker's own output; same at training and
          deployment (same state machine, just different position source)
        - Detector output = camera-derived at deployment (and at training
          when detector_kind="visual")
        - go_signal = Scheduler output, classical planner, same both paths
        - action + speed = policy output and wheel odometry

    The training-only geometric detector (detector_kind="geometric")
    produces deployment-shaped output from privileged inputs, so the
    reward formula is unchanged. When switching to the visual detector
    for the final training run, the reward code doesn't move.

Composition:
    Stack order:
        IsaacDirectEnv -> AgentEnvWrapper -> IntersectionRewardWrapper -> SB3

    AgentEnvWrapper must be INSIDE this wrapper so info is already
    enriched with Worker substate when this wrapper runs its shaping.

Weight tuning:
    All weights in IntersectionRewardConfig are placeholders. Tune
    AFTER the first clean training run validates the base reward
    isn't dominated by shaping. Start with small magnitudes (<1.0)
    relative to per-step lane reward (~1.0 in isaac_direct_env).

Author: Aaron Hamil
Date: 04/20/26
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


# Config

@dataclass
class IntersectionRewardConfig:
    """
    Weights + thresholds for intersection reward shaping.

    All magnitudes are placeholders pending the first training run.
    Values are chosen conservatively (<= per-step lane reward ~1.0)
    to avoid dominating the base reward before tuning.

    Attributes (gating):
        enabled: Master switch. If False, wrapper is a no-op.
        approach_zone_m: Only reward brake action when the detected
            stop line is within this ahead-of-car distance. Outside
            this, braking is premature and we don't encourage it.
        stop_tolerance_m: Absolute distance error below which the
            agent gets the perfect-stop bonus. Matches
            IntersectionLayout.stop_line_tolerance by default; this
            is a separate field so reward and Worker can be tuned
            independently.
        violation_speed_mps: If the agent is above this speed while
            go_signal=0, the gate-running penalty fires.
        min_detection_confidence: Only apply detector-derived shaping
            when confidence >= this. Matches the Worker's detector
            threshold but duplicated here to decouple reward from
            Worker tuning.

    Attributes (per-step weights):
        brake_incentive: Reward per unit of brake action when in
            approach zone with detected line. Typical scale: 0.1-0.5
            (applied per step at 20 Hz => 2-10 reward/sec if holding
            full brake).
        running_line_penalty: Per-step penalty for moving while
            go_signal=0. Negative. Scale: -0.5 to -2.0 per step.

    Attributes (one-shot weights):
        perfect_stop_bonus: One-shot bonus for stopping within
            stop_tolerance_m of the line. Scale: 5.0-20.0.
        overshoot_weight: Per-meter penalty for stopping PAST the
            line (distance_m < 0). Scale: 10.0-30.0. Intentionally
            HIGHER than undershoot (overshooting is a safety issue).
        undershoot_weight: Per-meter penalty for stopping short of
            the line (distance_m > tolerance). Scale: 3.0-10.0.
        correct_exit_bonus: One-shot reward when the agent exits on
            the road the Worker committed to. Scale: 5.0-20.0.
        wrong_exit_penalty: One-shot penalty for wrong exit.
            Negative. Scale: -10.0 to -30.0.
    """
    enabled: bool = True
    approach_zone_m: float = 1.5
    stop_tolerance_m: float = 0.08
    violation_speed_mps: float = 0.25
    min_detection_confidence: float = 0.3

    # Per-step
    brake_incentive: float = 0.25
    running_line_penalty: float = -1.0

    # One-shot
    perfect_stop_bonus: float = 10.0
    overshoot_weight: float = 20.0
    undershoot_weight: float = 5.0
    correct_exit_bonus: float = 10.0
    wrong_exit_penalty: float = -15.0


# Wrapper

class IntersectionRewardWrapper(gym.Wrapper):
    """
    Additive reward shaping for intersection behavior.

    Adds to (does not replace) the inner environment's reward:
        - Per-step brake encouragement near the stop line
        - Per-step running-the-line penalty
        - One-shot final-stop proximity bonus/penalty
        - One-shot correct/wrong exit bonus/penalty

    The Worker's `use_stop_line=False` legacy path produces no
    APPROACHING/STOPPING/EXITED transitions, so the one-shot rewards
    naturally never fire. Per-step terms also gate on substate so
    they're no-ops in legacy mode — making the wrapper safe to leave
    stacked even when running legacy Worker.

    Observation and action spaces pass through unchanged.

    Usage:
        env = IsaacDirectEnv(config)
        env = AgentEnvWrapper(env, graph=graph, agent_config=cfg)
        env = IntersectionRewardWrapper(env, config=reward_cfg)
    """

    # Info key for the shaping delta (useful for logging separately
    # from base reward in TensorBoard)
    INFO_KEY = "intersection_reward"

    # Obs / info field names we read
    _SPEED_IDX = 3  # vec[3] from TELEMETRY_INDICES

    def __init__(
        self,
        env: gym.Env,
        config: Optional[IntersectionRewardConfig] = None,
    ):
        super().__init__(env)
        self.config = config or IntersectionRewardConfig()

        # Edge-trigger state: track previous substate to fire one-shots
        # exactly once per transition.
        self._prev_substate: str = "none"
        self._prev_state: str = "cruising"

        # One-shot latches (reset at each CRUISING -> DECIDING promotion):
        self._stop_evaluated: bool = False
        self._exit_evaluated: bool = False

        # Track the intersection cycle we're in, so re-entering the same
        # intersection (after cooldown) re-arms the latches.
        self._cycle_intersection: Optional[str] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset and clear latches."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_substate = info.get("worker_substate", "none")
        self._prev_state = info.get("worker_state", "cruising")
        self._stop_evaluated = False
        self._exit_evaluated = False
        self._cycle_intersection = info.get("intersection")

        # Expose zero delta on reset for consistent logging
        info[self.INFO_KEY] = 0.0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step inner env, add intersection shaping to reward."""
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        if not self.config.enabled:
            info[self.INFO_KEY] = 0.0
            return obs, base_reward, terminated, truncated, info

        # Read everything from info / obs / action (all non-privileged)
        delta = self._compute_shaping(obs, action, info)

        # Update prev-state for next tick (AFTER computing shaping)
        self._prev_state = info.get("worker_state", "cruising")
        self._prev_substate = info.get("worker_substate", "none")

        info[self.INFO_KEY] = float(delta)
        return obs, float(base_reward) + float(delta), terminated, truncated, info

    ### Core shaping

    def _compute_shaping(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Compute the additive reward delta for this step.

        Gates:
            - Per-step terms only while in DECIDING
            - One-shot stop evaluation only on APPROACHING -> STOPPING
            - One-shot exit evaluation only on TRAVERSING -> EXITED
        """
        cfg = self.config
        state = info.get("worker_state", "cruising")
        substate = info.get("worker_substate", "none")
        intersection = info.get("intersection")

        # Re-arm latches on fresh intersection entry
        if (
            self._prev_state == "cruising"
            and state == "deciding"
        ):
            self._stop_evaluated = False
            self._exit_evaluated = False
            self._cycle_intersection = intersection

        # Also re-arm if the intersection id changed (shouldn't happen
        # without passing through CRUISING, but defensive)
        if intersection is not None and intersection != self._cycle_intersection:
            self._stop_evaluated = False
            self._exit_evaluated = False
            self._cycle_intersection = intersection

        delta = 0.0

        # Per-step gate: only in DECIDING. COMMITTED means we've already
        # released through the intersection, no further stop shaping.
        if state == "deciding":
            delta += self._per_step_terms(obs, action, info)

        # One-shot: final stop evaluation on APPROACHING -> STOPPING
        if (
            not self._stop_evaluated
            and self._prev_substate == "approaching"
            and substate == "stopping"
        ):
            delta += self._final_stop_bonus(info)
            self._stop_evaluated = True

        # One-shot: exit evaluation on TRAVERSING -> EXITED
        if (
            not self._exit_evaluated
            and self._prev_substate == "traversing"
            and substate == "exited"
        ):
            delta += self._exit_bonus(info)
            self._exit_evaluated = True

        return delta

    # Per-step terms

    def _per_step_terms(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Per-step shaping while in DECIDING.

        Two additive components:
            1. Brake encouragement near the detected line
            2. Running-the-line penalty (go_signal=0 but moving)
        """
        cfg = self.config
        delta = 0.0

        detected = bool(info.get("stop_line_detected", False))
        distance = float(info.get("stop_line_distance_m", 0.0))
        confidence = float(info.get("stop_line_confidence", 0.0))

        # Brake incentive: only when detector sees a line ahead,
        # within the approach zone, and with acceptable confidence.
        if (
            detected
            and confidence >= cfg.min_detection_confidence
            and 0.0 < distance <= cfg.approach_zone_m
            and action.shape[-1] >= 3
        ):
            brake_action = float(np.clip(action[2], 0.0, 1.0))
            delta += cfg.brake_incentive * brake_action

        # Running-the-line penalty: scheduler says wait but we're moving
        go_signal = float(info.get("go_signal", 1.0))
        speed = self._read_speed(obs)
        if go_signal < 0.5 and speed > cfg.violation_speed_mps:
            delta += cfg.running_line_penalty

        return delta

    ### One-shot: final stop proximity

    def _final_stop_bonus(self, info: Dict[str, Any]) -> float:
        """
        Evaluate stop placement at the APPROACHING -> STOPPING transition.

        Reads the detector's last-reported distance at the moment the
        Worker latched STOPPING. This IS the "distance at stop" — the
        Worker promotes to STOPPING only when speed < stopped_threshold.

        Reward shape is asymmetric: overshoot (past the line, distance<0)
        is penalized more heavily than undershoot, per NA safety
        convention — crossing the line is worse than stopping early.

        If the detector wasn't confident at the stop moment, we fall
        back to no shaping (neither reward nor penalty). We don't
        penalize agents for the detector missing a line.
        """
        cfg = self.config
        detected = bool(info.get("stop_line_detected", False))
        confidence = float(info.get("stop_line_confidence", 0.0))

        if not detected or confidence < cfg.min_detection_confidence:
            return 0.0

        distance = float(info.get("stop_line_distance_m", 0.0))

        if abs(distance) < cfg.stop_tolerance_m:
            return cfg.perfect_stop_bonus
        elif distance < 0.0:
            # Past the line (overshoot)
            return -cfg.overshoot_weight * abs(distance)
        else:
            # Short of the line (undershoot)
            return -cfg.undershoot_weight * (distance - cfg.stop_tolerance_m)

    # One-shot: exit validation

    def _exit_bonus(self, info: Dict[str, Any]) -> float:
        """
        One-shot reward at TRAVERSING -> EXITED.

        Uses info["exit_correct"] which the Worker sets to True/False
        based on comparing the detected exit road_id with the road the
        turn_token committed to.
        """
        cfg = self.config
        exit_correct = info.get("exit_correct")
        if exit_correct is True:
            return cfg.correct_exit_bonus
        elif exit_correct is False:
            return cfg.wrong_exit_penalty
        # None = exit not yet evaluated (shouldn't happen on this
        # transition, but defensive)
        return 0.0

    # Helpers

    def _read_speed(self, obs: Dict[str, np.ndarray]) -> float:
        """Pull speed from obs["vec"][3] with defensive checks."""
        if not isinstance(obs, dict):
            return 0.0
        vec = obs.get("vec")
        if vec is None or not hasattr(vec, "__getitem__"):
            return 0.0
        try:
            return float(vec[self._SPEED_IDX])
        except (IndexError, TypeError):
            return 0.0
