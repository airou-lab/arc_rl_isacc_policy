# ARCPro RL: Policy Architecture & Training Stability

This directory contains the reinforcement learning policy definitions and observation/action protocols for the ARCPro system.

## Policy: HierarchicalPathPlanningPolicy

The `HierarchicalPathPlanningPolicy` is a custom `RecurrentActorCriticPolicy` (from `sb3-contrib`) designed for vision-based autonomous driving.

### Architecture Overview
1.  **Vision Encoder:** 160x90 RGB images are processed via a `FusionFeaturesExtractor` (CNN).
2.  **Telemetry Integration:** A 12-float vector (speed, yaw rate, last actions, etc.) is concatenated with the flattened visual features (256-dim) to form a 268-dim latent state.
3.  **LSTM Layer:** A 256-unit LSTM processes the latent state to maintain temporal context (crucial for overcoming visual occlusion or sudden maneuvers).
4.  **Hierarchical Heads:**
    *   **Planning Head:** Predicts 5 waypoints as deviations from kinematic anchors (curved paths based on turn bias).
    *   **Control Head:** Generates final [steering, throttle, brake] actions conditioned on both the LSTM state and the planned waypoints.

### Technical Fix: Gradient Continuity (Optimizer Patch)
A critical "no-gradient" bug was resolved by recreating the optimizer in `__init__` after all sub-networks (Planning/Control) are built. 

**The Issue:** The standard Stable Baselines 3 `super().__init__` call initializes the optimizer before the custom hierarchical heads are added. Consequently, parameters in `planning_head` and `control_head` were never registered with the optimizer, resulting in zero-gradient updates for the most important parts of the network.

**The Fix:**
```python
# Recreate optimizer AFTER all hierarchical heads are built.
self.optimizer = self.optimizer_class(
    self.parameters(),
    lr=lr_schedule(1),
    **self.optimizer_kwargs,
)
```

## Observation Protocol (12-Float Vector)
The policy expects a strictly ordered 12-float telemetry vector:
- `0`: Turn Bias (Continuous [-1.0, 1.0])
- `1`: Reserved (0.0)
- `2`: Goal Distance (Masked to 0 during training)
- `3`: Speed (m/s)
- `4`: Yaw Rate (rad/s)
- `5-7`: Previous [Steer, Throttle, Brake]
- `8-11`: Lateral/Heading Error, Curvature, and Distance Travelled.
