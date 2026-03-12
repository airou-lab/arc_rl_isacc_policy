# ARCPro RL Reward Architectures

This document explains the two reward strategies developed for the ARCPro system, which are being ported to the **Isaac Lab Reward Manager**.

---

## 1. Original Evolution (Default)
**Config:** `reward_mode: "original"`

This strategy is a direct evolution of the original Unity/ROS2 training logic, modified to fix the "passivity trap" where the robot refused to move.

### The Math:
```python
if speed < 0.1: return -1.0  # Static Penalty

# Lane Bonus
if abs(lateral_error) < 0.5: reward += 1.0
else: reward -= abs(lateral_error) * 2.0

# Boosted Speed
reward += speed * 2.0  # Multiplier increased from 0.3 to 2.0
```

### Why it works:
In the original model, the reward for staying still (+1.0) was nearly identical to the reward for driving (+1.3). By boosting the speed multiplier and adding a static penalty, driving fast becomes the only way to achieve a high score.

---

## 2. Hybrid Racer
**Config:** `reward_mode: "hybrid"`

A modern racing-focused strategy that uses a Gaussian "Magnetic" Lane center to encourage precision.

### The Math:
```python
# Gaussian Lane Center
lane_reward = 2.0 * exp(-(lateral_error^2) / 0.25)

# Momentum
speed_reward = speed * 2.0
```

### Why it works:
Instead of a simple "In/Out" bonus, the Hybrid Racer uses a bell curve. 
- **Perfectly Centered:** +2.0 reward.
- **Slightly Off-center:** +1.5 reward.
- **Near the Edge:** +0.2 reward.

This providing a continuous gradient that "pulls" the AI toward the center of the lane, leading to much smoother racing lines and higher stability at high speeds.
