"""
Smoke Test — Validate All Components
======================================

Run from project root:
    python test_all.py

Tests everything that doesn't require Isaac Sim to be running:
    1. Config layer (save/load/diff)
    2. DAVE-2 model (forward pass, param count)
    3. Dataset (synthetic data load)
    4. ScriptedExpert (PD controller)
    5. KeyboardExpert (ramping logic)
    6. AckermannComputer (geometry math)
    7. Mini training loop (3 epochs on synthetic data)
    8. isaac_direct_env.py (syntax + config only, no sim)

Author: Aaron Hamil
Date: 03/05/26
"""

import sys
import os
import ast
import tempfile
import shutil
import csv
from pathlib import Path

import numpy as np

# Make sure we're at project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

passed = 0
failed = 0


def test(name, func):
    global passed, failed
    try:
        func()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


# ================================================================
#  1. CONFIG LAYER
# ================================================================
print("\n=== 1. Config Layer ===")

def test_config_create():
    from config.experiment import (
        ExperimentConfig, SimConfig, TrainingConfig,
        PolicyConfig, BaselineConfig, TELEMETRY_INDICES, TELEMETRY_DIM,
    )
    assert TELEMETRY_DIM == 12
    assert len(TELEMETRY_INDICES) == 12
    config = ExperimentConfig(name="test", method="rl")
    assert config.sim.camera_width == 160
    assert config.sim.camera_height == 90
    assert config.policy.lstm_hidden_size == 256

test("Config creation + defaults", test_config_create)

def test_config_save_load():
    from config.experiment import ExperimentConfig, BaselineConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="roundtrip_test", method="bc", seed=123,
            baseline=BaselineConfig(epochs=30, learning_rate=1e-4),
        )
        path = config.save(f"{tmpdir}/config.yaml")
        loaded = ExperimentConfig.load(str(path))
        assert loaded.name == "roundtrip_test"
        assert loaded.method == "bc"
        assert loaded.seed == 123
        assert loaded.baseline.epochs == 30
        assert loaded.baseline.learning_rate == 1e-4

test("Config save/load roundtrip", test_config_save_load)

def test_config_diff():
    from config.experiment import ExperimentConfig, BaselineConfig
    a = ExperimentConfig(name="a", baseline=BaselineConfig(epochs=30))
    b = ExperimentConfig(name="b", baseline=BaselineConfig(epochs=50))
    diffs = a.diff(b)
    assert "name" in diffs
    assert "baseline.epochs" in diffs

test("Config diff", test_config_diff)

def test_config_summary():
    from config.experiment import ExperimentConfig
    rl = ExperimentConfig(name="rl_run", method="rl")
    bc = ExperimentConfig(name="bc_run", method="bc")
    assert "RL" in rl.summary()
    assert "BC" in bc.summary()

test("Config summary", test_config_summary)


# ================================================================
#  2. DAVE-2 MODEL
# ================================================================
print("\n=== 2. DAVE-2 Model ===")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  [SKIP] PyTorch not installed — skipping model tests")

if HAS_TORCH:
    def test_dave2_forward():
        from baselines.dave2.model import DAVE2Net
        model = DAVE2Net(input_height=66, input_width=200, num_outputs=1)
        x = torch.randn(4, 3, 66, 200)
        out = model(x)
        assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"

    test("DAVE2Net forward (66x200, 1 output)", test_dave2_forward)

    def test_dave2_multi_output():
        from baselines.dave2.model import DAVE2Net
        model = DAVE2Net(input_height=66, input_width=200, num_outputs=3)
        x = torch.randn(2, 3, 66, 200)
        out = model(x)
        assert out.shape == (2, 3)

    test("DAVE2Net forward (3 outputs)", test_dave2_multi_output)

    def test_dave2_param_count():
        from baselines.dave2.model import DAVE2Net
        model = DAVE2Net(input_height=66, input_width=200, num_outputs=1)
        total, trainable = model.count_parameters()
        assert 200_000 < total < 400_000, f"Unexpected param count: {total}"
        assert total == trainable

    test("DAVE2Net param count (~250K)", test_dave2_param_count)

    def test_dave2_with_speed():
        from baselines.dave2.model import DAVE2NetWithSpeed
        model = DAVE2NetWithSpeed(input_height=66, input_width=200, num_outputs=2)
        img = torch.randn(4, 3, 66, 200)
        speed = torch.randn(4, 1)
        out = model(img, speed)
        assert out.shape == (4, 2)

    test("DAVE2NetWithSpeed forward", test_dave2_with_speed)

    def test_dave2_batchnorm():
        from baselines.dave2.model import DAVE2Net
        model = DAVE2Net(use_batchnorm=True)
        x = torch.randn(4, 3, 66, 200)
        out = model(x)
        assert out.shape == (4, 1)

    test("DAVE2Net with batchnorm", test_dave2_batchnorm)

    def test_dave2_summary():
        from baselines.dave2.model import DAVE2Net
        model = DAVE2Net()
        summary = model.architecture_summary()
        assert "DAVE-2 CNN" in summary
        assert "250" in summary or "parameters" in summary

    test("DAVE2Net architecture_summary()", test_dave2_summary)


# ================================================================
#  3. DATASET (synthetic data)
# ================================================================
print("\n=== 3. Dataset ===")

if HAS_TORCH:
    def _create_synthetic_dataset(tmpdir, num_frames=20):
        """Create a minimal fake dataset matching collect.py's output format."""
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir(parents=True)

        # Write fake PNG frames (just solid color images)
        import cv2
        for i in range(num_frames):
            img = np.random.randint(0, 255, (90, 160, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), img)

        # Write labels CSV
        with open(Path(tmpdir) / "labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "steering", "throttle", "brake", "speed"])
            for i in range(num_frames):
                writer.writerow([
                    f"frame_{i:06d}",
                    f"{np.random.uniform(-1, 1):.6f}",
                    f"{np.random.uniform(0, 1):.6f}",
                    f"{0.0:.6f}",
                    f"{np.random.uniform(0, 2):.6f}",
                ])

    def test_dataset_load():
        from baselines.dave2.dataset import DrivingDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_synthetic_dataset(tmpdir, num_frames=10)
            ds = DrivingDataset(
                data_dir=tmpdir,
                output_height=66, output_width=200,
                augment=False,
            )
            assert len(ds) == 10
            img, action = ds[0]
            assert img.shape == (3, 66, 200), f"Got {img.shape}"
            assert action.shape[0] >= 1

    test("DrivingDataset load synthetic", test_dataset_load)

    def test_dataset_augmentation():
        from baselines.dave2.dataset import DrivingDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_synthetic_dataset(tmpdir, num_frames=10)
            ds = DrivingDataset(
                data_dir=tmpdir,
                output_height=66, output_width=200,
                augment=True,
            )
            img1, act1 = ds[0]
            img2, act2 = ds[0]
            # Augmentation is random so two reads of same index may differ
            assert img1.shape == (3, 66, 200)

    test("DrivingDataset with augmentation", test_dataset_augmentation)

    def test_dataset_with_speed():
        from baselines.dave2.dataset import DrivingDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_synthetic_dataset(tmpdir, num_frames=10)
            ds = DrivingDataset(
                data_dir=tmpdir,
                output_height=66, output_width=200,
                include_speed=True,
            )
            result = ds[0]
            assert len(result) == 3  # (image, action, speed)

    test("DrivingDataset include_speed", test_dataset_with_speed)

    def test_dataset_splits():
        from baselines.dave2.dataset import DrivingDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_synthetic_dataset(tmpdir, num_frames=20)
            train_ds, val_ds = DrivingDataset.create_splits(
                tmpdir, train_ratio=0.8,
            )
            assert len(train_ds) == 16
            assert len(val_ds) == 4

    test("DrivingDataset train/val splits", test_dataset_splits)


# ================================================================
#  4. SCRIPTED EXPERT
# ================================================================
print("\n=== 4. ScriptedExpert ===")

def test_expert_straight():
    from baselines.dave2.collect import ScriptedExpert
    expert = ScriptedExpert()
    telemetry = np.zeros(12, dtype=np.float32)
    telemetry[3] = 1.0  # speed
    action = expert.compute_action(telemetry)
    assert action.shape == (3,)
    assert abs(action[0]) < 0.01  # no steering needed
    assert action[1] > 0.0        # should accelerate (below target speed)

test("ScriptedExpert straight driving", test_expert_straight)

def test_expert_lateral_correction():
    from baselines.dave2.collect import ScriptedExpert
    expert = ScriptedExpert()
    telemetry = np.zeros(12, dtype=np.float32)
    telemetry[3] = 1.0   # speed
    telemetry[8] = 0.5    # lateral error (drifted right)
    telemetry[9] = 0.1    # heading error
    action = expert.compute_action(telemetry)
    assert action[0] < -0.5  # should steer left (negative)

test("ScriptedExpert lateral correction", test_expert_lateral_correction)

def test_expert_braking():
    from baselines.dave2.collect import ScriptedExpert
    expert = ScriptedExpert()
    telemetry = np.zeros(12, dtype=np.float32)
    telemetry[3] = 3.0  # speed >> target (1.5 * 1.5 = 2.25)
    action = expert.compute_action(telemetry)
    assert action[2] > 0.0  # should brake
    assert action[1] == 0.0  # throttle cut

test("ScriptedExpert braking when too fast", test_expert_braking)


# ================================================================
#  5. KEYBOARD EXPERT (ramping logic, no terminal needed)
# ================================================================
print("\n=== 5. KeyboardExpert ===")

def test_keyboard_no_input():
    from baselines.dave2.collect import KeyboardExpert
    expert = KeyboardExpert(step_dt=0.1)
    action = expert.compute_action(np.zeros(12, dtype=np.float32))
    assert np.allclose(action, [0, 0, 0])

test("KeyboardExpert no keys = zero action", test_keyboard_no_input)

def test_keyboard_steer_ramp():
    from baselines.dave2.collect import KeyboardExpert
    expert = KeyboardExpert(step_dt=0.1)
    expert._key_left = True
    for _ in range(5):
        action = expert.compute_action(np.zeros(12))
    assert action[0] < -0.5  # ramped left

test("KeyboardExpert steering ramp", test_keyboard_steer_ramp)

def test_keyboard_decay():
    from baselines.dave2.collect import KeyboardExpert
    expert = KeyboardExpert(step_dt=0.1)
    expert._key_left = True
    for _ in range(5):
        expert.compute_action(np.zeros(12))
    expert._key_left = False
    for _ in range(10):
        action = expert.compute_action(np.zeros(12))
    assert abs(action[0]) < 0.1  # decayed toward zero

test("KeyboardExpert steering decay", test_keyboard_decay)

def test_keyboard_emergency_brake():
    from baselines.dave2.collect import KeyboardExpert
    expert = KeyboardExpert(step_dt=0.1)
    expert._brake = 1.0
    expert._throttle = 0.0
    expert._key_down = True
    action = expert.compute_action(np.zeros(12))
    assert action[2] >= 1.0
    assert action[1] == 0.0

test("KeyboardExpert emergency brake", test_keyboard_emergency_brake)

def test_keyboard_status_line():
    from baselines.dave2.collect import KeyboardExpert
    expert = KeyboardExpert(step_dt=0.1)
    hud = expert.status_line()
    assert "Steer" in hud and "Thr" in hud and "Brk" in hud
    expert._paused = True
    hud2 = expert.status_line()
    assert "PAUSED" in hud2

test("KeyboardExpert status_line", test_keyboard_status_line)


# ================================================================
#  6. ACKERMANN COMPUTER
# ================================================================
print("\n=== 6. AckermannComputer ===")

def test_ackermann_straight():
    from isaac_direct_env import AckermannComputer
    ack = AckermannComputer(wheelbase=0.33, track_width=0.28, wheel_radius=0.05)
    angles, velocities = ack.compute(steering_angle=0.0, speed=1.0)
    assert np.allclose(angles, [0, 0])
    expected_omega = 1.0 / 0.05  # speed / radius = 20 rad/s
    assert np.allclose(velocities, expected_omega)

test("Ackermann straight (equal wheels)", test_ackermann_straight)

def test_ackermann_turn():
    from isaac_direct_env import AckermannComputer
    ack = AckermannComputer(wheelbase=0.33, track_width=0.28, wheel_radius=0.05)
    angles, velocities = ack.compute(steering_angle=0.3, speed=1.0)
    # Right turn: right wheel (inner) should turn more
    assert angles[1] > angles[0]  # right > left angle
    # Inner wheel slower than outer
    assert velocities[1] < velocities[0]  # FR < FL

test("Ackermann right turn (inner/outer)", test_ackermann_turn)

def test_ackermann_left_turn():
    from isaac_direct_env import AckermannComputer
    ack = AckermannComputer(wheelbase=0.33, track_width=0.28, wheel_radius=0.05)
    angles, velocities = ack.compute(steering_angle=-0.3, speed=1.0)
    # Left turn: left wheel is inner
    assert abs(angles[0]) > abs(angles[1])

test("Ackermann left turn", test_ackermann_left_turn)

def test_ackermann_zero_speed():
    from isaac_direct_env import AckermannComputer
    ack = AckermannComputer(wheelbase=0.33, track_width=0.28, wheel_radius=0.05)
    angles, velocities = ack.compute(steering_angle=0.3, speed=0.0)
    assert np.allclose(velocities, 0.0, atol=0.01)

test("Ackermann zero speed", test_ackermann_zero_speed)


# ================================================================
#  7. MINI TRAINING LOOP (synthetic data, 3 epochs)
# ================================================================
print("\n=== 7. Mini Training Loop ===")

if HAS_TORCH:
    def test_mini_train():
        from baselines.dave2.model import DAVE2Net
        from baselines.dave2.dataset import DrivingDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic dataset
            _create_synthetic_dataset(tmpdir, num_frames=32)

            # Load dataset
            ds = DrivingDataset(
                data_dir=tmpdir,
                output_height=66, output_width=200,
                augment=False,
            )
            loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

            # Create model
            model = DAVE2Net(input_height=66, input_width=200, num_outputs=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            # Train 3 epochs
            model.train()
            losses = []
            for epoch in range(3):
                epoch_loss = 0.0
                for images, actions in loader:
                    pred = model(images)
                    target = actions[:, :1]  # steering only
                    loss = loss_fn(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss)

            # Loss should decrease (or at least not explode)
            assert losses[-1] < losses[0] * 10, f"Loss exploded: {losses}"
            assert not any(np.isnan(l) for l in losses), "NaN loss"

    test("Mini training loop (3 epochs, synthetic data)", test_mini_train)


# ================================================================
#  8. SYNTAX CHECK ALL FILES
# ================================================================
print("\n=== 8. Syntax Check All Files ===")

all_files = [
    "config/__init__.py",
    "config/experiment.py",
    "baselines/dave2/model.py",
    "baselines/dave2/dataset.py",
    "baselines/dave2/collect.py",
    "baselines/dave2/train.py",
    "isaac_direct_env.py",
    "isaac_ros2_env.py",
    "train_policy_ros2.py",
    "lane_detector.py",
    "inference_server_ros2.py",
    "policies/fusion_policy.py",
    "policies/hierarchical_policy.py",
    "losses/waypoint_losses.py",
    "wrappers/waypoint_tracking_wrapper.py",
]

for filepath in all_files:
    def make_syntax_test(fp):
        def _test():
            full = project_root / fp
            if not full.exists():
                raise FileNotFoundError(fp)
            with open(full) as f:
                ast.parse(f.read())
        return _test
    test(f"Syntax: {filepath}", make_syntax_test(filepath))


# ================================================================
#  SUMMARY
# ================================================================
total = passed + failed
print(f"\n{'='*50}")
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    print("\n  Fix the failures above before committing.")
    sys.exit(1)
else:
    print("\n  All clear — safe to commit and push.")
    sys.exit(0)
