"""
Expert Data Collection for Behavioral Cloning

Collects (image, action) pairs from an expert controller driving in simulation.
The collected data is saved in a format compatible with DrivingDataset for training DAVE-2.

Expert sources:
    1. "scripted" - PD controller following a known reference path.
       This is the default and most reproducible option. The path is defined by waypoints in the Isaac Sim scene.
    2. "teleop" - Human driving via keyboard (FUTURE: or bluetooth controller).
       Higher quality demonstrations but harder to reproduce.
    3. "pid" - PID controller tracking lane center from lane_detector.
       Self-referential (uses our own CV module) but demonstrates what perfect lane-following looks like.

Output format:
    {output_dir}/
    |-- metadata.yaml        # Collection config + statistics
    |-- frames/
    |   |-- frame_000000.png # Saved at collection_hz rate
    |   |-- frame_000001.png
    |   |-- ...
    |-- labels.csv           # frame_id, steering, throttle, brake, speed

The collector saves at a fixed rate (default 10 Hz) independent of the sim's physics rate. This ensures consistent
temporal spacing in the dataset regardless of sim performance.

Enviroment Decoupling:
    The core collection functions (collect_from_gym_env, collect_teleop_from_gym_env) accept ANY Gymnasium
    environment that follows the Dict observation contract:
        obs["image"] -> (H, W, 3) uint8 RGB
        obs["vec"]   -> (12,) float32 telemetry

    This means the same collector works with:
        - IsaacROS2Env (current ROS2-couple env, for quick testing)
        - Future direct-API Isaac Sim env (BaseSimEnv adapter)
        - Future CARLA env
        - Any mock/test env that follows the contract

Usage:
    # Collect 5 minutes of scripted expert data:
    python -m baselines.dave2.collect \\
        --output data/expert_001 \\
        --episodes 20 \\
        --duration 300 \\
        --expert scripted

    # Collect with keyboard teleop:
    python -m baselines.dave2.collect \\
        --output data/expert_teleop \\
        --episodes 10 \\
        --expert teleop

Keyboard Controls (--expert teleop):

    |===================================================|
    | W / up arrow        Throttle (increase while held)|
    | S / down arrow      Brake (increase while held)   |
    | A / left arrow      Steer left                    |
    | D / right arrow     Steer right                   |
    | SPACE               Emergency brake (full stop)   |
    | R                   Reset steering to center      |
    | Q                   Quit collection and save      |
    | P                   Pause/resume recording        |
    |===================================================|

    Controls use smooth ramping: holding a key gradually increases the command value,
                                 releasing it decays back to zero.
    This produces smoother demonstrations than bang-bang keyboard input.

    Steering: ramps at 2.0/sec while held, decays at 3.0/sec on release.
    Throttle: ramps at 1.5/sec while held, decays at 2.0/sec on release.
    Brake:    applied instantly, decays at 2.0/sec on release

    The terminal must be in focus for key capture. The collector uses raw terminal mode (termios) on Linux
    - your terminal will be restored to normal on exit or Ctrl+C.

    Tips for good teleop data:
    - Drive at a steady moderate speed (1-2 m/s)
    - Make smooth turns, avoid jerky corrections
    - Include recovery maneuvers (drift off-center, then correct)
    - COllect at least 10-15 minutes for reasonable BC training

Dependencies:
    - OpenCV for image saving
    - NumPy
    - PyYAML (for metadata)
    - termios + tty (Linux stdlib, for keyboard teleop)
    - Gymnasium-compatible environment (passed in by caller)

Author: Aaron Hamil
Date: 03/02/26
Updated: 03/03/26
"""

import argparse
import csv
import time
import logging
import threading
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import numpy as np
import cv2

logger = logging.getLogger(__name__)

#=============================================================================================#
#                                   SCRIPTED EXPERT CONTROLLER                                #
# A simple PD controller that follows the reference path.                                     #
# This doesn't need ROS2 - it takes the current vehicle state and outputs Ackermann commands. #
#=============================================================================================#

class ScriptedExpert:
    """
    PD controller that generates expert steering + throttle commands from the vehicle's current lateral error and
    heading error.

    This reads the same telemetry vector that the RL environment provides, specifically the lateral_error (idx 8)
    and heading_error (idx 9) fields.

    The controller is intentionally simple; its not trying to be optimal, just good enough to generate demonstrations
    for behavioral cloning. A BC model that matches this controller's performance has learned basic lane following.
    Our RL model should exceed it.
    """

    def __init__(
        self,
        kp_steer: float = 2.0,
        kd_steer: float = 0.5,
        target_speed: float = 1.5,
        kp_throttle: float = 0.5,
    ):
        """
        Args:
            kp_steer: Proportional gain for steering on lateral_error.
            kd_steer: Derivative gain for steering on heading error.
            target_speed: Desired cruising speed in m/s.
            kp_throttle: Proportional gain for speed control.
        """
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.target_speed = target_speed
        self.kp_throttle = kp_throttle

    def compute_action(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Compute expert action from current telemetry.

        Args:
            telemetry: 12-element float array matching the vec observation protocol defined in config/experiment.py

        Returns:
            [steering, throttle, brake] in [-1, 1] range.
        """
        # Extract relevant signals (matching TELEMETRY_INDICES)
        speed = telemetry[3]        # IDX_SPEED
        lateral_err = telemetry[8]  # IDX_LAT_ERR
        heading_err = telemetry[9]  # IDX_HDG_ERR

        # PD steering: correct lateral offset and heading angle
        steering = -(self.kp_steer * lateral_err + self.kd_steer * heading_err)
        steering = np.clip(steering, -1.0, 1.0)

        # P throttle: maintain target speed
        speed_err = self.target_speed - speed
        throttle = np.clip(self.kp_throttle * speed_err, 0.0, 1.0)

        # Brake if going too fast
        brake = 0.0
        if speed > self.target_speed * 1.5:
            brake = 0.3
            throttle = 0.0

        return np.array([steering, throttle, brake], dtype=np.float32)


#===========================================================================================
#                                   KEYBOARD TELEOP EXPERT                                 #
# Non-blocking keyboard reader on a background thread.                                     #
# Uses raw terminal mode (termios) on Linux for instant key reads without requiring Enter. #
# Terminal is restored on cleanup.                                                         #
#==========================================================================================#

class KeyboardExpert:
    """
    Human-driven expert controller via keyboard input.

    Reads keys on a background daemon thread using raw terminal mode (Linux termios).
    The main env loop calls compute_action() each step, which returns the current command state after applying
    smooth ramping and decay.

    Lifecycle:
        expert = KeyboardExpert()
        expert = expert.compute_action(telemetry) # begins listening (prints controls)
        action = expert.compute_action(telemetry) # call each step
        expert.stop()                             # restores terminal, joins thread

    The telemetry argument is accepted for interface compatibility with ScriptedExpert but is not
    used - the human is the controller.
    """

    # Ramp rates (units per second) - tuned for 10 Hz step rate
    STEER_RAMP_RATE = 2.0     # How fast steering increases while held
    STEER_DECAY_RATE = 3.0    # How fast steering returns to center on release
    THROTTLE_RAMP_RATE = 1.5  # How fast throttle increases while held
    THROTTLE_DECAY_RATE = 2.0 # How fast throttle decays on release
    BRAKE_RAMP_RATE = 3.0     # Brake ramps quickly for safety
    BRAKE_DECAY_RATE = 2.0    # Brake decays on release

    def __init__(self, step_dt: float = 0.1):
        """
        Args:
            step_dt: Expected time between compute_action() calls in (seconds).
                Used to scale ramp rates. Default 0.1 = 10 Hz collection.
        """
        self.step_dt = step_dt

        # Current command state (what compute_action returns)
        self._steering = 0.0
        self._throttle = 0.0
        self._brake = 0.0

        # Key-held state (set True while key is pressed)
        self._key_left = False
        self._key_right = False
        self._key_up = False
        self._key_down = False

        # Control flags
        self._running = False
        self._paused = False
        self.quit_requested = False

        # Thread and terminal state
        self._thread = None
        self._old_terminal_settings = None
        self._lock = threading.Lock()

    def start(self):
        """
        Begin keyboard listening. Switches terminal to raw mode and starts the background key-reader thread.

        Call stop() when done to restore the terminal.
        """
        self._running = True
        self.quit_requested = False

        # Print controls banner
        print("\n" + "=" * 55)
        print("  KEYBOARD TELEOP - Human Expert Data Collection")
        print("=" * 55)
        print("  W / ↑    Throttle     A / ←    Steer left")
        print("  S / ↓    Brake        D / →    Steer right")
        print("  SPACE    Emergency Stop  R      Reset steering")
        print("  P        Pause/resumt    Q      Quit & save")
        print("=" * 55)
        print("  Recording... (press P to pause)\n")

        # Start background key reader
        self._thread = threading.Thread(
            target=self._key_reader_loop,
            daemon=True,
            name="keyboard-teleop",
        )
        self._thread.start()

    def stop(self):
        """Stop keyboard listening and restore terminal settings."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Restore terminal (handled inside _key_reader_loop's finally block,
        # but calls explicitly in case thread died unexpectedly)
        self._restore_terminal()

    def compute_action(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Return current smoothed action from keyboard state.

        Called each env step (~10 Hz). Applies ramping to whatever keys are currently held,
        and decay to released keys. The telemetry arg is accepted for interface compatibility but not used.

        Args:
            telemetry: 12-element float array (unused - human is driving).

        Returns:
            [steering, throttle, brake] in [-1, 1] range.
        """
        with self._lock:
            dt = self.step_dt

            # === Steering: ramp toward direction, decay towards center ===
            if self._key_left and not self._key_right:
                # Steer left (negative steering)
                self._steering -= self.STEER_RAMP_RATE * dt
            elif self._key_right and not self._key_left:
                # Steer right (positive steering)
                self._steering += self.STEER_RAMP_RATE * dt
            else:
                # No steering key held - decay toward center
                if abs(self._steering) < self.STEER_DECAY_RATE * dt:
                    self._steering = 0.0
                elif self._steering > 0:
                    self._steering -= self.STEER_DECAY_RATE * dt
                else:
                    self._steering += self.STEER_DECAY_RATE * dt

            # === Throttle: ramp while held, decay on release ===
            if self._key_up:
                self._throttle += self.THROTTLE_RAMP_RATE * dt
            else:
                self._throttle -= self.THROTTLE_DECAY_RATE * dt

            # === Brake: ramp while held, decay on release ===
            if self._key_down:
                self._brake += self.BRAKE_RAMP_RATE * dt
                self._throttle = 0.0 # Can't throttle and brake simultaneously
            else:
                self._brake -= self.BRAKE_DECAY_RATE * dt

            # Clamp all values
            self._steering = float(np.clip(self._steering, -1.0, 1.0))
            self._throttle = float(np.clip(self._throttle, 0.0, 1.0))
            self._brake = float(np.clip(self._brake, 0.0, 1.0))

        return np.array(
            [self._steering, self._throttle, self._brake], dtype=np.float32
        )

    @property
    def is_paused(self) -> bool:
        """Whether frame recording is paused (P key toggle)."""
        return self._paused

    def status_line(self) -> str:
        """One-line HUD string for terminal output during collection."""
        pause_str = " [PAUSED]" if self._paused else ""
        return (
            f"Steer: {self._steering:+.2f}  "
            f"Thr: {self._throttle:.2f}  "
            f"Brk: {self._brake:.2f}{pause_str}"
        )

    # === Background Key Reader ===
    def _key_reader_loop(self):
        """
        Background thread: read keys in raw terminal mode.

        Uses termios to switch stdin to raw mode (no echo, no line buffering)
        so we get each keypress instantly. Arrow keys arrive as 3-byte escape sequences (ESC [ A/B/C/D]).

        The terminal is restored in the finally block even if the thread crashes or is interrupted.
        """
        try:
            import tty
            import termios
            import select
        except ImportError:
            logger.error(
                "termios/tty not available - keyboard teleop requires Linux. "
                "On Windows, use a gamepad or the scripted expert instead."
            )
            self._running = False
            return

        # Save current terminal settings so we can restore them
        fd = sys.stdin.fileno()
        try:
            self._old_terminal_settings = termios.tcgetattr(fd)
        except termios.error:
            logger.error(
                "Cannot access terminal settings. Are you running in a "
                "terminal with stdin attached? Keyboard teleop won't work "
                "in non-interactive environments (e.g., piped input, IDE "
                "run configs without terminal allocation)."
            )
            self._running = False
            return

        try:
            # Switch to raw mode: instant key reads, no echo
            tty.setraw(fd)

            while self._running:
                # select() with 50ms timeout - responsive but not busy-wait
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)

                if not readable:
                    # No key pressed this cycle - release all held keys
                    # (raw mode doesn't give us key-up events, so we treat
                    # "no key this cycle" as "key released")
                    with self._lock:
                        self._key_left = False
                        self._key_right = False
                        self._key_up = False
                        self._key_down = False
                    continue

                ch = sys.stdin.read(1)

                if ch == 'q' or ch == 'Q':
                    logger.info("Quit requested via keyboard")
                    self.quit_requested = True
                    self._running = False
                    break

                with self._lock:
                    self._handle_key(ch)

        finally:
            # ALWAYS restore terminal - even on crash or KeyboardInterrupt
            self._restore_terminal()

    def _handle_key(self, ch: str):
        """
        Process a single keypress. Must be called with self._lock held.

        Handles both WASD and arrow keys. Arrow keys arrive as 3-byte escape sequences:
        ESC (\\x1b) then '[' then A/B/C/D
        """
        # === Arrow key escape sequences ===
        if ch == '\x1b':
            # Read the next two bytes of the escape equence
            try:
                import select as _sel
                # Check if more bytes are availale (they should be for arrow keys. but not for bare ESC press)
                r, _, _ = _sel.select([sys.stdin], [], [], 0.01)
                if r:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        r2, _, _ = _sel.select([sys.stdin], [], [], 0.01)
                        if r2:
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A':    # Up arrow
                                self._key_up = True
                                return
                            elif ch3 == 'B':  # Down arrow
                                self._key_down = True
                                return
                            elif ch3 == 'C':  # Right arrow
                                self._key_right = True
                                return
                            elif ch3 == 'D':  # Left arrow
                                self._key_left = True
                                return
            except Exception:
                pass
            return # Bare ESC or unrecognized sequence

        # === WASD keys ===
        lower = ch.lower()
        if lower == 'w':
            self._key_up = True
        elif lower == 's':
            self._key_down = True
        elif lower == 'a':
            self._key_left = True
        elif lower == 'd':
            self._key_right = True

        # === Special Keys ===
        elif ch == ' ':
            # Emergency brake - instant full brake, zero throttle
            self._brake = 1.0
            self._throttle = 0.0
            self._key_down = True
        elif lower == 'r':
            # Reset steering to center
            self._steering = 0.0
        elif lower == 'p':
            # Toggle pause
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "RECORDING"
            # Print outside lock would be better but this works fine for feedback
            print(f"\r  [{state}]", end="", flush=True)

    def _restore_terminal(self):
        """Restore terminal to original settings (cooked mode)."""
        if self._old_terminal_settings is not None:
            try:
                import termios
                fd = sys.stdin.fileno()
                termios.tcsetattr(
                    fd, termios.TCSADRAIN, self._old_terminal_settings
                )
                self._old_terminal_settings = None
            except Exception:
                pass # Best effort - don't crash on cleanup

#============================================================#
#                      DATA COLLECTOR                        #
# Env-agnostic: saves images + labels from numpy arrays.     #
# No ROS2 dependency - works with any observation source.    #
#============================================================#

class DataCollector:
    """
    Collects expert driving data and saves to disk.

    This is a pure data recorder - it takes numpy arrays (image + telemetry + action)
    and write PNG frames + CSV labels. It has zero knowledge of where the data comes from
    (ROS2, direct API, replay, mock env, etc.).

    The expert controller is optional - if provided, the collector computes the expert action from telemetry.
    If not, the caller passes the actio directly via collect_from_arrays_with_action().
    """

    def __init__(
        self,
        output_dir: str,
        collection_hz: int = 10,
        expert: Optional[ScriptedExpert] = None,
        img_width: int = 160,
        img_height: int = 90,
    ):
        """
        Args:
            output_dir: Directory to save collected data.
            collection_hz: Rate at which to save frames (Hz).
            expert: Expert controller instance. Defaults to ScriptedExpert.
                Only used by collect_from_arrays() - collect_from_arrays_with_action()
                ignores this and uses the caller-provided action.
            img_width: Expected camera image width (for metadata only).
            img_height: Expected camera image height (for metadata only).
        """
        self.output_dir = Path(output_dir)
        self.collection_hz = collection_hz
        self.expert = expert or ScriptedExpert()
        self.img_width = img_width
        self.img_height = img_height

        # Create output directory structure
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._frame_count = 0
        self._csv_writer = None
        self._csv_file = None

    def collect_from_arrays(
        self,
        image: np.ndarray,
        telemetry: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Record a frame using the internal expert to compute the action.

        This is used by scripted collection where the expert controller determines the action from telemetry.

        Args:
            image: (H, W, 3) RGB uint8 camera image.
            telemetry: (12,) float32 telemetry vector.

        Returns:
            Expert action [steering, throttle, brake].
        """
        action = self.expert.compute_action(telemetry)
        self._save_frame(image, action, telemetry)
        return action

    def collect_from_arrays_with_action(
        self,
        image: np.ndarray,
        telemetry: np.ndarray,
        action: np.ndarray,
    ):
        """
        Record a frame with a caller-provided action.

        This is used by teleop collection where the human provides the action and we just record
        what they did.

        Args:
            image: (H, W, 3) RGB uint8 camera image.
            telemetry: (12,) float32 telemetry vector.
            action: (3,) float32 [steering, throttle, brake] from the human.
        """
        self._save_frame(image, action, telemetry)

    def _save_frame(
        self,
        image: np.ndarray,
        action: np.ndarray,
        telemetry: np.ndarray,
    ):
        """Internal: write one image + label row to disk."""
        frame_id = f"frame_{self._frame_count:06d}"

        # Save image as PNG (lossless), RGB -> BGR for OpenCV
        frame_path = self.frames_dir / f"{frame_id}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Write label row
        if self._csv_writer is None:
            self._csv_file = open(self.output_dir / "labels.csv", "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame_id", "steering", "throttle", "brake", "speed"
            ])

        self._csv_writer.writerow([
            frame_id,
            f"{action[0]:.6f}",
            f"{action[1]:.6f}",
            f"{action[2]:.6f}",
            f"{telemetry[3]:.6f}", # speed
        ])

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.info(f"Collected {self._frame_count} frames")

    def save_metadata(self, duration: float = 0.0, expert_name: str = ""):
        """Save collection metadata as YAML."""
        import yaml

        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_frames": self._frame_count,
            "collection_hz": self.collection_hz,
            "duration_seconds": duration,
            "image_resolution": [self.img_width, self.img_height],
            "expert_type": expert_name or self.expert.__class__.__name__,
            "expert_params": {
                "kp_steer": getattr(self.expert, "kp_steer", None),
                "kd_steer": getattr(self.expert, "kd_steer", None),
                "target_speed": getattr(self.expert, "target_speed", None),
            },
        }

        with open(self.output_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(
            f"Collection complete: {self._frame_count} frames "
            f"saved to {self.output_dir}"
        )

    def close(self):
        """Flush and close output files."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


#=================================================================================#
#               ENV-AGNOSTIC COLLECTION FUNCTIONS                                 #
# These accept ANY Gymnasium env with Dict obs {"image": ..., "vec": ...}.        #
# No ROS2 imports. No simulator-specific code. The CLI wires in the concrete env. #
#=================================================================================#

def collect_from_gym_env(
    env,
    output_dir: str,
    expert=None,
    num_episodes: int = 20,
    max_steps_per_episode: int = 300,
    collection_hz: int = 10,
):
    """
    Collect scripted expert demonstrations from any Gymnasium environment.

    This is the core collection loop, fully decoupled from any specific simulator.
    It works with any env that provides Dict observations with 'image' and 'vec' keys.

    Args:
        env: Gymnasium environment instance. Must provide:
            obs["image"] -> (H, W, 3) uint8 RGB
            obs["vec"]   -> (12,) float32 telemetry
            env.step(action) -> standard Gymnasium 5-tuple
            env.reset()  -> (obs, info)
        output_dir: Where to save the dataset.
        expert: Expert controller with compute_action(telemetry) -> action.
                Defaults to ScriptedExpert().
        num_episodes: Number of driving episodes to collect.
        max_steps_per_episode: Maximum number of steps per episode.
        collection_hz: Frames per second to record (for metadata only - actual rate is determined by env.step() speed).
    """
    if expert is None:
        expert = ScriptedExpert()

    collector = DataCollector(
        output_dir=utput_dir,
        collection_hz=collection_hz,
        expert=expert,
    )

    total_frames = 0
    start_time = time.time()

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                # Expert computes action from telemetry, collector saves frame + action
                telemetry = obs["vec"]
                action = collector.collect_from_arrays(obs["image"], telemetry)

                # Step environment with expert action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                total_frames += 1

            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"{step} steps, {total_frames} total frames"
            )

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user.")
    finally:
        duration = time.time() - start_time
        collector.save_metadata(duration=duration)
        collector.close()

    logger.info(
        f"Collected {total_frames} frames across {num_episodes} episodes "
        f"in {duration:.1f}s"
    )


def collect_teleop_from_gym_env(
    env,
    output_dir: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 300,
    collection_hz: int = 10,
):
    """
    Collect human expert demonstrations via keyboard teleop from any Gymnasium environment.

    Same contract as collect_from_gym_env but uses KeyboardExpert for human input.

    A status HUD is printed each step showing current steering, throttle, and brake values.
    Press P to pause recording (the vehicle keeps driving but frames aren't saved).
    Press Q to quit and save the dataset.

    Args:
        env: Gymnasium environment instance (same contract as collect_from_gym_env).
        output_dir: Where to save the dataset.
        num_episodes: Number of driving episodes to collect.
        max_steps_per_episode: Maximum steps per episode.
        collection_hz: Frames per second to record.
    """
    expert = KeyboardExpert(step_dt=1.0 / collection_hz)
    collector = DataCollector(output_dir=output_dir, collection_hz=collection_hz)

    total_frames = 0
    start_time = time.time()

    # Start keyboard listener (switches terminal to raw mode)
    expert.start()

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            print(f"\n Episode {episode + 1}/{num_episodes}")

            while not done and step < max_steps_per_episode:
                # Check if human pressed Q
                if expert.quit_requested:
                    logger.info("Quit requested - saving and exiting.")
                    break

                # Get human action from keyboard state
                telemetry = obs["vec"]
                action = expert.compute_action(telemetry)

                # Record frame (unless paused)
                if not expert.is_paused:
                    collector.collect_from_arrays_with_action(
                        obs["image"], telemetry, action
                    )
                    # Override recorded action with human's action
                    # (the collector calls expert.compute_action internally
                    # via ScriptedExpert, but we want the human's action)
                    total_frames += 1

                # Print HUD status line (overwrites same line)
                hud = expert.status_line()
                print(f"\r  Ep{episode + 1} Step{step:4d} | {hud} | "
                      f"Frames: {total_frames}", end="", flush=True)

                # Step environment with human's action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            if expert.quit_requested:
                break

    except KeyboardInterrupt:
        print() # Newline after the HUD line
        logger.info("Collection interrupted by Ctrl+C.")
    finally:
        # CRITICAL: restore terminal before any other cleanup
        expert.stop()
        print() # Clean newline after raw mode

        duration = time.time() - start_time
        collector.save_metadata(duration=duration, expert_name="KeyboardExpert")
        collector.close()
        env.close()

    logger.info(
        f"Teleop collection complete: {total_frames} frames "
        f"in {duration:.1f}s"
    )

#===============================================================================#
#          CONVENIENCE: CREATE DEFAULT ENV                                      #
# The CLI uses this to create an IsaacROS2Env if no env is passed.              #
# This is the only place with a ROS2 import - isolated from the core functions. #
#===============================================================================#
def _create_default_env(max_steps_per_episode: int = 300, collection_hz: int = 10):
    """
    Create the default IsaacROS2Env for CLI usage.

    This is the only function in the file that imports ROS2. It exists as a convenience for the CLI entrypoint.
    When the abstract env layer (BaseSimEnv + registry) is built, this will be replaced with
    SimFactory.create(config.sim).

    Returns:
        A Gymnasium-compatible environment instance.
    """
    try:
        from isaac_ros2_env import IsaacROS2Env, IsaacROS2Config
    except ImportError:
        logger.error(
            "isaac_ros2_env not found. Make sure ROS2 environment "
            "is available and Isaac Sim is running.\n"
            "Alternatively, pass a Gymnasium env directly to "
            "collect_from_gym_env() or collect_teleop_from_gym_env()."
        )
        raise

    config = IsaacROS2Config(
        img_width=160,
        img_height=90,
        episode_timeout=max_steps_per_episode / collection_hz,
    )

    return IsaacROS2Env(config=config)


#================#
# CLI ENTRYPOINT #
#================#

def main():
    parser = argparse.ArgumentParser(
        description="Collect expert driving data for behavioral cloning"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of driving episodes to collect"
    )
    parser.add_argument(
        "--max-steps", type=int, default=300,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--hz", type=int, default=10,
        help="Collection rate in Hz"
    )
    parser.add_argument(
        "--expert", type=str, default="scripted",
        choices=["scripted", "teleop", "pid"],
        help="Expert controller type (see module docstring for keyboard controls)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create the default environment (ROS2-coupled, for now)
    # When BaseSimEnv + registry exists, this becomes:
    #  config = ExperimentConfig.load(args.config)
    #  env = SimFactory.create(config.sim)
    env = _create_default_env(
        max_steps_per_episode=args.max_steps,
        collection_hz=args.hz,
    )

    try:
        if args.expert == "scripted":
            collect_from_gym_env(
                env=env,
                output_dir=args.output,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                collection_hz=args.hz,
            )
        elif args.expert == "teleop":
            collect_teleop_from_gym_env(
                env=env,
                output_dir=args.output,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                collection_hz=args.hz,
            )
        elif args.expert == "pid":
            logger.error("PID expert requires lane_detector integration - not yet implemented")
            raise NotImplementedError(
                "PID expert needs lane_detector.py to provide real-time "
                "lateral offset for the PID conreoller. This will be added "
                "when the direct-API environment is implemented."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
