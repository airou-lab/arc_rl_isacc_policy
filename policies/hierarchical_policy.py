"""
Hierarchical Path Planning Policy for Isaac Sim

Two-stage architecture for vision-based autonomous driving:
    1. Planning Head: predicts waypoint deviations from kinematic anchors
    2. Control Head: converts planned waypoints into Ackermann commands

Architecture:
    FusionFeatureExtractor (CNN + Physics) -> 268-dim features
        -> LSTM Actor -> 256-dim latent
        -> LSTM Critic -> 256-dim latent (separate)

    Actor path:
        latent_pi -> MLP Extractor -> planning_head -> waypoint deviations
        kinematic_anchors(obs_vec) + deviations -> final waypoints
        cat(latent_pi, norm_waypoints) -> control_head -> action_net -> [steer, throttle, brake]

    Critic path:
        latent_vf -> MLP Extractor -> value_net -> V(s)

    The planning head predicts DEVIATIONS from kinematic anchor points, not absolute positions. This gives the network an
    inductive bias towards smooth curved paths while allowing learned corrections.

Kinematic Anchors:
    Dynamic curved paths computed from turn_bias (navigation command) blended with current steering for smooth transitions.
    Supports continuous turn_bias in [-1, 1] instead of discrete {-1, 0, 1}.

Key Design Decisions:
    - Waypoint normalization (/ WAYPOINT_NORM_SCALE) is centralized in _compute_control_features() to prevent the forward/evaluate_actions
      inconsistency bug that caused training instability.
    - Planning head final layer is initialized near-zero so the policy starts by following kinematic anchors (straight/curved) paths before
      learning corrections.
    - predict_values() receives lstm_states as a raw (h, c) tuple from SB3-contrib, NOT an RNNStates object. This is a known SB3-contrib
      convention that caused AttributeError in earlier versions.
    - get_distribution() handles both RNNStates and bare tuple formats for compatibility with different SB3-contrib call sites.

Dependencies:
    - policies/fusion_policy.py (FusionFeatureExtractor)
    - sb3_contrib (RecurrentPPO, RecurrentActorCriticPolicy)
    - stable_baselines3
    - PyTorch

Used by:
    - train_policy_ros2.py (training entrypoint)
    - inference_server_ros2.py (deployment node)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Type, Any, Union
from gymnasium import spaces
import numpy as np

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution

class HierarchicalPathPlanningPolicy(RecurrentActorCriticPolicy):
    """
    Hierarchical recurrent policy with waypoint planning for autonomous driving.

    The policy predicts a sequence of waypoints representing the intended trajectory, then generates control actions conditioned on both the
    LSTM hidden state and the planned waypoints.
    """
    # Normalization constant for waypoints (meters -> normalized space)
    WAYPOINT_NORM_SCALE = 20.0

    # Telemetry vector indicies (must match isaac_ros2_env.py protocol)
    # These are identital and ensures the same 12-float observation vector.
    IDX_TURN_BIAS = 0 # Continuous turn command [-1, 1]
    IDX_RESERVED = 1 # Reserved (always 0)
    IDX_GOAL_DIST = 2 # Goal distance (masked to 0 during training)
    IDX_SPEED = 3 # Vehicle speed (m/s)
    IDX_YAW_RATE = 4 # Yaw rate (rad/s)
    IDX_LAST_STEER = 5 # Previous steering command
    IDX_LAST_THR = 6 # Previous throttle command
    IDX_LAST_BRK = 7 # Previous brake command
    IDX_LAT_ERR = 8 # Lateral error from path (m)
    IDX_HDG_ERR = 9 # Heading error from path (rad)
    IDX_KAPPA = 10 # Path curvature (1/m)
    IDX_DS = 11 # Distance traveled / cumulative odometry

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        # Hierarchical planning parameters
        num_waypoints: int = 5,
        waypoint_horizon: float = 2.5,
        repulsion_weight: float = 2.0,
        waypoint_loss_weight: float = 0.15,
        # Kinematic anchor parameters
        use_kinematic_anchors: bool = True,
        curvature_gain: float = 0.18,
        command_blend_factor: float = 0.6,
        steering_blend_factor: float = 0.4,
        progressive_curvature_exp: float = 1.15,
        max_deviation_meters: float = 8.0,
    ):
        """
        Initialize the hierarchical policy.

        Args:
            observation_space: Gymnasium observation space (Dict with 'image' and 'vec').
            action_space: Gymnasium action space (Box with 3 dims: steer, throttle, brake).
            lr_schedule: Learning rate schedule function.
            num_waypoints: Number of waypoints predicted ahead.
            waypoint_horion: Total forward distance (meters) spanned by waypoints.
            repulsion_weight: Weight for crash-avoidance auxiliary loss term.
            waypoint_loss_weight: Weight for waypoint auxiliary loss vs PPO loss.
            use_kinematic_anchors: If True, compute curved anchors from turn_bias + steering.
                                   If False use static straight-line anchors.
            curvature_gain: How aggressively anchors curve (radians per unit curvature signal)
            command_blend_factor: Weight given to turn_bias vs current steering.
            progressive_curvature_exp: Exponent for increasing curvature at distance.
            max_deviation_meters: Maximum learned deviation from anchor path.
        """

        # Store hierarchical parameters before super().__init__
        self.num_waypoints = num_waypoints
        self.waypoint_dim = num_waypoints * 2 # (x, y) per waypoint
        self.waypoint_horizon = waypoint_horizon
        self.repulsion_weight = repulsion_weight
        self.waypoint_loss_Weight = waypoint_loss_weight

        # Kinematic anchor config
        self.use_kinematic_anchors = use_kinematic_anchors
        self.curvature_gain = curvature_gain
        self.command_blend_factor = command_blend_factor
        self.steering_blend_factor = steering_blend_factor
        self.progressive_curvature_exp = progressive_curvature_exp

        # Head dimensions
        self.planning_hidden_dim = 128
        self.control_hidden_dim = 128

        # Compute spacing: e.g. horizon=2.5m, 5 waypoints -> 0.5m apart
        self.waypoint_spacing = waypoint_horizon / num_waypoints

        # Cap max deviation to be reasonable for the horizon
        self.max_deviation_meters = min(max_deviation_meters, waypoint_horizon * 0.8)

        if net_arch is None:
            net_arch = dict(pi=[64], vf=[64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        # Fix LSTM input dimension if FusionFeatureExtractor changed it
        if self.lstm_actor.input_size != self.features_dim:
            self.lstm_actor = nn.LSTM(
                self.features_dim,
                self.lstm_hidden_size,
                num_layers=self.n_lstm_layers,
                batch_first=False,
            )

        if self.enable_critic_lstm and self.lstm_critic.input_size != self.features_dim:
            self.lstm_criric = nn.LSTM(
                self.features_dim,
                self.lstm_hidden_size,
                num_layers=self.n_lstm_layers,
                batch_first=False,
            )

        # Build planning + control heads
        self._build_hierarchical_heads()

        # Buffer: straight-line anchors (fallback when kinematic anchors disabled)
        self.register_buffer("static_anchors", self._create_static_anchors())

        # Runtime state for logging / visualization
        self.last_waypoints = None
        self.last_anchors = None

    def _create_static_anchors(self) -> torch.Tensor:
        """Create straight-line anchors (0, dist) in vehicle frame."""
        anchors = torch.zeros(1, self.num_waypoints, 2)
        for i in range(self.num_waypoints):
            anchors[0, i, 0] = 0.0 # X = 0 (center)
            anchors[0, i, 1] = (i + 1) * self.waypoint_spacing # Y = forward
        return anchors

    def _build_hierarchical_heads(self) -> None:
        """Build the planning and control neural network heads."""
        if self.mlp_extractor is not None:
            head_input_dim = self.mlp_extractor.latent_dim_pi
        else:
            head_input_dim = self.lstm_output_dim
        # Planning Head: LSTM features -> waypoint deviations
        self.planning_head = nn.Sequential(
            nn.Linear(head_input_dim, self.planning_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.planning_hidden_dim, self.planning_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.planning_hidden_dim // 2, self.waypoint_dim),
        )

        # Initialize final layer near-zero so policy starts following anchors
        with torch.no_grad():
            self.planning_head[-1].weight.mul_(0.01)
            self.planning_head[-1].bias.fill_(0.0)

        # Control Head: LSTM features + planned waypoints -> control features
        control_input_dim = head_input_dim + self.waypoint_dim
        self.control_head = nn.Sequential(
            nn.Linear(control_input_dim, self.control_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.control_hidden_dim, self.control_hidden_dim // 2),
            nn.ReLU(),
        )

        # Final action projection
        self.action_net = nn.Linear(
            self.control_hidden_dim // 2, self.action_space.shape[0]
        )

        # Value net (standard, only if MLP extractor is not handling it)
        if self.mlp_extractor is None:
            self.value_net = nn.Linear(self.lstm_output_dim, 1)


    def _compute_kinematic_anchors(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute curved anchor points from turn command and current steering.

        This creates an inductive bias toward continuing the current maneuver while respecting
        the high-level navigation intent from turn_bias.

        Args:
            obs_vec: (batch_size, 12) telemetry vector.

        Returns:
            anchors: (batch_size, num_waypoints, 2) in vehicle frame [X=lateral, Y=forward].
        """
        batch_size = obs_vec.shape[0]
        device = obs_vec.device

        # Extract relevant signals
        turn_bias = obs_vec[:, self.IDX_TURN_BIAS] # [-1, 1] navigation command
        last_steer = obs_vec[:, self.IDX_LAST_STEER] # [-1, 1] current steering

        # Adaptive blending: strong command -> trust navigation intent,
        #                    weak command -> trust current steering for smooth driving
        command_strength = torch.abs(turn_bias)
        adaptive_command_weight = self.command_blend_factor + 0.3 * command_strength
        adaptive_steer_weight = 1.0 - adaptive_command_weight

        # Compute effective curvature signal
        effective_curvature = (
            adaptive_command_weight * turn_bias
            + adaptive_steer_weight * last_steer
        )
        effective_curvature = torch.clamp(effective_curvature, -1.0, 1.0)

        # Generate curved anchors
        anchor = torch.zeros(batch_size, self.num_waypoints, 2, device=device)

        for i in range (self.num_waypoints):
            dist = (i + 1) * self.waypoint_spacing

            # Progressive curvature: curves more at distance
            progressive_factor = (i + 1) ** self.progressive_curvature_exp
            angle = effective_curvature * self.curvature_gain * progressive_factor

            # Vehicle frame: X = lateral (+ right), Y = forward (+ ahead)
            anchor[:, i, 0] = dist * torch.sin(angle) # Lateral
            anchor[:, i, 1] = dist * torch.cos(angle) # Forward

        return anchor

    def _compute_waypoints(self, latent_pi: torch.Tensor, obs_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute final waypoints as anchors + learned deviations.

        Args:
            latent_pi: (batch_size, latent_dim) LSTM output.
            obs_vec: (batch_size, 12) telemetry vector.

        Returns:
            waypoints: (batch_size, num_waypoints, 2) positions in meters.
        """
        batch_size = latent_pi.shape[0]

        # Get anchors (kinematic or static)
        if self.use_kinematic_anchors:
            anchors = self._compute_kinematic_anchors(obs_vec)
        else:
            anchors = self.static_anchors.expand(batch_size, -1, -1).clone()

        self.last_anchors = anchors # Save for visualization / debugging

        # Predict deviations from anchors
        deviations = self.planning_head(latent_pi).reshape(
            -1, self.num_waypoints, 2
        )

        # Constrain deviations (reduced since anchors handle the major curve)
        deviations = torch.tanh(deviations) * self.max_deviation_meters

        # Final waypoints = anchors + learned corrections
        return anchors + deviations

    def _compute_control_features(self, latent_pi: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Fuse LSTM memory with planned waypoints for control.

        CRITICAL: All waypoint normalization happens HERE to guarentee consistency between forward(), evaluate_actions() and
                  get_distribution(). This was the root cause of the training instability bug in the Unity codebase.

        Args:
            latent_pi: (batch_size, latent_dim) LSTM output.
            waypoints: (batch_size, num_waypoints, 2) waypoint positions.

        Returns: control_features: (batch_size, num_waypoints, 2) waypoint positions.
        """
        wp_flat = waypoints.reshape(-1, self.waypoint_dim)
        wp_norm = wp_flat / self.WAYPOINT_NORM_SCALE

        return self.control_head(torch.cat([latent_pi, wp_norm], dim=-1))

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        """Full forward pass. Called during rollout collection."""
        features = self.extract_features(obs)

        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.enable_critic_lstm:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        else:
            latent_vf = vf_features
            lstm_states_vf = lstm_states.vf

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Hierarchical planning step
        obs_vec = obs["vec"] if isinstance(obs, dict) else obs

        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        self.last_waypoints = waypoints

        control_features = self._compute_control_features(latent_pi, waypoints)

        distribution = self._get_action_dist_from_latent(control_features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return (actions, values, log_prob, RNNStates(pi=lstm_states_pi, vf=lstm_states_vf),)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """
        Evaluate actions for PPO gradient update.

        Uses _compute_control_features() for consistent waypoint normalization, matching forward() exactly.
        """
        features = self.extract_features(obs)

        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.enable_critic_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        else:
            latent_vf = vf_features

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        obs_vec = obs["vec"] if isinstance (obs, dict) else obs

        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        control_features = self._compute_control_features(latent_pi, waypoints)

        distribution = self._get_action_dist_from_latent(control_features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def get_distribution(self, obs, lstm_states, episode_starts):
        """
        Get action distribution for deterministic action selection.

        Handles both RNNStates and bare tuple formats for compatibility with different SB3-contrib call sites.
        """
        if isinstance(lstm_states, tuple) and not hasattr(lstm_states, "pi"):
            lstm_states = RNNStates(pi=lstm_states, vf=lstm_states)

        features = self.extract_features(obs)
        latent_pi, lstm_sates_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)

        obs_vec = obs["vec"] if isinstance(obs, dict) else obs

        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        self.last_waypoints = waypoints

        control_features = self._compute_control_features(latent_pi, waypoints)

        # Return bare tuple - SB3-contrib expects this from get_distribution
        return (self._get_action_dist_from_latent(control_features), lstm_states_pi,)

    def predict_values(self, obs, lstm_states, episode_starts):
        """
        Predict values for given observations.

        IMPORTANT: SB3-contrib calls this with lstm_states already being the vf component (a raw (h, c) tuple),
                   NOT the full RNNStates object. The call site in collect_rollouts does:
                       values = self.policy.predict_values(obs, lstm_states.vf, episode_starts)
                   so we use lstm_states directly as the critic LSTM state.
        """
        features = self.extract_features(obs)

        if self.share_features_extractor:
            vf_features = features
        else:
            _, vf_features = features

        if self.enalbe_critic_lstm:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states, episode_starts, self.lstm_critic)
        else:
            latent_vf = vf_features

        if self.mlp_extractor is not None:
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        return self.value_net(latent_vf)

    def predict_waypoints(self, obs: Dict[str, torch.Tensor], state: Optional[RNNStates] = None,
        episode_start: Optional[np.ndarray] = None, deterministic: bool = False,
        ) -> Tuple[torch.Tensor, Optional[RNNStates]]:
        """
        Predict waypoints only (for visualization during evaluation).

        Returns:
            waypoints: (batch, num_waypoints, 2) in meters, vehicle frame.
            new_state: Updated LSTM states.
        """

        self.set_training_mode(False)

        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs_tensor = self.obs_to_tensor(obs)[0]
            else:
                obs_tensor = obs

            if state is None:
                batch_size = 1
                if isinstance(obs_tensor, dict) and "image" in obs_tensor:
                    img = obs_tensor["image"]
                    batch_size = img.shape[0] if img.ndim > 3 else 1
                state = self.get_initial_states(base_size)

            if episode_start is None:
                episode_start = np.array([False])
            episode_starts_tensor = torch.Tensor(episode_start, dtype=torch.float32, device=self.device)

            features = self.extract_features(obs_tensor)
            if self.share_features_extractor:
                pi_features = features
            else:
                pi_features = features[0]

            latent_pi, new_lstm_states_pi = self._process_sequence(pi_features, state.pi, episode_starts_tensor, self.lstm_actor)

            if self.mlp_extractor is not None:
                latent_pi = self.mlp_extractor.forward_actor(latent_pi)

            obs_vec = (
                obs_tensor["vec"] if isinstance(obs_tensor, dict) else obs_tensor
            )
            waypoints = seld._compute_waypoints(latent_pi, obs_vec)

            new_state = RNNStates(pi=new_lstm_states_pi, vf=state.vf)
            return waypoints, new_state

    def get_waypoints(self) -> Optional[torch.Tensor]:
        """Get last computed waypoints for logging / visualization."""
        return self.last_waypoints
