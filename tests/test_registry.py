"""
Tests for the Simulator Environment Registry

Validates the factory/registry pattern independently of any simulator.
Uses mock gym.Env subclasses to test registration, lookup, factory
creation, lazy import fallback, and error handling.

Run:
    python -m pytest tests/test_registry.py -v
    # or standalone:
    python tests/test_registry.py

Author: Aaron Hamil
Date: 03/31/26
"""

import sys
import pytest
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure repo root is on path so `envs` package resolves
sys.path.insert(0, ".")

from envs.registry import SimRegistry, register_sim, create_env, list_sims


# Mock environment classes for testing

class MockEnvBase(gym.Env):
    """
    Minimal gym.Env matching our observation/action contract.
    Used by tests that need a constructable environment.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        self.kwargs = kwargs
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
            "vec": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def step(self, action):
        obs = {
            "image": np.zeros((90, 160, 3), dtype=np.uint8),
            "vec": np.zeros(12, dtype=np.float32),
        }
        return obs, 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        obs = {
            "image": np.zeros((90, 160, 3), dtype=np.uint8),
            "vec": np.zeros(12, dtype=np.float32),
        }
        return obs, {}


# Fixtures

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before each test to prevent cross-contamination."""
    SimRegistry.clear()
    yield
    SimRegistry.clear()


# Registration tests

class TestRegistration:
    """Tests for @register_sim decorator and SimRegistry.register()."""

    def test_register_and_retrieve(self):
        """Basic registration: class is retrievable by name."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        retrieved = SimRegistry.get("test_sim")
        assert retrieved is TestEnv

    def test_duplicate_same_class_is_idempotent(self):
        """Re-registering the same class under the same name is a no-op."""
        class TestEnv(MockEnvBase):
            pass

        SimRegistry.register("test_sim", TestEnv)
        # Should not raise — same class, same name
        SimRegistry.register("test_sim", TestEnv)
        assert SimRegistry.get("test_sim") is TestEnv

    def test_duplicate_different_class_raises(self):
        """Registering a different class under an existing name raises ValueError."""
        class EnvA(MockEnvBase):
            pass

        class EnvB(MockEnvBase):
            pass

        SimRegistry.register("test_sim", EnvA)
        with pytest.raises(ValueError, match="already registered"):
            SimRegistry.register("test_sim", EnvB)

    def test_decorator_returns_class_unchanged(self):
        """@register_sim doesn't modify the decorated class."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            custom_attr = 42

        assert TestEnv.custom_attr == 42
        assert TestEnv.__name__ == "TestEnv"

    def test_multiple_registrations(self):
        """Multiple different simulators can coexist in the registry."""
        @register_sim("sim_a")
        class EnvA(MockEnvBase):
            pass

        @register_sim("sim_b")
        class EnvB(MockEnvBase):
            pass

        assert SimRegistry.get("sim_a") is EnvA
        assert SimRegistry.get("sim_b") is EnvB


# Lookup and error handling tests

class TestLookup:
    """Tests for SimRegistry.get() and error paths."""

    def test_unknown_sim_raises_keyerror(self):
        """Requesting an unregistered name with no module map entry raises KeyError."""
        with pytest.raises(KeyError, match="Unknown simulator"):
            SimRegistry.get("nonexistent_sim")

    def test_keyerror_lists_available(self):
        """The KeyError message includes available simulator names."""
        @register_sim("real_sim")
        class RealEnv(MockEnvBase):
            pass

        try:
            SimRegistry.get("fake_sim")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            # The error message should mention available sims
            assert "real_sim" in str(e)

    def test_is_registered_true(self):
        """is_registered returns True for registered names."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        assert SimRegistry.is_registered("test_sim") is True

    def test_is_registered_false(self):
        """is_registered returns False for unknown names (no lazy import)."""
        assert SimRegistry.is_registered("nonexistent") is False


# Factory function tests

class TestCreateEnv:
    """Tests for create_env() factory function."""

    def test_create_env_returns_instance(self):
        """create_env instantiates the registered class."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        env = create_env("test_sim")
        assert isinstance(env, TestEnv)
        assert isinstance(env, gym.Env)

    def test_create_env_passes_kwargs(self):
        """create_env forwards **kwargs to the env constructor."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        sentinel = object()
        env = create_env("test_sim", config=sentinel)
        assert env.config is sentinel

    def test_create_env_obs_action_contract(self):
        """Factory-created env respects our observation/action contract."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        env = create_env("test_sim")

        # Observation space: Dict with image (90,160,3) and vec (12,)
        assert "image" in env.observation_space.spaces
        assert "vec" in env.observation_space.spaces
        assert env.observation_space["image"].shape == (90, 160, 3)
        assert env.observation_space["vec"].shape == (12,)

        # Action space: Box(3,) with correct bounds
        assert env.action_space.shape == (3,)

    def test_create_env_unknown_raises(self):
        """create_env raises KeyError for unregistered names."""
        with pytest.raises(KeyError):
            create_env("nonexistent_sim")


# Listing tests

class TestListSims:
    """Tests for list_sims() and list_available()."""

    def test_list_includes_registered(self):
        """list_sims includes names that have been registered."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        available = list_sims()
        assert "test_sim" in available

    def test_list_includes_module_map(self):
        """list_sims includes names from _module_map (lazy-importable)."""
        available = list_sims()
        # These come from _module_map even if not yet imported
        assert "isaac" in available
        assert "gazebo" in available

    def test_list_is_sorted(self):
        """list_sims returns a sorted list."""
        @register_sim("z_sim")
        class ZEnv(MockEnvBase):
            pass

        @register_sim("a_sim")
        class AEnv(MockEnvBase):
            pass

        available = list_sims()
        assert available == sorted(available)

    def test_list_deduplicates(self):
        """If a name is in both _registry and _module_map, it appears once."""
        # "isaac" is in _module_map. Register it manually too.
        @register_sim("isaac")
        class FakeIsaac(MockEnvBase):
            pass

        available = list_sims()
        assert available.count("isaac") == 1


# Clear / isolation tests

class TestClear:
    """Tests for registry isolation and cleanup."""

    def test_clear_empties_registry(self):
        """clear() removes all registrations."""
        @register_sim("test_sim")
        class TestEnv(MockEnvBase):
            pass

        assert SimRegistry.is_registered("test_sim")
        SimRegistry.clear()
        assert not SimRegistry.is_registered("test_sim")

    def test_clear_preserves_module_map(self):
        """clear() does not affect _module_map (static config)."""
        SimRegistry.clear()
        available = list_sims()
        # Module map entries should still be listed
        assert "isaac" in available
        assert "gazebo" in available


# Module map tests

class TestModuleMap:
    """Tests for the lazy import module map."""

    def test_module_map_has_known_sims(self):
        """_module_map includes our two known simulators."""
        assert "isaac" in SimRegistry._module_map
        assert "gazebo" in SimRegistry._module_map

    def test_module_map_paths_are_correct(self):
        """Module paths match actual file names in the repo."""
        assert SimRegistry._module_map["isaac"] == "isaac_direct_env"
        assert SimRegistry._module_map["gazebo"] == "gazebo_direct_env"


# Standalone runner

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
