"""
Simulator Environment Registry

Factory + registry pattern for simulator environment creation.

How it works:
    1. Each env file decorates its class with @register_sim("name")
    2. The decorator registers the class in SimRegistry._registry
    3. Training scripts call create_env("name", config=...) to get an env
    4. If the env module hasn't been imported yet, the registry does a
       lazy import via _module_map before instantiation

This solves three problems:
    - No hardcoded imports in training scripts (swap sim via config)
    - Lazy importing (Gazebo deps don't load on Isaac-only machines)
    - Extensibility (new sim = one file + @register_sim decorator)

The registry does NOT enforce a BaseSimEnv abstract class. Both
IsaacDirectEnv and GazeboDirectEnv extend gym.Env directly with the
same obs/action contract (Dict{"image": Box(90,160,3), "vec": Box(12,)},
action Box(3,)). The contract is enforced by convention and tests,
not inheritance — matching the project's "no abstract base class layer"
principle.

Usage:
    # In isaac_direct_env.py:
    from envs.registry import register_sim

    @register_sim("isaac")
    class IsaacDirectEnv(gym.Env):
        ...

    # In train_policy.py:
    from envs import create_env
    env = create_env(config.sim.sim_type, config=isaac_config)

Dependencies:
    - Python 3.10+ (typing features)
    - gymnasium (for type hints only — no runtime dependency)

Author: Aaron Hamil
Date: 03/31/26
"""

import importlib
import logging
from typing import Dict, Type, Optional, List

import gymnasium as gym

logger = logging.getLogger(__name__)


class SimRegistry:
    """
    Central registry mapping simulator names to environment classes.

    Each adapter registers itself via the @register_sim decorator when
    its module is imported. The registry is a simple dict: name -> class.

    The _module_map enables lazy importing — we only import the simulator
    module when someone actually requests that adapter, so heavy
    dependencies (omni.isaac, gz.transport13) are never loaded
    unnecessarily.
    """

    _registry: Dict[str, Type[gym.Env]] = {}

    # Map from sim name to the module path that contains the adapter.
    # These are root-level modules.
    _module_map: Dict[str, str] = {
        "isaac": "isaac_direct_env",
        "gazebo": "gazebo_direct_env",
    }

    @classmethod
    def register(cls, name: str, env_cls: Type[gym.Env]) -> None:
        """
        Register an environment class under a string name.

        Called by the @register_sim decorator at import time. Raises
        ValueError on duplicate registration to catch accidental
        re-imports that could silently swap classes.

        Args:
            name: Short identifier (e.g. "isaac", "gazebo", "carla").
            env_cls: The gym.Env subclass to register.
        """
        if name in cls._registry:
            existing = cls._registry[name].__name__
            if existing == env_cls.__name__:
                # Same class re-imported (e.g. module reload) — skip
                logger.debug(
                    "SimRegistry: '%s' already registered by %s (same class, skipping)",
                    name, existing,
                )
                return
            raise ValueError(
                f"Simulator '{name}' already registered by {existing}. "
                f"Cannot re-register with {env_cls.__name__}."
            )

        cls._registry[name] = env_cls
        logger.debug("SimRegistry: registered '%s' -> %s", name, env_cls.__name__)

    @classmethod
    def get(cls, name: str) -> Type[gym.Env]:
        """
        Retrieve an environment class by name.

        If the class hasn't been imported yet, attempts lazy import
        from _module_map. This is the mechanism that prevents loading
        gz-transport on Isaac machines and vice versa.

        Args:
            name: Simulator identifier (must match a registered name
                  or a key in _module_map).

        Returns:
            The gym.Env subclass for the requested simulator.

        Raises:
            KeyError: If the name isn't registered and isn't in _module_map.
            ImportError: If lazy import fails (simulator not installed).
        """
        if name not in cls._registry:
            if name in cls._module_map:
                module_path = cls._module_map[name]
                logger.info(
                    "SimRegistry: '%s' not yet imported, "
                    "lazy-loading from '%s'",
                    name, module_path,
                )
                try:
                    importlib.import_module(module_path)
                except ImportError as e:
                    raise ImportError(
                        f"Could not import environment for '{name}' from "
                        f"'{module_path}'. Is the simulator installed?\n\n"
                        f"Original error: {e}"
                    ) from e
            else:
                available = list(cls._registry.keys()) + list(cls._module_map.keys())
                raise KeyError(
                    f"Unknown simulator '{name}'. "
                    f"Available: {sorted(set(available))}"
                )

        # After lazy import, the decorator should have populated _registry
        if name not in cls._registry:
            raise KeyError(
                f"Module '{cls._module_map.get(name, '?')}' was imported "
                f"but did not register '{name}'. Did you forget the "
                f"@register_sim('{name}') decorator?"
            )

        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all simulator names that can be created.

        Includes both already-imported registrations and names in
        _module_map that could be lazy-imported. Useful for CLI
        --help output and config validation.

        Returns:
            Sorted list of simulator name strings.
        """
        return sorted(set(list(cls._registry.keys()) + list(cls._module_map.keys())))

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a name is registered (without triggering lazy import)."""
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        """
        Clear the registry. Only used in testing.

        Does NOT clear _module_map — that's static configuration.
        """
        cls._registry.clear()
        logger.debug("SimRegistry: cleared all registrations")


def register_sim(name: str):
    """
    Class decorator that registers a gym.Env subclass in SimRegistry.

    Usage:
        @register_sim("isaac")
        class IsaacDirectEnv(gym.Env):
            ...

    The class is returned unchanged — this only has a registration
    side effect. The decorator fires at import time, so lazy importing
    the module automatically populates the registry.

    Args:
        name: Short identifier string for this simulator.

    Returns:
        Decorator function that registers and returns the class.
    """
    def decorator(cls: Type[gym.Env]) -> Type[gym.Env]:
        SimRegistry.register(name, cls)
        return cls
    return decorator


def create_env(sim_type: str, **kwargs) -> gym.Env:
    """
    Factory function: create a simulator environment by name.

    This is the primary API for training scripts. Instead of:
        from isaac_direct_env import IsaacDirectEnv
        env = IsaacDirectEnv(config=my_config)

    You write:
        from envs import create_env
        env = create_env("isaac", config=my_config)

    The sim_type string typically comes from ExperimentConfig.sim.sim_type,
    so switching simulators is a YAML change, not a code change.

    Args:
        sim_type: Simulator name (e.g. "isaac", "gazebo").
        **kwargs: Passed directly to the env class constructor.
                  Typically includes config= with the simulator-specific
                  config dataclass (IsaacDirectConfig, GazeboDirectConfig).

    Returns:
        Instantiated gym.Env subclass for the requested simulator.

    Raises:
        KeyError: If sim_type is unknown.
        ImportError: If the simulator's dependencies aren't installed.
    """
    env_cls = SimRegistry.get(sim_type)
    logger.info(
        "create_env: instantiating %s for sim_type='%s'",
        env_cls.__name__, sim_type,
    )
    return env_cls(**kwargs)


def list_sims() -> List[str]:
    """
    List all available simulator names.

    Convenience wrapper around SimRegistry.list_available().
    Useful for argparse choices or config validation:

        parser.add_argument("--sim", choices=list_sims())

    Returns:
        Sorted list of simulator name strings.
    """
    return SimRegistry.list_available()
