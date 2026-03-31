"""
Environment Registry Package

Public API for simulator environment creation.

Training scripts import from here:
    from envs import create_env, list_sims

Environment files use the decorator:
    from envs.registry import register_sim

    @register_sim("my_sim")
    class MySimEnv(gym.Env):
        ....
"""

from envs.registry import create_env, register_sim, list_sims, SimRegistry

__all__ = [
    "create_env",
    "register_sim",
    "list_sims",
    "SimRegistry,"
]
