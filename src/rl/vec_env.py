"""Vectorised environment builder for parallel DQN training.

Creates N independent ``QKDWDMEnv`` instances wrapped with
``ActionMaskWrapper``, running in subprocesses via ``SubprocVecEnv``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .qkd_wdm_env import QKDWDMEnv, EnvConfig
from .wrappers import ActionMaskWrapper


def _make_env(
    env_config: EnvConfig,
    seed: int,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Return a thunk that creates a single wrapped environment."""

    def _init() -> gym.Env:
        env = QKDWDMEnv(config=env_config)
        env.reset(seed=seed + rank)
        env = ActionMaskWrapper(env)
        return env

    return _init


def make_vec_env(
    env_config: EnvConfig,
    n_envs: int = 16,
    seed: int = 42,
    use_subproc: bool = True,
) -> SubprocVecEnv | DummyVecEnv:
    """Build a vectorised environment for DQN training.

    Parameters
    ----------
    env_config : EnvConfig
        Configuration for the QKD WDM environment.
    n_envs : int
        Number of parallel environments.
    seed : int
        Base random seed (each worker gets seed + rank).
    use_subproc : bool
        If True, use SubprocVecEnv (true parallelism).
        If False, use DummyVecEnv (sequential, for debugging).

    Returns
    -------
    SubprocVecEnv or DummyVecEnv
        Vectorised environment compatible with SB3.
    """
    env_fns = [_make_env(env_config, seed, rank=i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)
