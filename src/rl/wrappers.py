"""Action-mask wrapper for Stable-Baselines3 compatibility.

SB3 does not natively support action masking. This wrapper stores the
action mask in the info dict under the key ``action_mask``, which a custom
`` MaskableDQN`` policy can read (see train_dqn.py).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ActionMaskWrapper(gym.Wrapper):
    """Wrap a Gymnasium env that emits ``action_mask`` in its info dict.

    The mask is exposed as ``self.action_mask`` after each step/reset so
    that a custom policy can read it during forward passes.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_mask: Optional[np.ndarray] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.action_mask = info.get("action_mask", None)
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_mask = info.get("action_mask", None)
        return obs, reward, terminated, truncated, info
