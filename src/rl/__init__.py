"""RL module for QKD-aware wavelength assignment in WDM networks.

Provides:
- QKDWDMEnv: Gymnasium environment for single-core WDM wavelength selection
- compute_reward: SKR-based composite reward function
- ActionMaskWrapper: SB3-compatible action masking for infeasible wavelengths
- make_vec_env: SubprocVecEnv builder for parallel training
"""

from .qkd_wdm_env import QKDWDMEnv
from .reward import compute_reward
from .wrappers import ActionMaskWrapper
from .vec_env import make_vec_env

__all__ = [
    "QKDWDMEnv",
    "compute_reward",
    "ActionMaskWrapper",
    "make_vec_env",
]
