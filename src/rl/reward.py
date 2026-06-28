"""Composite reward for QKD-aware wavelength assignment.

  blocked  → -3.0
  accepted → +1.0 + 2.0 * min(SKR_after / SKR_before, 1)  ∈ [+1, +3]

Break-even acceptance rate: 3/(3+2) = 60 % — well below the observed ~77 %,
so cumulative reward is positive at steady state, giving a clear learning signal.

The SKR ratio already captures spectral-distance effects implicitly: channels
closer to the quantum slot cause more Raman noise → lower post-assignment SKR
→ lower ratio → lower reward.  No separate spacing term is needed.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-30          # guard against division by zero when SKR_before ≈ 0
_BLOCK_PENALTY = -3.0
_ACCEPT_BASE   =  1.0
_SKR_WEIGHT    =  2.0


def compute_reward(
    accepted: bool,
    edge_skr_after: np.ndarray | None = None,
    edge_skr_before: np.ndarray | None = None,
    # legacy kwargs kept for backward-compat; ignored
    assigned_wavelength_idx: int = 0,
    quantum_channel_indices=None,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> float:
    if not accepted:
        return _BLOCK_PENALTY

    reward = _ACCEPT_BASE

    if edge_skr_after is not None and edge_skr_before is not None and len(edge_skr_before):
        # Bottleneck SKR ratio across path edges, clipped to [0, 1]
        ratio = np.where(
            edge_skr_before > 0,
            np.clip(edge_skr_after / (edge_skr_before + _EPS), 0.0, 1.0),
            1.0,   # edge had no key capacity → neutral
        )
        reward += _SKR_WEIGHT * float(np.min(ratio))

    return float(reward)
