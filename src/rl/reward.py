"""Composite reward function for QKD-aware wavelength assignment.

The reward balances three objectives:
1. **r_accept**: +1 for successful assignment, -10 if blocked
2. **r_SKR**: minimum residual key ratio across edges after assignment
3. **r_spacing**: spectral distance from quantum channels (reduces crosstalk)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def compute_reward(
    accepted: bool,
    edge_skr_residual: Optional[np.ndarray] = None,
    edge_skr_before: Optional[np.ndarray] = None,
    assigned_wavelength_idx: int = 0,
    quantum_channel_indices: Optional[List[int]] = None,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> float:
    """Compute composite reward for a wavelength assignment action.

    Parameters
    ----------
    accepted : bool
        Whether the request was successfully assigned a wavelength.
    edge_skr_residual : np.ndarray, optional
        Per-edge SKR after assignment [bits/s].
    edge_skr_before : np.ndarray, optional
        Per-edge SKR before assignment [bits/s].
    assigned_wavelength_idx : int
        Index of the assigned wavelength (for spacing reward).
    quantum_channel_indices : list of int, optional
        Wavelength indices of quantum channels.
    alpha : float
        Weight for SKR preservation reward.
    beta : float
        Weight for spectral spacing reward.

    Returns
    -------
    float
        Scalar reward value.
    """
    if not accepted:
        return -10.0

    reward = 1.0  # r_accept

    # r_SKR: preserve key capacity
    if (
        edge_skr_residual is not None
        and edge_skr_before is not None
        and len(edge_skr_before) > 0
    ):
        # Residual key ratio, clipped to [0, 1] for each edge
        # Avoid division by zero for edges with 0 initial SKR
        safe_before = np.where(edge_skr_before > 0, edge_skr_before, np.inf)
        residual_ratio = np.where(
            edge_skr_before > 0,
            np.clip(edge_skr_residual / safe_before, 0.0, 1.0),
            1.0,  # edge had no key capacity → no penalty
        )
        min_ratio = float(np.min(residual_ratio))
        reward += alpha * min_ratio

    # r_spacing: prefer classical wavelengths far from quantum
    if quantum_channel_indices:
        min_dist = min(
            abs(assigned_wavelength_idx - q_idx)
            for q_idx in quantum_channel_indices
        )
        # Normalize spacing reward: distance of 20+ channels gives max 1.0
        spacing_reward = min(min_dist / 20.0, 1.0)
        reward += beta * spacing_reward

    return float(reward)
