"""QKD key-capacity provider — abstract and actual-SKR modes.

Supported modes
---------------
``abstract`` (default)
    Exponential-decay model:  K = K0 * exp(-alpha * length_km)  [kb/s]
    Zero external dependencies.  For quick testing only.

``actual_skr``
    Self-contained approx-finite-key decoy-state BB84 with discrete
    FWM + SpRS noise model, implemented in ``physics.py``.
    Key capacity is **dynamic**: it depends on how many classical
    WDM channels are currently active on the link.
    This is the **recommended mode for course-paper results**.

    Usage::

        python run.py --qkd-capacity-mode actual_skr
"""

import math
from typing import Callable

import networkx as nx

from . import physics
from .config import SimulationConfig


def initialize_qkd_provider(
    mode: str,
    graph: nx.Graph,
    config: SimulationConfig,
) -> Callable[[float, int], float]:
    """Return a callable ``f(distance_m, n_active_channels) -> key_capacity_kbps``.

    The second argument ``n_active_channels`` is the number of classical
    channels currently consuming bandwidth on the link (0 = dark fibre).
    """
    if mode == "abstract":
        return _build_abstract_provider(config)

    if mode == "actual_skr":
        return _build_embedded_skr_provider(config)

    raise ValueError(
        f"Unknown QKD capacity mode: {mode!r}. "
        f"Supported modes: 'abstract', 'actual_skr'."
    )


# ---------------------------------------------------------------------------
# Abstract (exponential-decay) provider
# ---------------------------------------------------------------------------

def _build_abstract_provider(config: SimulationConfig) -> Callable[[float, int], float]:
    K0: float = config.abstract_K0_kbps
    alpha: float = config.abstract_alpha_per_km

    def abstract_skr(distance_m: float, _n_active: int = 0) -> float:
        """Exponential-decay key rate in kb/s (ignores n_active)."""
        length_km = distance_m / 1000.0
        return K0 * math.exp(-alpha * length_km)

    return abstract_skr


# ---------------------------------------------------------------------------
# Embedded SKR provider (physics.py)
# ---------------------------------------------------------------------------

def _build_embedded_skr_provider(config: SimulationConfig) -> Callable[[float, int], float]:
    """Use the self-contained finite-key decoy-state BB84 from physics.py.

    Key capacity is recomputed each time ``n_active`` changes.
    Classical channels occupy low C-band (190.0 THz + i×50 GHz);
    quantum sits at 193.5 THz ± 12.5 GHz.
    """

    n_quantum = getattr(config, 'n_quantum_channels', physics.N_QUANTUM_CHANNELS)

    def actual_skr(distance_m: float, n_active: int) -> float:
        """finite-key decoy BB84 + FWM/SpRS noise → kbps."""
        return physics.get_key_capacity_kbps(
            distance_m, num_classical_channels=n_active,
            n_quantum_channels=n_quantum,
        )

    n_max = getattr(config, 'n_max_classical_channels', physics.N_CLASSICAL_CHANNELS)
    print(
        "[skr_adapter] Dynamic embedded SKR provider "
        f"(finite-key decoy BB84, {n_max} classical ch, {n_quantum} quantum ch, "
        f"50 GHz grid, first-fit from {physics.CLASSICAL_BASE_FREQ_HZ/1e12:.1f} THz)"
    )
    return actual_skr
