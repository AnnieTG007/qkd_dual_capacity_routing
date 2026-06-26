"""QKD key-capacity provider — abstract and actual-SKR modes.

Supported modes
---------------
``abstract`` (default)
    Exponential-decay model:  K = K0 * exp(-alpha * length_km)  [kb/s]
    Zero external dependencies.  For quick testing only.

``actual_skr``
    Self-contained approx-finite-key decoy-state BB84 with simplified
    discrete FWM + SpRS noise model, implemented in ``physics.py``.
    This is the **recommended mode for course-paper results**.

    The noise depends on the number of classical channels active on a
    link.  By default a fixed channel count (40 ch × 50 GHz) is used,
    but it can be linked to the instantaneous classical utilisation.

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
) -> Callable[[float], float]:
    """Return a callable ``f(distance_m: float) -> key_capacity_kbps``."""
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

def _build_abstract_provider(config: SimulationConfig) -> Callable[[float], float]:
    K0: float = config.abstract_K0_kbps
    alpha: float = config.abstract_alpha_per_km

    def abstract_skr(distance_m: float) -> float:
        """Exponential-decay key rate in kb/s."""
        length_km = distance_m / 1000.0
        return K0 * math.exp(-alpha * length_km)

    return abstract_skr


# ---------------------------------------------------------------------------
# Embedded SKR provider (physics.py)
# ---------------------------------------------------------------------------

def _build_embedded_skr_provider(config: SimulationConfig) -> Callable[[float], float]:
    """Use the self-contained finite-key decoy-state BB84 from physics.py.

    Noise is computed with the simplified discrete FWM + SpRS model.
    The number of classical channels is fixed at the default defined in
    ``physics.N_CLASSICAL_CHANNELS`` (40 channels at 50 GHz spacing).
    """
    def actual_skr(distance_m: float) -> float:
        """approx_finite_key_rate + FWM/SpRS noise → kbps."""
        return physics.get_key_capacity_kbps(distance_m)

    print(
        "[skr_adapter] Using embedded approx_finite_key_rate "
        "(finite-key decoy BB84 + simplified FWM/SpRS noise)"
    )
    return actual_skr
