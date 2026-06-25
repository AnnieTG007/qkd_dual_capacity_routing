"""QKD key-capacity provider — abstract and actual-SKR modes.

Supported modes
---------------
``abstract`` (default, zero-dependency)
    Exponential-decay model:  K = K0 * exp(-alpha * length_km)  [kb/s]

``actual_skr``
    Dynamically imports a BB84 secret-key-rate function from an external
    QKD project.  Two projects are supported, searched in order:

    1. **qkd_optical_network** (preferred)
       ``approx_finite_key_rate`` — three-state decoy BB84 with Gaussian
       finite-key corrections.  More rigorous; recommended for the course
       paper.

    2. **QKD_Network** (fallback)
       ``BB84_SKR_infinite`` — single-intensity infinite-key BB84.
       Used when qkd_optical_network is not available.

    The old-project modules are loaded **at function-call time** via
    ``importlib`` — this module-level import succeeds even when the old
    projects are absent, as long as ``actual_skr`` mode is never requested.
"""

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

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
        return _build_actual_skr_provider(config)

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
# Actual SKR provider
# ---------------------------------------------------------------------------

def _build_actual_skr_provider(
    config: SimulationConfig,
) -> Callable[[float], float]:
    """Try to load a BB84 SKR function from external projects.

    Priority order:
    1. qkd_optical_network  → approx_finite_key_rate (finite-key + decoy)
    2. QKD_Network           → BB84_SKR_infinite    (infinite-key)
    """
    old_root = config.old_project_root

    # ---- Strategy 1: qkd_optical_network (finite-key decoy BB84) ----
    if old_root is not None:
        provider = _try_qkd_optical_network(old_root)
        if provider is not None:
            return provider

    # ---- Strategy 2: QKD_Network (infinite-key BB84) ----
    if old_root is not None:
        provider = _try_qkd_network_project(old_root)
        if provider is not None:
            return provider

    # ---- None found ----
    raise RuntimeError(
        "actual_skr mode requires a valid --old-project-root pointing to "
        "one of:\n"
        "  1. qkd_optical_network project (preferred — finite-key decoy BB84)\n"
        "  2. QKD_Network project (fallback — infinite-key BB84)\n\n"
        "Example:\n"
        '  python run.py --qkd-capacity-mode actual_skr '
        '--old-project-root "E:/.../01_simulation_work/qkd_optical_network"'
    )


# ---------------------------------------------------------------------------
# qkd_optical_network provider (approx_finite_key_rate)
# ---------------------------------------------------------------------------

def _try_qkd_optical_network(old_root: str) -> Optional[Callable[[float], float]]:
    """Try approx_finite_key_rate from qkd_optical_network project."""
    import importlib.util as iu

    root = Path(old_root)
    src_dir = root / "src"

    # The qkd_optical_network project has its package at src/qkd_sim/
    skr_module_path = (
        src_dir / "qkd_sim" / "physical" / "skr" / "skr_decoy_bb84.py"
    )
    if not skr_module_path.is_file():
        return None

    # Add src/ to sys.path so intra-package imports work
    src_str = str(src_dir.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    try:
        # Dynamic import of the approx_finite_key_rate function
        spec = iu.spec_from_file_location(
            "qkd_sim.physical.skr.skr_decoy_bb84", str(skr_module_path)
        )
        mod = iu.module_from_spec(spec)
        sys.modules["qkd_sim.physical.skr.skr_decoy_bb84"] = mod
        spec.loader.exec_module(mod)

        approx_finite_key_rate = getattr(mod, "approx_finite_key_rate", None)
        if approx_finite_key_rate is None:
            return None

        # Build FiberConfig and SKRConfig with default parameters
        fiber_cfg, skr_cfg = _build_qkd_optical_network_configs()

        def actual_skr_finite(distance_m: float) -> float:
            """approx_finite_key_rate → kbps (zero-noise baseline)."""
            skr_bps, _skr_bpp, _qber = approx_finite_key_rate(
                distance_m, fiber_cfg, skr_cfg, p_noise=0.0
            )
            return float(skr_bps) / 1000.0  # bps → kbps

        print(
            "[skr_adapter] Using approx_finite_key_rate "
            f"(finite-key decoy BB84) from {root.name}"
        )
        return actual_skr_finite

    except Exception as exc:
        print(
            f"[skr_adapter] Warning: failed to load qkd_optical_network SKR: {exc}",
            file=sys.stderr,
        )
        return None


def _build_qkd_optical_network_configs():
    """Build FiberConfig and SKRConfig matching the 'custom' profile defaults.

    These values mirror ``src/qkd_sim/config/defaults/skr_para/bb84_config.yaml``
    (custom profile) and ``fiber_para/fiber_smf.yaml``.
    """
    # Import schema classes (now on sys.path)
    from qkd_sim.config.schema import BlockLength, FiberConfig, SKRConfig  # noqa: E402

    # FiberConfig — only ``alpha`` is used by _channel_quantities
    # alpha_dB_per_km=0.2  →  alpha = 4.61e-5 m^-1  (standard SMF)
    fiber_cfg = FiberConfig(
        alpha_dB_per_km=0.2,
        gamma_per_W_km=1.3,
        D_ps_nm_km=17.0,
        D_slope_ps_nm2_km=0.056,
        L_km=50.0,
        A_eff=8.0e-11,
        rayleigh_coeff=4.8e-8,
        T_kelvin=300.0,
    )

    # SKRConfig — 'custom' profile from bb84_config.yaml
    skr_cfg = SKRConfig(
        eta_spd=0.25,
        IL_dB=6.0,
        dark_count_prob=1.0e-6,
        noise_count_prob=0.0,
        mu_signal=0.75,
        mu_decoy=0.2,
        e_det=0.01,
        f_ec=1.16,
        R_rep=5.0e7,
        q_sifting=0.5,
        p_signal=0.875,
        p_decoy=0.0625,
        block_length=BlockLength(mode="bob", N_alice=None, N_bob=1.0e7),
        gamma_ks=5.3,
        P_X_alice=0.5,
        P_X_bob=0.5,
        epsilon_cor=1.0e-12,
        epsilon_sec=1.0e-12,
        approx_finite_N_pulse=1.0e10,
        concentration_method="Hoeffding",
        improved_serfling=True,
        optimize_params=False,  # False for scalar mode (faster)
    )

    return fiber_cfg, skr_cfg


# ---------------------------------------------------------------------------
# QKD_Network provider (BB84_SKR_infinite) — fallback
# ---------------------------------------------------------------------------

def _try_qkd_network_project(old_root: str) -> Optional[Callable[[float], float]]:
    """Try BB84_SKR_infinite from the old QKD_Network project."""
    import importlib.util as iu

    root = Path(old_root)
    skr_module_path = root / "src" / "qkd" / "SKR_BB84_finite.py"

    if not skr_module_path.is_file():
        return None

    try:
        spec = iu.spec_from_file_location(
            "SKR_BB84_finite", str(skr_module_path)
        )
        mod = iu.module_from_spec(spec)
        sys.modules["SKR_BB84_finite"] = mod
        spec.loader.exec_module(mod)

        BB84_SKR_infinite = getattr(mod, "BB84_SKR_infinite", None)
        if BB84_SKR_infinite is None:
            return None

        def actual_skr_infinite(distance_m: float) -> float:
            """BB84_SKR_infinite → kbps (zero-noise baseline)."""
            _skr_per_pulse, skr_per_sec, _e_ave = BB84_SKR_infinite(
                distance_m, 0.0
            )
            return skr_per_sec / 1000.0  # bps → kbps

        print(
            "[skr_adapter] Using BB84_SKR_infinite "
            f"(infinite-key BB84) from {root.name}"
        )
        return actual_skr_infinite

    except Exception as exc:
        print(
            f"[skr_adapter] Warning: failed to load QKD_Network SKR: {exc}",
            file=sys.stderr,
        )
        return None
