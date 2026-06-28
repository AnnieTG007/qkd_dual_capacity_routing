"""Gymnasium environment for single-core WDM wavelength assignment.

Models a dynamic WDM optical network where each fiber link carries both
classical data channels and quantum (QKD) channels. The agent selects
wavelengths for arriving classical service requests, aiming to preserve
key-generation capacity across the network.

Key features:
- Discrete single-frequency FWM + SpRS noise models
- SKR computation via BB84 decoy-state with finite-key corrections
- Action masking for feasibility (wavelength occupancy + SKR constraint)
- Poisson arrivals + exponential holding times for dynamic traffic

State space (flattened vector):
  [occupancy_flat (|E|×W), path_mask (|E|), quantum_positions (W),
   request_bw, request_key]

Action space: Discrete(W) — wavelength index, masked for feasibility.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .reward import compute_reward

# ── physical constants (SSMF G.652) ────────────────────────────────────────

PLANCK: float = 6.62607015e-34     # J·s
C_LIGHT: float = 299792458.0       # m/s
BOLTZMANN: float = 1.380649e-23    # J/K
ALPHA_DB_PER_KM: float = 0.2
ALPHA_PER_KM: float = ALPHA_DB_PER_KM * (math.log(10) / 10.0)
GAMMA_PER_W_KM: float = 1.3e-3     # W⁻¹·km⁻¹
D_C_PS_NM_KM: float = 17.0         # ps/(nm·km)
D_SLOPE_PS_NM2_KM: float = 0.056
A_EFF_M2: float = 80e-12
RAYLEIGH_COEFF: float = 4.8e-8
T_KELVIN: float = 300.0

# ── QKD system parameters ──────────────────────────────────────────────────

MU_SIGNAL: float = 0.75
MU_DECOY: float = 0.2
P_SIGNAL: float = 0.875
P_DECOY: float = 0.0625
Q_SIFTING: float = 0.5
F_EC: float = 1.16
ETA_SPD: float = 0.25
E_DET: float = 0.01
DARK_COUNT_PROB: float = 1e-6
GATE_TIME_S: float = 1e-9
IL_DB: float = 6.0
IL_LINEAR: float = 10 ** (-IL_DB / 10)
R_REP: float = 50e6                 # pulse repetition rate [Hz]
N_PULSE: float = 1e10               # finite-key block size
GAMMA_KS: float = 5.3               # Gaussian confidence multiplier

# ── GNPy 92-point Raman gain table ─────────────────────────────────────────

_RAMAN_FREQ_OFFSET_THZ = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
    6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 12.75, 13.0, 13.25, 13.5, 14.0, 14.5, 14.75, 15.0,
    15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.25, 18.5, 18.75, 19.0,
    19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
    25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0,
    30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5,
    36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0,
    41.5, 42.0,
], dtype=np.float64)

_RAMAN_G0_PER_WM = np.array([
    0.00000000e+00, 1.12351610e-05, 3.47838074e-05, 5.79356636e-05,
    8.06921680e-05, 9.79845709e-05, 1.10454361e-04, 1.18735302e-04,
    1.24736889e-04, 1.30110053e-04, 1.41001273e-04, 1.46383247e-04,
    1.57011792e-04, 1.70765865e-04, 1.88408911e-04, 2.05914127e-04,
    2.24074028e-04, 2.47508283e-04, 2.77729174e-04, 3.08044243e-04,
    3.34764439e-04, 3.56481704e-04, 3.77127256e-04, 3.96269124e-04,
    4.10955175e-04, 4.18718761e-04, 4.19511263e-04, 4.17025384e-04,
    4.13565369e-04, 4.07726048e-04, 3.83671291e-04, 4.08564283e-04,
    3.69571936e-04, 3.14442090e-04, 2.16074535e-04, 1.23097823e-04,
    8.95457457e-05, 7.52470400e-05, 7.19806145e-05, 8.87961158e-05,
    9.30812065e-05, 9.37058268e-05, 8.45719619e-05, 6.90585286e-05,
    4.50407159e-05, 3.36521245e-05, 3.02292475e-05, 2.69376939e-05,
    2.60020897e-05, 2.82958958e-05, 3.08667558e-05, 3.66024657e-05,
    5.80610307e-05, 6.54797937e-05, 6.25022715e-05, 5.37806442e-05,
    3.94996621e-05, 2.68120644e-05, 2.33038554e-05, 1.79140757e-05,
    1.52472424e-05, 1.32707565e-05, 1.06541760e-05, 9.84649374e-06,
    9.13999627e-06, 9.08971012e-06, 1.04227525e-05, 1.50419271e-05,
    1.77838232e-05, 2.15810815e-05, 2.03744008e-05, 1.81939341e-05,
    1.31862121e-05, 9.65352116e-06, 8.62698322e-06, 9.18688016e-06,
    1.01737784e-05, 1.08017817e-05, 1.03903588e-05, 9.30040333e-06,
    8.30809173e-06, 6.90650401e-06, 5.52238029e-06, 3.90648708e-06,
    2.22908227e-06, 1.55796177e-06, 9.77218716e-07, 3.23477236e-07,
    1.60602454e-07, 7.97306386e-08,
], dtype=np.float64)

_RAMAN_F_REF: float = 206.184634112792e12
_RAMAN_A_EFF_REF: float = 75.74659443542413e-12


# ── topology data (NSFNET14) ────────────────────────────────────────────────

@dataclass
class Topology:
    """Simple graph topology."""
    num_nodes: int
    edges: List[Tuple[int, int]]       # (u, v) pairs
    edge_lengths_km: List[float]       # per-edge fibre length


def nsfnet14() -> Topology:
    """Return the NSFNET 14-node, 21-edge topology."""
    edges = [
        (0, 1), (0, 2), (0, 5),
        (1, 2), (1, 4),
        (2, 3), (2, 5), (2, 8),
        (3, 6), (3, 8),
        (4, 7), (4, 10),
        (5, 8), (5, 11),
        (6, 7), (6, 9),
        (7, 10), (7, 13),
        (8, 9), (8, 11),
        (9, 12),
        (10, 12), (10, 13),
        (11, 12),
    ]
    lengths = [
        52.5, 75.0, 60.0,
        55.0, 70.0,
        65.0, 50.0, 80.0,
        55.0, 75.0,
        60.0, 90.0,
        85.0, 100.0,
        50.0, 65.0,
        70.0, 55.0,
        60.0, 75.0,
        80.0,
        65.0, 50.0,
        70.0,
    ]
    return Topology(num_nodes=14, edges=edges, edge_lengths_km=lengths)


# ── math helpers ────────────────────────────────────────────────────────────

def _H2(x: float) -> float:
    """Binary Shannon entropy."""
    x = min(max(float(x), 1e-15), 1.0 - 1e-15)
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


# ── simplified FWM + SpRS noise models ─────────────────────────────────────

def _phase_mismatch(
    f2: float, f3: float, f4: float, f_ref: float = 193.5e12
) -> float:
    """Second-order phase mismatch Δβ (discrete model)."""
    lam = C_LIGHT / f2
    lam_ref = C_LIGHT / f_ref
    D_c = D_C_PS_NM_KM * 1e-6 * 1e3        # s/m²
    D_slope = D_SLOPE_PS_NM2_KM * 1e-6 * 1e3 * 1e-3  # s/m³
    D_c_f2 = D_c + D_slope * (lam - lam_ref)
    df32 = abs(f3 - f2)
    df42 = abs(f4 - f2)
    return (2.0 * math.pi * lam ** 2 / C_LIGHT) * df32 * df42 * (
        D_c_f2 + (lam ** 2 / (2 * C_LIGHT)) * (df32 + df42) * D_slope
    )


def _fwm_coefficient(da: float, db: float, L_km: float) -> float:
    """FWM efficiency η for distance L_km."""
    num = math.exp(-da * L_km) - 2 * math.exp(-da * L_km / 2) * math.cos(db * L_km) + 1
    denom = da ** 2 / 4 + db ** 2
    return num / max(denom, 1e-30)


def _phonon_occupation(delta_f: float, T: float) -> float:
    """Bose-Einstein phonon occupation factor n_th."""
    abs_df = abs(delta_f)
    if abs_df < 1e-6:
        return 0.0
    exponent = PLANCK * abs_df / (BOLTZMANN * T)
    if exponent > 700.0:
        return 0.0
    return 1.0 / (math.exp(exponent) - 1.0)


def _get_raman_gain(delta_f: float, f_pump: float) -> float:
    """Interpolate GNPy Raman gain g_R [1/(W·m)] at |Δf|."""
    abs_df = abs(delta_f)
    if abs_df > 42e12:
        return 0.0
    g0 = float(np.interp(
        abs_df, _RAMAN_FREQ_OFFSET_THZ * 1e12, _RAMAN_G0_PER_WM,
        left=0.0, right=0.0,
    ))
    return g0 * (f_pump / _RAMAN_F_REF) * (_RAMAN_A_EFF_REF / A_EFF_M2)


def _raman_cross_section(
    f_q: float, f_c: float, g_R: float, n_th: float, noise_bw_hz: float,
) -> float:
    """Raman cross-section σ with Stokes / anti-Stokes distinction."""
    if f_c > f_q:  # Stokes
        return 2.0 * PLANCK * f_q * g_R * (1.0 + n_th) * noise_bw_hz
    else:           # Anti-Stokes
        if abs(f_c - f_q) < 1e6:
            return 0.0
        return 2.0 * PLANCK * f_q * g_R * n_th * (f_q / f_c) * noise_bw_hz


def estimate_noise(
    classical_freqs_hz: np.ndarray,
    classical_power_w: float,
    quantum_freq_hz: float,
    length_km: float,
    noise_bw_hz: float,
) -> Tuple[float, float, float]:
    """Estimate FWM + SpRS noise power [W] at the quantum receiver.

    Returns (P_fwm, P_sprs, P_total) in watts.
    """
    L_m = length_km * 1e3
    alpha = ALPHA_PER_KM            # per km
    gamma = GAMMA_PER_W_KM / 1e3    # W⁻¹·m⁻¹
    alpha_per_m = alpha * 1e-3      # per m
    exp_neg_alpha_L = math.exp(-alpha_per_m * L_m)

    freqs = np.asarray(classical_freqs_hz, dtype=np.float64)
    N = len(freqs)
    if N < 2:
        p_s = _sprs_only(classical_freqs_hz, classical_power_w,
                         quantum_freq_hz, L_m, alpha_per_m, noise_bw_hz)
        return 0.0, p_s, p_s

    # ── FWM: enumerate (f3, f4) pairs, match f2 = f3+f4-f_q ──
    tol = 50e9 / 20.0               # 2.5 GHz at 50 GHz spacing
    f3_grid, f4_grid = np.meshgrid(freqs, freqs, indexing="ij")
    f2_needed = f3_grid + f4_grid - quantum_freq_hz  # (N, N)

    idx = np.searchsorted(freqs, f2_needed.ravel())
    idx = np.clip(idx, 1, N - 1)
    left = freqs[idx - 1]
    right = freqs[idx]
    nearest = np.where(
        np.abs(f2_needed.ravel() - left) < np.abs(f2_needed.ravel() - right),
        left, right,
    ).reshape(N, N)

    f2_match = np.abs(nearest - f2_needed) < tol
    not_spm = (
        (np.abs(nearest - f3_grid) > 1e6)
        & (np.abs(nearest - f4_grid) > 1e6)
    )
    valid = f2_match & not_spm

    total_fwm = 0.0
    if valid.any():
        valid_indices = np.argwhere(valid)
        for ri, ci in valid_indices:
            f3 = freqs[ri]
            f4 = freqs[ci]
            f2 = float(nearest[ri, ci])
            D_factor = 3.0 if abs(f3 - f4) < 1e6 else 6.0
            db = _phase_mismatch(f2, f3, f4)
            # Δα ≈ 0 (uniform C-band loss)
            eta = _fwm_coefficient(0.0, db, L_m)
            p_fwd = (
                exp_neg_alpha_L
                * (gamma ** 2 / 9.0)
                * (D_factor ** 2)
                * eta
                * classical_power_w ** 3
            )
            total_fwm += float(p_fwd)

    # ── SpRS: per-pump sum ──
    total_sprs = 0.0
    for f_pump in freqs:
        delta_f = f_pump - quantum_freq_hz
        if abs(delta_f) > 42e12:
            continue
        g_R = _get_raman_gain(delta_f, f_pump)
        n_th = _phonon_occupation(delta_f, T_KELVIN)
        sigma = _raman_cross_section(quantum_freq_hz, f_pump, g_R, n_th, noise_bw_hz)

        # Forward: α_c ≈ α_q → L·exp(-αL)
        p_fwd = classical_power_w * sigma * exp_neg_alpha_L * L_m
        # Backward
        sum_alpha = 2.0 * alpha_per_m
        p_bwd = classical_power_w * sigma * (1.0 - math.exp(-sum_alpha * L_m)) / sum_alpha
        total_sprs += float(p_fwd + p_bwd)

    return float(total_fwm), float(total_sprs), float(total_fwm + total_sprs)


def _sprs_only(
    classical_freqs_hz: np.ndarray,
    classical_power_w: float,
    quantum_freq_hz: float,
    L_m: float,
    alpha_per_m: float,
    noise_bw_hz: float,
) -> float:
    """SpRS-only noise when fewer than 2 classical channels."""
    exp_neg_alpha_L = math.exp(-alpha_per_m * L_m)
    total_sprs = 0.0
    for f_pump in np.asarray(classical_freqs_hz):
        delta_f = f_pump - quantum_freq_hz
        if abs(delta_f) > 42e12:
            continue
        g_R = _get_raman_gain(delta_f, f_pump)
        n_th = _phonon_occupation(delta_f, T_KELVIN)
        sigma = _raman_cross_section(quantum_freq_hz, f_pump, g_R, n_th, noise_bw_hz)
        p_fwd = classical_power_w * sigma * exp_neg_alpha_L * L_m
        sum_alpha = 2.0 * alpha_per_m
        p_bwd = classical_power_w * sigma * (1.0 - math.exp(-sum_alpha * L_m)) / sum_alpha
        total_sprs += float(p_fwd + p_bwd)
    return total_sprs


# ── SKR computation ─────────────────────────────────────────────────────────

def _channel_quantities(
    distance_km: float, mu: float, p_noise: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Per-intensity channel quantities: (eta, Y0, Q_mu, E_mu)."""
    alpha_per_m = ALPHA_PER_KM * 1e-3
    eta = float(ETA_SPD * math.exp(-alpha_per_m * distance_km * 1e3) * IL_LINEAR)
    p_background = _clip(0.0, 1.0, DARK_COUNT_PROB + p_noise)
    Y0 = float(1.0 - (1.0 - p_background) ** 2)
    exp_term = math.exp(-mu * eta)
    Q_mu = float(Y0 + 1.0 - exp_term)
    e0 = 0.5
    E_mu = float((e0 * Y0 + E_DET * (1.0 - exp_term)) / max(Q_mu, 1e-30))
    return eta, Y0, Q_mu, E_mu


def compute_skr(
    distance_km: float,
    p_noise: float = 0.0,
) -> Tuple[float, float]:
    """Compute BB84 decoy-state SKR with finite-key corrections.

    Returns (skr_bps, qber). skr_bps is in bits per second.
    """
    mu, nu = MU_SIGNAL, MU_DECOY
    p_mu, p_nu = P_SIGNAL, P_DECOY
    N = N_PULSE
    gamma = GAMMA_KS
    e0 = 0.5

    eta, Y0, Q_mu, E_mu = _channel_quantities(distance_km, mu, p_noise)
    _, _, Q_nu, E_nu = _channel_quantities(distance_km, nu, p_noise)

    # Gaussian finite-key corrections on decoy
    denom_Q = max(p_nu * Q_nu * N / 2.0, 1e-30)
    Q_nu_L = Q_nu * (1.0 - gamma / math.sqrt(denom_Q))
    Q_nu_L = max(Q_nu_L, 0.0)

    EnuQnu = E_nu * Q_nu
    denom_E = max(p_nu * EnuQnu * N / 2.0, 1e-30)
    EnuQnu_U = EnuQnu * (1.0 + gamma / math.sqrt(denom_E))
    EnuQnu_U = max(EnuQnu_U, 0.0)

    # Y₁ lower bound
    denom_Y1 = mu * nu - nu ** 2
    if abs(denom_Y1) < 1e-30:
        return 0.0, float(E_mu)
    Y1_L = (mu / denom_Y1) * (
        Q_nu_L * math.exp(nu)
        - (nu ** 2 / mu ** 2) * Q_mu * math.exp(mu)
        - (mu ** 2 - nu ** 2) / mu ** 2 * Y0
    )
    Y1_L = max(Y1_L, 0.0)

    # Q₁ lower bound, e₁ upper bound
    Q1_L = Y1_L * mu * math.exp(-mu)
    if Y1_L < 1e-30:
        return 0.0, float(E_mu)
    e1_U = (EnuQnu_U * math.exp(nu) - e0 * Y0) / (nu * Y1_L)
    e1_U = _clip(0.0, 0.5, e1_U)

    # Secure key rate [bits/pulse]
    skr = p_mu * Q_SIFTING * (
        -Q_mu * F_EC * _H2(E_mu) + Q1_L * (1.0 - _H2(e1_U))
    )
    skr_bps = float(max(0.0, skr)) * R_REP
    return skr_bps, float(E_mu)


# ── Environment config ──────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """Configuration for the QKD WDM environment."""

    # Topology
    topology: Topology = field(default_factory=nsfnet14)

    # WDM grid
    num_wavelengths: int = 32            # W
    channel_spacing_hz: float = 50e9     # 50 GHz
    base_freq_hz: float = 190.0e12       # lowest classical channel
    quantum_freq_hz: float = 190.75e12   # quantum channel at slot 15 (mid-grid)
    quantum_bw_hz: float = 25e9          # quantum receiver noise BW
    num_quantum_channels: int = 4        # parallel QKD channels

    # Classical channel parameters
    classical_power_dbm: float = -25.0   # per-channel launch power (matches physics.py)
    classical_bandwidth_gbps: float = 100.0

    # Traffic
    arrival_rate: float = 0.05           # Poisson λ (requests per second)
    mean_holding_time_s: float = 60.0    # 1/μ
    mean_holding_steps: int = 12         # expected steps before a channel departs
    max_steps: int = 2000                # episode length

    # Reward weights
    reward_alpha: float = 0.5            # SKR preservation weight
    reward_beta: float = 0.1             # spacing reward weight

    # Seed
    seed: int = 42

    @property
    def classical_power_w(self) -> float:
        return 1e-3 * 10 ** (self.classical_power_dbm / 10.0)

    @property
    def classical_freqs(self) -> np.ndarray:
        return np.array([
            self.base_freq_hz + i * self.channel_spacing_hz
            for i in range(self.num_wavelengths)
        ], dtype=np.float64)


# ── Gymnasium environment ───────────────────────────────────────────────────

class QKDWDMEnv(gym.Env):
    """Single-core WDM wavelength assignment environment.

    State (flattened vector):
      [occupancy_flat (|E|×W), path_mask (|E|), quantum_positions (W),
       request_bw, request_key]

    Action: wavelength index w ∈ {0, ..., W-1} (masked for feasibility).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.cfg = self.config

        topo = self.cfg.topology
        self.num_edges = len(topo.edges)
        self.edges = topo.edges
        self.edge_lengths_km = np.array(topo.edge_lengths_km, dtype=np.float64)
        self.W = self.cfg.num_wavelengths

        # Precompute quantum channel indices and frequencies
        # Quantum channels are placed symmetrically around quantum_freq_hz
        half = (self.cfg.num_quantum_channels - 1) / 2.0
        self.quantum_indices: List[int] = []
        self.quantum_freqs: List[float] = []
        for i in range(self.cfg.num_quantum_channels):
            offset = (i - half) * 25e9  # 25 GHz guard band each
            f_q = self.cfg.quantum_freq_hz + offset
            self.quantum_freqs.append(f_q)

        # Observation space
        obs_dim = (
            self.num_edges * self.W       # occupancy
            + self.num_edges              # path mask
            + self.W                      # quantum positions
            + 2                           # request (bw, key)
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1e6, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.W)

        # Internal state
        self.occupancy: Optional[np.ndarray] = None        # (|E|, W)
        self.edge_skr: Optional[np.ndarray] = None         # (|E|,) bits/s
        self.path_mask: Optional[np.ndarray] = None        # (|E|,)
        self.request_bw: float = 0.0
        self.request_key: float = 0.0
        self.step_count: int = 0
        self.np_random: np.random.Generator = np.random.default_rng(self.cfg.seed)

        # Request queue (pre-generated Poisson traffic)
        self._request_queue: List[Tuple[float, float]] = []
        self._pre_generate_requests()

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def _n_slots(self) -> int:
        """Number of contiguous wavelength slots the current request needs.

        Matches ``qkd_routing/resources.py`` (``ceil(bw / bw_per_ch)``): a
        400-Gbps request over a 100-Gbps grid needs 4 wavelengths.
        """
        return max(1, math.ceil(self.request_bw / self.cfg.classical_bandwidth_gbps))

    def _pre_generate_requests(self) -> None:
        """Pre-generate Poisson arrivals for deterministic replay.

        Bandwidth and key demands are drawn per security level to match the
        routing simulation (``qkd_routing/config.py``).  A 400-Gbps request
        therefore needs ``ceil(400 / 100) = 4`` wavelength slots, exercising
        the multi-slot allocation path (see :pyattr:`_n_slots`).
        """
        lam = self.cfg.arrival_rate
        rng = np.random.default_rng(self.cfg.seed)
        # (cumulative probability, bandwidth options [Gbps], key demand [kbps])
        # mirrors SimulationConfig.security_levels:
        #   Low  p=0.5  bw∈{10,40}   key=0.2
        #   Med  p=0.3  bw∈{40,100}  key=1.0
        #   High p=0.2  bw∈{100,400} key=2.0
        sec_levels = [
            (0.5, [10.0, 40.0], 0.2),
            (0.3, [40.0, 100.0], 1.0),
            (0.2, [100.0, 400.0], 2.0),
        ]
        cum = np.cumsum([p for p, _, _ in sec_levels])
        t = 0.0
        for _ in range(self.cfg.max_steps * 2):  # oversample
            t += rng.exponential(1.0 / lam)
            if t > self.cfg.max_steps * self.cfg.mean_holding_time_s * 2:
                break
            # Pick a security level, then bandwidth/key for that level.
            lvl = int(np.searchsorted(cum, rng.uniform() * cum[-1]))
            lvl = min(lvl, len(sec_levels) - 1)
            _, bw_options, key = sec_levels[lvl]
            bw = float(rng.choice(bw_options))
            self._request_queue.append((bw, key))

    def _skr_for_edge(self, e: int) -> float:
        """Compute SKR [bits/s] for a single edge using current occupancy."""
        length_km = self.edge_lengths_km[e]
        active = np.where(self.occupancy[e] > 0)[0]
        if len(active) == 0:
            p_noise = 0.0
        else:
            freqs = self.cfg.classical_freqs[active]
            _, _, p_total = estimate_noise(
                freqs, self.cfg.classical_power_w,
                self.cfg.quantum_freq_hz, length_km,
                self.cfg.quantum_bw_hz,
            )
            p_noise = float(
                p_total * GATE_TIME_S * ETA_SPD * IL_LINEAR
                / (PLANCK * self.cfg.quantum_freq_hz)
            )
        skr_bps, _ = compute_skr(length_km, p_noise)
        return skr_bps * self.cfg.num_quantum_channels

    def _compute_edge_skr(self) -> np.ndarray:
        """Compute per-edge SKR [bits/s] based on current occupancy."""
        return np.array(
            [self._skr_for_edge(e) for e in range(self.num_edges)],
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        occ_flat = self.occupancy.ravel().astype(np.float32)
        path_mask = self.path_mask.astype(np.float32) if self.path_mask is not None \
            else np.zeros(self.num_edges, dtype=np.float32)
        quantum_pos = np.zeros(self.W, dtype=np.float32)
        # Mark quantum positions
        for f_q in self.quantum_freqs:
            idx = int(round((f_q - self.cfg.base_freq_hz) / self.cfg.channel_spacing_hz))
            if 0 <= idx < self.W:
                quantum_pos[idx] = 1.0
        request = np.array([self.request_bw, self.request_key], dtype=np.float32)
        return np.concatenate([occ_flat, path_mask, quantum_pos, request])

    def action_masks(self) -> np.ndarray:
        """Required by sb3-contrib ActionMasker for MaskablePPO."""
        return self._get_action_mask()

    def _get_action_mask(self) -> np.ndarray:
        """Build action mask over feasible *start* wavelengths on the path.

        The action is the start index of a contiguous block of ``n_slots``
        wavelengths (one per 100-Gbps sub-channel of the request).  A start
        ``w`` is feasible only if every slot ``w .. w+n_slots-1`` is:
          * not the quantum slot or its guard bands,
          * free on all path edges, and
          * leaves the post-allocation SKR ≥ the key demand on all path edges
            once *all* ``n_slots`` channels are added.
        Start positions with ``w + n_slots > W`` are masked out.
        """
        n = self._n_slots

        # Per-slot usability: quantum guard band first.
        q_slot = int(round(
            (self.cfg.quantum_freq_hz - self.cfg.base_freq_hz)
            / self.cfg.channel_spacing_hz
        ))
        slot_usable = np.ones(self.W, dtype=bool)
        for gs in range(q_slot - 1, q_slot + 2):
            if 0 <= gs < self.W:
                slot_usable[gs] = False

        if self.path_mask is None:
            # No path context: a start is valid if its whole block is usable.
            mask = np.zeros(self.W, dtype=np.int8)
            for w in range(self.W - n + 1):
                if slot_usable[w:w + n].all():
                    mask[w] = 1
            return mask

        path_edges = np.where(self.path_mask > 0)[0]
        # Wavelength must be free on ALL path edges.
        for w in range(self.W):
            if not slot_usable[w]:
                continue
            for e in path_edges:
                if self.occupancy[e, w] > 0:
                    slot_usable[w] = False
                    break

        # Candidate starts: full block of n usable (free, non-guard) slots.
        mask = np.zeros(self.W, dtype=np.int8)
        for w in range(self.W - n + 1):
            if slot_usable[w:w + n].all():
                mask[w] = 1

        # SKR feasibility: adding all n channels must keep SKR ≥ key demand.
        # _skr_for_edge returns bps; request_key is kbps → convert.
        key_demand_bps = self.request_key * 1000.0
        for w in range(self.W):
            if mask[w] == 0:
                continue
            for e in path_edges:
                occ_before = self.occupancy[e, w:w + n].copy()
                self.occupancy[e, w:w + n] = 1.0
                new_skr = self._skr_for_edge(e)
                self.occupancy[e, w:w + n] = occ_before
                if new_skr < key_demand_bps:
                    mask[w] = 0
                    break

        return mask

    def block_reason(self) -> str:
        """Classify why all wavelengths are infeasible (call only when mask is all-zero).

        Returns
        -------
        'classical'   – no free slot exists on path edges (classical capacity full)
        'skr'         – free slots exist but adding any one would violate key demand
        'mixed'       – both constraints contribute
        """
        if self.path_mask is None:
            return "classical"

        n = self._n_slots
        path_edges = np.where(self.path_mask > 0)[0]
        q_slot = int(round(
            (self.cfg.quantum_freq_hz - self.cfg.base_freq_hz)
            / self.cfg.channel_spacing_hz
        ))

        occupancy_free = 0   # start positions with a full free, non-guard block
        skr_blocked = 0      # of those, how many fail the SKR constraint
        key_demand_bps = self.request_key * 1000.0

        for w in range(self.W - n + 1):
            block = range(w, w + n)
            # whole block must avoid the quantum guard band
            if any(abs(s - q_slot) <= 1 for s in block):
                continue
            # whole block must be free on all path edges
            occ_ok = all(
                self.occupancy[e, s] == 0 for e in path_edges for s in block
            )
            if not occ_ok:
                continue
            occupancy_free += 1
            # check SKR after adding all n channels
            skr_ok = True
            for e in path_edges:
                self.occupancy[e, w:w + n] = 1.0
                new_skr = self._skr_for_edge(e)
                self.occupancy[e, w:w + n] = 0.0
                if new_skr < key_demand_bps:
                    skr_ok = False
                    break
            if not skr_ok:
                skr_blocked += 1

        if occupancy_free == 0:
            return "classical"
        if skr_blocked == occupancy_free:
            return "skr"
        return "mixed"

    def _random_path(self) -> np.ndarray:
        """Generate a random path mask for the current request.

        Simplified: picks 2 random edges as a path.
        """
        mask = np.zeros(self.num_edges, dtype=np.float32)
        if self.num_edges >= 2:
            indices = self.np_random.choice(self.num_edges, size=2, replace=False)
            mask[indices] = 1.0
        return mask

    # ── Gym API ──────────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.occupancy = np.zeros((self.num_edges, self.W), dtype=np.float32)
        self.edge_skr = self._compute_edge_skr()
        self.step_count = 0

        # Draw first request
        if self._request_queue:
            self.request_bw, self.request_key = self._request_queue.pop(0)
        else:
            self.request_bw, self.request_key = 100.0, 1.0

        self.path_mask = self._random_path()

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def _simulate_departures(self) -> None:
        """Release occupied wavelengths with geometric holding-time probability.

        Each occupied slot departs independently each step with probability
        1 / mean_holding_steps, applied across all edges so the network
        reaches a dynamic steady state rather than filling monotonically.
        """
        p_dep = 1.0 / max(self.cfg.mean_holding_steps, 1)
        released = (self.occupancy > 0) & (
            self.np_random.random(self.occupancy.shape) < p_dep
        )
        if released.any():
            self.occupancy[released] = 0.0

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Simulate departures before processing the new arrival
        self._simulate_departures()

        path_edges = np.where(self.path_mask > 0)[0]

        # Check feasibility after departures (mask may have opened up)
        action_mask = self._get_action_mask()
        feasible = bool(action_mask[action] == 1)

        edge_skr_before = self.edge_skr.copy()

        if feasible:
            # Allocate the contiguous block of n_slots wavelengths.
            n = self._n_slots
            for e in path_edges:
                self.occupancy[e, action:action + n] = 1.0
            self.edge_skr = self._compute_edge_skr()

        # Reward
        reward = compute_reward(
            accepted=feasible,
            edge_skr_after=self.edge_skr if feasible else None,
            edge_skr_before=edge_skr_before if feasible else None,
        )

        # Next request
        if self._request_queue:
            self.request_bw, self.request_key = self._request_queue.pop(0)
        else:
            self.request_bw, self.request_key = 100.0, 1.0

        self.path_mask = self._random_path()

        terminated = self.step_count >= self.cfg.max_steps
        truncated = False

        obs = self._get_obs()
        next_mask = self._get_action_mask()
        info = {
            "action_mask": next_mask,
            "accepted": feasible,
            # classical utilization: fraction of wavelength slots occupied across edges
            "classical_util": float(np.mean(self.occupancy > 0)),
            # aggregate SKR across all edges (bps)
            "aggregate_skr": float(np.sum(self.edge_skr)),
            # block reason (only meaningful when not feasible)
            "block_reason": self.block_reason() if not feasible else "none",
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Simple text render."""
        n_occupied = int(np.sum(self.occupancy))
        total_skr = float(np.sum(self.edge_skr))
        print(
            f"Step {self.step_count}: "
            f"occupied={n_occupied} slots, "
            f"total_SKR={total_skr:.1f} bps"
        )
