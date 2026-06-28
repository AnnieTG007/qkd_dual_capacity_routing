"""Self-contained BB84 QKD physical-layer formulas with GNPy-based noise model.

The noise model mirrors the **discrete single-frequency FWM + SpRS**
implementation from ``qkd_optical_network/src/qkd_sim/physical/noise/``,
using GNPy-official Raman gain data and fibre parameters.

Noise sources
-------------
**FWM** — per valid frequency-triplet (f₂, f₃, f₄) with f₂ = f₃+f₄-f_q:
  η = [exp(-Δα·L) - 2·exp(-Δα·L/2)·cos(Δβ·L) + 1] / [(Δα)²/4 + (Δβ)²]
  P_fwd = exp(-α_q·L) · (γ²/9) · D² · η · P₃ · P₄ · P₂

**SpRS** — per pump channel, with phonon-occupation-corrected cross-section:
  Stokes (f_c > f_q):
    σ = 2·h·f_q · g_R · (1 + n_th) · B_noise
  Anti-Stokes (f_c < f_q):
    σ = 2·h·f_q · g_R · n_th · (f_q/f_c) · B_noise

**GNPy Raman gain** — 92-point SSMF table, frequency + area corrected:
  g_R = g0(|Δf|) · (f_pump/f_ref) · (A_eff_ref/A_eff)

SKR Models
----------
* ``approx_finite_key_rate`` — 3-state decoy BB84 + Gaussian finite-key
  corrections (**recommended for course paper**)
* ``infinite_key_rate`` — single-intensity infinite-key BB84 (comparison)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

# ============================================================================
# Physical constants
# ============================================================================

PLANCK: float = 6.62607015e-34  # J·s
C_LIGHT: float = 299792458.0  # m/s
BOLTZMANN: float = 1.380649e-23  # J/K

# ============================================================================
# Fibre parameters — SSMF (G.652) from fiber_para/fiber_smf.yaml
# ============================================================================

ALPHA_DB_PER_KM: float = 0.2  # attenuation [dB/km]
ALPHA_PER_M: float = ALPHA_DB_PER_KM * (math.log(10) / 10.0) * 1e-3  # ≈ 4.605e-5 m⁻¹
GAMMA_PER_W_M: float = 1.3e-3  # nonlinear coefficient [W⁻¹·m⁻¹]
D_C_S_PER_M2: float = 17.0e-6  # chromatic dispersion at 193.5 THz [s/m²]
D_SLOPE_S_PER_M3: float = 0.056e3  # dispersion slope [s/m³]
A_EFF_M2: float = 8.0e-11  # effective area [m²] (80 μm²)
RAYLEIGH_COEFF: float = 4.8e-8  # Rayleigh backscatter recapture [m⁻³]
T_KELVIN: float = 300.0  # fibre temperature [K]

# ============================================================================
# Detector / system — skr_para/bb84_config.yaml (custom profile)
# ============================================================================

ETA_SPD: float = 0.25  # SPD quantum efficiency
IL_DB: float = 6.0  # insertion loss [dB] (excl. fibre)
IL_LINEAR: float = 10.0 ** (-IL_DB / 10.0)  # ≈ 0.251
DARK_COUNT_PROB: float = 1.0e-6  # dark-count probability per gate
E_DET: float = 0.01  # detector intrinsic error rate
F_EC: float = 1.16  # error-correction efficiency
R_REP: float = 5.0e7  # pulse repetition rate [Hz]
Q_SIFTING: float = 0.5  # BB84 sifting efficiency
GATE_TIME_S: float = 1.0e-9  # SPD gate width [s]

# ============================================================================
# Decoy-state protocol
# ============================================================================

MU_SIGNAL: float = 0.75  # signal-state mean photon number (μ)
MU_DECOY: float = 0.2  # decoy-state mean photon number (ν)
P_SIGNAL: float = 0.875  # signal-state emission probability (p_μ)
P_DECOY: float = 0.0625  # decoy-state emission probability (p_ν)

# ============================================================================
# Finite-key parameters
# ============================================================================

N_PULSE: float = 1.0e10  # total transmitted pulses (Alice-side)
GAMMA_KS: float = 5.3  # Gaussian confidence multiplier

# ============================================================================
# Classical channel plan — WDM grid defaults
# ============================================================================
# Classical channels occupy low-frequency C-band (first-fit, lowest first).
# Quantum channel sits at the high end of C-band with a wide guard gap.
#
#   Classical        ←── guard ──→  Quantum
#   190.0 .. 191.55 THz             193.5 THz ± 12.5 GHz
#   (32 ch × 50 GHz)                (25 GHz BW)
#
# Effective guard: 193.5 − 191.5625 ≈ 1.94 THz

N_CLASSICAL_CHANNELS: int = 16          # max classical DWDM channels per edge
N_QUANTUM_CHANNELS: int = 16           # parallel QKD quantum channels per edge
CLASSICAL_SPACING_HZ: float = 50e9      # channel spacing [Hz] (50 GHz)
CLASSICAL_BANDWIDTH_PER_CH_GBPS: float = 100.0  # data-rate per classical ch [Gb/s]
CLASSICAL_POWER_PER_CH_W: float = 3.16e-6  # fibre launch power per ch [W] (−25 dBm)
CLASSICAL_BASE_FREQ_HZ: float = 190.0e12  # lowest classical centre freq [Hz]
QUANTUM_FREQ_HZ: float = 193.5e12       # quantum-channel centre frequency [Hz]
QUANTUM_BANDWIDTH_GHZ: float = 25.0     # quantum receiver noise bandwidth [GHz]

# ============================================================================
# GNPy 92-point SSMF Raman gain coefficient table
# ============================================================================
# Source: https://gnpy.readthedocs.io/en/master/model.html
# Reference conditions: f_ref = 206.185 THz (1454 nm), A_eff = 75.75 μm²
# Frequency correction:  g_R ∝ f_pump / f_ref
# Area correction:       g_R ∝ A_eff_ref / A_eff

_RAMAN_FREQ_OFFSET_THZ: np.ndarray = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
    6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 12.75, 13.0, 13.25, 13.5, 14.0, 14.5, 14.75, 15.0,
    15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.25, 18.5, 18.75, 19.0,
    19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
    25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0,
    30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5,
    36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0,
    41.5, 42.0,
], dtype=np.float64)  # [THz]

_RAMAN_G0_PER_WM: np.ndarray = np.array([
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
], dtype=np.float64)  # [1/(W·m)]

_RAMAN_F_REF: float = 206.184634112792e12  # [Hz] (1454 nm)
_RAMAN_A_EFF_REF: float = 75.74659443542413e-12  # [m²]


def _get_raman_gain(
    delta_f: float,
    f_pump: float,
) -> float:
    """Interpolate GNPy Raman gain g_R [1/(W·m)] at frequency offset |Δf|.

    g_R = g0(|Δf|) · (f_pump / f_ref) · (A_eff_ref / A_eff)

    Returns 0 for |Δf| > 42 THz (outside the GNPy table).
    """
    abs_df = abs(delta_f)
    if abs_df > 42e12:
        return 0.0
    g0 = float(
        np.interp(abs_df, _RAMAN_FREQ_OFFSET_THZ * 1e12, _RAMAN_G0_PER_WM,
                  left=0.0, right=0.0)
    )
    return g0 * (f_pump / _RAMAN_F_REF) * (_RAMAN_A_EFF_REF / A_EFF_M2)


# ============================================================================
# Math helpers
# ============================================================================

def _H2(x: float) -> float:
    """Binary Shannon entropy:  h(x) = -x·log₂x - (1-x)·log₂(1-x)."""
    x = min(max(float(x), 1e-15), 1.0 - 1e-15)
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================
# FWM: discrete single-frequency triplet enumeration (GNPy-aligned)
# ============================================================================

def _phase_mismatch(
    f2: float,
    f3: float,
    f4: float,
    f_ref: float = 193.5e12,
) -> float:
    """Phase mismatch Δβ for FWM triplet (f₂, f₃, f₄).

    From fwm_kernels._phase_mismatch (GNPy convention):
        Δβ = (2π λ²/c) · |f₃-f₂| · |f₄-f₂|
             · [D_c(f₂) + (λ²/(2c))·(|f₃-f₂|+|f₄-f₂|)·D_slope]
        λ = c / f₂
        D_c(f₂) = D_c + D_slope · (λ - λ_ref)
    """
    lam = C_LIGHT / f2
    lam_ref = C_LIGHT / f_ref
    D_c_f2 = D_C_S_PER_M2 + D_SLOPE_S_PER_M3 * (lam - lam_ref)
    df32 = abs(f3 - f2)
    df42 = abs(f4 - f2)
    return (
        (2.0 * math.pi * lam ** 2.0 / C_LIGHT)
        * df32 * df42
        * (D_c_f2 + (lam ** 2.0 / (2.0 * C_LIGHT)) * (df32 + df42) * D_SLOPE_S_PER_M3)
    )


def _fwm_coefficient(
    delta_alpha: float,
    delta_beta: float,
    L: float,
) -> float:
    """FWM efficiency η (GNPy convention, from fwm_kernels._fwm_coefficient).

    η = [exp(-Δα·L) - 2·exp(-Δα·L/2)·cos(Δβ·L) + 1] / [(Δα)²/4 + (Δβ)²]
    """
    da = delta_alpha
    db = delta_beta
    num = math.exp(-da * L) - 2.0 * math.exp(-da * L / 2.0) * math.cos(db * L) + 1.0
    denom = (da ** 2.0) / 4.0 + db ** 2.0
    return num / max(denom, 1e-30)


def _compute_fwm_noise(
    distance_m: float,
    classical_freqs: List[float],
    P_ch: float,
    f_quantum: float,
) -> float:
    """Sum FWM noise over valid (f₃, f₄) pairs matching f₂ = f₃+f₄-f_q.

    Uses the GNPy-aligned discrete single-frequency FWM formula from
    fwm_solver.DiscreteFWMSolver.  For each (f₃, f₄) classical pair,
    the required f₂ is matched against the classical grid with a
    tolerance of channel_spacing / 20 (~2.5 GHz at 50 GHz spacing).

    Uniform-loss approximation: all channels have the same α, so
    Δα = 0 (simplifies η).
    """
    L = distance_m
    alpha = ALPHA_PER_M
    gamma = GAMMA_PER_W_M
    tol = CLASSICAL_SPACING_HZ / 20.0  # ~2.5 GHz at 50 GHz spacing

    freqs_arr = np.array(classical_freqs, dtype=np.float64)
    N = len(freqs_arr)
    if N < 2:
        return 0.0

    total_fwm: float = 0.0

    # Meshgrid of (f3, f4) pairs
    f3_grid, f4_grid = np.meshgrid(freqs_arr, freqs_arr, indexing="ij")

    # f2_needed = f3 + f4 - f_q  (for each pair)
    f2_needed = f3_grid + f4_grid - f_quantum  # (N, N)

    # Find nearest classical channel to each f2_needed
    idx = np.searchsorted(freqs_arr, f2_needed.ravel())
    idx = np.clip(idx, 1, N - 1)
    left = freqs_arr[idx - 1]
    right = freqs_arr[idx]
    nearest = np.where(
        np.abs(f2_needed.ravel() - left) < np.abs(f2_needed.ravel() - right),
        left, right,
    ).reshape(N, N)

    # Valid mask: f2 matches within tolerance, and exclude SPM/XPM (f2==f3 or f2==f4)
    f2_match = np.abs(nearest - f2_needed) < tol
    not_spm = (
        (np.abs(nearest - f3_grid) > 1e6)  # f2 ≠ f3
        & (np.abs(nearest - f4_grid) > 1e6)  # f2 ≠ f4
    )
    valid = f2_match & not_spm

    if not valid.any():
        return 0.0

    # Iterate only over valid triplets (typically O(100-1000), not O(N³))
    valid_indices = np.argwhere(valid)
    exp_neg_alpha_L = math.exp(-alpha * L)

    for ri, ci in valid_indices:
        f3 = freqs_arr[ri]
        f4 = freqs_arr[ci]
        f2 = float(nearest[ri, ci])
        P3 = P_ch
        P4 = P_ch
        P2 = P_ch  # uses same per-channel power

        # Degeneracy factor
        D_factor = 3.0 if abs(f3 - f4) < 1e6 else 6.0

        # Phase mismatch at f2
        db = _phase_mismatch(f2, f3, f4)

        # FWM efficiency (Δα = 0 — uniform C-band loss)
        # Δα = α₄+α₃-α₂-α₁ ≈ 0 when all channels have same α
        eta = _fwm_coefficient(0.0, db, L)

        # Forward FWM noise power (GNPy formula)
        # P_fwd = exp(-α_q·L) · (γ²/9) · D² · η · P₃·P₄·P₂
        p_fwd = (
            exp_neg_alpha_L
            * (gamma ** 2.0 / 9.0)
            * (D_factor ** 2.0)
            * eta
            * P3 * P4 * P2
        )
        total_fwm += float(p_fwd)

    return total_fwm


# ============================================================================
# SpRS: per-pump Raman scattering (GNPy Raman gain + phonon occupation)
# ============================================================================

def _phonon_occupation(delta_f: float, T: float) -> float:
    """Bose-Einstein phonon occupation factor n_th.

    n_th(Δf) = 1 / [exp(h·Δf / (k·T)) - 1]

    Returns 0 at Δf = 0.
    """
    abs_df = abs(delta_f)
    if abs_df < 1e-6:
        return 0.0
    exponent = PLANCK * abs_df / (BOLTZMANN * T)
    # Clamp to avoid overflow
    if exponent > 700.0:
        return 0.0
    return 1.0 / (math.exp(exponent) - 1.0)


def _raman_cross_section(
    f_q: float,
    f_c: float,
    g_R: float,
    n_th: float,
    noise_bw_hz: float,
) -> float:
    """Raman cross-section σ [1/m] with Stokes / anti-Stokes distinction.

    From sprs_kernels._raman_cross_section (formulas_sprs.md §3.1.4):

    Stokes (f_c > f_q, pump above quantum, scattered photon is Stokes):
        σ = 2·h·f_q · g_R · (1 + n_th) · B_noise

    Anti-Stokes (f_c < f_q):
        σ = 2·h·f_q · g_R · n_th · (f_q / f_c) · B_noise
    """
    if f_c > f_q:
        # Stokes
        return 2.0 * PLANCK * f_q * g_R * (1.0 + n_th) * noise_bw_hz
    else:
        # Anti-Stokes
        if abs(f_c - f_q) < 1e6:
            return 0.0
        return 2.0 * PLANCK * f_q * g_R * n_th * (f_q / f_c) * noise_bw_hz


def _compute_sprs_noise(
    distance_m: float,
    classical_freqs: List[float],
    P_ch: float,
    f_quantum: float,
    noise_bw_hz: float,
) -> float:
    """Sum forward + backward SpRS noise over classical pump channels.

    Forward propagation (sparse_kernels._forward_propagation):
      P_fwd = P_pump · σ · exp(-α_q·L) · [1 − exp(-(α_c−α_q)·L)] / (α_c − α_q)

    Backward propagation (sparse_kernels._backward_propagation):
      P_bwd = P_pump · σ · [1 − exp(-(α_c+α_q)·L)] / (α_c + α_q)

    Uniform-loss approximation: α_c ≈ α_q = α → forward uses L'Hôpital limit.
    """
    L = distance_m
    alpha = ALPHA_PER_M
    exp_neg_alpha_L = math.exp(-alpha * L)
    total_sprs: float = 0.0

    for f_pump in classical_freqs:
        delta_f = f_pump - f_quantum
        if abs(delta_f) > 42e12:
            continue  # beyond GNPy Raman table

        g_R = _get_raman_gain(delta_f, f_pump)
        n_th = _phonon_occupation(delta_f, T_KELVIN)
        sigma = _raman_cross_section(f_quantum, f_pump, g_R, n_th, noise_bw_hz)

        # Forward: α_c ≈ α_q  →  integral → L · exp(-αL)
        p_fwd = P_ch * sigma * exp_neg_alpha_L * L

        # Backward
        sum_alpha = 2.0 * alpha
        p_bwd = P_ch * sigma * (1.0 - math.exp(-sum_alpha * L)) / sum_alpha

        total_sprs += float(p_fwd + p_bwd)

    return total_sprs


# ============================================================================
# Unified noise estimation
# ============================================================================

def estimate_noise_photon_prob(
    distance_m: float,
    num_classical_channels: Optional[int] = None,
    classical_power_per_ch_w: Optional[float] = None,
    classical_spacing_hz: Optional[float] = None,
    classical_base_freq_hz: Optional[float] = None,
    quantum_freq_hz: Optional[float] = None,
    noise_bandwidth_hz: Optional[float] = None,
) -> float:
    """Estimate total noise photon-count probability per gate for a QKD link.

    Uses GNPy-aligned discrete FWM + SpRS with 92-point Raman table.
    Classical channels are placed at **low C-band frequencies** (first-fit,
    lowest first) below the quantum channel.  The guard band is implicit:
    ``f_quantum − f_classical_last`` grows when fewer channels are active.

    Parameters
    ----------
    distance_m : float
        Fibre span length [m].
    num_classical_channels : int, optional
        Number of **active** classical DWDM channels (0 = dark fibre).
    classical_power_per_ch_w : float, optional
        Launch power per classical channel [W] (default −10 dBm).
    classical_spacing_hz : float, optional
        Classical channel spacing [Hz] (default 50 GHz).
    classical_base_freq_hz : float, optional
        Lowest classical channel centre frequency [Hz] (default 190.0 THz).
    quantum_freq_hz : float, optional
        Quantum channel centre frequency [Hz] (default 193.5 THz).
    noise_bandwidth_hz : float, optional
        Quantum receiver noise bandwidth [Hz] (default 25 GHz).

    Returns
    -------
    p_noise : float
        Noise photon-count probability per gate (dimensionless).
    """
    N_ch = num_classical_channels if num_classical_channels is not None else 0
    if N_ch <= 0:
        return 0.0

    P_ch = classical_power_per_ch_w or CLASSICAL_POWER_PER_CH_W
    Δf = classical_spacing_hz or CLASSICAL_SPACING_HZ
    f_q = quantum_freq_hz or QUANTUM_FREQ_HZ
    B_noise = noise_bandwidth_hz or (QUANTUM_BANDWIDTH_GHZ * 1e9)
    f_base = classical_base_freq_hz or CLASSICAL_BASE_FREQ_HZ

    # Classical channels: first-fit from low frequency upward
    classical_freqs = [f_base + i * Δf for i in range(N_ch)]

    P_fwm = _compute_fwm_noise(distance_m, classical_freqs, P_ch, f_q)
    P_sprs = _compute_sprs_noise(distance_m, classical_freqs, P_ch, f_q, B_noise)
    P_noise = P_fwm + P_sprs

    # Noise power → photon-count probability
    # N_photon = P_noise · τ_gate · η_SPD · IL_linear / (h · f_q)
    return P_noise * GATE_TIME_S * ETA_SPD * IL_LINEAR / (PLANCK * f_q)


# ============================================================================
# Channel model (per-intensity)
# ============================================================================

def _channel_quantities(
    distance_m: float,
    mu: float,
    p_noise: float = 0.0,
):
    """Channel intermediate quantities for a given intensity μ."""
    eta = ETA_SPD * math.exp(-ALPHA_PER_M * distance_m) * IL_LINEAR
    p_background = _clip(0.0, 1.0, DARK_COUNT_PROB + p_noise)
    Y0 = 1.0 - (1.0 - p_background) ** 2.0
    exp_term = math.exp(-mu * eta)
    Q_mu = float(Y0 + 1.0 - exp_term)
    e0 = 0.5
    E_mu = float((e0 * Y0 + E_DET * (1.0 - exp_term)) / max(Q_mu, 1e-30))
    return float(eta), float(Y0), Q_mu, E_mu


# ============================================================================
# Model 2 — approx finite-key (3-state decoy + Gaussian correction)
# ============================================================================

def approx_finite_key_rate(
    distance_m: float,
    p_noise: float = 0.0,
) -> tuple[float, float, float]:
    """Approx finite-key BB84 SKR with 3-state decoy analysis.

    Implements formulas §2 from docs/formulas_skr.md.
    Returns (skr_bps, skr_bit_per_pulse, qber).
    """
    mu = MU_SIGNAL
    nu = MU_DECOY
    p_mu = P_SIGNAL
    p_nu = P_DECOY
    N_pulse = N_PULSE
    gamma = GAMMA_KS
    e0 = 0.5

    eta, Y0, Q_mu, E_mu = _channel_quantities(distance_m, mu, p_noise)
    _, _, Q_nu, E_nu = _channel_quantities(distance_m, nu, p_noise)

    # Gaussian finite-key corrections on decoy
    denom_Q = max(p_nu * Q_nu * N_pulse / 2.0, 1e-30)
    Q_nu_L = Q_nu * (1.0 - gamma / math.sqrt(denom_Q))
    Q_nu_L = max(Q_nu_L, 0.0)

    EnuQnu = E_nu * Q_nu
    denom_E = max(p_nu * EnuQnu * N_pulse / 2.0, 1e-30)
    EnuQnu_U = EnuQnu * (1.0 + gamma / math.sqrt(denom_E))
    EnuQnu_U = max(EnuQnu_U, 0.0)

    # Y₁ lower bound
    denom_Y1 = mu * nu - nu ** 2.0
    if abs(denom_Y1) < 1e-30:
        return 0.0, 0.0, float(E_mu)
    Y1_L = (mu / denom_Y1) * (
        Q_nu_L * math.exp(nu)
        - (nu ** 2.0 / mu ** 2.0) * Q_mu * math.exp(mu)
        - (mu ** 2.0 - nu ** 2.0) / mu ** 2.0 * Y0
    )
    Y1_L = max(Y1_L, 0.0)

    # Q₁ lower bound, e₁ upper bound
    Q1_L = Y1_L * mu * math.exp(-mu)
    if Y1_L < 1e-30:
        return 0.0, 0.0, float(E_mu)
    e1_U = (EnuQnu_U * math.exp(nu) - e0 * Y0) / (nu * Y1_L)
    e1_U = _clip(0.0, 0.5, e1_U)

    # Secure key rate
    skr = p_mu * Q_SIFTING * (
        -Q_mu * F_EC * _H2(E_mu) + Q1_L * (1.0 - _H2(e1_U))
    )
    skr_bps = float(max(0.0, skr)) * R_REP
    return skr_bps, skr_bps / max(R_REP, 1.0), float(E_mu)


# ============================================================================
# Infinite-key (single-intensity, no decoy) — for comparison
# ============================================================================

def infinite_key_rate(
    distance_m: float,
    p_noise: float = 0.0,
) -> tuple[float, float, float]:
    """Infinite-key BB84 SKR (GLLP / Shor-Preskill)."""
    mu = MU_SIGNAL
    eta, Y0, Q_mu, E_mu = _channel_quantities(distance_m, mu, p_noise)
    e0 = 0.5
    Y1 = Y0 + eta
    Q1 = Y1 * mu * math.exp(-mu)
    e1 = (e0 * Y0 + E_DET * eta) / max(Y1, 1e-30)
    e1 = _clip(0.0, 0.5, e1)
    skr = Q_SIFTING * (-Q_mu * F_EC * _H2(E_mu) + Q1 * (1.0 - _H2(e1)))
    skr_bps = float(max(0.0, skr)) * R_REP
    return skr_bps, skr_bps / max(R_REP, 1.0), float(E_mu)


# ============================================================================
# Convenience — key-capacity provider
# ============================================================================

def get_key_capacity_kbps(
    distance_m: float,
    num_classical_channels: int = 0,
    n_quantum_channels: int = 4,
) -> float:
    """Return QKD key capacity in kb/s for a link of given length.

    ``num_classical_channels`` is the number of **currently active**
    classical channels on this link (0 = dark fibre).  When >0 the
    GNPy-based FWM + SpRS noise model is engaged.

    ``n_quantum_channels`` is the number of parallel QKD quantum
    channels multiplexed on the same fibre (default 4).  Total key
    capacity scales linearly with the channel count.

    Parameters
    ----------
    distance_m : float
        Fibre span length [m].
    num_classical_channels : int
        Active classical channels (0 = dark fibre).
    n_quantum_channels : int
        Number of parallel QKD quantum channels.

    Returns
    -------
    float
        Key capacity in kb/s (≥ 0).
    """
    if num_classical_channels > 0:
        p_noise = estimate_noise_photon_prob(
            distance_m,
            num_classical_channels=num_classical_channels,
        )
    else:
        p_noise = 0.0

    skr_bps, _bpp, _qber = approx_finite_key_rate(distance_m, p_noise)
    return float(max(skr_bps, 0.0)) / 1000.0 * n_quantum_channels
