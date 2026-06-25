"""Configuration management for QKD dual-capacity routing simulation."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SecurityLevelConfig:
    """Configuration for one security level of traffic demands."""

    probability: float = 0.5
    bandwidth_options: List[float] = field(default_factory=lambda: [10.0, 40.0])
    key_rate_kbps: float = 1.0


@dataclass
class SimulationConfig:
    """Central configuration for the simulation."""

    # ---- Topology ----
    num_nodes: int = 14

    # ---- Classical capacity mode ----
    classical_capacity_mode: str = "constant"
    # "constant" | "csv" | "gnpy_csv" | "gnpy_optional"
    constant_classical_capacity_gbps: float = 400.0
    classical_capacity_csv_path: Optional[str] = None
    gnpy_result_csv_path: Optional[str] = None
    # GSNR-to-capacity parameters (for gnpy_csv with GSNR data)
    gnpy_bandwidth_ghz: float = 75.0
    gnpy_osnr_margin_db: float = 3.0

    # ---- QKD key capacity mode ----
    qkd_capacity_mode: str = "abstract"
    # "abstract" | "actual_skr"
    abstract_K0_kbps: float = 1000.0
    abstract_alpha_per_km: float = 0.02
    old_project_root: Optional[str] = None

    # ---- Traffic parameters ----
    load_values: List[float] = field(
        default_factory=lambda: [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    )
    mean_holding_time: float = 1.0
    num_requests: int = 5000
    warmup_ratio: float = 0.2

    security_levels: List[SecurityLevelConfig] = field(
        default_factory=lambda: [
            SecurityLevelConfig(
                probability=0.5,
                bandwidth_options=[10.0, 40.0],
                key_rate_kbps=1.0,
            ),
            SecurityLevelConfig(
                probability=0.3,
                bandwidth_options=[40.0, 100.0],
                key_rate_kbps=5.0,
            ),
            SecurityLevelConfig(
                probability=0.2,
                bandwidth_options=[100.0, 400.0],
                key_rate_kbps=10.0,
            ),
        ]
    )

    # ---- Routing ----
    k_paths: int = 5
    strategies: List[str] = field(
        default_factory=lambda: [
            "min_hop",
            "min_distance",
            "key_capacity_aware",
            "dual_capacity_aware",
        ]
    )

    # ---- Random seed ----
    random_seed: int = 2026

    # ---- Output ----
    output_dir: str = "results"

    # ---- Float tolerance ----
    eps: float = 1e-9


DEFAULT_CONFIG = SimulationConfig()
