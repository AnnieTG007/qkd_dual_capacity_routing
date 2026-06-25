"""Classical-capacity provider — constant, CSV, and GNPy-based modes.

Provides a **factory** that returns a callable for per-edge classical
capacity (Gbps).  Supported modes:

``constant`` (default)
    Every edge gets the same fixed capacity (e.g. 400 Gb/s).

``csv``
    Read per-edge capacities from a CSV file with columns
    ``u, v, classical_capacity_gbps``.

``gnpy_csv``
    Read GNPy / GN-model pre-computed results from CSV.  Two CSV formats
    are accepted:

    * Direct capacity:
        ``u, v, classical_capacity_gbps``
    * GSNR-based (detected by presence of a ``gsnr_db`` column):
        ``u, v, gsnr_db, bandwidth_ghz``

    GSNR values are mapped to capacity via a Shannon-like abstraction::

        gsnr_linear = 10 ** (gsnr_db / 10)
        capacity_gbps = bandwidth_hz * log2(1 + gsnr_linear / margin) / 1e9

    Note: This formula is a **course-level abstraction** for converting
    GNPy QoT results into link capacity; it is not a vendor net-rate model.

``gnpy_optional``
    Placeholder — same semantics as ``gnpy_csv`` but prints a reminder to
    pre-compute capacities with GNPy offline if the result file is absent.
"""

import csv
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import networkx as nx

from .config import SimulationConfig
from .utils import canonical_edge


def initialize_classical_provider(
    mode: str,
    graph: nx.Graph,
    config: SimulationConfig,
) -> Callable[[Tuple[int, int]], float]:
    """Return a callable ``f(edge: (int,int)) -> classical_capacity_gbps``.

    Parameters
    ----------
    mode : str
        One of ``"constant"``, ``"csv"``, ``"gnpy_csv"``, ``"gnpy_optional"``.
    graph : nx.Graph
        The topology graph.
    config : SimulationConfig
        Full simulation configuration.

    Returns
    -------
    callable
        Function taking a canonical edge tuple ``(u, v)`` and returning
        classical link capacity in **Gb/s**.
    """
    if mode == "constant":
        return _build_constant_provider(config)

    if mode == "csv":
        return _build_csv_provider(config)

    if mode == "gnpy_csv":
        return _build_gnpy_csv_provider(config)

    if mode == "gnpy_optional":
        return _build_gnpy_optional_provider(config)

    raise ValueError(
        f"Unknown classical capacity mode: {mode!r}. "
        f"Supported: 'constant', 'csv', 'gnpy_csv', 'gnpy_optional'."
    )


# ---------------------------------------------------------------------------
# Constant provider
# ---------------------------------------------------------------------------

def _build_constant_provider(
    config: SimulationConfig,
) -> Callable[[Tuple[int, int]], float]:
    cap = config.constant_classical_capacity_gbps

    def constant_cap(edge: Tuple[int, int]) -> float:
        return cap

    return constant_cap


# ---------------------------------------------------------------------------
# CSV provider
# ---------------------------------------------------------------------------

def _build_csv_provider(
    config: SimulationConfig,
) -> Callable[[Tuple[int, int]], float]:
    path = config.classical_capacity_csv_path
    if path is None:
        raise RuntimeError(
            "CSV classical capacity mode requires --classical-capacity-csv PATH."
        )
    edge_caps = _read_capacity_csv(path)
    return _make_dict_provider(edge_caps, "classical capacity CSV")


# ---------------------------------------------------------------------------
# GNPy CSV provider
# ---------------------------------------------------------------------------

def _build_gnpy_csv_provider(
    config: SimulationConfig,
) -> Callable[[Tuple[int, int]], float]:
    path = config.gnpy_result_csv_path
    if path is None:
        raise RuntimeError(
            "gnpy_csv mode requires --gnpy-result-csv PATH."
        )
    edge_caps = _read_gnpy_csv(
        path,
        bandwidth_ghz=config.gnpy_bandwidth_ghz,
        osnr_margin_db=config.gnpy_osnr_margin_db,
    )
    return _make_dict_provider(edge_caps, "GNPy result CSV")


def _build_gnpy_optional_provider(
    config: SimulationConfig,
) -> Callable[[Tuple[int, int]], float]:
    path = config.gnpy_result_csv_path
    if path is None:
        print(
            "[gnpy_optional] No GNPy CSV path given. "
            "Falling back to constant capacity (400 Gbps).\n"
            "To use GNPy results, first run GNPy offline and provide:\n"
            "  --gnpy-result-csv data/gnpy_capacity.csv",
            file=sys.stderr,
        )
        return _build_constant_provider(config)

    csv_path = Path(path)
    if not csv_path.is_file():
        print(
            f"[gnpy_optional] GNPy CSV not found at {csv_path}. "
            f"Falling back to constant capacity (400 Gbps).\n"
            f"Generate the file offline with GNPy, then re-run.",
            file=sys.stderr,
        )
        return _build_constant_provider(config)

    edge_caps = _read_gnpy_csv(
        path,
        bandwidth_ghz=config.gnpy_bandwidth_ghz,
        osnr_margin_db=config.gnpy_osnr_margin_db,
    )
    return _make_dict_provider(edge_caps, "GNPy optional")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dict_provider(
    edge_caps: Dict[Tuple[int, int], float],
    description: str,
) -> Callable[[Tuple[int, int]], float]:
    """Wrap a per-edge capacity dictionary into a callable with fallback."""

    def provider(edge: Tuple[int, int]) -> float:
        key = canonical_edge(*edge)
        if key in edge_caps:
            return edge_caps[key]
        # If not found, return 0 — the edge effectively has no capacity
        print(
            f"[{description}] Warning: edge {key} not found in data; "
            f"returning 0 Gbps.",
            file=sys.stderr,
        )
        return 0.0

    return provider


def _read_capacity_csv(
    csv_path: str,
) -> Dict[Tuple[int, int], float]:
    """Read ``u, v, classical_capacity_gbps`` CSV."""
    edge_caps: Dict[Tuple[int, int], float] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["u"])
            v = int(row["v"])
            cap = float(row["classical_capacity_gbps"])
            edge_caps[canonical_edge(u, v)] = cap
    return edge_caps


def _read_gnpy_csv(
    csv_path: str,
    bandwidth_ghz: float = 75.0,
    osnr_margin_db: float = 3.0,
) -> Dict[Tuple[int, int], float]:
    """Read GNPy CSV — auto-detect direct-capacity vs GSNR format.

    Direct-capacity format:
        ``u, v, classical_capacity_gbps``

    GSNR format:
        ``u, v, gsnr_db, bandwidth_ghz``
        (bandwidth_ghz column is **per-row**, overriding the global default
        when present).

    GSNR → capacity mapping (Shannon-like abstraction)::

        gsnr_linear  = 10 ** (gsnr_db / 10)
        margin_linear = 10 ** (osnr_margin_db / 10)
        capacity_gbps = B_hz * log2(1 + gsnr_linear / margin_linear) / 1e9

    This is a **course-level abstraction** for linking GNPy QoT results
    to link capacity; it is not a line-rate or net-rate model.
    """
    edge_caps: Dict[Tuple[int, int], float] = {}
    margin_linear = 10.0 ** (osnr_margin_db / 10.0)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            u = int(row["u"])
            v = int(row["v"])
            key = canonical_edge(u, v)

            if "gsnr_db" in fieldnames:
                # GSNR → Shannon capacity
                gsnr_db = float(row["gsnr_db"])
                gsnr_linear = 10.0 ** (gsnr_db / 10.0)

                bw_ghz = (
                    float(row["bandwidth_ghz"])
                    if "bandwidth_ghz" in fieldnames
                    else bandwidth_ghz
                )
                bw_hz = bw_ghz * 1e9

                # Shannon-like capacity in Gbps
                cap = bw_hz * math.log2(1.0 + gsnr_linear / margin_linear) / 1e9
            else:
                # Direct capacity
                cap = float(row["classical_capacity_gbps"])

            edge_caps[key] = cap

    return edge_caps
