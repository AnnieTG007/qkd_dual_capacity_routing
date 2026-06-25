"""Traffic request generation — Poisson arrivals + exponential holding times.

Generates the complete request sequence upfront for reproducibility.
The same sequence is reused across all routing strategies at a given load.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np

from .config import SecurityLevelConfig, SimulationConfig


@dataclass
class Request:
    """A single connection request with dual classical + QKD-key demands."""

    request_id: int
    src: int
    dst: int
    arrival_time: float
    holding_time: float
    bandwidth_gbps: float
    key_rate_kbps: float
    security_level: str  # "Low", "Medium", "High"

    @property
    def departure_time(self) -> float:
        return self.arrival_time + self.holding_time


def generate_request_sequence(
    graph: nx.Graph,
    num_requests: int,
    offered_load: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> List[Request]:
    """Generate *num_requests* with Poisson arrivals and exponential holding.

    Parameters
    ----------
    graph : nx.Graph
        Used to filter source-destination pairs that have at least one path.
    num_requests : int
        Total number of requests to generate.
    offered_load : float
        rho = lambda / mu  (Erlangs).  Determines the mean arrival rate.
    config : SimulationConfig
        Mean holding time and security-level definitions.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list of Request
        Requests ordered by arrival_time.
    """
    mean_hold = config.mean_holding_time
    mu = 1.0 / mean_hold  # departure rate
    lam = offered_load * mu  # arrival rate
    mean_inter_arrival = 1.0 / lam

    # Nodes that are in the graph
    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    # Security-level cumulative probabilities
    sec_levels: List[SecurityLevelConfig] = config.security_levels
    sec_probs = np.array([s.probability for s in sec_levels], dtype=float)
    sec_probs /= sec_probs.sum()  # normalise
    sec_cumprobs = np.cumsum(sec_probs)

    security_names = ["Low", "Medium", "High"]

    requests: List[Request] = []

    current_time = 0.0
    for req_id in range(num_requests):
        # Inter-arrival time (exponential)
        inter_arrival = rng.exponential(mean_inter_arrival)
        current_time += inter_arrival

        # Holding time (exponential)
        holding_time = rng.exponential(mean_hold)

        # Random source != destination (uniform over all ordered pairs)
        src, dst = _pick_src_dst(graph, nodes, n_nodes, rng)

        # Security level
        p = rng.uniform()
        level_idx = int(np.searchsorted(sec_cumprobs, p))
        level_idx = min(level_idx, len(sec_levels) - 1)
        lev = sec_levels[level_idx]

        # Bandwidth — pick uniformly from options
        bandwidth_gbps = float(rng.choice(lev.bandwidth_options))
        key_rate_kbps = lev.key_rate_kbps

        requests.append(
            Request(
                request_id=req_id,
                src=src,
                dst=dst,
                arrival_time=current_time,
                holding_time=holding_time,
                bandwidth_gbps=bandwidth_gbps,
                key_rate_kbps=key_rate_kbps,
                security_level=security_names[level_idx],
            )
        )

    return requests


def _pick_src_dst(
    graph: nx.Graph,
    nodes: List[int],
    n_nodes: int,
    rng: np.random.Generator,
):
    """Pick src != dst such that a path exists."""
    for _ in range(200):  # safety limit
        src = int(rng.integers(0, n_nodes))
        dst = int(rng.integers(0, n_nodes))
        if src != dst and nx.has_path(graph, src, dst):
            return src, dst
    # Fallback: iterate
    for src in nodes:
        for dst in nodes:
            if src != dst and nx.has_path(graph, src, dst):
                return src, dst
    raise RuntimeError("Graph has no connected node pairs!")
