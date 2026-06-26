"""Per-edge dual-capacity resource tracking with dynamic QKD-key capacity.

Manages two scalar resources on every undirected edge:

* ``classical_capacity_gbps``  — classical communication bandwidth (Gb/s)
* ``key_capacity_kbps``         — QKD secret-key rate (kb/s), **dynamic**

Key capacity depends on the number of *active* classical channels on the
link because each co-propagating channel contributes FWM + SpRS noise.
Channels are first-fit from low frequency upward: the N-th channel becomes
active once ``classical_used > (N-1) × bandwidth_per_ch``.
"""

import math
from typing import Callable, Dict, List, Tuple

from .utils import EPS, canonical_edge


class EdgeResources:
    """Track classical + QKD key capacity on one undirected link.

    Classical capacity is divided into ``n_max`` channels of
    ``bandwidth_per_ch_gbps`` each (50 GHz WDM grid).  As classical
    utilisation grows, more channels become active and the QKD key
    rate is recomputed on the fly.
    """

    def __init__(
        self,
        edge: Tuple[int, int],
        distance_m: float,
        n_max_channels: int,
        bandwidth_per_ch_gbps: float,
        skr_provider: Callable[[float, int], float],
    ):
        """
        Parameters
        ----------
        edge : (u, v)
            Canonical undirected edge tuple.
        distance_m : float
            Fibre span length [m] — used for SKR computation.
        n_max_channels : int
            Maximum number of classical DWDM channels on this link.
        bandwidth_per_ch_gbps : float
            Data-rate per classical channel [Gb/s] (e.g. 200).
        skr_provider : callable
            ``f(distance_m, n_active_channels) -> key_capacity_kbps``.
        """
        self.edge = edge
        self.distance_m = distance_m
        self.n_max = n_max_channels
        self.bandwidth_per_ch = bandwidth_per_ch_gbps
        self.classical_total: float = n_max_channels * bandwidth_per_ch_gbps

        self.classical_used: float = 0.0
        self.key_used: float = 0.0

        self._skr_provider = skr_provider
        self._cached_n_active: int = 0
        self._key_total: float = skr_provider(distance_m, 0)

    # ------------------------------------------------------------------
    # Active classical channel count
    # ------------------------------------------------------------------

    def _n_active_for(self, classical_used: float) -> int:
        """Number of active channels for a given classical-usage level."""
        if classical_used <= 0.0:
            return 0
        return min(self.n_max, max(0, math.ceil(classical_used / self.bandwidth_per_ch)))

    def _n_active(self) -> int:
        return self._n_active_for(self.classical_used)

    # ------------------------------------------------------------------
    # Key capacity (dynamic)
    # ------------------------------------------------------------------

    @property
    def key_total(self) -> float:
        return self._key_total

    def _update_key_total(self):
        """Recompute key capacity if the active channel count changed."""
        n = self._n_active()
        if n != self._cached_n_active:
            self._cached_n_active = n
            self._key_total = self._skr_provider(self.distance_m, n)

    def _future_key_total(self, additional_classical: float) -> float:
        """Key capacity **after** allocating *additional_classical* Gb/s."""
        future_used = self.classical_used + additional_classical
        future_n = self._n_active_for(future_used)
        if future_n == self._cached_n_active:
            return self._key_total  # no recomputation needed
        return self._skr_provider(self.distance_m, future_n)

    # ------------------------------------------------------------------
    # Residual properties
    # ------------------------------------------------------------------

    @property
    def classical_residual(self) -> float:
        return self.classical_total - self.classical_used

    @property
    def key_residual(self) -> float:
        return self._key_total - self.key_used

    # ------------------------------------------------------------------
    # Capacity checks
    # ------------------------------------------------------------------

    def can_accommodate(
        self,
        classical_demand_gbps: float,
        key_demand_kbps: float,
    ) -> bool:
        """Check if **both** resources suffice.

        Key capacity is evaluated at the *future* active channel count
        (i.e. after the classical demand is hypothetically allocated).
        """
        # Classical check
        if self.classical_residual < classical_demand_gbps - EPS:
            return False
        # Key check — use future n_active
        future_key_cap = self._future_key_total(classical_demand_gbps)
        return future_key_cap - self.key_used - key_demand_kbps >= -EPS

    def classical_ok(self, classical_demand_gbps: float) -> bool:
        """Check classical capacity only (at current n_active)."""
        return self.classical_residual >= classical_demand_gbps - EPS

    def key_ok(self, key_demand_kbps: float) -> bool:
        """Check key capacity only (at current n_active)."""
        return self.key_residual >= key_demand_kbps - EPS

    # ------------------------------------------------------------------
    # Allocate / release
    # ------------------------------------------------------------------

    def allocate(self, classical_demand_gbps: float, key_demand_kbps: float):
        """Deduct resources.  Caller must have checked feasibility."""
        self.classical_used += classical_demand_gbps
        self.key_used += key_demand_kbps
        self._update_key_total()
        # Sanity checks
        if self.classical_used > self.classical_total + EPS:
            raise RuntimeError(
                f"Edge {self.edge}: classical over-allocated "
                f"(used={self.classical_used:.4f}, total={self.classical_total})"
            )
        if self.key_used > self._key_total + EPS:
            raise RuntimeError(
                f"Edge {self.edge}: key over-allocated "
                f"(used={self.key_used:.4f}, total={self._key_total:.4f}, "
                f"n_active={self._cached_n_active})"
            )

    def release(self, classical_demand_gbps: float, key_demand_kbps: float):
        """Return resources."""
        self.classical_used -= classical_demand_gbps
        self.key_used -= key_demand_kbps
        # Clamp tiny negatives from float rounding
        if self.classical_used < 0.0:
            self.classical_used = 0.0
        if self.key_used < 0.0:
            self.key_used = 0.0
        self._update_key_total()

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    @property
    def classical_utilization(self) -> float:
        if self.classical_total < EPS:
            return 0.0
        return self.classical_used / self.classical_total

    @property
    def key_utilization(self) -> float:
        if self._key_total < EPS:
            return 0.0
        return self.key_used / self._key_total


class NetworkResources:
    """Manages EdgeResources across the whole network graph.

    All lookups use canonical undirected edge keys ``(min(u,v), max(u,v))``.
    """

    def __init__(self, graph, classical_provider, qkd_provider, config):
        """
        Parameters
        ----------
        graph : nx.Graph
            Topology with ``length_km`` on each edge.
        classical_provider : callable
            ``f(edge_tuple) -> classical_capacity_gbps``
            (still used for non-dynamic classical modes like csv).
        qkd_provider : callable
            ``f(distance_m, n_active) -> key_capacity_kbps``
            Signature changed: now accepts n_active as 2nd arg.
        config : SimulationConfig
            Provides n_max_classical_channels and bandwidth_per_ch.
        """
        self._edges: Dict[Tuple[int, int], EdgeResources] = {}
        self._n_max = getattr(config, 'n_max_classical_channels', 32)
        self._bw_per_ch = getattr(config, 'classical_bandwidth_per_ch_gbps', 200.0)

        for u, v, data in graph.edges(data=True):
            key = canonical_edge(u, v)
            length_km = data.get("length_km", 0.0)
            distance_m = length_km * 1000.0

            # Classical total capacity (overrideable via provider for csv modes)
            classical_cap = classical_provider(key)
            edge_res = EdgeResources(
                edge=key,
                distance_m=distance_m,
                n_max_channels=self._n_max,
                bandwidth_per_ch_gbps=self._bw_per_ch,
                skr_provider=qkd_provider,
            )
            # Override total if provider gave a different value
            if abs(classical_cap - edge_res.classical_total) > EPS:
                edge_res.classical_total = classical_cap
            self._edges[key] = edge_res

    # ------------------------------------------------------------------
    # Single-edge access
    # ------------------------------------------------------------------

    def get_edge(self, u: int, v: int) -> EdgeResources:
        return self._edges[canonical_edge(u, v)]

    # ------------------------------------------------------------------
    # Path-level operations
    # ------------------------------------------------------------------

    def can_allocate_path(
        self, path: List[int], classical_demand: float, key_demand: float
    ) -> bool:
        """True if every edge on *path* has sufficient residual capacity."""
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if not edge.can_accommodate(classical_demand, key_demand):
                return False
        return True

    def allocate_path(
        self, path: List[int], classical_demand: float, key_demand: float
    ):
        """Deduct resources along *path*.  Call after feasibility check."""
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            edge.allocate(classical_demand, key_demand)

    def release_path(
        self, path: List[int], classical_demand: float, key_demand: float
    ):
        """Return resources along *path*."""
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            edge.release(classical_demand, key_demand)

    # ------------------------------------------------------------------
    # Path feasibility classification (used by routing)
    # ------------------------------------------------------------------

    def classify_path_feasibility(
        self, path: List[int], classical_demand: float, key_demand: float
    ) -> str:
        """Classify a single path w.r.t. dual-capacity constraints.

        Returns one of:
            'feasible'
            'classical_insufficient'
            'key_insufficient'
            'joint_insufficient'
        """
        classical_ok = True
        key_ok = True
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if not edge.classical_ok(classical_demand):
                classical_ok = False
            # Key check: use future n_active (after classical allocation)
            future_key_cap = edge._future_key_total(classical_demand)
            if future_key_cap - edge.key_used - key_demand < -EPS:
                key_ok = False
            if not classical_ok and not key_ok:
                break

        if classical_ok and key_ok:
            return "feasible"
        if not classical_ok and not key_ok:
            return "joint_insufficient"
        if not classical_ok:
            return "classical_insufficient"
        return "key_insufficient"

    # ------------------------------------------------------------------
    # Current total used (for time-weighted accumulation)
    # ------------------------------------------------------------------

    def get_total_used(self):
        """Return (total_classical_used, total_key_used) summed over all edges."""
        c_used = sum(e.classical_used for e in self._edges.values())
        k_used = sum(e.key_used for e in self._edges.values())
        return c_used, k_used

    def get_total_capacity(self):
        """Return (total_classical_capacity, total_key_capacity) summed over edges."""
        c_total = sum(e.classical_total for e in self._edges.values())
        k_total = sum(e.key_total for e in self._edges.values())
        return c_total, k_total

    def get_per_edge_utilization(self):
        """Return two dicts: {edge: classical_util}, {edge: key_util}."""
        c_util = {}
        k_util = {}
        for key, e in self._edges.items():
            c_util[key] = e.classical_utilization
            k_util[key] = e.key_utilization
        return c_util, k_util

    # ------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------

    def items(self):
        return self._edges.items()
