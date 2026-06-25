"""Per-edge dual-capacity resource tracking.

Manages two scalar resources on every undirected edge:

* ``classical_capacity_gbps``  — classical communication bandwidth (Gb/s)
* ``key_capacity_kbps``         — QKD secret-key rate (kb/s)
"""

from typing import Dict, List, Tuple

from .utils import EPS, canonical_edge


class EdgeResources:
    """Track classical + QKD key capacity on one undirected link."""

    def __init__(
        self,
        edge: Tuple[int, int],
        classical_total_gbps: float,
        key_total_kbps: float,
    ):
        self.edge = edge
        self.classical_total: float = classical_total_gbps
        self.key_total: float = key_total_kbps
        self.classical_used: float = 0.0
        self.key_used: float = 0.0

    # -- residual properties -------------------------------------------------
    @property
    def classical_residual(self) -> float:
        return self.classical_total - self.classical_used

    @property
    def key_residual(self) -> float:
        return self.key_total - self.key_used

    # -- capacity check ------------------------------------------------------
    def can_accommodate(
        self,
        classical_demand_gbps: float,
        key_demand_kbps: float,
    ) -> bool:
        """Check if both resources are sufficient (with float tolerance)."""
        return (
            self.classical_residual >= classical_demand_gbps - EPS
            and self.key_residual >= key_demand_kbps - EPS
        )

    def classical_ok(self, classical_demand_gbps: float) -> bool:
        """Check classical capacity only."""
        return self.classical_residual >= classical_demand_gbps - EPS

    def key_ok(self, key_demand_kbps: float) -> bool:
        """Check key capacity only."""
        return self.key_residual >= key_demand_kbps - EPS

    # -- allocate / release --------------------------------------------------
    def allocate(self, classical_demand_gbps: float, key_demand_kbps: float):
        """Deduct resources.  Caller must have already checked feasibility."""
        self.classical_used += classical_demand_gbps
        self.key_used += key_demand_kbps
        # Sanity check
        if self.classical_used > self.classical_total + EPS:
            raise RuntimeError(
                f"Edge {self.edge}: classical over-allocated "
                f"(used={self.classical_used}, total={self.classical_total})"
            )
        if self.key_used > self.key_total + EPS:
            raise RuntimeError(
                f"Edge {self.edge}: key over-allocated "
                f"(used={self.key_used}, total={self.key_total})"
            )

    def release(self, classical_demand_gbps: float, key_demand_kbps: float):
        """Return resources.  Sanity-checks for double-release."""
        self.classical_used -= classical_demand_gbps
        self.key_used -= key_demand_kbps
        if self.classical_used < -EPS:
            raise RuntimeError(
                f"Edge {self.edge}: classical double-release "
                f"(used={self.classical_used})"
            )
        if self.key_used < -EPS:
            raise RuntimeError(
                f"Edge {self.edge}: key double-release "
                f"(used={self.key_used})"
            )
        # Clamp tiny negative values from float rounding
        if self.classical_used < 0.0:
            self.classical_used = 0.0
        if self.key_used < 0.0:
            self.key_used = 0.0

    @property
    def classical_utilization(self) -> float:
        if self.classical_total < EPS:
            return 0.0
        return self.classical_used / self.classical_total

    @property
    def key_utilization(self) -> float:
        if self.key_total < EPS:
            return 0.0
        return self.key_used / self.key_total


class NetworkResources:
    """Manages EdgeResources across the whole network graph.

    All lookups use canonical undirected edge keys ``(min(u,v), max(u,v))``.
    """

    def __init__(self, graph, classical_provider, qkd_provider):
        """
        Parameters
        ----------
        graph : nx.Graph
            Topology with ``length_km`` on each edge.
        classical_provider : callable
            ``f(edge_tuple) -> classical_capacity_gbps``
        qkd_provider : callable
            ``f(distance_m) -> key_capacity_kbps``
        """
        self._edges: Dict[Tuple[int, int], EdgeResources] = {}
        for u, v, data in graph.edges(data=True):
            key = canonical_edge(u, v)
            length_km = data.get("length_km", 0.0)
            classical_cap = classical_provider(key)
            key_cap = qkd_provider(length_km * 1000.0)
            self._edges[key] = EdgeResources(key, classical_cap, key_cap)

    # -- single-edge access --------------------------------------------------
    def get_edge(self, u: int, v: int) -> EdgeResources:
        return self._edges[canonical_edge(u, v)]

    # -- path-level operations -----------------------------------------------
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

    # -- path feasibility classification (used by routing) -------------------
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
            if not edge.key_ok(key_demand):
                key_ok = False
            if not classical_ok and not key_ok:
                # Early exit — both already failed
                break

        if classical_ok and key_ok:
            return "feasible"
        if not classical_ok and not key_ok:
            return "joint_insufficient"
        if not classical_ok:
            return "classical_insufficient"
        return "key_insufficient"

    # -- utilization snapshot ------------------------------------------------
    def get_avg_utilization(self):
        """Return (avg_classical_util, avg_key_util) across all edges."""
        if not self._edges:
            return 0.0, 0.0
        n = len(self._edges)
        c_util = sum(e.classical_utilization for e in self._edges.values()) / n
        k_util = sum(e.key_utilization for e in self._edges.values()) / n
        return c_util, k_util

    def get_max_utilization(self):
        """Return (max_classical_util, max_key_util) across all edges."""
        if not self._edges:
            return 0.0, 0.0
        c_util = max(e.classical_utilization for e in self._edges.values())
        k_util = max(e.key_utilization for e in self._edges.values())
        return c_util, k_util

    # -- iterate all edges ---------------------------------------------------
    def items(self):
        return self._edges.items()
