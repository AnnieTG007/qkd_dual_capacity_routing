"""Per-edge dual-capacity resource tracking with wavelength continuity.

Each edge tracks:
* ``slots_occupied``  — set of wavelength slot indices in use (wavelength continuity)
* ``key_used``        — QKD key consumed (kbps), dynamic total from SKR model
"""

import math
from typing import Callable, Dict, List, Optional, Set, Tuple

def _n_slots_for(bandwidth_gbps: float, bw_per_ch: float) -> int:
    """Number of wavelength slots needed for ``bandwidth_gbps`` demand."""
    return max(1, math.ceil(bandwidth_gbps / max(bw_per_ch, 1e-9)))

from .utils import EPS, canonical_edge


class EdgeResources:
    """Track wavelength slot occupancy + QKD key capacity on one undirected link.

    Classical capacity is divided into ``n_max`` discrete wavelength slots.
    Each slot carries ``bandwidth_per_ch_gbps`` Gbps.  The QKD key rate is
    recomputed whenever the number of active (occupied) slots changes.
    """

    def __init__(
        self,
        edge: Tuple[int, int],
        distance_m: float,
        n_max_channels: int,
        bandwidth_per_ch_gbps: float,
        skr_provider: Callable[[float, int], float],
    ):
        self.edge = edge
        self.distance_m = distance_m
        self.n_max = n_max_channels
        self.bandwidth_per_ch = bandwidth_per_ch_gbps
        self.classical_total: float = n_max_channels * bandwidth_per_ch_gbps

        self.slots_occupied: Set[int] = set()   # occupied wavelength indices
        self.key_used: float = 0.0

        self._skr_provider = skr_provider
        self._cached_n_active: int = 0
        self._key_total: float = skr_provider(distance_m, 0)

    # ------------------------------------------------------------------
    # Slot / bandwidth helpers
    # ------------------------------------------------------------------

    def _n_active(self) -> int:
        return len(self.slots_occupied)

    def slots_free(self) -> int:
        return self.n_max - len(self.slots_occupied)

    def is_slot_free(self, w: int) -> bool:
        return w not in self.slots_occupied

    @property
    def classical_used(self) -> float:
        return len(self.slots_occupied) * self.bandwidth_per_ch

    @property
    def classical_residual(self) -> float:
        return self.slots_free() * self.bandwidth_per_ch

    # ------------------------------------------------------------------
    # Key capacity (dynamic — recalculated when n_active changes)
    # ------------------------------------------------------------------

    @property
    def key_total(self) -> float:
        return self._key_total

    def _update_key_total(self):
        n = len(self.slots_occupied)
        if n != self._cached_n_active:
            self._cached_n_active = n
            self._key_total = self._skr_provider(self.distance_m, n)

    def _future_key_total(self, extra_slots: int = 1) -> float:
        """Key capacity after hypothetically adding ``extra_slots`` channels."""
        future_n = min(len(self.slots_occupied) + extra_slots, self.n_max)
        if future_n == self._cached_n_active:
            return self._key_total
        return self._skr_provider(self.distance_m, future_n)

    @property
    def key_residual(self) -> float:
        return self._key_total - self.key_used

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def is_slot_feasible(self, w: int, key_demand: float) -> bool:
        """True if slot w is free AND key capacity will hold after allocation."""
        if not self.is_slot_free(w) or w >= self.n_max:
            return False
        future_key = self._future_key_total(1)
        return future_key - self.key_used - key_demand >= -EPS

    def can_accommodate(self, classical_demand_gbps: float, key_demand_kbps: float,
                        bw_per_ch: float = 100.0) -> bool:
        """True if enough free slots exist AND key capacity is sufficient."""
        n = _n_slots_for(classical_demand_gbps, bw_per_ch)
        if self.slots_free() < n:
            return False
        future_key = self._future_key_total(n)
        return future_key - self.key_used - key_demand_kbps >= -EPS

    def classical_ok(self, classical_demand_gbps: float, bw_per_ch: float = 100.0) -> bool:
        """True if enough wavelength slots are free for the bandwidth demand."""
        return self.slots_free() >= _n_slots_for(classical_demand_gbps, bw_per_ch)

    def key_ok(self, key_demand_kbps: float) -> bool:
        return self.key_residual >= key_demand_kbps - EPS

    # ------------------------------------------------------------------
    # Allocate / release (slot-aware)
    # ------------------------------------------------------------------

    def allocate(self, wavelengths: List[int], key_demand: float):
        """Occupy wavelength slots and deduct key capacity."""
        for w in wavelengths:
            self.slots_occupied.add(w)
        self.key_used += key_demand
        self._update_key_total()

    def release(self, wavelengths: List[int], key_demand: float):
        """Free wavelength slots and return key capacity."""
        for w in wavelengths:
            self.slots_occupied.discard(w)
        self.key_used -= key_demand
        if self.key_used < 0.0:
            self.key_used = 0.0
        self._update_key_total()

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    @property
    def classical_utilization(self) -> float:
        return len(self.slots_occupied) / max(self.n_max, 1)

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
        self._edges: Dict[Tuple[int, int], EdgeResources] = {}
        self._n_max = getattr(config, 'n_max_classical_channels', 8)
        self._bw_per_ch = getattr(config, 'classical_bandwidth_per_ch_gbps', 100.0)

        for u, v, data in graph.edges(data=True):
            key = canonical_edge(u, v)
            distance_m = data.get("length_km", 0.0) * 1000.0
            edge_res = EdgeResources(
                edge=key,
                distance_m=distance_m,
                n_max_channels=self._n_max,
                bandwidth_per_ch_gbps=self._bw_per_ch,
                skr_provider=qkd_provider,
            )
            # Allow classical_provider override (csv modes)
            classical_cap = classical_provider(key)
            if abs(classical_cap - edge_res.classical_total) > EPS:
                edge_res.classical_total = classical_cap
            self._edges[key] = edge_res

    # ------------------------------------------------------------------
    # Single-edge access
    # ------------------------------------------------------------------

    def get_edge(self, u: int, v: int) -> EdgeResources:
        return self._edges[canonical_edge(u, v)]

    # ------------------------------------------------------------------
    # Wavelength continuity
    # ------------------------------------------------------------------

    def find_free_wavelengths(
        self, path: List[int], n_slots: int
    ) -> Optional[List[int]]:
        """First-Fit: return the first ``n_slots`` wavelength indices all free
        on ALL edges of path (each sub-channel satisfies wavelength continuity).

        Slots need not be contiguous — each 100-Gbps lightpath sub-channel uses
        an independent wavelength on the same physical path.
        Returns None if fewer than ``n_slots`` common free wavelengths exist.
        """
        free: List[int] = []
        for w in range(self._n_max):
            if all(
                self.get_edge(path[i], path[i + 1]).is_slot_free(w)
                for i in range(len(path) - 1)
            ):
                free.append(w)
                if len(free) == n_slots:
                    return free
        return None

    # ------------------------------------------------------------------
    # Path-level operations
    # ------------------------------------------------------------------

    def can_allocate_path(
        self, path: List[int], classical_demand: float, key_demand: float
    ) -> bool:
        """True if every edge has sufficient capacity AND enough common free
        wavelengths exist to carry ``classical_demand`` Gbps."""
        n_slots = _n_slots_for(classical_demand, self._bw_per_ch)
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if not edge.can_accommodate(classical_demand, key_demand, self._bw_per_ch):
                return False
        return self.find_free_wavelengths(path, n_slots) is not None

    def allocate_path(
        self, path: List[int], wavelengths: List[int], key_demand: float
    ):
        """Occupy ``wavelengths`` on every edge of path and deduct key capacity."""
        for i in range(len(path) - 1):
            self.get_edge(path[i], path[i + 1]).allocate(wavelengths, key_demand)

    def release_path(
        self, path: List[int], wavelengths: List[int], key_demand: float
    ):
        """Release ``wavelengths`` on every edge of path and restore key capacity."""
        for i in range(len(path) - 1):
            self.get_edge(path[i], path[i + 1]).release(wavelengths, key_demand)

    # ------------------------------------------------------------------
    # Feasibility classification (used for blocking-reason reporting)
    # ------------------------------------------------------------------

    def classify_path_feasibility(
        self, path: List[int], classical_demand: float, key_demand: float
    ) -> str:
        """Classify a single path w.r.t. dual-capacity + wavelength continuity.

        Returns one of:
            'feasible'
            'wavelength_blocking'   — capacity OK but no common free slots
            'classical_insufficient'
            'key_insufficient'
            'joint_insufficient'
        """
        n_slots = _n_slots_for(classical_demand, self._bw_per_ch)
        classical_ok = True
        key_ok = True
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if not edge.classical_ok(classical_demand, self._bw_per_ch):
                classical_ok = False
            future_key = edge._future_key_total(n_slots)
            if future_key - edge.key_used - key_demand < -EPS:
                key_ok = False
            if not classical_ok and not key_ok:
                break

        if not classical_ok and not key_ok:
            return "joint_insufficient"
        if not classical_ok:
            return "classical_insufficient"
        if not key_ok:
            return "key_insufficient"
        # Both per-edge checks pass; now check wavelength continuity
        if self.find_free_wavelengths(path, n_slots) is None:
            return "wavelength_blocking"
        return "feasible"

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def get_total_used(self):
        c_used = sum(e.classical_used for e in self._edges.values())
        k_used = sum(e.key_used for e in self._edges.values())
        return c_used, k_used

    def get_total_capacity(self):
        c_total = sum(e.classical_total for e in self._edges.values())
        k_total = sum(e.key_total for e in self._edges.values())
        return c_total, k_total

    def get_per_edge_utilization(self):
        c_util = {key: e.classical_utilization for key, e in self._edges.items()}
        k_util = {key: e.key_utilization for key, e in self._edges.items()}
        return c_util, k_util

    def items(self):
        return self._edges.items()
