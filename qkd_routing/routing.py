"""Routing strategies for dual-capacity QKD networks.

Four strategies are implemented:

1. **Min-hop**      — fewest hops from K-shortest paths.
2. **Min-distance** — shortest total length from K-shortest paths.
3. **Key-capacity-aware** — maximise the bottleneck residual key ratio.
4. **Dual-capacity-aware** — maximise the normalised dual-resource bottleneck.

Each strategy also performs blocking-reason classification over the
K candidate paths so that the simulation can report *why* a request
was blocked.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .resources import NetworkResources
from .traffic import Request


# ---------------------------------------------------------------------------
# Public helper — classify blocking over K candidate paths
# ---------------------------------------------------------------------------

def classify_blocking_type(
    candidate_paths: List[Tuple[List[int], float]],
    request: Request,
    network: NetworkResources,
) -> str:
    """Determine the dominant blocking reason across *candidate_paths*.

    For each path we check whether classical capacity, key capacity, both,
    or neither would be sufficient.

    Returns one of:
        'topology_blocking'
        'classical_blocking'
        'key_blocking'
        'joint_blocking'
    """
    if not candidate_paths:
        return "topology_blocking"

    has_classical_feasible = False
    has_key_feasible = False
    has_both_feasible = False

    for path, _cost in candidate_paths:
        status = network.classify_path_feasibility(
            path, request.bandwidth_gbps, request.key_rate_kbps
        )
        if status == "feasible":
            has_both_feasible = True
            has_classical_feasible = True
            has_key_feasible = True
            break  # at least one fully feasible → not "blocked" in classification
        elif status == "classical_insufficient":
            has_key_feasible = True
        elif status == "key_insufficient":
            has_classical_feasible = True
        # joint_insufficient contributes to neither

    if has_both_feasible:
        # Should not be called for a successful routing, but if it is:
        return "success"

    if has_classical_feasible and not has_key_feasible:
        return "key_blocking"
    if has_key_feasible and not has_classical_feasible:
        return "classical_blocking"
    return "joint_blocking"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RoutingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def find_path(
        self,
        request: Request,
        network: NetworkResources,
        all_pairs_paths: Dict[Tuple[int, int], List[Tuple[List[int], float]]],
    ) -> Tuple[Optional[List[int]], str]:
        """Return (path, blocking_reason).

        path is None if blocked.  blocking_reason is one of the strings
        returned by :func:`classify_blocking_type`.
        """
        ...


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class MinHopRouting(RoutingStrategy):
    """Select the path with the fewest hops; ties broken by shortest distance."""

    name = "min_hop"

    def find_path(self, request, network, all_pairs_paths):
        candidates = all_pairs_paths.get((request.src, request.dst), [])
        if not candidates:
            return None, "topology_blocking"

        # Sort by (hops, distance)
        sorted_candidates = sorted(candidates, key=lambda pc: (len(pc[0]), pc[1]))

        for path, _cost in sorted_candidates:
            if network.can_allocate_path(
                path, request.bandwidth_gbps, request.key_rate_kbps
            ):
                return path, "success"

        reason = classify_blocking_type(candidates, request, network)
        return None, reason


class MinDistanceRouting(RoutingStrategy):
    """Select the shortest-distance path; ties broken by fewest hops."""

    name = "min_distance"

    def find_path(self, request, network, all_pairs_paths):
        candidates = all_pairs_paths.get((request.src, request.dst), [])
        if not candidates:
            return None, "topology_blocking"

        # Sort by (distance, hops)
        sorted_candidates = sorted(candidates, key=lambda pc: (pc[1], len(pc[0])))

        for path, _cost in sorted_candidates:
            if network.can_allocate_path(
                path, request.bandwidth_gbps, request.key_rate_kbps
            ):
                return path, "success"

        reason = classify_blocking_type(candidates, request, network)
        return None, reason


class KeyCapacityAwareRouting(RoutingStrategy):
    """Among feasible K-shortest paths, pick the one with the highest
    bottleneck **key** residual ratio::

        maximise  min_e (residual_key[e] / request.key_rate_kbps)

    Ties are broken by shortest total distance.
    """

    name = "key_capacity_aware"

    def find_path(self, request, network, all_pairs_paths):
        candidates = all_pairs_paths.get((request.src, request.dst), [])
        if not candidates:
            return None, "topology_blocking"

        best_path = None
        best_score = -1.0
        best_distance = float("inf")

        for path, distance in candidates:
            if not network.can_allocate_path(
                path, request.bandwidth_gbps, request.key_rate_kbps
            ):
                continue
            # Compute bottleneck key ratio
            min_key_ratio = float("inf")
            for i in range(len(path) - 1):
                edge = network.get_edge(path[i], path[i + 1])
                ratio = edge.key_residual / max(request.key_rate_kbps, 1e-9)
                if ratio < min_key_ratio:
                    min_key_ratio = ratio
            # Compare: higher ratio is better; tie → shorter distance
            if min_key_ratio > best_score or (
                abs(min_key_ratio - best_score) < 1e-12
                and distance < best_distance
            ):
                best_score = min_key_ratio
                best_distance = distance
                best_path = path

        if best_path is not None:
            return best_path, "success"

        reason = classify_blocking_type(candidates, request, network)
        return None, reason


class DualCapacityAwareRouting(RoutingStrategy):
    """Among feasible K-shortest paths, minimise hop count first, then
    maximise the normalised dual-resource bottleneck within the same hop tier::

        1. Group candidates by hop count.
        2. In the shortest-hop tier that has at least one feasible path,
           select the path with the highest  min_e min(cl_frac, key_frac).
        3. Fall back to the next hop tier only if the current tier is
           fully infeasible.

    Using total-capacity fractions (not raw demand ratios) keeps both
    dimensions on the same [0, 1] scale regardless of traffic mix.
    Hop-tier grouping prevents unnecessary detours that drain key capacity.
    """

    name = "dual_capacity_aware"

    def find_path(self, request, network, all_pairs_paths):
        candidates = all_pairs_paths.get((request.src, request.dst), [])
        if not candidates:
            return None, "topology_blocking"

        # Group candidates by hop count, preserving distance order within
        from collections import defaultdict
        by_hops: dict = defaultdict(list)
        for path, distance in candidates:
            by_hops[len(path)].append((path, distance))

        for hop_count in sorted(by_hops):
            best_path = None
            best_score = -1.0
            best_distance = float("inf")

            for path, distance in by_hops[hop_count]:
                if not network.can_allocate_path(
                    path, request.bandwidth_gbps, request.key_rate_kbps
                ):
                    continue
                # Bottleneck dual fraction: min over path edges of
                # min(classical_residual/classical_total, key_residual/key_total)
                min_dual_frac = float("inf")
                for i in range(len(path) - 1):
                    edge = network.get_edge(path[i], path[i + 1])
                    cl_frac = edge.classical_residual / max(edge.classical_total, 1e-9)
                    key_frac = edge.key_residual / max(edge.key_total, 1e-9)
                    dual = min(cl_frac, key_frac)
                    if dual < min_dual_frac:
                        min_dual_frac = dual
                if min_dual_frac > best_score or (
                    abs(min_dual_frac - best_score) < 1e-12
                    and distance < best_distance
                ):
                    best_score = min_dual_frac
                    best_distance = distance
                    best_path = path

            if best_path is not None:
                return best_path, "success"
            # No feasible path in this hop tier → try next tier

        reason = classify_blocking_type(candidates, request, network)
        return None, reason


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_strategy(name: str) -> RoutingStrategy:
    """Return a routing strategy instance by name."""
    registry = {
        "min_hop": MinHopRouting,
        "min_distance": MinDistanceRouting,
        "key_capacity_aware": KeyCapacityAwareRouting,
        "dual_capacity_aware": DualCapacityAwareRouting,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown strategy {name!r}. Options: {list(registry.keys())}"
        )
    return registry[name]()
