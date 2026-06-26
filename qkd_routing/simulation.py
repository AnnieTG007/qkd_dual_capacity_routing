"""Discrete-event simulation engine for dual-capacity QKD network routing.

Uses a min-heap priority queue for arrival and departure events.
Each strategy runs on a deep-copied initial network so that allocations
from one strategy do not affect another.
"""

import copy
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx

from .config import SimulationConfig
from .utils import EPS
from .resources import NetworkResources
from .routing import RoutingStrategy
from .traffic import Request


@dataclass(order=True)
class Event:
    """Thin wrapper for the heapq — ordered by time."""

    time: float
    event_type: str = field(compare=False)  # 'arrival' | 'departure'
    request_id: int = field(compare=False)


class SimulationRun:
    """Runs one simulation for a given offered load and routing strategy."""

    def __init__(
        self,
        config: SimulationConfig,
        base_graph: nx.Graph,
        all_pairs_paths: Dict[Tuple[int, int], List[Tuple[List[int], float]]],
        requests: List[Request],
        strategy: RoutingStrategy,
        classical_provider,
        qkd_provider,
    ):
        self.config = config
        self.base_graph = base_graph
        self.all_pairs_paths = all_pairs_paths
        self.requests = requests
        self.strategy = strategy
        self.classical_provider = classical_provider
        self.qkd_provider = qkd_provider

    def run(self) -> Dict[str, Any]:
        """Execute the event-driven simulation and return result metrics."""
        config = self.config
        total_requests = len(self.requests)

        # Number of arrivals treated as warmup (not counted in stats)
        warmup_count = int(total_requests * config.warmup_ratio)

        # Deep-copy the base graph so strategies are independent
        graph = copy.deepcopy(self.base_graph)

        # Initialise fresh resources on the copied graph
        network = NetworkResources(
            graph, self.classical_provider, self.qkd_provider, config
        )

        # Build event heap — one arrival + one departure per request
        events: List[Event] = []
        for req in self.requests:
            events.append(Event(req.arrival_time, "arrival", req.request_id))
            events.append(
                Event(req.departure_time, "departure", req.request_id)
            )
        heapq.heapify(events)

        # Request lookup
        request_map: Dict[int, Request] = {
            r.request_id: r for r in self.requests
        }

        # Active connections: request_id → path
        active_connections: Dict[int, List[int]] = {}

        # ---- Statistics (only counted for arrivals >= warmup_count) ----
        arrivals_seen: int = 0  # total arrivals processed so far
        num_accepted: int = 0
        num_blocked: int = 0
        blocking_counts: Dict[str, int] = {
            "classical_blocking": 0,
            "key_blocking": 0,
            "joint_blocking": 0,
            "topology_blocking": 0,
        }
        total_hops: int = 0
        total_path_length_km: float = 0.0

        # Time-weighted utilisation tracking
        last_time: float = 0.0
        integrated_c_used: float = 0.0  # ∫ classical_used(t) dt
        integrated_k_used: float = 0.0  # ∫ key_used(t) dt
        total_c_capacity: float = 0.0
        total_k_capacity: float = 0.0
        # Pre-fetch total capacities (constant over simulation)
        total_c_capacity, total_k_capacity = network.get_total_capacity()
        # Track per-edge peak utilisation
        peak_c_util: float = 0.0
        peak_k_util: float = 0.0

        # Process events in time order
        while events:
            event = heapq.heappop(events)

            # ---- Accumulate time-weighted utilisation ----
            delta = event.time - last_time
            if delta > 0:
                c_used_now, k_used_now = network.get_total_used()
                integrated_c_used += c_used_now * delta
                integrated_k_used += k_used_now * delta
                # Track peak per-edge utilisation
                if total_c_capacity > EPS:
                    peak_c_util = max(peak_c_util, c_used_now / total_c_capacity)
                if total_k_capacity > EPS:
                    peak_k_util = max(peak_k_util, k_used_now / total_k_capacity)
            last_time = event.time

            if event.event_type == "departure":
                # Release resources for connections that are ending
                if event.request_id in active_connections:
                    req = request_map[event.request_id]
                    path = active_connections.pop(event.request_id)
                    network.release_path(
                        path, req.bandwidth_gbps, req.key_rate_kbps
                    )
                continue

            # ---- Arrival event ----
            req = request_map.get(event.request_id)
            if req is None:
                continue

            arrivals_seen += 1
            in_measurement = arrivals_seen > warmup_count

            # Attempt routing
            path, reason = self.strategy.find_path(
                req, network, self.all_pairs_paths
            )

            if path is not None:
                # Success — allocate resources and track the connection
                network.allocate_path(
                    path, req.bandwidth_gbps, req.key_rate_kbps
                )
                active_connections[req.request_id] = path

                if in_measurement:
                    num_accepted += 1
                    total_hops += len(path) - 1
                    dist = 0.0
                    for i in range(len(path) - 1):
                        dist += graph[path[i]][path[i + 1]]["length_km"]
                    total_path_length_km += dist
            else:
                # Blocked
                if in_measurement:
                    num_blocked += 1
                    blocking_counts[reason] = (
                        blocking_counts.get(reason, 0) + 1
                    )

        # ---- Finalise metrics ----
        measured = total_requests - warmup_count
        if measured < 1:
            measured = 1

        # Time-weighted average utilisation
        sim_duration = last_time if last_time > 0 else 1.0
        if sim_duration > 0 and total_c_capacity > EPS:
            avg_c_util = (integrated_c_used / sim_duration) / total_c_capacity
        else:
            avg_c_util = 0.0
        if sim_duration > 0 and total_k_capacity > EPS:
            avg_k_util = (integrated_k_used / sim_duration) / total_k_capacity
        else:
            avg_k_util = 0.0

        # Per-type blocking rates
        classical_blocking_rate = (
            blocking_counts.get("classical_blocking", 0) / measured
        )
        key_blocking_rate = (
            blocking_counts.get("key_blocking", 0) / measured
        )
        joint_blocking_rate = (
            blocking_counts.get("joint_blocking", 0) / measured
        )
        topology_blocking_rate = (
            blocking_counts.get("topology_blocking", 0) / measured
        )

        # Path stats
        avg_hops = total_hops / max(num_accepted, 1)
        avg_path_length_km = total_path_length_km / max(num_accepted, 1)

        return {
            "num_requests": measured,
            "num_accepted": num_accepted,
            "num_blocked": num_blocked,
            "blocking_rate": num_blocked / measured,
            "classical_blocking_rate": classical_blocking_rate,
            "key_blocking_rate": key_blocking_rate,
            "joint_blocking_rate": joint_blocking_rate,
            "topology_blocking_rate": topology_blocking_rate,
            "blocking_counts": dict(blocking_counts),
            "avg_hops": avg_hops,
            "avg_path_length_km": avg_path_length_km,
            "avg_classical_utilization": avg_c_util,
            "avg_key_utilization": avg_k_util,
            "max_classical_utilization": peak_c_util,
            "max_key_utilization": peak_k_util,
        }
