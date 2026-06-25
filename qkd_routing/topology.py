"""NSFNET 14-node topology construction and K-shortest-path precomputation.

The NSFNET14 topology is extracted from the existing QKD_Network project's
data/topology/NSFNET14.json and topology.py (topology1 matrix). Edge distances
are in kilometres.
"""

from typing import Dict, List, Tuple

import networkx as nx

# NSFNET 14-node edge list: (u, v, length_km)
# Source: QKD_Network project data/topology/NSFNET14.json
NSFNET14_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 52.5),
    (0, 2, 75.0),
    (0, 7, 90.0),
    (1, 2, 30.0),
    (1, 3, 30.0),
    (2, 5, 75.0),
    (3, 4, 30.0),
    (3, 10, 75.0),
    (4, 5, 65.0),
    (4, 6, 30.0),
    (5, 9, 52.5),
    (5, 13, 75.0),
    (6, 7, 37.5),
    (7, 8, 37.5),
    (8, 9, 40.0),
    (8, 11, 15.0),
    (8, 12, 15.0),
    (10, 11, 25.0),
    (10, 12, 37.5),
    (11, 13, 40.0),
    (12, 13, 15.0),
]


def build_nsfnet_graph() -> nx.Graph:
    """Build an undirected NSFNET 14-node NetworkX graph.

    Each edge carries the attribute 'length_km' (float).

    Returns
    -------
    nx.Graph
        Undirected graph with nodes 0..13 and 21 edges.
    """
    G = nx.Graph()
    G.add_nodes_from(range(14))
    for u, v, dist in NSFNET14_EDGES:
        G.add_edge(u, v, length_km=dist)
    return G


def compute_all_pairs_k_shortest_paths(
    graph: nx.Graph,
    k: int,
    weight: str = "length_km",
) -> Dict[Tuple[int, int], List[Tuple[List[int], float]]]:
    """Pre-compute K-shortest paths for every (src, dst) node pair.

    Uses NetworkX's built-in shortest_simple_paths generator.  For each
    ordered pair with src != dst and a connected path, the *k* shortest
    (by *weight*) paths are stored together with their total weight.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph with edge weight attribute.
    k : int
        Number of candidate shortest paths to keep per pair.
    weight : str
        Edge attribute to use as cost (default "length_km").

    Returns
    -------
    dict
        Mapping (src, dst) -> list of (node_list, total_cost).
        Node lists include both endpoints.
    """
    all_paths: Dict[Tuple[int, int], List[Tuple[List[int], float]]] = {}
    nodes = list(graph.nodes())

    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            # Check connectivity
            if not nx.has_path(graph, src, dst):
                all_paths[(src, dst)] = []
                continue
            # Generate k-shortest simple paths
            path_gen = nx.shortest_simple_paths(
                graph, source=src, target=dst, weight=weight
            )
            k_paths: List[Tuple[List[int], float]] = []
            for i, path in enumerate(path_gen):
                if i >= k:
                    break
                cost = sum(
                    graph[path[j]][path[j + 1]][weight]
                    for j in range(len(path) - 1)
                )
                k_paths.append((path, cost))
            all_paths[(src, dst)] = k_paths

    return all_paths
