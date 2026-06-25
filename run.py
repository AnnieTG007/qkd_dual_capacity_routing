#!/usr/bin/env python
"""QKD Dual-Capacity Routing Simulation — Course Paper Project.

Entry point for running the discrete-event simulation that compares four
routing strategies under classical + QKD-key dual-capacity constraints
on the NSFNET 14-node topology.

Usage
-----
.. code-block:: bash

    # Default mode (abstract QKD + constant 400 Gbps classical)
    python run.py

    # Actual SKR mode (requires old QKD_Network project)
    python run.py --qkd-capacity-mode actual_skr --old-project-root ../QKD_Network

    # CSV-based classical capacity
    python run.py --classical-capacity-mode csv --classical-capacity-csv data/my_caps.csv

    # GNPy CSV mode
    python run.py --classical-capacity-mode gnpy_csv --gnpy-result-csv data/gnpy_results.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure the package is importable when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from qkd_routing.config import DEFAULT_CONFIG, SimulationConfig
from qkd_routing.gnpy_adapter import initialize_classical_provider
from qkd_routing.metrics import build_results_dataframe, save_results
from qkd_routing.plotting import generate_all_plots
from qkd_routing.routing import get_strategy
from qkd_routing.simulation import SimulationRun
from qkd_routing.skr_adapter import initialize_qkd_provider
from qkd_routing.topology import build_nsfnet_graph, compute_all_pairs_k_shortest_paths
from qkd_routing.traffic import generate_request_sequence


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QKD Dual-Capacity Routing Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --qkd-capacity-mode actual_skr --old-project-root ../QKD_Network
  python run.py --classical-capacity-mode csv --classical-capacity-csv data/caps.csv
  python run.py --classical-capacity-mode gnpy_csv --gnpy-result-csv data/gnpy.csv
        """,
    )

    # ---- QKD capacity ----
    parser.add_argument(
        "--qkd-capacity-mode",
        default=DEFAULT_CONFIG.qkd_capacity_mode,
        choices=["abstract", "actual_skr"],
        help="QKD key-capacity model (default: %(default)s)",
    )
    parser.add_argument(
        "--abstract-K0",
        type=float,
        default=DEFAULT_CONFIG.abstract_K0_kbps,
        help="K0 for abstract SKR model (kb/s at 0 km, default: %(default)s)",
    )
    parser.add_argument(
        "--abstract-alpha",
        type=float,
        default=DEFAULT_CONFIG.abstract_alpha_per_km,
        help="Decay coefficient per km (default: %(default)s)",
    )
    parser.add_argument(
        "--old-project-root",
        default=None,
        help="Path to QKD_Network project (required for actual_skr mode)",
    )

    # ---- Classical capacity ----
    parser.add_argument(
        "--classical-capacity-mode",
        default=DEFAULT_CONFIG.classical_capacity_mode,
        choices=["constant", "csv", "gnpy_csv", "gnpy_optional"],
        help="Classical capacity model (default: %(default)s)",
    )
    parser.add_argument(
        "--constant-classical-cap",
        type=float,
        default=DEFAULT_CONFIG.constant_classical_capacity_gbps,
        help="Per-edge constant classical capacity in Gb/s (default: %(default)s)",
    )
    parser.add_argument(
        "--classical-capacity-csv",
        default=None,
        help="CSV file for csv classical capacity mode",
    )
    parser.add_argument(
        "--gnpy-result-csv",
        default=None,
        help="CSV file for gnpy_csv / gnpy_optional mode",
    )
    parser.add_argument(
        "--gnpy-bandwidth-ghz",
        type=float,
        default=DEFAULT_CONFIG.gnpy_bandwidth_ghz,
        help="Bandwidth for GSNR→capacity mapping (default: %(default)s GHz)",
    )
    parser.add_argument(
        "--gnpy-osnr-margin-db",
        type=float,
        default=DEFAULT_CONFIG.gnpy_osnr_margin_db,
        help="OSNR margin for GSNR→capacity mapping (default: %(default)s dB)",
    )

    # ---- Simulation parameters ----
    parser.add_argument(
        "--load-start",
        type=float,
        default=20.0,
        help="Starting offered load (Erlang, default: %(default)s)",
    )
    parser.add_argument(
        "--load-end",
        type=float,
        default=160.0,
        help="Ending offered load (Erlang, default: %(default)s)",
    )
    parser.add_argument(
        "--load-step",
        type=float,
        default=20.0,
        help="Load step size (default: %(default)s)",
    )
    parser.add_argument(
        "--mean-holding-time",
        type=float,
        default=DEFAULT_CONFIG.mean_holding_time,
        help="Mean holding time (default: %(default)s)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_CONFIG.num_requests,
        help="Number of requests per run (default: %(default)s)",
    )
    parser.add_argument(
        "--k-paths",
        type=int,
        default=DEFAULT_CONFIG.k_paths,
        help="K for K-shortest paths (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.random_seed,
        help="Base random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_CONFIG.warmup_ratio,
        help="Fraction of requests for warmup (default: %(default)s)",
    )

    # ---- Strategy selection ----
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_CONFIG.strategies,
        choices=[
            "min_hop",
            "min_distance",
            "key_capacity_aware",
            "dual_capacity_aware",
        ],
        help="Routing strategies to compare (default: all four)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CONFIG.output_dir,
        help="Output directory for results (default: %(default)s)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build config from CLI args
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> SimulationConfig:
    config = SimulationConfig()

    config.qkd_capacity_mode = args.qkd_capacity_mode
    config.abstract_K0_kbps = args.abstract_K0
    config.abstract_alpha_per_km = args.abstract_alpha
    config.old_project_root = args.old_project_root

    config.classical_capacity_mode = args.classical_capacity_mode
    config.constant_classical_capacity_gbps = args.constant_classical_cap
    config.classical_capacity_csv_path = args.classical_capacity_csv
    config.gnpy_result_csv_path = args.gnpy_result_csv
    config.gnpy_bandwidth_ghz = args.gnpy_bandwidth_ghz
    config.gnpy_osnr_margin_db = args.gnpy_osnr_margin_db

    config.mean_holding_time = args.mean_holding_time
    config.num_requests = args.num_requests
    config.k_paths = args.k_paths
    config.random_seed = args.seed
    config.warmup_ratio = args.warmup_ratio
    config.strategies = args.strategies

    # Build load_values from range
    config.load_values = list(
        np.arange(args.load_start, args.load_end + 1e-9, args.load_step)
    )

    config.output_dir = args.output_dir

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 62)
    print("  QKD Dual-Capacity Routing Simulation")
    print("=" * 62)
    print(f"  QKD mode:           {config.qkd_capacity_mode}")
    print(f"  Classical mode:     {config.classical_capacity_mode}")
    print(f"  Strategies:         {', '.join(config.strategies)}")
    print(f"  Loads:              {config.load_values}")
    print(f"  Requests/run:       {config.num_requests}")
    print(f"  K paths:            {config.k_paths}")
    print(f"  Seed:               {config.random_seed}")
    print(f"  Output:             {config.output_dir}/")
    print("=" * 62)

    # 1. Build topology
    print("\n[1/6] Building NSFNET topology ...")
    graph = build_nsfnet_graph()
    print(f"       Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # 2. Pre-compute K-shortest paths
    print("[2/6] Computing K-shortest paths ...")
    all_pairs_paths = compute_all_pairs_k_shortest_paths(graph, config.k_paths)
    n_pairs = sum(1 for v in all_pairs_paths.values() if v)
    print(f"       {n_pairs} reachable S-D pairs with up to {config.k_paths} paths each")

    # 3. Initialize capacity providers
    print("[3/6] Initializing capacity providers ...")
    classical_provider = initialize_classical_provider(
        config.classical_capacity_mode, graph, config
    )
    qkd_provider = initialize_qkd_provider(
        config.qkd_capacity_mode, graph, config
    )

    # Show a sample
    sample_edge = (0, 1)
    sample_dist = graph[0][1]["length_km"] * 1000.0
    print(f"       Edge (0,1): classical={classical_provider(sample_edge):.1f} Gbps, "
          f"QKD={qkd_provider(sample_dist):.1f} kbps")

    # 4. Run sweep
    print("[4/6] Running simulation sweep ...")
    all_results: List[Dict[str, Any]] = []

    for load in config.load_values:
        # Each load gets its own request sequence (deterministic seed)
        rng = np.random.default_rng(config.random_seed + int(load))
        requests = generate_request_sequence(
            graph, config.num_requests, load, config, rng
        )
        print(f"       Load={load:.0f} Erlang: {len(requests)} requests generated")

        for strat_name in config.strategies:
            strategy = get_strategy(strat_name)
            sim = SimulationRun(
                config=config,
                base_graph=graph,
                all_pairs_paths=all_pairs_paths,
                requests=requests,
                strategy=strategy,
                classical_provider=classical_provider,
                qkd_provider=qkd_provider,
            )
            result = sim.run()
            result["load"] = load
            result["strategy"] = strat_name
            result["qkd_mode"] = config.qkd_capacity_mode
            result["classical_mode"] = config.classical_capacity_mode

            all_results.append(result)
            print(
                f"         {strat_name:22s}  "
                f"accepted={result['num_accepted']:5d}  "
                f"blocking={result['blocking_rate']:.4f}  "
                f"c_block={result['classical_blocking_rate']:.4f}  "
                f"k_block={result['key_blocking_rate']:.4f}  "
                f"j_block={result['joint_blocking_rate']:.4f}  "
                f"t_block={result['topology_blocking_rate']:.4f}"
            )

    # 5. Build DataFrame & save
    print("[5/6] Saving results ...")
    df = build_results_dataframe(all_results)
    csv_path = f"{config.output_dir}/simulation_results.csv"
    save_results(df, csv_path)

    # 6. Plot
    print("[6/6] Generating plots ...")
    generate_all_plots(df, config.output_dir)

    print("\n" + "=" * 62)
    print("  Simulation complete.")
    print(f"  Results: {csv_path}")
    print(f"  Plots:   {config.output_dir}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
