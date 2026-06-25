"""Results aggregation and CSV output."""

import os
from typing import Any, Dict, List

import pandas as pd

# Column order for the results CSV
RESULT_COLUMNS = [
    "load",
    "strategy",
    "qkd_mode",
    "classical_mode",
    "num_requests",
    "num_accepted",
    "num_blocked",
    "blocking_rate",
    "classical_blocking_rate",
    "key_blocking_rate",
    "joint_blocking_rate",
    "topology_blocking_rate",
    "avg_hops",
    "avg_path_length_km",
    "avg_classical_utilization",
    "avg_key_utilization",
    "max_classical_utilization",
    "max_key_utilization",
]


def build_results_dataframe(
    run_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Convert a list of per-run result dicts into a DataFrame.

    Each dict should contain the keys listed in ``RESULT_COLUMNS`` plus
    optional metadata (qkd_mode, classical_mode, load, strategy).
    """
    rows = []
    for r in run_results:
        row = {col: r.get(col, None) for col in RESULT_COLUMNS}
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def save_results(df: pd.DataFrame, output_path: str):
    """Save the results DataFrame to CSV, creating parent directories."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"[metrics] Results saved to {output_path}")
