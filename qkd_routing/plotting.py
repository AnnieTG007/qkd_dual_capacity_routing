"""Matplotlib-based plotting for QKD dual-capacity routing results.

Generates six publication-quality figures suitable for a course paper.
"""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless PNG generation

import matplotlib.pyplot as plt
import pandas as pd

# Strategy display names and line styles
STRATEGY_STYLES: Dict[str, dict] = {
    "min_hop": {
        "label": "Min-Hop",
        "marker": "o",
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "min_distance": {
        "label": "Min-Distance",
        "marker": "s",
        "linestyle": "--",
        "color": "#ff7f0e",
    },
    "key_capacity_aware": {
        "label": "Key-Capacity-Aware",
        "marker": "^",
        "linestyle": "-.",
        "color": "#2ca02c",
    },
    "dual_capacity_aware": {
        "label": "Dual-Capacity-Aware",
        "marker": "D",
        "linestyle": ":",
        "color": "#d62728",
    },
}


def _setup_figure(title: str, xlabel: str, ylabel: str):
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    return fig, ax


def _plot_strategy_lines(ax, df: pd.DataFrame, x_col: str, y_col: str,
                         max_load: float = 400.0):
    """Plot one line per strategy, focused on low-to-medium load (<=max_load)."""
    df = df[df[x_col] <= max_load]
    strategies = df["strategy"].unique()
    all_y = []
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values(x_col)
        style = STRATEGY_STYLES.get(strat, {})
        ax.plot(
            sub[x_col],
            sub[y_col],
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            label=style.get("label", strat),
            linewidth=1.8,
            markersize=6,
        )
        all_y.extend(sub[y_col].tolist())
    ax.legend(fontsize=9, loc="best")
    # Auto-scale y-axis to data range with margin so differences are visible
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        margin = max((ymax - ymin) * 0.20, 0.005)
        ax.set_ylim(max(0, ymin - margin), ymax + margin)


def plot_blocking_rate_vs_load(df: pd.DataFrame, output_dir: str):
    """Overall blocking probability vs offered load."""
    fig, ax = _setup_figure(
        "Overall Blocking Probability vs Offered Load",
        "Offered Load (Erlang)",
        "Blocking Probability",
    )
    _plot_strategy_lines(ax, df, "load", "blocking_rate")
    path = os.path.join(output_dir, "blocking_rate_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_key_blocking_rate_vs_load(df: pd.DataFrame, output_dir: str):
    """Key-capacity blocking rate vs offered load."""
    fig, ax = _setup_figure(
        "Key-Capacity Blocking Rate vs Offered Load",
        "Offered Load (Erlang)",
        "Key Blocking Rate",
    )
    _plot_strategy_lines(ax, df, "load", "key_blocking_rate")
    path = os.path.join(output_dir, "key_blocking_rate_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_classical_blocking_rate_vs_load(df: pd.DataFrame, output_dir: str):
    """Classical-capacity blocking rate vs offered load."""
    fig, ax = _setup_figure(
        "Classical-Capacity Blocking Rate vs Offered Load",
        "Offered Load (Erlang)",
        "Classical Blocking Rate",
    )
    _plot_strategy_lines(ax, df, "load", "classical_blocking_rate")
    path = os.path.join(output_dir, "classical_blocking_rate_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_avg_path_length_vs_load(df: pd.DataFrame, output_dir: str):
    """Average accepted path length vs offered load."""
    fig, ax = _setup_figure(
        "Average Path Length vs Offered Load",
        "Offered Load (Erlang)",
        "Average Path Length (km)",
    )
    _plot_strategy_lines(ax, df, "load", "avg_path_length_km")
    path = os.path.join(output_dir, "avg_path_length_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_classical_utilization_vs_load(df: pd.DataFrame, output_dir: str):
    """Average classical capacity utilization vs offered load."""
    fig, ax = _setup_figure(
        "Average Classical Capacity Utilization vs Offered Load",
        "Offered Load (Erlang)",
        "Avg Classical Utilization",
    )
    _plot_strategy_lines(ax, df, "load", "avg_classical_utilization")
    path = os.path.join(output_dir, "classical_utilization_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_key_utilization_vs_load(df: pd.DataFrame, output_dir: str):
    """Average key capacity utilization vs offered load."""
    fig, ax = _setup_figure(
        "Average Key Capacity Utilization vs Offered Load",
        "Offered Load (Erlang)",
        "Avg Key Utilization",
    )
    _plot_strategy_lines(ax, df, "load", "avg_key_utilization")
    path = os.path.join(output_dir, "key_utilization_vs_load.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def generate_all_plots(df: pd.DataFrame, output_dir: str):
    """Generate all six result plots."""
    os.makedirs(output_dir, exist_ok=True)
    plot_blocking_rate_vs_load(df, output_dir)
    plot_key_blocking_rate_vs_load(df, output_dir)
    plot_classical_blocking_rate_vs_load(df, output_dir)
    plot_avg_path_length_vs_load(df, output_dir)
    plot_classical_utilization_vs_load(df, output_dir)
    plot_key_utilization_vs_load(df, output_dir)
    print(f"[plot] All figures saved to {output_dir}/")
