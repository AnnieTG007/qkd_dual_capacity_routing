#!/usr/bin/env python
"""Generate ``results/skr_vs_distance.png`` for the paper.

Finite-key decoy-state BB84 secure-key rate (per quantum channel) vs. fibre
distance for 0-8 co-propagating classical channels at -25 dBm/channel, using
the self-contained ``qkd_routing/physics.py`` model.  Reproduces Fig.
``fig:skr_distance`` in ``paper_qkd_coexistence.tex``.

Usage::

    python -m scripts.plot_skr_vs_distance
    python scripts/plot_skr_vs_distance.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qkd_routing import physics  # noqa: E402

# Distances 1-320 km; per-quantum-channel SKR so the y-axis is the raw R_SKR.
DISTANCES_KM = np.linspace(1.0, 320.0, 120)
CLASSICAL_COUNTS = [0, 2, 4, 6, 8]
COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"]


def main() -> None:
    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_cl, color in zip(CLASSICAL_COUNTS, COLORS):
        skr_kbps = [
            physics.get_key_capacity_kbps(
                d_km * 1000.0,
                num_classical_channels=n_cl,
                n_quantum_channels=1,
            )
            for d_km in DISTANCES_KM
        ]
        label = "Dark fibre (0 ch)" if n_cl == 0 else f"{n_cl} classical ch"
        ax.plot(DISTANCES_KM, skr_kbps, color=color, linewidth=2.0, label=label)

    ax.set_yscale("log")
    ax.set_ylim(1e-2, ax.get_ylim()[1])
    ax.set_xlim(DISTANCES_KM[0], DISTANCES_KM[-1])
    ax.set_title(
        "Finite-Key Decoy-State BB84 SKR vs. Fibre Distance",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Fibre Distance (km)", fontsize=11)
    ax.set_ylabel("Secure Key Rate (kb/s)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="best", title="Co-propagating load")

    path = out_dir / "skr_vs_distance.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


if __name__ == "__main__":
    main()
