# CLAUDE.md — qkd_dual_capacity_routing

## Project Overview

Course paper simulation project studying blocking performance of four routing
strategies under **dual classical + QKD key capacity constraints** on the
NSFNET 14-node topology.

## Getting Started

```bash
cd qkd_dual_capacity_routing
.venv\Scripts\activate
python run.py
```

Default mode: `abstract` QKD + `constant` 400 Gbps classical — runs without
any external project dependencies.

## Key Commands

```bash
# Default run
python run.py

# Actual SKR from qkd_optical_network (preferred — finite-key decoy BB84)
python run.py --qkd-capacity-mode actual_skr --old-project-root "E:/.../01_simulation_work/qkd_optical_network"

# CSV classical capacity
python run.py --classical-capacity-mode csv --classical-capacity-csv data/my_caps.csv

# Custom simulation parameters
python run.py --load-start 20 --load-end 200 --load-step 20 --num-requests 10000 --seed 42
```

## Architecture

```
run.py                        # CLI entry, argparse, sweep loop
qkd_routing/
  config.py                   # SimulationConfig dataclass
  topology.py                 # NSFNET14 graph + KSP precomputation
  traffic.py                  # Request dataclass + Poisson generation
  routing.py                  # 4 strategies + blocking classification
  resources.py                # EdgeResources + NetworkResources
  simulation.py               # heapq-driven discrete-event simulation
  metrics.py                  # DataFrame + CSV export
  plotting.py                 # 6 matplotlib figures (Agg backend)
  skr_adapter.py              # abstract / actual_skr (finite-key decoy BB84)
  gnpy_adapter.py             # constant / csv / gnpy_csv / gnpy_optional
  utils.py                    # EPS, canonical_edge, safe_div
```

## Reference Projects (READ ONLY)

- `E:\王雨婷个人文件夹\01：仿真代码合集\QKD_Network` — NSFNET14 topology, BB84_SKR_infinite
- `E:\王雨婷个人文件夹\学校统一事务\研一\研究工作\解川-交接材料\代码\KeyConsumption_24node` — topology matrices, KSP, event sim
- `E:\王雨婷个人文件夹\01_simulation_work\qkd_optical_network` — approx_finite_key_rate (finite-key decoy BB84)

**Never modify, delete, or overwrite files in these directories.**

## SKR Model Details

The `actual_skr` mode searches for external projects in order:

1. **qkd_optical_network** (preferred): `approx_finite_key_rate()` — three-state
   decoy BB84 (signal μ + decoy ν + vacuum) with Gaussian finite-key corrections.
   More rigorous; recommended for course paper results.

2. **QKD_Network** (fallback): `BB84_SKR_infinite()` — single-intensity
   infinite-key BB84.

## Edge Direction Convention

All resource dictionaries use canonical edge keys: `(min(u,v), max(u,v))`.
This ensures undirected edge handling regardless of path traversal direction.

## Float Tolerance

All capacity comparisons use `residual >= demand - EPS` with `EPS = 1e-9`.

## Output

- `results/simulation_results.csv` — per (load, strategy) metrics
- `results/*.png` — 6 publication-quality figures

## Dependencies

Python ≥ 3.10: networkx, numpy, pandas, matplotlib
