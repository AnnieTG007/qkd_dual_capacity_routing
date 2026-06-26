# CLAUDE.md — qkd_dual_capacity_routing

## Project Overview

Course paper simulation project studying blocking performance of four routing
strategies under **dual classical + QKD key capacity constraints** on the
NSFNET 14-node topology.

## Getting Started

```bash
cd "E:\王雨婷个人文件夹\课程资料\通信网理论\qkd_network_routing\qkd_dual_capacity_routing"
conda activate qkd_env
python run.py
```

Default mode: `abstract` QKD + `constant` 400 Gbps classical.

## Key Commands

```bash
# Default run
python run.py

# Actual SKR mode (self-contained — finite-key decoy BB84 + GNPy FWM/SpRS noise)
python run.py --qkd-capacity-mode actual_skr

# CSV classical capacity
python run.py --classical-capacity-mode csv --classical-capacity-csv data/my_caps.csv

# GNPy CSV classical capacity
python run.py --classical-capacity-mode gnpy_csv --gnpy-result-csv data/gnpy.csv

# Custom simulation parameters
python run.py --load-start 20 --load-end 200 --load-step 20 --num-requests 10000 --seed 42

# Compare only two strategies
python run.py --strategies min_hop dual_capacity_aware
```

## Architecture

```
run.py                        # CLI entry, argparse, sweep loop
qkd_routing/
  config.py                   # SimulationConfig dataclass + CLI args
  topology.py                 # NSFNET14 graph (21 edges) + KSP precomputation
  traffic.py                  # Request dataclass + Poisson generation (3 security levels)
  routing.py                  # 4 strategies + blocking classification
  resources.py                # EdgeResources + NetworkResources (canonical edge key)
  simulation.py               # heapq-driven discrete-event simulation + time-weighted util
  physics.py                  # Self-contained: finite-key decoy BB84 + GNPy FWM/SpRS noise
  skr_adapter.py              # abstract / actual_skr QKD capacity provider
  gnpy_adapter.py             # constant / csv / gnpy_csv / gnpy_optional classical capacity
  metrics.py                  # DataFrame + CSV export
  plotting.py                 # 6 matplotlib figures (Agg backend, no Tk needed)
  utils.py                    # EPS=1e-9, canonical_edge, safe_div
data/
  README.md
results/                      # Output CSV + PNGs
```

## Reference Projects (READ ONLY)

- `E:\王雨婷个人文件夹\01：仿真代码合集\QKD_Network` — NSFNET14 topology, BB84_SKR_infinite
- `E:\王雨婷个人文件夹\学校统一事务\研一\研究工作\解川-交接材料\代码\KeyConsumption_24node` — topology matrices, KSP, event sim
- `E:\王雨婷个人文件夹\01_simulation_work\qkd_optical_network` — approx_finite_key_rate, GNPy noise model reference

**Never modify, delete, or overwrite files in these directories.**

## SKR Model Details

The `actual_skr` mode uses the **self-contained** `physics.py` module (no external
project dependencies):

- **BB84**: 3-state decoy protocol (signal μ=0.75 + decoy ν=0.2 + vacuum),
  finite-key Gaussian corrections (γ_KS=5.3, N=10¹⁰ pulses)
- **FWM noise**: Discrete single-frequency triplet enumeration with GNPy
  efficiency coefficient η, phase matching Δβ, pump frequencies at 50 GHz grid
- **SpRS noise**: Phonon-occupation-corrected Raman cross-section with
  Stokes/anti-Stokes distinction, 92-point GNPy SSMF Raman gain table
- **Dark-fibre reference**: 10km→185 kbps, 50km→28 kbps, 90km→4 kbps

See `qkd_routing/physics.py` for full parameter set and formulas.

## Edge Direction Convention

All resource dictionaries use canonical edge keys: `(min(u,v), max(u,v))`.
This ensures undirected edge handling regardless of path traversal direction.

## Float Tolerance

All capacity comparisons use `residual >= demand - EPS` with `EPS = 1e-9`.

## Utilization Tracking

Time-weighted average utilization: `∫ total_used(t) dt / simulation_duration`,
normalized by total network capacity. Peak utilization tracks the maximum
instantaneous per-edge value across all events.

## Output

- `results/simulation_results.csv` — per (load, strategy) metrics
- `results/*.png` — 6 publication-quality figures

## Dependencies

Python ≥ 3.10 (conda env `qkd_env`): networkx, numpy, pandas, matplotlib, pyyaml
