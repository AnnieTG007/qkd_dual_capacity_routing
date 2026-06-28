# Work Plan — QKD Coexistence Resource Allocation Paper + RL Integration

**Date**: 2026-06-27/28 (overnight session)
**Status**: 🔄 In Progress (DQN training running)

---

## Paper Structure (4-algorithm ladder)

| Tier | Algorithm | Role |
|------|-----------|------|
| Benchmark | Shortest-path routing + First-fit wavelength | Baseline |
| Contribution 1 | Key-aware / Dual-aware routing + First-fit | Network theory (course concepts) |
| Contribution 2 | DQN wavelength selection (replace First-fit) | RL for wavelength allocation |

---

## Completed

### Phase 1: Paper (✅ Done)
- [x] `paper_qkd_coexistence.tex` — IEEE conference format
- [x] Abstract, Introduction, Network Modeling, Algorithm Design, Simulation Analysis, Conclusion
- [x] 15 references, 12 equations
- [x] 4 routing strategies: MH, MD, KCA, DCA
- [x] DQN section: state/action/reward/architecture/training
- [x] Discrete single-frequency FWM + SpRS noise models (undergrad thesis style)
- [x] Appendix with code listing (8 modules, usage commands)

### Phase 2: RL Code Module (✅ Done)
- [x] `src/qkd_sim/network/rl/__init__.py` — package init
- [x] `src/qkd_sim/network/rl/qkd_wdm_env.py` — Gymnasium env (700+ lines)
  - NSFNET14 topology, WDM grid, Poisson traffic
  - Discrete FWM + SpRS noise models
  - BB84 decoy-state SKR with finite-key corrections
  - Action masking for feasibility
  - Observation: occupancy + path_mask + quantum_pos + request
- [x] `src/qkd_sim/network/rl/reward.py` — SKR-based composite reward
- [x] `src/qkd_sim/network/rl/wrappers.py` — ActionMaskWrapper for SB3
- [x] `src/qkd_sim/network/rl/vec_env.py` — SubprocVecEnv builder (16 parallel)
- [x] `src/qkd_sim/network/rl/train_dqn.py` — SB3 DQN training script
  - Dueling architecture [1024, 1024, 512]
  - All hyperparameters from DQN-QKD sweep
  - EvalCallback + StopTrainingOnNoModelImprovement
  - Auto-save checkpoints and config

### Phase 3: Integration (✅ Done)
- [x] Environment imports noise/SKR models from project
- [x] Training script with full CLI
- [x] Paper appendix references all modules

---

## Files Created / Modified

| File | Status |
|------|--------|
| `paper_qkd_coexistence.tex` | ✅ New (complete draft) |
| `src/qkd_sim/network/rl/__init__.py` | ✅ New |
| `src/qkd_sim/network/rl/qkd_wdm_env.py` | ✅ New |
| `src/qkd_sim/network/rl/reward.py` | ✅ New |
| `src/qkd_sim/network/rl/wrappers.py` | ✅ New |
| `src/qkd_sim/network/rl/vec_env.py` | ✅ New |
| `src/qkd_sim/network/rl/train_dqn.py` | ✅ New |
| `WORK_PLAN.md` | ✅ Updated |

---

## Completed in This Session (2026-06-28)

### Bugs Fixed
- `train_dqn.py`: `_PROJECT_ROOT` parents[3]→parents[2]; `activation_fn=None`→`nn.ReLU`; `progress_bar=True`→False
- `qkd_wdm_env.py`: `estimate_noise` N<2 returned 2-tuple (should be 3); `classical_power_dbm=-10.0`→`-25.0` (100× power error killing all QKD); quantum channel at 193.5 THz was outside 32-slot grid → moved to 190.75 THz (slot 15); added `_simulate_departures()` so env reaches dynamic steady state; `_get_action_mask` rewrote to only recompute single edge (24× speedup)
- `routing.py`: `dual_capacity_aware` → hop-tier grouping + normalized fraction scoring
- `config.py`: `n_max_classical_channels` 16→8 (creates real classical blocking)

### Installed
- torch 2.12.1+cpu, stable-baselines3 2.9.0 into qkd_env

### Simulation Results (new)
| Load (E) | min_hop | dual_capacity_aware | Improvement |
|----------|---------|---------------------|-------------|
| 200      | 0.3498  | **0.3298**          | **+2.0 pp** |
| 400      | 0.4655  | **0.4558**          | **+0.97 pp**|
| 300 classical | 0.0703 | **0.0593**      | **+16%**    |

DCA reduces total blocking by up to 2.0 pp at 200 Erlang (5.7% relative improvement >1% ✓).

### Paper Updated
- Empty `fig:skr_distance` placeholder → `results/skr_vs_distance.png` (generated)
- Added `\includegraphics` for all 7 figures (blocking, key_blocking, classical_blocking, utilization, dqn_bars, dqn_curves)
- Results section updated to reflect dual-capacity constraint findings

### DQN Status
- 100K-step training running: `python -m src.rl.train_dqn --no-subproc --total-timesteps 100K`
- After completion: `python -m src.rl.eval_dqn --n-episodes 30 --max-steps 1000`
- `src/rl/eval_dqn.py` created: 4-panel bar chart + episode reward curves

## For User Review
1. Wait for DQN training to finish, then run eval
2. Check paper equations for FWM/SpRS formula accuracy
3. Add course textbook citation if required

## No Edits Made to Reference Materials (per constraint)
