# GPU Device Handoff — DQN Retraining

## Quick Start

```bash
git clone https://github.com/AnnieTG007/qkd_dual_capacity_routing.git
cd qkd_dual_capacity_routing
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate qkd_env
```

## What Needs Fixing Before Retraining

### Critical: multi-slot environment mismatch

The routing simulation (`qkd_routing/resources.py`) now allocates
`n_slots = ceil(bandwidth_gbps / 100)` wavelengths per request (e.g., a
400 Gbps request needs 4 slots).  The DQN environment
(`src/rl/qkd_wdm_env.py`) still allocates **exactly 1 slot per action**,
regardless of request bandwidth.  This inconsistency must be fixed.

**Recommended fix in `src/rl/qkd_wdm_env.py`:**

1. Add helper in `QKDWDMEnv`:
   ```python
   @property
   def _n_slots(self) -> int:
       return max(1, math.ceil(self.request_bw / self.cfg.classical_bandwidth_gbps))
   ```

2. In `_get_action_mask`: wavelength `w` is feasible only if the next
   `n_slots` wavelengths `w, w+1, ..., w+n_slots-1` are ALL free on all
   path edges AND satisfy the SKR constraint for all added channels.
   (Use contiguous slots for simplicity; matches flex-grid intuition.)

   ```python
   n = self._n_slots
   for w in range(self.W - n + 1):       # only valid start positions
       if any mask bit already 0: skip
       for e in path_edges:
           if any(self.occupancy[e, w:w+n] > 0):
               mask[w] = 0; break
       # SKR check: simulate adding all n channels
       ...
   # zero out slots too close to end to avoid index overflow
   for w in range(self.W - n + 1, self.W):
       mask[w] = 0
   ```

3. In `step`: allocate all `n_slots` wavelengths when action is chosen:
   ```python
   n = self._n_slots
   for e in path_edges:
       self.occupancy[e, action:action+n] = 1.0
   ```

4. In `_simulate_departures`: departures already work slot-by-slot —
   no change needed.

5. **Reward function** (`src/rl/reward.py`): no change needed. The SKR
   ratio `SKR_after / SKR_before` already captures the impact of adding
   n_slots classical channels (more noise → lower ratio → lower reward).

## Training Command

```bash
python -m src.rl.train_dqn \
    --total-timesteps 500000 \
    --learning-rate 3e-4 \
    --learning-starts 10000 \
    --target-update-interval 5000 \
    --exploration-final-eps 0.05 \
    --eval-freq 5000 \
    --n-eval-episodes 10 \
    --max-steps 2000
```

> **Note**: Remove `--no-subproc` on GPU device — subprocess parallelism
> (`SubprocVecEnv`) will use multiple CPU cores for environment rollouts
> and is much faster.  The default is already parallel (the flag disables it).

Expected training time on GPU + 8-core CPU: ~30–60 minutes for 500K steps.

## After Training: Generate Figures

```bash
# 1. Evaluate DQN vs First-Fit (30 episodes x 1000 steps)
python -m src.rl.eval_rl --seed 100

# 2. Re-run routing simulation (5000 requests/load, 50-400 Erlang)
python run.py --qkd-capacity-mode actual_skr \
    --load-start 50 --load-end 400 --load-step 50 \
    --num-requests 5000 --seed 42

# 3. Compile paper
pdflatex paper_qkd_coexistence.tex
pdflatex paper_qkd_coexistence.tex   # second pass for cross-references
```

## Current State (as of this commit)

| Item | Status |
|------|--------|
| Routing simulation (wavelength continuity, 32 slots) | Done, results in `results/` |
| Routing figures (blocking vs load, c_block=0) | Generated |
| DQN environment (multi-slot fix) | **TODO** |
| DQN training | Incomplete (CPU-only, ~30K steps) |
| DQN eval figures (bar chart, training curve) | Stale — regenerate after fix |
| Paper LaTeX | Compiled, 6 pages |

## Key Finding (Routing Simulation)

With W=32 wavelength slots per edge (3200 Gbps), **classical blocking is
zero** across all loads (50–400 Erlang).  All blocking is from QKD key
capacity.  `min_distance` routing consistently outperforms `min_hop`
(shorter paths → lower FWM/Raman noise → higher SKR).

The DQN's job is wavelength assignment: it should learn to prefer
wavelengths far from the quantum channel (slot 15 at 190.75 THz) to
minimize Raman scattering onto the quantum receiver.
