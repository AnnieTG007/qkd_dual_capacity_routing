"""Joint routing + wavelength-assignment evaluation.

For each arriving request, a routing strategy ranks the K shortest candidate
paths; wavelength assignment is then attempted on each path **in order** until
one succeeds (first-feasible wins), otherwise the request is blocked.  We
compare two wavelength-assignment (WA) policies on the *same* candidate paths:

  * First-Fit  — lowest feasible start wavelength
  * DQN        — masked argmax over the trained Q-network

A single :class:`QKDWDMEnv` instance is used as the physics / feasibility
oracle (occupancy grid, multi-slot action mask, per-edge SKR), so training and
evaluation share identical dynamics.  Routing strategies are ranked using the
oracle's current per-edge SKR and free-slot counts.

Usage::

    python -m src.rl.eval_joint_rwa --strategies dual_capacity_aware
    python -m src.rl.eval_joint_rwa --n-episodes 10 --max-steps 1000 --seed 100
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN

from .qkd_wdm_env import EnvConfig, QKDWDMEnv, nsfnet14

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DQN_CKPT_DIR = _PROJECT_ROOT / "data" / "rl_checkpoints"
_RESULTS_DIR = _PROJECT_ROOT / "results"

# Security-level traffic mix (matches qkd_routing/config.py and the env).
_SEC_LEVELS = [
    (0.5, [10.0, 40.0], 0.2),
    (0.3, [40.0, 100.0], 1.0),
    (0.2, [100.0, 400.0], 2.0),
]

STRATEGY_LABELS = {
    "min_hop": "Min-Hop",
    "min_distance": "Min-Distance",
    "key_capacity_aware": "Key-Capacity-Aware",
    "dual_capacity_aware": "Dual-Capacity-Aware",
}


# ── request stream ────────────────────────────────────────────────────────────

@dataclass
class JointRequest:
    src: int
    dst: int
    bw: float
    key: float


def _generate_requests(env: QKDWDMEnv, n: int, rng: np.random.Generator) -> List[JointRequest]:
    cum = np.cumsum([p for p, _, _ in _SEC_LEVELS])
    reqs: List[JointRequest] = []
    for _ in range(n):
        s, d = env._node_pairs[int(rng.integers(len(env._node_pairs)))]
        lvl = min(int(np.searchsorted(cum, rng.uniform() * cum[-1])), len(_SEC_LEVELS) - 1)
        _, bw_opts, key = _SEC_LEVELS[lvl]
        reqs.append(JointRequest(s, d, float(rng.choice(bw_opts)), key))
    return reqs


# ── routing-strategy path ranking ─────────────────────────────────────────────

def _path_len_km(env: QKDWDMEnv, nodes: List[int]) -> float:
    return float(sum(env.edge_lengths_km[e] for e in env.path_edge_indices(nodes)))


def rank_paths(
    strategy: str,
    env: QKDWDMEnv,
    candidates: List[List[int]],
    edge_skr: np.ndarray,
    key_demand_bps: float,
    n_slots: int,
    top_k: int = 3,
) -> List[List[int]]:
    """Order candidate paths by a routing strategy, return the top ``top_k``."""
    if not candidates:
        return []

    def hops(p):
        return len(p) - 1

    def bottleneck_key_ratio(p):
        idxs = env.path_edge_indices(p)
        return min(edge_skr[e] / max(key_demand_bps, 1e-9) for e in idxs)

    def bottleneck_dual_frac(p):
        idxs = env.path_edge_indices(p)
        out = float("inf")
        for e in idxs:
            free = int(np.sum(env.occupancy[e] == 0))
            cl_frac = min(free / max(env.W, 1), 1.0)
            key_frac = min(edge_skr[e] / max(key_demand_bps, 1e-9), 1.0)
            out = min(out, cl_frac, key_frac)
        return out

    if strategy == "min_hop":
        ordered = sorted(candidates, key=lambda p: (hops(p), _path_len_km(env, p)))
    elif strategy == "min_distance":
        ordered = sorted(candidates, key=lambda p: (_path_len_km(env, p), hops(p)))
    elif strategy == "key_capacity_aware":
        ordered = sorted(candidates, key=lambda p: (-bottleneck_key_ratio(p), _path_len_km(env, p)))
    elif strategy == "dual_capacity_aware":
        # Shortest hop tier first; within a tier, highest dual bottleneck first.
        ordered = sorted(candidates, key=lambda p: (hops(p), -bottleneck_dual_frac(p), _path_len_km(env, p)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return ordered[:top_k]


# ── wavelength-assignment policies ────────────────────────────────────────────

def _ff_choice(mask: np.ndarray) -> int:
    feas = np.where(mask == 1)[0]
    return int(feas[0]) if len(feas) else -1


def _dqn_choice(model: DQN, obs: np.ndarray, mask: np.ndarray) -> int:
    feas = np.where(mask == 1)[0]
    if len(feas) == 0:
        return -1
    obs_t = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        q = model.q_net(obs_t).squeeze(0).cpu().numpy()
    q[mask == 0] = -np.inf
    return int(np.argmax(q))


# ── episode runner ────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    total: int = 0
    blocked: int = 0
    skr_samples: List[float] = field(default_factory=list)

    @property
    def blocking_rate(self) -> float:
        return self.blocked / max(self.total, 1)

    @property
    def mean_skr_kbps(self) -> float:
        return float(np.mean(self.skr_samples) / 1000.0) if self.skr_samples else 0.0


def run_episode(
    env: QKDWDMEnv,
    strategy: str,
    wa_policy: Callable[[np.ndarray, np.ndarray], int],
    requests: List[JointRequest],
    seed: int,
) -> Metrics:
    env.np_random = np.random.default_rng(seed)
    env.occupancy = np.zeros((env.num_edges, env.W), dtype=np.float32)
    env.path_mask = np.zeros(env.num_edges, dtype=np.float32)
    env.edge_skr = env._compute_edge_skr()
    m = Metrics()

    for req in requests:
        env._simulate_departures()
        env.edge_skr = env._compute_edge_skr()
        env.request_bw, env.request_key = req.bw, req.key
        key_bps = req.key * 1000.0
        n_slots = env._n_slots

        ranked = rank_paths(
            strategy, env, env.candidate_paths(req.src, req.dst),
            env.edge_skr, key_bps, n_slots,
        )

        placed = False
        for nodes in ranked:
            env.path_mask = env.path_to_mask(nodes)
            mask = env._get_action_mask()
            obs = env._get_obs()
            action = wa_policy(obs, mask)
            if action >= 0:
                idxs = env.path_edge_indices(nodes)
                for e in idxs:
                    env.occupancy[e, action:action + n_slots] = 1.0
                env.edge_skr = env._compute_edge_skr()
                placed = True
                break

        m.total += 1
        if not placed:
            m.blocked += 1
        m.skr_samples.append(float(np.sum(env.edge_skr)))

    return m


# ── main evaluation ───────────────────────────────────────────────────────────

def _best_zip() -> Optional[Path]:
    best = _DQN_CKPT_DIR / "best_model.zip"
    if best.exists():
        return best
    zips = sorted(_DQN_CKPT_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def evaluate(strategies: List[str], n_episodes: int, max_steps: int,
             base_seed: int, holding_steps: int) -> Dict[str, Dict[str, Metrics]]:
    cfg = EnvConfig(topology=nsfnet14(), max_steps=max_steps,
                    mean_holding_steps=holding_steps)
    env = QKDWDMEnv(config=cfg)

    dqn_path = _best_zip()
    dqn = DQN.load(str(dqn_path)) if dqn_path else None
    print(f"DQN model: {dqn_path}")

    policies: Dict[str, Callable] = {"First-Fit": lambda o, m: _ff_choice(m)}
    if dqn is not None:
        policies["DQN"] = lambda o, m, mdl=dqn: _dqn_choice(mdl, o, m)

    # results[strategy][policy] = aggregated Metrics across episodes
    results: Dict[str, Dict[str, List[Metrics]]] = {
        s: {p: [] for p in policies} for s in strategies
    }

    req_rng = np.random.default_rng(base_seed)
    episode_reqs = [_generate_requests(env, max_steps, req_rng) for _ in range(n_episodes)]

    for strat in strategies:
        for ep in range(n_episodes):
            seed = base_seed + ep
            for pname, pol in policies.items():
                results[strat][pname].append(
                    run_episode(env, strat, pol, episode_reqs[ep], seed)
                )
        # progress
        line = []
        for pname in policies:
            br = np.mean([m.blocking_rate for m in results[strat][pname]])
            line.append(f"{pname} block={br:.3f}")
        print(f"  {STRATEGY_LABELS.get(strat, strat):<22} " + "  ".join(line))

    return results


def print_table(results) -> None:
    print("\n" + "=" * 78)
    print(f"{'Strategy':<22}{'Policy':<12}{'Blocking':>12}{'Agg SKR(kbps)':>16}")
    print("-" * 78)
    for strat, polmap in results.items():
        ff = polmap.get("First-Fit", [])
        ff_block = np.mean([m.blocking_rate for m in ff]) if ff else float("nan")
        for pname, eps in polmap.items():
            br = np.mean([m.blocking_rate for m in eps])
            skr = np.mean([m.mean_skr_kbps for m in eps])
            tag = ""
            if pname == "DQN" and ff:
                rel = (br - ff_block) / max(ff_block, 1e-9) * 100
                tag = f"  ({rel:+.1f}% vs FF)"
            print(f"{STRATEGY_LABELS.get(strat, strat):<22}{pname:<12}{br:>12.4f}{skr:>16.1f}{tag}")
    print("=" * 78)


def plot_bars(results) -> None:
    strats = list(results.keys())
    x = np.arange(len(strats))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    def series(metric):
        ff = [np.mean([getattr(m, metric) for m in results[s]["First-Fit"]]) for s in strats]
        dq = [np.mean([getattr(m, metric) for m in results[s]["DQN"]]) for s in strats]
        return ff, dq

    ff_b, dq_b = series("blocking_rate")
    ax1.bar(x - w / 2, ff_b, w, label="First-Fit", color="#55A868")
    ax1.bar(x + w / 2, dq_b, w, label="DQN", color="#4C72B0")
    ax1.set_ylabel("Blocking Rate")
    ax1.set_title("Blocking Rate by Routing Strategy", fontweight="bold", fontsize=10)

    ff_s, dq_s = series("mean_skr_kbps")
    ax2.bar(x - w / 2, ff_s, w, label="First-Fit", color="#55A868")
    ax2.bar(x + w / 2, dq_s, w, label="DQN", color="#4C72B0")
    ax2.set_ylabel("Mean Aggregate SKR (kbps)")
    ax2.set_title("Aggregate SKR by Routing Strategy", fontweight="bold", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strats], rotation=15, fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Joint Routing + WA: DQN vs First-Fit wavelength assignment (k=3 candidate paths)",
                 fontsize=11)
    plt.tight_layout()
    out = _RESULTS_DIR / "eval_joint_rwa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+",
                   default=["dual_capacity_aware", "min_distance", "min_hop", "key_capacity_aware"])
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--holding-steps", type=int, default=12,
                   help="mean holding steps (higher = heavier load)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    _RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Joint RWA eval: {args.n_episodes} episodes x {args.max_steps} steps, "
          f"holding={args.holding_steps}, seed={args.seed}")
    results = evaluate(args.strategies, args.n_episodes, args.max_steps,
                       args.seed, args.holding_steps)
    print_table(results)
    if not args.no_plot and "DQN" in next(iter(results.values())):
        plot_bars(results)


if __name__ == "__main__":
    main()
