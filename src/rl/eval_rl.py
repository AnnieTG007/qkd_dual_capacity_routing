"""Compare PPO, DQN, and First-Fit on the QKD-WDM environment.

Metrics (no spacing — SKR implicitly captures frequency proximity):
  1. Blocking rate  (overall + breakdown: classical / skr / mixed)
  2. Classical utilisation  (avg fraction of wavelength×edge slots occupied)
  3. Average aggregate SKR  (sum over all edges, bps → converted to kbps)
  4. Mean episode reward

Figures saved:
  results/eval_rl_bars.png   — 4-panel comparison bar chart
  results/eval_rl_curves.png — training reward curves from evaluations.npz

Usage:
    python -m src.rl.eval_rl
    python -m src.rl.eval_rl --dqn data/rl_checkpoints/best_model.zip
    python -m src.rl.eval_rl --ppo data/ppo_checkpoints/best_model.zip
    python -m src.rl.eval_rl --n-episodes 30 --max-steps 1000 --seed 100
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .qkd_wdm_env import EnvConfig, QKDWDMEnv, nsfnet14

_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
_DQN_CKPT_DIR    = _PROJECT_ROOT / "data" / "rl_checkpoints"
_PPO_CKPT_DIR    = _PROJECT_ROOT / "data" / "ppo_checkpoints"
_DQN_LOG_DIR     = _PROJECT_ROOT / "data" / "rl_logs"
_PPO_LOG_DIR     = _PROJECT_ROOT / "data" / "ppo_logs"
_RESULTS_DIR     = _PROJECT_ROOT / "results"


# ── data containers ───────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    total_steps:      int   = 0
    blocked_steps:    int   = 0
    cumulative_reward: float = 0.0
    classical_util:   List[float] = field(default_factory=list)
    skr_samples:      List[float] = field(default_factory=list)
    block_reasons:    Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def blocking_rate(self) -> float:
        return self.blocked_steps / max(self.total_steps, 1)

    @property
    def mean_classical_util(self) -> float:
        return float(np.mean(self.classical_util)) if self.classical_util else 0.0

    @property
    def mean_skr_kbps(self) -> float:
        return float(np.mean(self.skr_samples) / 1000.0) if self.skr_samples else 0.0


# ── policies ──────────────────────────────────────────────────────────────────

def _first_fit_action(mask: np.ndarray) -> int:
    feasible = np.where(mask == 1)[0]
    return int(feasible[0]) if len(feasible) > 0 else 0


def _dqn_masked_action(model: DQN, obs: np.ndarray, mask: np.ndarray) -> int:
    """DQN inference with manual action masking via Q-value override."""
    import torch
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        q_vals = model.q_net(obs_tensor).squeeze(0).cpu().numpy()
    q_vals[mask == 0] = -np.inf
    feasible = np.where(mask == 1)[0]
    if len(feasible) == 0:
        return 0
    return int(np.argmax(q_vals))


def _ppo_masked_action(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> int:
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)


# ── episode runner ────────────────────────────────────────────────────────────

def _run_episode(
    env: QKDWDMEnv,
    policy: Callable[[np.ndarray, np.ndarray], int],
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    result = EpisodeResult()

    for _ in range(max_steps):
        mask   = info.get("action_mask", np.ones(env.W, dtype=np.int8))
        action = policy(obs, mask)

        obs, reward, terminated, truncated, info = env.step(action)
        result.total_steps       += 1
        result.cumulative_reward += reward
        result.classical_util.append(info.get("classical_util", 0.0))
        result.skr_samples.append(info.get("aggregate_skr", 0.0))

        if not info.get("accepted", reward >= 0):
            result.blocked_steps += 1
            reason = info.get("block_reason", "classical")
            result.block_reasons[reason] += 1

        if terminated or truncated:
            break

    return result


# ── main evaluation loop ──────────────────────────────────────────────────────

def _best_zip(ckpt_dir: Path) -> Optional[Path]:
    best = ckpt_dir / "best_model.zip"
    if best.exists():
        return best
    zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def evaluate(
    dqn_path: Optional[Path],
    ppo_path: Optional[Path],
    n_episodes: int,
    max_steps: int,
    base_seed: int,
) -> Dict[str, List[EpisodeResult]]:
    cfg = EnvConfig(topology=nsfnet14(), max_steps=max_steps)

    # Load models
    policies: Dict[str, Callable] = {}

    if dqn_path and dqn_path.exists():
        dqn = DQN.load(str(dqn_path))
        print(f"DQN model: {dqn_path}")
        policies["DQN"] = lambda obs, mask, m=dqn: _dqn_masked_action(m, obs, mask)
    else:
        print("DQN model not found — skipping.")

    if ppo_path and ppo_path.exists():
        ppo = MaskablePPO.load(str(ppo_path))
        print(f"PPO model: {ppo_path}")
        policies["PPO"] = lambda obs, mask, m=ppo: _ppo_masked_action(m, obs, mask)
    else:
        print("PPO model not found — skipping.")

    policies["First-Fit"] = lambda obs, mask: _first_fit_action(mask)

    results: Dict[str, List[EpisodeResult]] = {name: [] for name in policies}

    print(f"\nEvaluating {n_episodes} episodes × {max_steps} steps each")
    for ep in range(n_episodes):
        seed = base_seed + ep
        for name, policy in policies.items():
            env = QKDWDMEnv(config=cfg)
            r   = _run_episode(env, policy, seed, max_steps)
            results[name].append(r)
        if (ep + 1) % 5 == 0:
            # Quick progress line
            parts = []
            for name in policies:
                br = np.mean([r.blocking_rate for r in results[name]])
                parts.append(f"{name} block={br:.3f}")
            print(f"  Ep {ep+1:3d}/{n_episodes}  " + "  ".join(parts))

    return results


# ── figures ───────────────────────────────────────────────────────────────────

def _ci95(vals: List[float]) -> float:
    return 1.96 * float(np.std(vals)) / max(np.sqrt(len(vals)), 1)


def plot_bar_comparison(results: Dict[str, List[EpisodeResult]]) -> None:
    names  = list(results.keys())
    colors = {"DQN": "#4C72B0", "PPO": "#DD8452", "First-Fit": "#55A868"}
    c      = [colors.get(n, "#888888") for n in names]
    x      = np.arange(len(names))
    w      = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("DQN vs PPO vs First-Fit  (30 eval episodes × 1000 steps)", fontsize=11)

    def _bar(ax, vals_by_name, ylabel, title, pct=False, higher_better=True):
        means = [np.mean(vals_by_name[n]) for n in names]
        errs  = [_ci95(vals_by_name[n]) for n in names]
        bars  = ax.bar(x, means, w, color=c, yerr=errs, capsize=4,
                       error_kw=dict(elinewidth=1.2))
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        # Annotate relative improvement vs First-Fit
        if "First-Fit" in names:
            ff_mean = means[names.index("First-Fit")]
            for i, (n, m) in enumerate(zip(names, means)):
                if n == "First-Fit" or abs(ff_mean) < 1e-9:
                    continue
                rel = (m - ff_mean) / abs(ff_mean) * 100
                sign = "+" if rel > 0 else ""
                col  = "green" if (rel > 0) == higher_better else "red"
                ax.text(i, m + errs[i] + abs(m) * 0.02,
                        f"{sign}{rel:.1f}%", ha="center", va="bottom",
                        fontsize=7, color=col, fontweight="bold")
        return bars

    # 1. Blocking rate
    _bar(axes[0],
         {n: [r.blocking_rate for r in results[n]] for n in names},
         "Blocking Rate", "Blocking Rate", higher_better=False)

    # 2. Classical utilisation
    _bar(axes[1],
         {n: [r.mean_classical_util * 100 for r in results[n]] for n in names},
         "Utilisation (%)", "Classical Utilisation")

    # 3. Avg aggregate SKR
    _bar(axes[2],
         {n: [r.mean_skr_kbps for r in results[n]] for n in names},
         "Agg. SKR (kbps)", "Mean Aggregate SKR")

    # 4. Episode reward
    _bar(axes[3],
         {n: [r.cumulative_reward for r in results[n]] for n in names},
         "Cumul. Reward", "Episode Reward")

    plt.tight_layout()
    out = _RESULTS_DIR / "eval_rl_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_training_curves() -> None:
    """Plot eval reward curves from SB3 EvalCallback evaluations.npz logs."""
    sources = {
        "DQN":  _DQN_LOG_DIR / "evaluations.npz",
        "PPO":  _PPO_LOG_DIR / "evaluations.npz",
    }
    colors = {"DQN": "#4C72B0", "PPO": "#DD8452"}

    fig, ax = plt.subplots(figsize=(8, 4))
    found_any = False
    for label, path in sources.items():
        if not path.exists():
            continue
        data = np.load(str(path))
        steps   = data["timesteps"]
        rewards = data["results"]          # shape (n_evals, n_episodes)
        mean_r  = rewards.mean(axis=1)
        std_r   = rewards.std(axis=1)
        ax.plot(steps, mean_r, label=label, color=colors[label], linewidth=1.8)
        ax.fill_between(steps, mean_r - std_r, mean_r + std_r,
                        alpha=0.18, color=colors[label])
        found_any = True

    if not found_any:
        ax.text(0.5, 0.5, "No evaluations.npz found yet.\nTrain DQN/PPO first.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray", style="italic")

    ax.set_xlabel("Training Steps", fontsize=10)
    ax.set_ylabel("Mean Episode Reward", fontsize=10)
    ax.set_title("Training Reward Curves: PPO vs DQN", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = _RESULTS_DIR / "eval_rl_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def print_summary(results: Dict[str, List[EpisodeResult]]) -> None:
    ff = results.get("First-Fit", [])
    ff_block = np.mean([r.blocking_rate for r in ff]) if ff else 0
    ff_skr   = np.mean([r.mean_skr_kbps for r in ff]) if ff else 0
    ff_util  = np.mean([r.mean_classical_util for r in ff]) if ff else 0

    header = f"{'Metric':<32} {'DQN':>12} {'PPO':>12} {'First-Fit':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    def row(label, key_fn, fmt=".4f", higher_better=True):
        vals = {}
        for n in results:
            v = np.mean([key_fn(r) for r in results[n]])
            vals[n] = v
        line = f"{label:<32}"
        for n in ("DQN", "PPO", "First-Fit"):
            v = vals.get(n, float("nan"))
            line += f" {v:>12{fmt}}"
        print(line)
        ff_v = vals.get("First-Fit", float("nan"))
        for n in ("DQN", "PPO"):
            v = vals.get(n, float("nan"))
            if not (np.isnan(v) or np.isnan(ff_v) or abs(ff_v) < 1e-9):
                rel = (v - ff_v) / abs(ff_v) * 100
                sign = "+" if rel > 0 else ""
                better = (rel > 0) == higher_better
                tag = "OK" if better and abs(rel) >= 1 else ("~" if abs(rel) < 1 else "!!")
                print(f"  {n} vs FF: {sign}{rel:.2f}%  {tag}")

    row("Blocking Rate",          lambda r: r.blocking_rate,        higher_better=False)
    row("Classical Utilisation",  lambda r: r.mean_classical_util,  fmt=".4f")
    row("Avg Aggregate SKR (kbps)", lambda r: r.mean_skr_kbps,      fmt=".1f")
    row("Episode Reward",         lambda r: r.cumulative_reward,     fmt=".1f")

    # Blocking reason breakdown
    print("\nBlocking reason breakdown (% of all blocked steps):")
    for name, eps in results.items():
        total_blocked = sum(r.blocked_steps for r in eps)
        if total_blocked == 0:
            print(f"  {name}: no blocks")
            continue
        reasons = defaultdict(int)
        for r in eps:
            for k, v in r.block_reasons.items():
                reasons[k] += v
        parts = [f"{k}={v/total_blocked*100:.1f}%" for k, v in sorted(reasons.items())]
        print(f"  {name}: {', '.join(parts)}")
    print("=" * len(header))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dqn",        type=str, default=None)
    p.add_argument("--ppo",        type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--max-steps",  type=int, default=1000)
    p.add_argument("--seed",       type=int, default=100)
    args = p.parse_args()

    _RESULTS_DIR.mkdir(exist_ok=True)

    dqn_path = Path(args.dqn) if args.dqn else _best_zip(_DQN_CKPT_DIR)
    ppo_path = Path(args.ppo) if args.ppo else _best_zip(_PPO_CKPT_DIR)

    results = evaluate(dqn_path, ppo_path, args.n_episodes, args.max_steps, args.seed)
    print_summary(results)
    plot_bar_comparison(results)
    plot_training_curves()


if __name__ == "__main__":
    main()
