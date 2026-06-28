"""Evaluate trained DQN vs first-fit wavelength assignment.

Runs both policies on the same environment seeds and reports:
  - Blocking rate (requests where no feasible wavelength exists)
  - Average aggregate SKR preserved after assignment
  - Average wavelength spacing from quantum channel
  - Per-episode cumulative reward

Produces two figures saved to results/:
  eval_dqn_bars.png  — bar chart comparing key metrics
  eval_dqn_curves.png — per-episode reward curves

Usage:
    python -m src.rl.eval_dqn                          # auto-find latest checkpoint
    python -m src.rl.eval_dqn --model data/rl_checkpoints/best_model.zip
    python -m src.rl.eval_dqn --n-episodes 50 --max-steps 2000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from .qkd_wdm_env import QKDWDMEnv, EnvConfig, nsfnet14
from .wrappers import ActionMaskWrapper

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "rl_checkpoints"
_RESULTS_DIR = _PROJECT_ROOT / "results"


# ── policy helpers ────────────────────────────────────────────────────────────

def _first_fit_action(action_mask: np.ndarray) -> int:
    """Pick the lowest-index feasible wavelength (first-fit)."""
    feasible = np.where(action_mask == 1)[0]
    if len(feasible) == 0:
        return 0  # all blocked — any action triggers -10
    return int(feasible[0])


def _dqn_action(model: DQN, obs: np.ndarray) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ── episode metrics ───────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    total_steps: int = 0
    blocked_steps: int = 0
    accepted_steps: int = 0
    cumulative_reward: float = 0.0
    skr_samples: List[float] = field(default_factory=list)
    spacing_samples: List[float] = field(default_factory=list)

    @property
    def blocking_rate(self) -> float:
        return self.blocked_steps / max(self.total_steps, 1)

    @property
    def mean_skr(self) -> float:
        return float(np.mean(self.skr_samples)) if self.skr_samples else 0.0

    @property
    def mean_spacing(self) -> float:
        return float(np.mean(self.spacing_samples)) if self.spacing_samples else 0.0


def _run_episode(
    env: QKDWDMEnv,
    policy,           # callable(obs, mask) -> int
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    result = EpisodeResult()

    # Quantum slot index for spacing computation
    base_env = env.unwrapped
    q_slot = int(round(
        (base_env.cfg.quantum_freq_hz - base_env.cfg.base_freq_hz)
        / base_env.cfg.channel_spacing_hz
    ))

    for _ in range(max_steps):
        mask = info.get("action_mask", np.ones(base_env.W, dtype=np.int8))
        action = policy(obs, mask)

        obs, reward, terminated, truncated, info = env.step(action)
        result.total_steps += 1
        result.cumulative_reward += reward

        if reward < -5.0:   # blocked (-10)
            result.blocked_steps += 1
        else:                # accepted (>=1)
            result.accepted_steps += 1
            # SKR: sum over all edges after assignment
            skr_total = float(np.sum(base_env.edge_skr)) if base_env.edge_skr is not None else 0.0
            result.skr_samples.append(skr_total)
            # Spacing: distance of chosen wavelength from quantum slot
            spacing = abs(action - q_slot)
            result.spacing_samples.append(spacing)

        if terminated or truncated:
            break

    return result


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model_path: Optional[Path],
    n_episodes: int,
    max_steps: int,
    base_seed: int,
) -> Tuple[List[EpisodeResult], List[EpisodeResult]]:
    """Run DQN and first-fit for n_episodes each; return (dqn_results, ff_results)."""

    # Load model — prefer best_model.zip (EvalCallback's best), else newest by mtime
    if model_path is None:
        best = _CHECKPOINT_DIR / "best_model.zip"
        if best.exists():
            model_path = best
        else:
            candidates = sorted(_CHECKPOINT_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(
                    f"No .zip checkpoint found in {_CHECKPOINT_DIR}. "
                    "Train first with: python -m src.rl.train_dqn"
                )
            model_path = candidates[-1]

    print(f"Loading model: {model_path}")
    model = DQN.load(str(model_path))

    cfg = EnvConfig(topology=nsfnet14(), max_steps=max_steps)

    dqn_results: List[EpisodeResult] = []
    ff_results: List[EpisodeResult] = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        # DQN episode
        env_dqn = ActionMaskWrapper(QKDWDMEnv(config=cfg))
        dqn_policy = lambda obs, mask: _dqn_action(model, obs)
        r_dqn = _run_episode(env_dqn, dqn_policy, seed, max_steps)
        dqn_results.append(r_dqn)

        # First-fit episode (same seed)
        env_ff = ActionMaskWrapper(QKDWDMEnv(config=cfg))
        ff_policy = lambda obs, mask: _first_fit_action(mask)
        r_ff = _run_episode(env_ff, ff_policy, seed, max_steps)
        ff_results.append(r_ff)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Episode {ep+1:3d}/{n_episodes}  "
                f"DQN block={r_dqn.blocking_rate:.3f}  "
                f"FF  block={r_ff.blocking_rate:.3f}"
            )

    return dqn_results, ff_results


# ── plotting ──────────────────────────────────────────────────────────────────

def _ci95(values: List[float]) -> float:
    """95% confidence interval half-width."""
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(n)


def plot_bar_comparison(
    dqn_results: List[EpisodeResult],
    ff_results: List[EpisodeResult],
    out_path: Path,
) -> None:
    """Four-panel bar chart: blocking, SKR, spacing, reward."""

    metrics = {
        "Blocking Rate\n(lower is better)": (
            [r.blocking_rate for r in dqn_results],
            [r.blocking_rate for r in ff_results],
        ),
        "Mean Aggregate SKR (bps)\n(higher is better)": (
            [r.mean_skr for r in dqn_results],
            [r.mean_skr for r in ff_results],
        ),
        "Mean Wavelength Spacing\nfrom Quantum (slots, higher better)": (
            [r.mean_spacing for r in dqn_results],
            [r.mean_spacing for r in ff_results],
        ),
        "Mean Episode Reward\n(higher is better)": (
            [r.cumulative_reward for r in dqn_results],
            [r.cumulative_reward for r in ff_results],
        ),
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("DQN vs First-Fit Wavelength Assignment", fontsize=13, fontweight="bold")

    colors = {"DQN": "#2196F3", "First-Fit": "#FF7043"}
    x = np.array([0, 1])
    width = 0.5

    for ax, (title, (dqn_vals, ff_vals)) in zip(axes, metrics.items()):
        dqn_mean = np.mean(dqn_vals)
        ff_mean = np.mean(ff_vals)
        dqn_err = _ci95(dqn_vals)
        ff_err = _ci95(ff_vals)

        bars = ax.bar(
            x, [dqn_mean, ff_mean],
            width=width,
            color=[colors["DQN"], colors["First-Fit"]],
            yerr=[dqn_err, ff_err],
            capsize=5,
            error_kw={"linewidth": 1.5},
        )

        # Annotate improvement
        if ff_mean != 0:
            pct = (dqn_mean - ff_mean) / abs(ff_mean) * 100
            sign = "+" if pct >= 0 else ""
            ax.text(
                0.5, 0.95, f"DQN: {sign}{pct:.1f}%",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#1565C0" if pct >= 0 else "#B71C1C",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["DQN", "First-Fit"], fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_reward_curves(
    dqn_results: List[EpisodeResult],
    ff_results: List[EpisodeResult],
    out_path: Path,
) -> None:
    """Line chart of per-episode cumulative reward for DQN and first-fit."""

    dqn_rewards = [r.cumulative_reward for r in dqn_results]
    ff_rewards = [r.cumulative_reward for r in ff_results]
    episodes = np.arange(1, len(dqn_rewards) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, dqn_rewards, "o-", color="#2196F3", label="DQN", linewidth=1.5, markersize=4)
    ax.plot(episodes, ff_rewards, "s--", color="#FF7043", label="First-Fit", linewidth=1.5, markersize=4)

    # Running averages
    window = max(3, len(episodes) // 10)
    if len(episodes) >= window:
        dqn_avg = np.convolve(dqn_rewards, np.ones(window) / window, mode="valid")
        ff_avg = np.convolve(ff_rewards, np.ones(window) / window, mode="valid")
        x_avg = episodes[window - 1:]
        ax.plot(x_avg, dqn_avg, color="#0D47A1", linewidth=2.5, label=f"DQN (MA-{window})")
        ax.plot(x_avg, ff_avg, color="#BF360C", linewidth=2.5, label=f"FF (MA-{window})")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=11)
    ax.set_title("DQN vs First-Fit: Per-Episode Reward", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def print_summary(
    dqn_results: List[EpisodeResult],
    ff_results: List[EpisodeResult],
) -> None:
    headers = ["Metric", "DQN", "First-Fit", "Improvement"]
    rows = []

    def pct(a, b):
        if b == 0:
            return "N/A"
        return f"{(a - b) / abs(b) * 100:+.2f}%"

    dqn_block = np.mean([r.blocking_rate for r in dqn_results])
    ff_block = np.mean([r.blocking_rate for r in ff_results])
    rows.append(["Blocking Rate", f"{dqn_block:.4f}", f"{ff_block:.4f}",
                 pct(-dqn_block, -ff_block)])  # lower is better → negate

    dqn_skr = np.mean([r.mean_skr for r in dqn_results])
    ff_skr = np.mean([r.mean_skr for r in ff_results])
    rows.append(["Mean Aggregate SKR (bps)", f"{dqn_skr:.1f}", f"{ff_skr:.1f}",
                 pct(dqn_skr, ff_skr)])

    dqn_sp = np.mean([r.mean_spacing for r in dqn_results])
    ff_sp = np.mean([r.mean_spacing for r in ff_results])
    rows.append(["Mean Wavelength Spacing", f"{dqn_sp:.2f}", f"{ff_sp:.2f}",
                 pct(dqn_sp, ff_sp)])

    dqn_rew = np.mean([r.cumulative_reward for r in dqn_results])
    ff_rew = np.mean([r.cumulative_reward for r in ff_results])
    rows.append(["Mean Episode Reward", f"{dqn_rew:.1f}", f"{ff_rew:.1f}",
                 pct(dqn_rew, ff_rew)])

    col_w = [max(len(h), max(len(r[i]) for r in rows)) + 2
             for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    print("\n" + sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DQN vs first-fit")
    p.add_argument("--model", type=str, default=None,
                   help="Path to trained model .zip (auto-selects latest if omitted)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model) if args.model else None

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {args.n_episodes} episodes × {args.max_steps} steps each …")
    dqn_results, ff_results = evaluate(
        model_path, args.n_episodes, args.max_steps, args.seed
    )

    print_summary(dqn_results, ff_results)
    plot_bar_comparison(dqn_results, ff_results, _RESULTS_DIR / "eval_dqn_bars.png")
    plot_reward_curves(dqn_results, ff_results, _RESULTS_DIR / "eval_dqn_curves.png")


if __name__ == "__main__":
    main()
