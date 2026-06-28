#!/usr/bin/env python
"""Train a DQN agent for QKD-aware wavelength assignment.

Uses Stable-Baselines3 DQN with a vectorized environment.
The trained model can be evaluated against first-fit and random
baselines using ``eval_dqn.py``.

Usage:
    python train_dqn.py                          # default config
    python train_dqn.py --total-timesteps 5M     # shorter training
    python train_dqn.py --n-envs 8 --seed 123    # custom parallelism
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from .qkd_wdm_env import QKDWDMEnv, EnvConfig, nsfnet14
from .vec_env import make_vec_env
from .wrappers import ActionMaskWrapper

# ── default paths ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # qkd_dual_capacity_routing/
_CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "rl_checkpoints"
_LOG_DIR = _PROJECT_ROOT / "data" / "rl_logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN for QKD wavelength assignment")
    p.add_argument("--total-timesteps", type=str, default="10M",
                   help="Total training steps (e.g. 1M, 10M)")
    p.add_argument("--n-envs", type=int, default=16,
                   help="Number of parallel environments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=50_000)
    p.add_argument("--target-update-interval", type=int, default=10_000)
    p.add_argument("--exploration-fraction", type=float, default=0.70)
    p.add_argument("--exploration-initial-eps", type=float, default=0.9)
    p.add_argument("--exploration-final-eps", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Max steps per episode")
    p.add_argument("--n-eval-episodes", type=int, default=10,
                   help="Evaluation episodes")
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="Evaluate every N steps")
    p.add_argument("--no-subproc", action="store_true",
                   help="Disable subprocess parallelism (debug)")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device (auto/cpu/cuda)")
    p.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                   help="Resume from checkpoint .zip (default: auto-find best_model.zip)")
    return p.parse_args()


def _parse_timesteps(s: str) -> int:
    """Parse human-readable timestep string: 1M → 1_000_000, 10M → 10_000_000."""
    s = s.strip().upper()
    multiplier = 1
    if s.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith("K"):
        multiplier = 1_000
        s = s[:-1]
    return int(float(s) * multiplier)


def build_config(args: argparse.Namespace) -> EnvConfig:
    """Build environment config from CLI args."""
    topo = nsfnet14()
    return EnvConfig(
        topology=topo,
        max_steps=args.max_steps,
        seed=args.seed,
    )


# ── DQN policy kwargs (dueling architecture) ────────────────────────────────


def build_policy_kwargs() -> Dict:
    return dict(
        net_arch=[1024, 1024, 512],
        activation_fn=nn.ReLU,
    )


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    total_timesteps = _parse_timesteps(args.total_timesteps)

    # Create directories
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build config
    env_config = build_config(args)

    print("=" * 60)
    print("QKD DQN Training")
    print("=" * 60)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs:   {args.n_envs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Buffer size:      {args.buffer_size:,}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Network:          [1024, 1024, 512] (dueling)")
    print(f"  Episode length:   {env_config.max_steps}")
    print(f"  Checkpoint dir:   {_CHECKPOINT_DIR}")
    print(f"  Log dir:          {_LOG_DIR}")
    print("=" * 60)

    # Build vectorised environment
    vec_env = make_vec_env(
        env_config,
        n_envs=args.n_envs,
        seed=args.seed,
        use_subproc=not args.no_subproc,
    )

    # Build separate eval environment (single, with Monitor)
    eval_env = ActionMaskWrapper(QKDWDMEnv(config=env_config))
    eval_env = Monitor(eval_env)

    # Stop training if no improvement for 100 eval steps
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,
        min_evals=50,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(_CHECKPOINT_DIR),
        log_path=str(_LOG_DIR),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        callback_after_eval=stop_callback,
    )

    # Build or resume DQN model
    if args.resume is not None or (args.resume is None and (_CHECKPOINT_DIR / "best_model.zip").exists() and False):
        # explicit resume path
        resume_path = args.resume
    else:
        resume_path = None

    if args.resume is not None:
        # Load checkpoint; if not a path, auto-find best_model.zip
        ckpt = Path(args.resume) if args.resume != "auto" else _CHECKPOINT_DIR / "best_model.zip"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"Resuming from: {ckpt}")
        model = DQN.load(
            str(ckpt),
            env=vec_env,
            device=args.device,
            verbose=1,
        )
        # Override exploration to current (post-training) epsilon
        model.exploration_rate = args.exploration_initial_eps
        model.exploration_schedule = lambda progress: (
            args.exploration_initial_eps
            + (args.exploration_final_eps - args.exploration_initial_eps)
            * (1 - progress)
        )
    else:
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            target_update_interval=args.target_update_interval,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            max_grad_norm=10.0,
            policy_kwargs=build_policy_kwargs(),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )
    elapsed = time.time() - start_time

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = _CHECKPOINT_DIR / f"dqn_qkd_{timestamp}.zip"
    model.save(str(final_path))

    # Save training config
    config_path = _CHECKPOINT_DIR / f"dqn_qkd_{timestamp}_config.json"
    config_dict = {
        "total_timesteps": total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "network": [1024, 1024, 512],
        "elapsed_seconds": round(elapsed, 1),
        "final_model": str(final_path),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Model saved to: {final_path}")
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
