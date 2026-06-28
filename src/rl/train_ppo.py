#!/usr/bin/env python
"""Train a MaskablePPO agent for QKD-aware wavelength assignment.

Reward (v2): block=-3, accept=1+2*min(SKR_after/SKR_before, 1)
Break-even acceptance rate: 60% — well below observed ~77%, so reward is
positive at steady state and the learning signal is strong.

Usage:
    python -m src.rl.train_ppo
    python -m src.rl.train_ppo --total-timesteps 300K
    python -m src.rl.train_ppo --resume auto --total-timesteps 200K
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .qkd_wdm_env import EnvConfig, QKDWDMEnv, nsfnet14

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "ppo_checkpoints"
_LOG_DIR = _PROJECT_ROOT / "data" / "ppo_logs"


# ── convergence callback ──────────────────────────────────────────────────────

class ConvergenceCallback(BaseCallback):
    """Stop training when eval reward has converged.

    Convergence criterion: the relative standard deviation of the last
    `window` eval rewards is below `rel_tol`.  Only activates after
    `min_evals` evaluations so early noise doesn't trigger premature stopping.
    """

    def __init__(self, window: int = 10, rel_tol: float = 0.02, min_evals: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.window = window
        self.rel_tol = rel_tol
        self.min_evals = min_evals
        self._rewards: list = []

    def _on_step(self) -> bool:
        # Hook into EvalCallback's last_mean_reward after each evaluation
        if hasattr(self.parent, "last_mean_reward"):
            r = self.parent.last_mean_reward
            if len(self._rewards) == 0 or r != self._rewards[-1]:
                self._rewards.append(r)
                if len(self._rewards) >= self.min_evals:
                    window = self._rewards[-self.window:]
                    mean = float(np.mean(window))
                    std  = float(np.std(window))
                    rel_std = std / max(abs(mean), 1e-6)
                    if rel_std < self.rel_tol:
                        if self.verbose:
                            print(f"\nConvergence detected: rel_std={rel_std:.4f} < {self.rel_tol}"
                                  f" over last {self.window} evals. Stopping.")
                        return False   # signal SB3 to stop
        return True


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskablePPO for QKD wavelength assignment")
    p.add_argument("--total-timesteps", type=str, default="1M")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Steps collected per env per update")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10,
                   help="PPO epochs per update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="Entropy coefficient (encourages exploration)")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="Max steps per episode")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                   help="Resume from checkpoint .zip; 'auto' uses best_model.zip")
    return p.parse_args()


def _parse_timesteps(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


# ── environment builders ──────────────────────────────────────────────────────

def _mask_fn(env: QKDWDMEnv):
    return env.action_masks()


def make_env(cfg: EnvConfig, seed: int):
    def _init():
        env = QKDWDMEnv(config=cfg)
        env = ActionMasker(env, _mask_fn)   # MaskablePPO needs this wrapper
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    total_timesteps = _parse_timesteps(args.total_timesteps)

    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(topology=nsfnet14(), max_steps=args.max_steps, seed=args.seed)

    print("=" * 60)
    print("QKD PPO Training  (reward v2: block=-3, accept=1+2*SKR_ratio)")
    print("=" * 60)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  n_steps/update:  {args.n_steps}  ×  {args.n_envs} envs = {args.n_steps * args.n_envs} per update")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  PPO epochs:      {args.n_epochs}")
    print(f"  Entropy coef:    {args.ent_coef}")
    print(f"  Episode length:  {cfg.max_steps}")
    print(f"  Checkpoint dir:  {_CHECKPOINT_DIR}")
    print("=" * 60)

    # Vectorised train env
    vec_env = DummyVecEnv([make_env(cfg, args.seed + i) for i in range(args.n_envs)])

    # Separate eval env (single, unwrapped from DummyVec)
    eval_env = make_env(cfg, seed=args.seed + 999)()

    conv_cb = ConvergenceCallback(window=10, rel_tol=0.02, min_evals=20, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(_CHECKPOINT_DIR),
        log_path=str(_LOG_DIR),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        callback_after_eval=conv_cb,
    )

    # net_arch must be a dict (not list of dict) since SB3 v1.8
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=nn.ReLU,
    )

    if args.resume is not None:
        ckpt = _CHECKPOINT_DIR / "best_model.zip" if args.resume == "auto" else Path(args.resume)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"Resuming from: {ckpt}")
        model = MaskablePPO.load(str(ckpt), env=vec_env, device=args.device, verbose=1)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    elapsed = time.time() - start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = _CHECKPOINT_DIR / f"ppo_qkd_{timestamp}.zip"
    model.save(str(final_path))

    cfg_path = _CHECKPOINT_DIR / f"ppo_qkd_{timestamp}_config.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "total_timesteps": total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "elapsed_seconds": round(elapsed, 1),
            "final_model": str(final_path),
        }, f, indent=2)

    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
