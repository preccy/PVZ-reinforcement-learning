from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from pvz_env import PvZEnv
from pvz_env.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on simplified PvZ simulator")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    p.add_argument("--run-name", type=str, default="ppo_pvz")
    p.add_argument("--quick", action="store_true", help="quick smoke training (50k timesteps)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 50_000)

    set_global_seed(args.seed)

    def make_env():
        return PvZEnv(seed=args.seed, difficulty=args.difficulty)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    vec_env = VecMonitor(vec_env)

    run_dir = Path("logs") / args.run_name
    model_dir = Path("models")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(save_freq=max(5_000 // args.n_envs, 1), save_path=str(model_dir / args.run_name), name_prefix="ppo_ckpt")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(run_dir),
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
    )

    print(f"Training for {args.timesteps} timesteps on difficulty={args.difficulty} with {args.n_envs} envs")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)

    final_path = model_dir / f"{args.run_name}_final"
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
