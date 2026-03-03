from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from pvz_env import PvZEnv
from pvz_env.utils import set_global_seed


def action_mask_fn(env: PvZEnv):
    return env.get_action_mask()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on simplified PvZ simulator")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    p.add_argument("--run-name", type=str, default="ppo_pvz")
    p.add_argument("--quick", action="store_true", help="quick smoke training (50k timesteps)")
    p.add_argument("--load-model", type=Path, default=None, help="path to existing model .zip to resume from")
    p.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="reset SB3 timestep counter when calling learn() while resuming",
    )
    p.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="final model save name (without .zip); defaults to <run-name>_final",
    )
    p.add_argument("--n-steps", type=int, default=1024, help="rollout steps per env")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=6)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--torch-threads", type=int, default=6)
    p.add_argument("--torch-interop", type=int, default=2)
    p.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    p.add_argument("--masking", action="store_true", help="enable invalid action masking with MaskablePPO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 50_000)

    if args.load_model is not None and not args.load_model.exists():
        raise FileNotFoundError(
            f"Load model path does not exist: {args.load_model}. "
            "Provide a valid .zip created by Stable-Baselines3."
        )

    set_global_seed(args.seed)

    def make_env():
        env = PvZEnv(seed=args.seed, difficulty=args.difficulty)
        if args.masking:
            env = ActionMasker(env, action_mask_fn)
        return env

    # Use subprocess vectorization with spawn so env stepping can scale across CPU cores on Windows safely.
    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    vec_env = VecMonitor(vec_env)
    print("VEC ENV TYPE:", type(vec_env))
    print("NUM ENVS:", getattr(vec_env, "num_envs", "??"))

    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        print(
            f"Warning: batch_size ({args.batch_size}) > n_steps*n_envs ({rollout_size}); "
            f"clamping batch_size to {rollout_size}."
        )
        args.batch_size = rollout_size

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop)

    algo_name = "MaskablePPO" if args.masking else "PPO"

    print(
        "TRAIN CONFIG:",
        f"algo={algo_name}",
        f"n_envs={args.n_envs}",
        f"n_steps={args.n_steps}",
        f"batch_size={args.batch_size}",
        f"n_epochs={args.n_epochs}",
        f"torch_threads={args.torch_threads}",
        f"device={args.device}",
    )

    run_dir = Path("logs") / args.run_name
    model_dir = Path("models")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(model_dir / args.run_name),
        name_prefix="ppo_ckpt",
    )

    algo_cls = MaskablePPO if args.masking else PPO

    if args.load_model is not None:
        print(f"Resuming {algo_name} training from {args.load_model}")
        model = algo_cls.load(
            args.load_model,
            env=vec_env,
            tensorboard_log=str(run_dir),
            device=args.device,
        )
    else:
        print(f"Starting fresh {algo_name} training")
        model = algo_cls(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(run_dir),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
        )

    print(
        f"Training for {args.timesteps} timesteps on difficulty={args.difficulty} with {args.n_envs} envs using {algo_name}"
    )
    if args.load_model is not None:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            progress_bar=True,
            reset_num_timesteps=args.reset_timesteps,
        )
    else:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)

    final_name = args.save_name or f"{args.run_name}_final"
    final_path = model_dir / final_name
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
