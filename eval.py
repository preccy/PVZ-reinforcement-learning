from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pvz_env import PvZEnv, ScriptedBaselinePolicy
from pvz_env.render import format_text_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained/random/scripted policies")
    p.add_argument("--model", type=str, default="models/ppo_pvz_final.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    p.add_argument("--policy", choices=["ppo", "random", "scripted"], default="ppo")
    p.add_argument("--render", action="store_true", help="print step-by-step text output")
    p.add_argument("--stochastic", action="store_true", help="sample PPO actions instead of deterministic mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = PvZEnv(seed=args.seed, difficulty=args.difficulty)

    model = None
    scripted = None
    if args.policy == "ppo":
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(str(model_path))
    elif args.policy == "scripted":
        scripted = ScriptedBaselinePolicy()

    rewards = []
    wins = 0
    lengths = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            if args.policy == "ppo":
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            elif args.policy == "random":
                action = env.action_space.sample()
            else:
                action = scripted.predict(obs)

            obs, reward, term, trunc, info = env.step(int(action))
            done = term or trunc
            ep_reward += reward
            ep_len += 1
            if args.render:
                print(f"a={info['action_name']:>12} r={reward: .3f} | {format_text_snapshot(info['snapshot'])}")

        win = info["snapshot"]["win"]
        wins += int(win)
        lengths.append(ep_len)
        rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, steps={ep_len}, win={win}")

    print("-" * 60)
    print(f"Policy: {args.policy}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average steps: {np.mean(lengths):.1f}")
    print(f"Win rate: {wins / args.episodes:.2%}")


if __name__ == "__main__":
    main()
