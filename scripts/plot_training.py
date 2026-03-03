from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Monitor.csv rewards from SB3 VecMonitor logs")
    p.add_argument("--monitor", type=str, default="logs/ppo_pvz/monitor.csv")
    p.add_argument("--out", type=str, default="plots/training_curve.png")
    return p.parse_args()


def load_monitor(path: Path):
    rewards = []
    lengths = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rewards.append(float(parts[0]))
            lengths.append(float(parts[1]))
    return np.array(rewards), np.array(lengths)


def main() -> None:
    args = parse_args()
    path = Path(args.monitor)
    if not path.exists():
        raise FileNotFoundError(path)

    r, l = load_monitor(path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    window = min(30, len(r))
    if window == 0:
        raise RuntimeError("No episodes found in monitor file")
    smooth = np.convolve(r, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 4))
    plt.plot(r, alpha=0.4, label="episode reward")
    plt.plot(np.arange(window - 1, len(r)), smooth, lw=2.0, label=f"moving avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PvZ PPO Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
