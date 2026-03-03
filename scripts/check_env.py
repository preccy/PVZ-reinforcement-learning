from __future__ import annotations

import sys
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvz_env import PvZEnv


if __name__ == "__main__":
    env = PvZEnv(seed=0)
    check_env(env, warn=True)
    print("SB3 env checker passed")
