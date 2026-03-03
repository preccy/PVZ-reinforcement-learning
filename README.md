# PvZ Reinforcement Learning (Simplified Simulator + PPO)

A complete, runnable **Plants vs Zombies-inspired** reinforcement learning project using a custom Gymnasium environment and Stable-Baselines3 PPO.

> This is **not** the original PvZ game. It's a compact simulator designed for fast RL iteration, reproducibility, and extensibility.

## Why simulator-first?

Training directly on a real game is brittle (screen scraping, latency, nondeterminism, difficult action timing). This project starts with a structured simulator so PPO can actually learn policy behavior quickly, then you can scale mechanics later.

## Features

- 3-lane x 6-column PvZ-like simulator
- Plants: Sunflower, Peashooter, Wall-nut
- Zombies: Normal + Conehead
- Sun economy, cooldowns, sky sun, sunflower sun
- Mowers per lane (single-use emergency clear)
- High-level discrete action space for faster PPO learning
- Dense reward shaping in `pvz_env/config.py`
- PPO training pipeline with vectorized envs + checkpoints
- Evaluation modes: PPO, random, scripted baseline
- Pygame replay viewer for demos
- Basic tests + SB3 env checker
- Optional curriculum via `--difficulty easy|normal|hard`

## Repository layout

```
.
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── replay.py
├── Makefile
├── scripts/
│   └── plot_training.py
├── pvz_env/
│   ├── __init__.py
│   ├── config.py
│   ├── sim.py
│   ├── env.py
│   ├── render.py
│   └── utils.py
└── tests/
    └── test_env.py
```

`models/`, `logs/`, and `plots/` are generated at runtime and gitignored.

## Setup (Windows/macOS/Linux)

### 1) Create environment

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start commands

```bash
# smoke training (~50k steps)
python train.py --quick --timesteps 50000 --n-envs 4 --run-name quick_test

# full-ish first run
python train.py --timesteps 300000 --n-envs 8 --run-name ppo_pvz

# evaluate trained model
python eval.py --policy ppo --model models/ppo_pvz_final.zip --episodes 10

# compare random baseline
python eval.py --policy random --episodes 10

# compare scripted baseline
python eval.py --policy scripted --episodes 10

# replay one episode with pygame
python replay.py --policy ppo --model models/ppo_pvz_final.zip
```

## Observation space (fixed, normalized float32)

Observation length: **27**

- sun (1)
- plant cooldowns sunflower/peashooter/wallnut (3)
- lane danger summaries (3)
- nearest zombie distance per lane (3)
- zombie count per lane (3)
- per-lane plant counts [sunflower, peashooter, wallnut] x 3 lanes (9)
- mower flags per lane (3)
- loose sun count (1)
- time progress (1)

## Action space (Discrete 11)

0. noop
1. collect_sun
2-4. econ lane 0/1/2 (sunflower intent)
5-7. defend lane 0/1/2 (peashooter intent)
8-10. panic lane 0/1/2 (wall-nut or fallback defense)

The env translates these intents into legal placements with fallback tile selection. PPO still learns *when and where* to use each intent.

## Reward shaping (dense)

Centralized in `pvz_env/config.py`:

- +step survival reward
- +kill bonus
- +sun collection bonus
- -invalid action penalty
- -mower consumed penalty
- -large loss penalty
- +win bonus

Tune these coefficients to trade off econ greed vs safety.

## Training details

`train.py` uses:
- `stable_baselines3.PPO(MlpPolicy)`
- vectorized environments (`--n-envs`, default 8)
- checkpoint callback
- tensorboard log dir in `logs/<run-name>/`
- final model save in `models/<run-name>_final.zip`

TensorBoard:
```bash
tensorboard --logdir logs
```

## Evaluation and baselines

`eval.py` supports:
- `--policy ppo` (load trained model)
- `--policy random`
- `--policy scripted`
- optional `--render` text step trace

Use this to verify RL > random and compare against a simple heuristic baseline.

## Replay / visualization

`replay.py` uses pygame and draws:
- grid lanes and columns
- plant circles by type
- zombies moving right->left
- loose sun orbs
- HUD: step, sun, action name, reward, mower flags

## Curriculum option

Use difficulty levels:
- `easy`: lower zombie spawn pressure
- `normal`: default
- `hard`: higher pressure, less free economy tempo

Examples:
```bash
python train.py --difficulty easy
python eval.py --difficulty hard --policy ppo --model models/ppo_pvz_final.zip
```

## Testing

```bash
pytest -q
```

Coverage includes reset/step, observation shape/dtype, invalid action handling, non-negative resources/cooldowns, smoke episode completion, and SB3 env checker.

## Troubleshooting

- **pygame window fails on headless server**: use `eval.py --render` text mode instead of replay.
- **TensorBoard missing**: install from requirements or skip; training still works.
- **Slow training**: reduce `--n-envs`, `--timesteps`, or try `--difficulty easy`.
- **No model found in eval/replay**: verify path or run training first.

## Known limitations

- Simplified combat and movement timings (not frame-accurate PvZ)
- Single zombie archetype behaviors (only HP/speed differ)
- No projectiles rendered separately
- Action abstraction hides exact tile-level placement decisions

## Recommended first experiments

1. Increase kill bonus and reduce invalid penalty to encourage aggression.
2. Add action variants for explicit lane+plant combinations.
3. Extend wave schedule (progressive spawn ramps instead of fixed probabilities).
4. Expand board to 5 lanes and add new zombie classes.
5. Train curriculum: easy -> normal -> hard model transfer.
6. Compare PPO hyperparameters (`n_steps`, `ent_coef`, `gamma`) using short sweeps.

## Next steps

- 5-lane full board with more plants/zombies
- wave scripting and event-based spawns
- imitation learning warm-start from scripted policy
- computer-vision adapter for real game prototyping
- multi-objective rewards and risk-sensitive policies
