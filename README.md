# PvZ Reinforcement Learning (Simplified Simulator + PPO)

A complete, runnable **Plants vs Zombies-inspired** reinforcement learning project using a custom Gymnasium environment and Stable-Baselines3 PPO.

> This is **not** the original PvZ game. It's a compact simulator designed for fast RL iteration, reproducibility, and extensibility.

## Why simulator-first?

Training directly on a real game is brittle (screen scraping, latency, nondeterminism, difficult action timing). This project starts with a structured simulator so PPO can actually learn policy behavior quickly, then you can scale mechanics later.

## Features

- 5-lane x 9-column PvZ-like simulator
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

# recommended fresh run on CPU
python train.py --timesteps 400000 --n-envs 8 --difficulty normal --run-name ppo_pvz_5x9

# recommended masked run (MaskablePPO + invalid action masking)
python train.py --timesteps 400000 --n-envs 8 --difficulty normal --run-name maskable_pvz_5x9 --masking

# evaluate trained model
python eval.py --policy ppo --model models/ppo_pvz_final.zip --episodes 10

# evaluate a masked model
python eval.py --policy ppo --algo maskable --model models/maskable_pvz_5x9_final.zip --episodes 10

# evaluate with stochastic action sampling (diagnostics)
python eval.py --policy ppo --model models/ppo_pvz_final.zip --episodes 10 --stochastic

# compare random baseline
python eval.py --policy random --episodes 10

# compare scripted baseline
python eval.py --policy scripted --episodes 10

# replay one episode with pygame
python replay.py --policy ppo --model models/ppo_pvz_final.zip

# replay a maskable model
python replay.py --policy ppo --algo maskable --model models/maskable_pvz_5x9_final.zip

# replay with stochastic sampling
python replay.py --policy ppo --model models/ppo_pvz_final.zip --stochastic

# replay with shape-only fallback (ignore PNG assets)
python replay.py --policy ppo --no-sprites

# replay speed/length controls
python replay.py --policy ppo --fps 30 --max-steps 600
```


## Compatibility note

The environment now uses a **5x9 board**, **17 actions**, and **42-dim observations**.
Older saved models from the 3x6 version are not compatible and should be retrained from scratch.

## Wave scheduling (high-level)

Zombie spawning is now driven by a simple timeline:
- trickle spawns happen throughout the episode
- flag-wave spikes occur at 25%, 50%, 75%, and 100% progress
- a final-wave spike happens near the end
- conehead share ramps up over time based on difficulty

Win condition remains surviving `EPISODE_STEPS`; wave completion is tracked in `info["snapshot"]` for debugging.

## Observation space (fixed, normalized float32)

Observation length: **42**

- sun (1)
- plant cooldowns sunflower/peashooter/wallnut (3)
- per lane (5 lanes): zombie_count, nearest_zombie_distance, threat_score, sunflower_count, peashooter_count, wallnut_count, mower_available (35 total)
- loose sun count (1)
- time progress (1)
- wave completion progress (1)

## Action space (Discrete 17)

0. noop
1. collect_sun
2-6. econ lane 0..4 (sunflower intent)
7-11. defend lane 0..4 (peashooter intent)
12-16. panic lane 0..4 (wall-nut or fallback defense)

The env translates these intents into legal placements with fallback tile selection. PPO still learns *when and where* to use each intent.

## Reward shaping (dense)

Centralized in `pvz_env/config.py`:

- +step survival reward
- +kill bonus
- +placement bonuses (any placement, sunflower economy, defender placement) so planting is discoverable
- +optional capped sun collection bonus (`sun_collect_bonus`, default `0.0`) to avoid collect-only reward farming
- -sun hoarding penalty above a threshold to discourage sitting on unspent economy
- -invalid action penalty
- -mower consumed penalty
- -large loss penalty
- +win bonus

Tune these coefficients to trade off econ greed vs safety.


Key anti-degeneracy knobs in `RewardConfig`:
- `sun_collect_bonus` + `sun_collect_cap`: optional capped collect reward (default bonus is zero)
- `empty_collect_penalty`: penalty for collect with no loose sun
- `tiny_collect_threshold` + `tiny_collect_penalty`: penalty for collecting very small amounts
- `place_any_bonus`: reward for any successful placement
- `place_sunflower_bonus`: extra reward for sunflower economy placement
- `place_defender_bonus`: extra reward for peashooter/wallnut placement
- `sun_hoard_threshold` + `sun_hoard_penalty`: small per-step penalty for hoarding too much sun



## Action masking

This repo supports invalid action masking via **sb3-contrib MaskablePPO**.

Why it helps:
- invalid actions are removed *before sampling*, so the agent does not waste updates on actions that cannot succeed
- training becomes more sample-efficient because exploration is focused on executable actions
- deterministic replay/eval becomes more reliable because argmax is taken over valid actions only

Enable masking in training:
```bash
python train.py --masking --timesteps 400000 --n-envs 8 --difficulty normal --run-name maskable_pvz_5x9
```

Evaluate/replay masked models:
```bash
python eval.py --policy ppo --algo maskable --model models/maskable_pvz_5x9_final.zip --episodes 10
python replay.py --policy ppo --algo maskable --model models/maskable_pvz_5x9_final.zip
```

`--stochastic` remains supported for both PPO and MaskablePPO; for masked models it samples from the masked action distribution.

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
- optional `--stochastic` action sampling for PPO diagnosis

Use this to verify RL > random and compare against a simple heuristic baseline.

## Replay / visualization

For MaskablePPO checkpoints, replay with:
```bash
python replay.py --algo maskable --policy ppo --model models/xxx.zip
```

`replay.py` uses pygame and draws:
- grid lanes and columns
- plant circles by type
- zombies moving right->left
- loose sun orbs
- HUD: step, sun, action name, reward, mower flags

### Sprites

You can optionally drop your own PNGs under:

```text
assets/
  plants/
    sunflower.png
    peashooter.png
    wallnut.png
  zombies/
    zombie.png
    cone_zombie.png
  ui/
    sun.png   (optional)
```

Sprite behavior in replay:
- if a PNG exists, replay uses it
- if it is missing or fails to load, replay falls back to built-in shape rendering for that entity only
- plant sprites are scaled to roughly 90% of tile width/height
- zombie sprites are scaled by height to roughly 95% of tile height while preserving aspect ratio
- `sun.png` is optional; missing file falls back to a yellow circle
- `--sprite-scale` multiplies these default target sizes
- `--no-sprites` disables PNG loading entirely and forces fallback shapes

## Curriculum option

Use difficulty levels:
- `easy`: lower trickle rate and gentler conehead ramp
- `normal`: default
- `hard`: higher trickle rate, larger wave multipliers, faster conehead ramp

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
3. Add more wave archetypes and scripted mixed-lane pushes.
4. Expand with new plant and zombie classes.
5. Train curriculum: easy -> normal -> hard model transfer.
6. Compare PPO hyperparameters (`n_steps`, `ent_coef`, `gamma`) using short sweeps.

## Next steps

- 5-lane full board with more plants/zombies
- wave scripting and event-based spawns
- imitation learning warm-start from scripted policy
- computer-vision adapter for real game prototyping
- multi-objective rewards and risk-sensitive policies
