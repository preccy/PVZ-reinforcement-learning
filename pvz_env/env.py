from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .sim import PvZSimulator


@dataclass
class StepInfo:
    invalid_action: bool = False
    action_name: str = "noop"
    sun_collected: int = 0
    placed: bool = False
    placed_kind: Optional[str] = None


class PvZEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: Optional[int] = None, difficulty: str = "normal"):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.sim = PvZSimulator(self.rng, difficulty=difficulty)
        self.difficulty = difficulty
        self.action_space = spaces.Discrete(len(config.ACTION_MEANINGS))
        obs_size = 1 + 3 + (7 * config.LANES) + 1 + 1 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sim.rng = self.rng
        if options and "difficulty" in options:
            self.sim = PvZSimulator(self.rng, difficulty=options["difficulty"])
        self.sim.reset()
        return self._build_obs(), self._build_info(StepInfo())

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index {action}")

        step_info = self._apply_high_level_action(action)
        collected = 0
        if action == 1:
            if len(self.sim.loose_sun) == 0:
                step_info.invalid_action = True
            else:
                collected = self.sim.collect_sun()

        reward = config.REWARDS.step_survival
        if step_info.invalid_action:
            reward += config.REWARDS.invalid_action_penalty
        if action == 1 and collected == 0:
            reward += config.REWARDS.empty_collect_penalty
        if action == 1 and 0 < collected < config.REWARDS.tiny_collect_threshold:
            reward += config.REWARDS.tiny_collect_penalty

        if step_info.placed:
            reward += config.REWARDS.place_any_bonus
            if step_info.placed_kind == "sunflower":
                reward += config.REWARDS.place_sunflower_bonus
            if step_info.placed_kind in ("peashooter", "wallnut"):
                reward += config.REWARDS.place_defender_bonus

        sim_out = self.sim.step()
        if action == 1:
            step_info.sun_collected = collected
            reward += collected * config.REWARDS.sun_collect_bonus
            sim_out["sun_collected"] = collected

        reward += sim_out["kills"] * config.REWARDS.kill_bonus
        reward += sim_out["mower_used"] * config.REWARDS.mower_consumed_penalty
        if sim_out["lost"]:
            reward += config.REWARDS.lose_penalty
        if sim_out["won"]:
            reward += config.REWARDS.win_bonus

        terminated = self.sim.done
        truncated = False
        return self._build_obs(), float(reward), terminated, truncated, self._build_info(step_info)

    def render(self):
        snap = self.sim.snapshot()
        return (
            f"step={snap['step']} sun={snap['sun']} zombies={len(snap['zombies'])} "
            f"plants={len(snap['plants'])} mowers={snap['mowers']} wave={snap['wave_completion']:.2f}"
        )

    def _apply_high_level_action(self, action: int) -> StepInfo:
        name = config.ACTION_MEANINGS[action]
        info = StepInfo(action_name=name)
        if action in (0, 1):
            return info

        lane = int(name.rsplit("_", 1)[-1])
        if name.startswith("econ"):
            placed = self._try_place_lane("sunflower", lane, preferred_cols=[1, 0, 2])
            info.invalid_action = not placed
            info.placed = placed
            info.placed_kind = "sunflower" if placed else None
        elif name.startswith("defend"):
            placed = self._try_place_lane("peashooter", lane, preferred_cols=[2, 3, 1, 4])
            info.invalid_action = not placed
            info.placed = placed
            info.placed_kind = "peashooter" if placed else None
        elif name.startswith("panic"):
            danger = self._lane_danger(lane)
            placed_kind: Optional[str] = None
            placed = False
            if danger > 0.6:
                placed = self._try_place_lane("wallnut", lane, preferred_cols=[4, 3, 5, 2, 6])
                if placed:
                    placed_kind = "wallnut"
                else:
                    placed = self._try_place_lane("peashooter", lane, preferred_cols=[2, 3, 1, 4])
                    if placed:
                        placed_kind = "peashooter"
            else:
                placed = self._try_place_lane("peashooter", lane, preferred_cols=[3, 2, 1, 4])
                if placed:
                    placed_kind = "peashooter"
            info.invalid_action = not placed
            info.placed = placed
            info.placed_kind = placed_kind
        return info

    def _try_place_lane(self, kind: str, lane: int, preferred_cols: list[int]) -> bool:
        for col in preferred_cols:
            if self.sim.place(kind, lane, col):
                return True
        return False

    def _build_obs(self) -> np.ndarray:
        sun_norm = min(1.0, self.sim.state.sun / config.MAX_SUN)
        cooldowns = [
            self.sim.state.cooldowns[k] / max(1, config.PLANTS[k].cooldown)
            for k in ("sunflower", "peashooter", "wallnut")
        ]

        lane_features = []
        for lane in range(config.LANES):
            lane_plants = [p for p in self.sim.grid[lane] if p]
            lane_features.extend(
                [
                    self._lane_zombie_count(lane),
                    self._nearest_zombie_dist(lane),
                    self._lane_danger(lane),
                    min(1.0, sum(1 for p in lane_plants if p.kind == "sunflower") / config.COLS),
                    min(1.0, sum(1 for p in lane_plants if p.kind == "peashooter") / config.COLS),
                    min(1.0, sum(1 for p in lane_plants if p.kind == "wallnut") / config.COLS),
                    1.0 if self.sim.state.mowers[lane] else 0.0,
                ]
            )

        loose_sun = min(1.0, len(self.sim.loose_sun) / config.MAX_LOOSE_SUN)
        time_prog = min(1.0, self.sim.state.step_idx / config.EPISODE_STEPS)
        wave_prog = float(np.clip(self.sim.wave_completion_ratio, 0.0, 1.0))
        obs = np.array([sun_norm, *cooldowns, *lane_features, loose_sun, time_prog, wave_prog], dtype=np.float32)
        return obs

    def _lane_danger(self, lane: int) -> float:
        score = 0.0
        for z in self.sim.zombies:
            if z.lane == lane:
                zcfg = config.ZOMBIES[z.kind]
                closeness = 1.0 - min(1.0, z.x / config.COLS)
                score += (z.hp / zcfg.hp) * (0.5 + closeness)
        return min(1.0, score / 3.0)

    def _nearest_zombie_dist(self, lane: int) -> float:
        lane_z = [z for z in self.sim.zombies if z.lane == lane]
        if not lane_z:
            return 1.0
        nearest = min(lane_z, key=lambda z: z.x)
        return float(np.clip(nearest.x / config.COLS, 0.0, 1.0))

    def _lane_zombie_count(self, lane: int) -> float:
        return min(1.0, sum(1 for z in self.sim.zombies if z.lane == lane) / 8.0)

    def _build_info(self, step_info: StepInfo) -> dict[str, Any]:
        return {
            "action_name": step_info.action_name,
            "invalid_action": step_info.invalid_action,
            "sun_collected": step_info.sun_collected,
            "placed": step_info.placed,
            "placed_kind": step_info.placed_kind,
            "snapshot": self.sim.snapshot(),
        }


class ScriptedBaselinePolicy:
    def predict(self, obs: np.ndarray) -> int:
        sun = obs[0]
        lane_start = 4
        lane_block = 7
        threat_offsets = [lane_start + lane * lane_block + 2 for lane in range(config.LANES)]
        zombie_count_offsets = [lane_start + lane * lane_block for lane in range(config.LANES)]

        if obs[-3] > 0.25:
            return 1

        threats = np.array([obs[idx] for idx in threat_offsets], dtype=np.float32)
        max_lane = int(np.argmax(threats))
        if threats[max_lane] > 0.55:
            return 2 + (2 * config.LANES) + max_lane

        if sun > 0.25:
            zombie_counts = np.array([obs[idx] for idx in zombie_count_offsets], dtype=np.float32)
            defend_lane = int(np.argmin(zombie_counts))
            return 2 + config.LANES + defend_lane

        econ_lane = int(np.argmin(threats))
        return 2 + econ_lane
