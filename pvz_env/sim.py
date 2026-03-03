from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config


@dataclass
class Plant:
    kind: str
    lane: int
    col: int
    hp: float
    cooldown_tick: int = 0


@dataclass
class Zombie:
    id: int
    kind: str
    lane: int
    x: float
    hp: float


@dataclass
class LooseSun:
    lane: int
    x: float
    amount: int
    ttl: int


@dataclass
class SpawnEvent:
    step: int
    multiplier: float
    is_flag_wave: bool = False
    is_final_wave: bool = False


@dataclass
class SimState:
    step_idx: int = 0
    sun: int = config.INITIAL_SUN
    cooldowns: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in config.PLANTS.keys()})
    mowers: List[bool] = field(default_factory=lambda: [True] * config.LANES)


class PvZSimulator:
    def __init__(self, rng: np.random.Generator, difficulty: str = "normal"):
        self.rng = rng
        self.difficulty = difficulty
        self.wave_cfg = config.WAVES[difficulty]
        self.state = SimState()
        self.grid: List[List[Optional[Plant]]] = [[None for _ in range(config.COLS)] for _ in range(config.LANES)]
        self.zombies: List[Zombie] = []
        self.loose_sun: List[LooseSun] = []
        self.done = False
        self.win = False
        self.wave_schedule: dict[int, SpawnEvent] = {}
        self.final_wave_step = int(config.EPISODE_STEPS * 0.94)
        self.wave_completion_step: Optional[int] = None
        self.wave_completion_ratio = 0.0
        self._next_zombie_id = 1

    def reset(self) -> None:
        self.state = SimState()
        self.grid = [[None for _ in range(config.COLS)] for _ in range(config.LANES)]
        self.zombies = []
        self.loose_sun = []
        self.done = False
        self.win = False
        self.wave_schedule = self._build_wave_schedule()
        self.final_wave_step = max(
            self.final_wave_step,
            max((s for s, event in self.wave_schedule.items() if event.is_final_wave), default=self.final_wave_step),
        )
        self.wave_completion_step = None
        self.wave_completion_ratio = 0.0
        self._next_zombie_id = 1

    def can_place(self, kind: str, lane: int, col: int) -> bool:
        if self.done:
            return False
        if kind not in config.PLANTS or lane < 0 or lane >= config.LANES or col < 0 or col >= config.COLS:
            return False
        cfg = config.PLANTS[kind]
        if self.state.cooldowns[kind] > 0 or self.state.sun < cfg.cost:
            return False
        if self.grid[lane][col] is not None:
            return False
        return True

    def place(self, kind: str, lane: int, col: int) -> bool:
        if not self.can_place(kind, lane, col):
            return False
        cfg = config.PLANTS[kind]
        self.state.sun -= cfg.cost
        self.state.cooldowns[kind] = cfg.cooldown
        self.grid[lane][col] = Plant(kind=kind, lane=lane, col=col, hp=cfg.hp)
        return True

    def collect_sun(self) -> int:
        total = sum(s.amount for s in self.loose_sun)
        self.state.sun = int(min(config.MAX_SUN, self.state.sun + total))
        self.loose_sun.clear()
        return total

    def step(self) -> dict:
        if self.done:
            return {
                "kills": 0,
                "mower_used": 0,
                "sun_collected": 0,
                "lost": False,
                "won": self.win,
                "wave_completion": self.wave_completion_ratio,
                "final_wave_step": self.final_wave_step,
                "events": [],
            }

        self.state.step_idx += 1
        for k in self.state.cooldowns:
            self.state.cooldowns[k] = max(0, self.state.cooldowns[k] - 1)

        kills, events = self._plants_attack()
        self._spawn_sky_sun()
        self._spawn_zombies()
        mower_used = self._zombie_behaviour()
        self._sunflower_production()
        self._tick_loose_sun()
        self._update_wave_completion()

        lost = False
        if any(z.x <= config.HOUSE_X and not self.state.mowers[z.lane] for z in self.zombies):
            self.done = True
            lost = True

        if self.state.step_idx >= config.EPISODE_STEPS and not self.done:
            self.done = True
            self.win = True

        return {
            "kills": kills,
            "mower_used": mower_used,
            "sun_collected": 0,
            "lost": lost,
            "won": self.win,
            "wave_completion": self.wave_completion_ratio,
            "final_wave_step": self.final_wave_step,
            "events": events,
        }

    def _build_wave_schedule(self) -> dict[int, SpawnEvent]:
        schedule: dict[int, SpawnEvent] = {}
        spacing = 28
        trickle_start = 12
        for step in range(trickle_start, config.EPISODE_STEPS + 1, spacing):
            progress = min(1.0, step / config.EPISODE_STEPS)
            schedule[step] = SpawnEvent(step=step, multiplier=1.0 + 0.7 * progress)

        for ratio in (0.25, 0.50, 0.75, 1.00):
            step = max(1, min(config.EPISODE_STEPS, int(round(config.EPISODE_STEPS * ratio))))
            schedule[step] = SpawnEvent(step=step, multiplier=self.wave_cfg.flag_wave_multiplier, is_flag_wave=True)

        final_step = max(1, int(config.EPISODE_STEPS * 0.94))
        schedule[final_step] = SpawnEvent(
            step=final_step,
            multiplier=self.wave_cfg.final_wave_multiplier,
            is_flag_wave=True,
            is_final_wave=True,
        )
        self.final_wave_step = final_step
        return schedule

    def _spawn_sky_sun(self) -> None:
        if self.state.step_idx % self.wave_cfg.sky_sun_interval == 0:
            if len(self.loose_sun) < config.MAX_LOOSE_SUN:
                lane = int(self.rng.integers(0, config.LANES))
                x = float(self.rng.uniform(0, config.COLS - 1))
                self.loose_sun.append(LooseSun(lane=lane, x=x, amount=self.wave_cfg.sky_sun_amount, ttl=55))

    def _spawn_zombie(self, kind: str, lane: int, x: float) -> None:
        zcfg = config.ZOMBIES[kind]
        self.zombies.append(Zombie(id=self._next_zombie_id, kind=kind, lane=lane, x=x, hp=zcfg.hp))
        self._next_zombie_id += 1

    def _pick_zombie_kind(self, progress: float) -> str:
        cone_ratio = float(np.clip(0.10 + self.wave_cfg.conehead_ramp * progress, 0.0, 0.80))
        # Buckethead ramps from 0 in early game to configured cap; normal stays most common early.
        bucket_ratio = float(np.clip((progress - 0.33) * 1.5 * self.wave_cfg.buckethead_ramp, 0.0, 0.45))

        roll = float(self.rng.random())
        if roll < bucket_ratio:
            return "buckethead"
        if roll < bucket_ratio + cone_ratio:
            return "conehead"
        return "normal"

    def _spawn_zombies(self) -> None:
        schedule_event = self.wave_schedule.get(self.state.step_idx)
        progress = min(1.0, self.state.step_idx / config.EPISODE_STEPS)
        active_multiplier = schedule_event.multiplier if schedule_event else 1.0

        base_rate = self.wave_cfg.base_trickle_rate
        lane_spawn_prob = min(0.95, base_rate * active_multiplier * (0.65 + 0.5 * progress))

        for lane in range(config.LANES):
            if self.rng.random() >= lane_spawn_prob:
                continue
            self._spawn_zombie(kind=self._pick_zombie_kind(progress), lane=lane, x=config.COLS + 0.8)

        if schedule_event and schedule_event.is_flag_wave:
            burst = max(1, int(round(active_multiplier)))
            for _ in range(burst):
                lane = int(self.rng.integers(0, config.LANES))
                self._spawn_zombie(kind=self._pick_zombie_kind(progress), lane=lane, x=config.COLS + 0.8)

    def _plants_attack(self) -> tuple[int, list[dict]]:
        kills = 0
        events: list[dict] = []
        for lane in range(config.LANES):
            lane_zombies = [z for z in self.zombies if z.lane == lane]
            if not lane_zombies:
                continue
            nearest = min(lane_zombies, key=lambda z: z.x)
            for col in range(config.COLS):
                plant = self.grid[lane][col]
                if not plant or plant.kind != "peashooter":
                    continue
                plant.cooldown_tick += 1
                pcfg = config.PLANTS[plant.kind]
                if plant.cooldown_tick >= pcfg.attack_interval and nearest.x >= col:
                    nearest.hp -= pcfg.attack_damage
                    plant.cooldown_tick = 0
                    events.append(
                        {
                            "type": "pea_shot",
                            "lane": lane,
                            "x0": float(col + 0.5),
                            "x1": float(nearest.x),
                            "target_id": nearest.id,
                        }
                    )
            dead = [z for z in lane_zombies if z.hp <= 0]
            for z in dead:
                kills += 1
                if len(self.loose_sun) < config.MAX_LOOSE_SUN and self.rng.random() < 0.5:
                    self.loose_sun.append(LooseSun(lane=z.lane, x=max(0.0, z.x), amount=25, ttl=45))
                self.zombies.remove(z)
        return kills, events

    def _zombie_behaviour(self) -> int:
        mower_used = 0
        for z in list(self.zombies):
            plant_target = None
            plant_col = None
            for col in range(config.COLS):
                plant = self.grid[z.lane][col]
                if plant and z.x <= col + 0.25:
                    plant_target = plant
                    plant_col = col
                    break
            if plant_target is not None:
                zcfg = config.ZOMBIES[z.kind]
                plant_target.hp -= zcfg.dps * 0.1
                if plant_target.hp <= 0 and plant_col is not None:
                    self.grid[z.lane][plant_col] = None
            else:
                zcfg = config.ZOMBIES[z.kind]
                z.x -= zcfg.speed

            if z.x <= 0 and self.state.mowers[z.lane]:
                self.state.mowers[z.lane] = False
                mower_used += 1
                for other in list(self.zombies):
                    if other.lane == z.lane and other.x <= config.COLS + 1.2:
                        self.zombies.remove(other)
        return mower_used

    def _sunflower_production(self) -> None:
        for lane in range(config.LANES):
            for col in range(config.COLS):
                plant = self.grid[lane][col]
                if not plant or plant.kind != "sunflower":
                    continue
                plant.cooldown_tick += 1
                pcfg = config.PLANTS[plant.kind]
                if plant.cooldown_tick >= pcfg.sun_interval:
                    if len(self.loose_sun) < config.MAX_LOOSE_SUN:
                        self.loose_sun.append(LooseSun(lane=lane, x=col, amount=pcfg.sun_amount, ttl=60))
                    plant.cooldown_tick = 0

    def _tick_loose_sun(self) -> None:
        for s in self.loose_sun:
            s.ttl -= 1
        self.loose_sun = [s for s in self.loose_sun if s.ttl > 0]

    def _update_wave_completion(self) -> None:
        if self.wave_completion_step is not None:
            self.wave_completion_ratio = 1.0
            return
        if self.state.step_idx >= self.final_wave_step and len(self.zombies) == 0:
            self.wave_completion_step = self.state.step_idx
            self.wave_completion_ratio = 1.0
        elif self.state.step_idx < self.final_wave_step:
            self.wave_completion_ratio = 0.0
        else:
            span = max(1, config.EPISODE_STEPS - self.final_wave_step)
            progress = (self.state.step_idx - self.final_wave_step) / span
            self.wave_completion_ratio = float(np.clip(progress, 0.0, 0.99))

    def snapshot(self) -> dict:
        lane_sunflower_counts = [
            sum(1 for plant in self.grid[lane] if plant is not None and plant.kind == "sunflower")
            for lane in range(config.LANES)
        ]
        return {
            "step": self.state.step_idx,
            "sun": self.state.sun,
            "cooldowns": dict(self.state.cooldowns),
            "mowers": list(self.state.mowers),
            "plants": [
                {"kind": p.kind, "lane": p.lane, "col": p.col, "hp": p.hp}
                for lane in self.grid
                for p in lane
                if p is not None
            ],
            "zombies": [{"id": z.id, "kind": z.kind, "lane": z.lane, "x": z.x, "hp": z.hp} for z in self.zombies],
            "loose_sun": [{"lane": s.lane, "x": s.x, "amount": s.amount, "ttl": s.ttl} for s in self.loose_sun],
            "sunflowers_total": int(sum(lane_sunflower_counts)),
            "sunflowers_per_lane": lane_sunflower_counts,
            "wave_completion": self.wave_completion_ratio,
            "final_wave_step": self.final_wave_step,
            "done": self.done,
            "win": self.win,
        }
