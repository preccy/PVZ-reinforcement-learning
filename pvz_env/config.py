from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantConfig:
    cost: int
    hp: float
    cooldown: int
    attack_damage: float = 0.0
    attack_interval: int = 1
    sun_interval: int = 0
    sun_amount: int = 0


@dataclass(frozen=True)
class ZombieConfig:
    hp: float
    speed: float
    dps: float
    reward_on_kill: float


@dataclass(frozen=True)
class RewardConfig:
    step_survival: float = 0.03
    kill_bonus: float = 0.7
    sun_collect_bonus: float = 0.04
    invalid_action_penalty: float = -0.15
    mower_consumed_penalty: float = -2.0
    lose_penalty: float = -10.0
    win_bonus: float = 10.0


@dataclass(frozen=True)
class WaveConfig:
    sky_sun_interval: int
    sky_sun_amount: int
    base_trickle_rate: float
    flag_wave_multiplier: float
    final_wave_multiplier: float
    conehead_ramp: float


LANES = 5
COLS = 9
HOUSE_X = -0.2
INITIAL_SUN = 150
MAX_SUN = 500
EPISODE_STEPS = 900
MAX_LOOSE_SUN = 8

PLANTS = {
    "sunflower": PlantConfig(cost=50, hp=60, cooldown=20, sun_interval=25, sun_amount=25),
    "peashooter": PlantConfig(cost=100, hp=80, cooldown=15, attack_damage=20, attack_interval=5),
    "wallnut": PlantConfig(cost=75, hp=300, cooldown=25),
}

ZOMBIES = {
    "normal": ZombieConfig(hp=120, speed=0.04, dps=14.0, reward_on_kill=1.0),
    "conehead": ZombieConfig(hp=220, speed=0.03, dps=15.0, reward_on_kill=1.5),
}

WAVES = {
    "easy": WaveConfig(
        sky_sun_interval=18,
        sky_sun_amount=25,
        base_trickle_rate=0.10,
        flag_wave_multiplier=2.0,
        final_wave_multiplier=3.0,
        conehead_ramp=0.20,
    ),
    "normal": WaveConfig(
        sky_sun_interval=20,
        sky_sun_amount=25,
        base_trickle_rate=0.13,
        flag_wave_multiplier=2.3,
        final_wave_multiplier=3.3,
        conehead_ramp=0.30,
    ),
    "hard": WaveConfig(
        sky_sun_interval=23,
        sky_sun_amount=25,
        base_trickle_rate=0.16,
        flag_wave_multiplier=2.7,
        final_wave_multiplier=3.8,
        conehead_ramp=0.42,
    ),
}

REWARDS = RewardConfig()

ACTION_MEANINGS = [
    "noop",
    "collect_sun",
    "econ_lane_0",
    "econ_lane_1",
    "econ_lane_2",
    "econ_lane_3",
    "econ_lane_4",
    "defend_lane_0",
    "defend_lane_1",
    "defend_lane_2",
    "defend_lane_3",
    "defend_lane_4",
    "panic_lane_0",
    "panic_lane_1",
    "panic_lane_2",
    "panic_lane_3",
    "panic_lane_4",
]
