from __future__ import annotations

import numpy as np
import pytest

from pvz_env import PvZEnv
from pvz_env import config
from pvz_env.sim import LooseSun, Zombie


def test_reset_step_shapes_and_dtype():
    env = PvZEnv(seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.action_space.n == 17
    assert env.observation_space.shape == (42,)

    next_obs, reward, term, trunc, step_info = env.step(0)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "snapshot" in step_info


def test_invalid_action_raises():
    env = PvZEnv(seed=2)
    env.reset()
    with pytest.raises(ValueError):
        env.step(env.action_space.n)


def test_no_negative_sun_after_many_steps():
    env = PvZEnv(seed=3)
    env.reset()
    panic_lane0 = 2 + (2 * config.LANES)
    for _ in range(150):
        env.step(panic_lane0)
        assert env.sim.state.sun >= 0
        assert all(v >= 0 for v in env.sim.state.cooldowns.values())


def test_random_episode_runs_full_length_without_crash():
    env = PvZEnv(seed=4, difficulty="easy")
    obs, _ = env.reset(seed=4)
    done = False
    steps = 0
    while not done and steps < config.EPISODE_STEPS + 50:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        done = term or trunc
        steps += 1
    assert done
    assert steps <= config.EPISODE_STEPS


def test_termination_signals_for_loss_and_win_paths():
    loss_env = PvZEnv(seed=10)
    loss_env.reset(seed=10)
    lane = 0
    loss_env.sim.state.mowers[lane] = False
    zcfg = config.ZOMBIES["normal"]
    loss_env.sim.zombies = [Zombie(kind="normal", lane=lane, x=config.HOUSE_X - 0.01, hp=zcfg.hp)]
    _, _, term, trunc, info = loss_env.step(0)
    assert term is True
    assert trunc is False
    assert info["snapshot"]["win"] is False

    win_env = PvZEnv(seed=11)
    win_env.reset(seed=11)
    win_env.sim.state.step_idx = config.EPISODE_STEPS - 1
    win_env.sim.zombies.clear()
    _, _, term, trunc, info = win_env.step(0)
    assert term is True
    assert trunc is False
    assert info["snapshot"]["win"] is True


def test_empty_collect_is_invalid_and_worse_than_noop():
    steps = 8

    collect_env = PvZEnv(seed=21)
    collect_env.reset(seed=21)
    saw_invalid = False
    collect_total = 0.0
    for _ in range(steps):
        _, reward, _, _, info = collect_env.step(1)
        collect_total += reward
        if len(info["snapshot"]["loose_sun"]) == 0 and info["invalid_action"]:
            saw_invalid = True

    noop_env = PvZEnv(seed=21)
    noop_env.reset(seed=21)
    noop_total = 0.0
    for _ in range(steps):
        _, reward, _, _, _ = noop_env.step(0)
        noop_total += reward

    assert saw_invalid
    assert collect_total < noop_total


def test_reward_config_has_empty_collect_penalty():
    assert hasattr(config.REWARDS, "empty_collect_penalty")

    env = PvZEnv(seed=31)
    env.reset(seed=31)
    _, reward, _, _, info = env.step(1)
    assert isinstance(reward, float)
    assert "invalid_action" in info



def test_collect_reward_not_dominant_with_default_config():
    env = PvZEnv(seed=51)
    env.reset(seed=51)
    env.sim.loose_sun = [LooseSun(lane=0, x=1.0, amount=100, ttl=10)]

    _, reward, _, _, info = env.step(1)

    collect_only_gain = reward - config.REWARDS.step_survival
    assert info["sun_collected"] == 100
    assert collect_only_gain <= 0.2


def test_econ_action_can_yield_positive_reward_with_placement_bonus():
    env = PvZEnv(seed=52)
    env.reset(seed=52)
    env.sim.zombies.clear()
    env.sim.loose_sun.clear()
    env.sim.state.sun = max(env.sim.state.sun, config.PLANTS["sunflower"].cost)
    env.sim.state.cooldowns["sunflower"] = 0
    before_plants = len(env.sim.snapshot()["plants"])

    _, reward, _, _, info = env.step(2)
    after_plants = len(info["snapshot"]["plants"])

    assert after_plants == before_plants + 1
    assert info["placed"] is True
    assert info["placed_kind"] == "sunflower"
    assert reward > 0.0
    assert reward >= config.REWARDS.step_survival + config.REWARDS.place_any_bonus + config.REWARDS.place_sunflower_bonus
