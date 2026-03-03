from __future__ import annotations

import numpy as np
import pytest

from pvz_env import PvZEnv


def test_reset_step_shapes_and_dtype():
    env = PvZEnv(seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

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
    for _ in range(150):
        env.step(8)
        assert env.sim.state.sun >= 0
        assert all(v >= 0 for v in env.sim.state.cooldowns.values())


def test_smoke_episode_runs_to_end():
    env = PvZEnv(seed=4)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 1200:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
    assert done
    assert steps > 1
