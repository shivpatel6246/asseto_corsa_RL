"""Offline tests (no AC needed): python -m pytest tests"""
import ctypes
import math

import numpy as np
import pytest

from ac_rl.shared_memory import SPageFileGraphic, SPageFilePhysics, SPageFileStatic
from ac_rl.track import ReferenceLine
from tests.fake_ac import R, make_env


def test_struct_layout_sanity():
    # Offsets that are easy to verify by hand against the AC SDK header.
    assert SPageFilePhysics.velocity.offset == 32
    assert SPageFileGraphic.completedLaps.offset == 132
    assert SPageFileStatic.numberOfSessions.offset == 60
    assert ctypes.sizeof(SPageFilePhysics) % 4 == 0


def test_reference_line_features():
    a = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    line = ReferenceLine(np.stack([R * np.cos(a), R * np.sin(a)], axis=1))
    assert line.length == pytest.approx(2 * math.pi * R, rel=1e-3)

    # 3 m outside the circle at angle 0, moving along the CCW tangent (+z).
    lat, herr, turn = line.features((R + 3, 0.0), (0.0, 20.0), [10, 50])
    assert abs(lat) == pytest.approx(3.0, abs=0.2)
    assert abs(herr) < 0.05
    # CCW circle turns left by d / R radians.
    assert turn[0] == pytest.approx(10 / R, abs=0.03)
    assert turn[1] == pytest.approx(50 / R, abs=0.03)

    # Inside vs outside give opposite lateral signs.
    lat_in, _, _ = line.features((R - 3, 0.0), (0.0, 20.0), [10])
    assert np.sign(lat_in) == -np.sign(lat)


def test_env_follows_circle(tmp_path):
    env, sim = make_env(tmp_path=tmp_path)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert sim.resets == 1

    laps, total = [], 0.0
    for _ in range(env.max_steps):
        # Hand-written controller: hold ~15 m/s, steer for curvature 1/R plus lateral correction.
        speed = sim.v
        r = math.hypot(sim.x, sim.z)
        radial_v = (sim.x * math.cos(sim.psi) + sim.z * math.sin(sim.psi)) * speed / r
        steer = (1 / R) / 0.05 + 0.05 * (r - R) + 0.1 * radial_v
        pedal = 0.5 if speed < 15 else 0.0
        obs, reward, term, trunc, info = env.step(np.array([steer, pedal]))
        total += reward
        assert env.observation_space.contains(obs)
        if "lap_time_ms" in info:
            laps.append(info)
        assert not term, info
        if trunc:
            break
    assert trunc
    assert total > 0
    assert info["progress_m"] > 2 * math.pi * R  # more than one lap
    assert laps and laps[-1]["lap_valid"]


def test_env_terminates_offtrack(tmp_path):
    env, sim = make_env(tmp_path=tmp_path)
    env.reset()
    for _ in range(400):
        _, r, term, trunc, info = env.step(np.array([-1.0, 1.0]))  # full right lock, full throttle
        if term:
            break
    assert term and info["termination_reason"] == "offtrack"


def test_env_terminates_stuck(tmp_path):
    env, sim = make_env(tmp_path=tmp_path)
    env.reset()
    for _ in range(400):
        _, r, term, trunc, info = env.step(np.array([0.0, -1.0]))
        if term:
            break
    assert term and info["termination_reason"] == "stuck"


def test_env_without_line(tmp_path):
    env, _ = make_env(with_line=False, tmp_path=tmp_path)
    obs, _ = env.reset()
    assert obs.shape == (22,)


def test_sac_smoke(tmp_path):
    from stable_baselines3 import SAC

    env, _ = make_env(tmp_path=tmp_path)
    model = SAC("MlpPolicy", env, train_freq=(1, "episode"), gradient_steps=-1, learning_starts=100,
                buffer_size=5000, batch_size=64, policy_kwargs={"net_arch": [64, 64]})
    model.learn(600)
    model.save(tmp_path / "m.zip")
    model.save_replay_buffer(tmp_path / "rb.pkl")
