"""Offline tests: real upstream env + our wrapper/baseline/scripts/SAC, against tests/fake_client.py.

Needs the `acrl` env (Python 3.9 + upstream requirements) and `python scripts/prepare_assets.py`
(the overlay and occupancy grid). Run: python -m pytest tests -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from acrl_indy.centerline import Centerline  # noqa: E402
from acrl_indy.config import baseline_config, load_config, repo_path  # noqa: E402
from acrl_indy.paths import AC_CONFIGS_OVERLAY, add_upstream_to_path  # noqa: E402
from tests import fake_client  # noqa: E402

pytestmark = pytest.mark.skipif(not (AC_CONFIGS_OVERLAY / "tracks" / "config.yaml").exists(),
                                reason="run scripts/prepare_assets.py first")


@pytest.fixture
def cl():
    return Centerline.from_csv(repo_path(load_config().centerline.file))


@pytest.fixture(autouse=True)
def fake_ac(cl, monkeypatch):
    fake_client.install_import_stubs()
    add_upstream_to_path()
    import AssettoCorsaEnv.ac_env as ac_env

    fake_client.SIM = fake_client.Sim(cl)
    monkeypatch.setattr(ac_env, "Client", fake_client.FakeClient)
    return fake_client.SIM


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_script(name, monkeypatch, tmp_path, argv=()):
    mod = _load_script(name)
    monkeypatch.setattr(mod, "RUNS", tmp_path, raising=False)
    monkeypatch.setattr(sys, "argv", [name, *argv])
    with pytest.raises(SystemExit) as e:
        mod.main()
    return e.value.code


def test_centerline_projection(cl):
    s, lat, i = cl.project(cl.point_at(1234.5))
    assert s == pytest.approx(1234.5, abs=0.05) and abs(lat) < 0.05
    # 2 m to the left of the line (CCW of heading) gives lateral +2.
    h = cl.heading_at(500.0)
    p = cl.point_at(500.0) + 2.0 * np.array([-np.sin(h), np.cos(h)])
    assert cl.project(p)[1] == pytest.approx(2.0, abs=0.05)
    assert Centerline.progress_delta(1.0, cl.length - 1.0, cl.length) == pytest.approx(2.0)


def test_upstream_env_uses_centerline(tmp_path):
    from acrl_indy.envs import make_env

    env = make_env(load_config(), tmp_path)
    assert Path(env.ref_lap_file).name == "indianapolis_sp-centerline.csv"
    obs = env.reset()
    assert obs.shape == env.observation_space.shape
    # upstream's own gap is now measured to the centerline; it should match our lateral error.
    env.set_actions(np.array([0.0, 0.5, -1.0]))
    _, _, _, info = env.step(None)
    assert abs(abs(env.state["gap"]) - abs(info["lateral_error"])) < 0.2


def test_smoke_gym_detects_steer_sign(monkeypatch, tmp_path, capsys):
    assert _run_script("smoke_gym", monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert f"set baseline.steer_sign: {fake_client.STEER_SIGN_TRUE}" in out


def test_smoke_reset(monkeypatch, tmp_path, fake_ac):
    assert _run_script("smoke_reset", monkeypatch, tmp_path) == 0
    assert fake_ac.resets >= 2


def test_baseline_completes_lap_with_correct_sign(monkeypatch, tmp_path, capsys):
    code = _run_script("run_baseline", monkeypatch, tmp_path, [f"baseline.steer_sign={fake_client.STEER_SIGN_TRUE}"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "OVERALL       : PASS" in out
    assert list(tmp_path.glob("baseline_*/lateral_speed_vs_progress.png"))


def test_baseline_wrong_sign_fails(monkeypatch, tmp_path, capsys):
    code = _run_script("run_baseline", monkeypatch, tmp_path, [f"baseline.steer_sign={-fake_client.STEER_SIGN_TRUE}"])
    assert code == 1


def test_reward_and_lateral_termination(tmp_path):
    from acrl_indy.envs import make_env

    cfg = baseline_config(load_config())
    env = make_env(cfg, tmp_path)
    env.reset()
    rewards, info, done = [], {}, False
    for _ in range(25 * 20):
        env.set_actions(np.array([0.05, 0.0, -1.0]))  # constant small steer: drifts off the centerline
        _, r, done, info = env.step(None)
        rewards.append(r)
        if done:
            break
    assert done and info["termination_reason"] in ("lateral_error", "out_of_track")
    assert info["termination_reason"] == "lateral_error"  # 4 m limit is inside the ~6.5 m half-width
    assert env.state["reward"] == pytest.approx(rewards[-1])
    stats = env.close()
    assert "lat_rms" in stats and "progress_m" in stats and stats["ep_reward"] == pytest.approx(sum(rewards[-len(env.states):]), rel=1e-6)


def test_upstream_sac_agent_runs(tmp_path):
    import torch

    from acrl_indy.envs import make_env
    from discor.agent import Agent
    from discor.algorithm import SAC
    from omegaconf import OmegaConf

    cfg = load_config(None, ["AssettoCorsa.max_episode_py_time=6", "Agent.start_steps=60",
                             "Agent.batch_size=32", "Agent.memory_size=2000", "Agent.checkpoint_freq=0"])
    env = make_env(cfg, tmp_path)
    algo = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
               device=torch.device("cpu"), seed=0, **OmegaConf.to_container(cfg.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir=str(tmp_path) + "/", device=torch.device("cpu"),
                  seed=0, **cfg.Agent, wandb_logger=None)
    for _ in range(2):
        agent.train_episode()
    assert agent._steps > 60
    summary = (tmp_path / "summary.csv").read_text()
    assert "lat_rms" in summary and "progress_m" in summary
    agent.save(str(tmp_path / "model" / "final"), save_buffer=True)
    assert (tmp_path / "model" / "final" / "policy_net.pth").exists()
