"""Phase 4: pure-pursuit + PI speed baseline, one lap from the hotlap spawn.

    python scripts/run_baseline.py
    python scripts/run_baseline.py baseline.target_speed_kmh=80 baseline.steer_sign=-1

Writes runs/baseline_<time>/{trace.csv, metrics.json, lateral_speed_vs_progress.png}.
Pass criteria = the project success metrics (lap, lateral RMS < 1 m, max < 3 m, no out-of-track).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrl_indy.config import baseline_config, load_config  # noqa: E402
from acrl_indy.envs import make_env, max_steer_deg  # noqa: E402
from acrl_indy.metrics import format_metrics  # noqa: E402
from acrl_indy.paths import RUNS  # noqa: E402
from acrl_indy.pure_pursuit import PurePursuitController  # noqa: E402
from acrl_indy.rollout import run_lap, save_outputs  # noqa: E402


def main():
    cfg = baseline_config(load_config(None, sys.argv[1:]))
    out = RUNS / f"baseline_{datetime.now():%Y%m%d_%H%M%S}"
    env = make_env(cfg, out / "env")
    assert not env.use_relative_actions, "baseline expects absolute actions"
    ctrl = PurePursuitController(env.cl, cfg.baseline, max_steer_deg(cfg))
    dt = 1.0 / env.ctrl_rate

    def policy(obs, state):
        return ctrl.act(state, dt)

    print(f"Baseline: target {cfg.baseline.target_speed_kmh} km/h, steer_sign {cfg.baseline.steer_sign}")
    trace, m = run_lap(env, policy, cfg.eval.max_time_s)
    save_outputs(out, trace, m, "pure pursuit baseline")
    print(f"\nResult ({m['termination_reason']}):\n{format_metrics(m)}\nSaved {out}")
    sys.exit(0 if m["success"] else 1)


if __name__ == "__main__":
    main()
