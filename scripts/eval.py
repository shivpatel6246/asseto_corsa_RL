"""Phase 5 eval: load a SAC checkpoint, drive one lap deterministically, report success metrics,
and plot lateral error vs progress next to the baseline.

    python scripts/eval.py runs/<run>/model/best_reward
    python scripts/eval.py runs/<run>/model/checkpoints/step_00100000 --baseline runs/baseline_<time>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from acrl_indy.config import load_config  # noqa: E402
from acrl_indy.envs import make_env  # noqa: E402
from acrl_indy.metrics import format_metrics  # noqa: E402
from acrl_indy.paths import RUNS, add_upstream_to_path  # noqa: E402
from acrl_indy.rollout import run_lap, save_outputs  # noqa: E402
from acrl_indy.train import pick_device  # noqa: E402


def latest_baseline():
    runs = sorted(RUNS.glob("baseline_*/trace.csv"))
    return runs[-1].parent if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="dir containing policy_net.pth")
    ap.add_argument("--config", default=None, help="defaults to the run's saved config.yml")
    ap.add_argument("--baseline", type=Path, default=None, help="baseline run dir (default: latest)")
    ap.add_argument("overrides", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    run_dir = next((p for p in [args.model_dir, *args.model_dir.parents] if (p / "config.yml").exists()), None)
    cfg = load_config(args.config or (run_dir / "config.yml" if run_dir else None), args.overrides)

    add_upstream_to_path()
    from discor.algorithm.eval import EvalAlgorithm

    out = args.model_dir / "eval"
    env = make_env(cfg, out / "env")
    algo = EvalAlgorithm(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                         device=pick_device(cfg.device), policy_hidden_units=list(cfg.SAC.policy_hidden_units))
    algo.load_models(str(args.model_dir))

    def policy(obs, state):
        action, _ = algo.exploit(obs)
        return action, None

    trace, m = run_lap(env, policy, cfg.eval.max_time_s)
    base_dir = args.baseline or latest_baseline()
    compare = ("baseline", pd.read_csv(base_dir / "trace.csv")) if base_dir else None
    save_outputs(out, trace, m, "SAC", compare=compare)
    print(f"\nResult ({m['termination_reason']}):\n{format_metrics(m)}\nSaved {out}"
          + (f" (compared with {base_dir})" if base_dir else " (no baseline found to compare)"))
    sys.exit(0 if m["success"] else 1)


if __name__ == "__main__":
    main()
