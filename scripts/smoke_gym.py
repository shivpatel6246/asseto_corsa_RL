"""Phase 2 smoke test 2 (from upstream test_gym.ipynb): reset, small constant throttle with zero
steer for ~5 s, confirm the car moves and observations change. Then a short steering probe that
measures which way +steer turns the car in upstream's frame and prints `baseline.steer_sign`.

Uses absolute actions (the baseline's env settings) so the commands mean what they say.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from acrl_indy.config import baseline_config, load_config  # noqa: E402
from acrl_indy.envs import make_env  # noqa: E402
from acrl_indy.paths import RUNS  # noqa: E402

THROTTLE = 0.2  # fraction of full throttle


def heading(st):
    return math.atan2(st["velocity_x"], st["velocity_z"])  # upstream frame: x = AC z, y = AC x


def main():
    cfg = baseline_config(load_config(None, sys.argv[1:]))
    env = make_env(cfg, RUNS / "smoke_gym")
    hz = env.ctrl_rate
    problems = []

    obs0 = env.reset()
    st0 = dict(env.state)
    obs = obs0
    for _ in range(int(5 * hz)):
        env.set_actions(np.array([0.0, 2 * THROTTLE - 1, -1.0]))
        obs, _, done, info = env.step(None)
        if done:
            problems.append(f"episode ended during straight-line test: {info}")
            break
    st = env.state
    moved = math.hypot(st["world_position_x"] - st0["world_position_x"], st["world_position_y"] - st0["world_position_y"])
    dobs = float(np.linalg.norm(np.asarray(obs) - np.asarray(obs0)))
    print(f"after 5 s: speed {st['speed'] * 3.6:.1f} km/h, moved {moved:.1f} m, |obs - obs0| = {dobs:.3f}, "
          f"lateral {info.get('lateral_error', float('nan')):+.2f} m, obs dim {len(obs)}")
    if moved < 5:
        problems.append("car moved < 5 m: check vJoy preset is active and AC has focus")
    if dobs < 1e-3:
        problems.append("observations did not change")

    # Steering probe: +0.1 steer for 1 s while rolling; measure heading change from velocity.
    sign = None
    if st["speed"] > 4 and not problems:
        h0, yaw0 = heading(st), st["yaw"]
        for _ in range(int(1 * hz)):
            env.set_actions(np.array([0.1, 2 * THROTTLE - 1, -1.0]))
            _, _, done, _ = env.step(None)
            if done:
                break
        st = env.state
        dh = (heading(st) - h0 + math.pi) % (2 * math.pi) - math.pi
        dyaw = (st["yaw"] - yaw0 + math.pi) % (2 * math.pi) - math.pi
        sign = 1 if dh > 0 else -1
        print(f"steer +0.1 for 1 s: velocity-heading change {math.degrees(dh):+.1f} deg, "
              f"plugin yaw change {math.degrees(dyaw):+.1f} deg")
        print(f"==> set baseline.steer_sign: {sign} in configs/indy.yml "
              f"(currently {cfg.baseline.steer_sign}); record it in docs/SETUP_LOG.md")
        if abs(math.degrees(dh)) < 0.5:
            problems.append("heading barely changed during steer probe; inconclusive")

    for _ in range(int(2 * hz)):  # stop the car
        env.set_actions(np.array([0.0, -1.0, 1.0]))
        env.step(None)
    env.recover_car()
    env.close()
    print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
