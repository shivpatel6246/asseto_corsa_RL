"""Phase 2 smoke test 3: drive off track on purpose, confirm the episode terminates, then confirm
the CSP reset puts the car back at the spawn at rest.
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


def main():
    cfg = baseline_config(load_config(None, sys.argv[1:]))
    env = make_env(cfg, RUNS / "smoke_reset")
    hz = env.ctrl_rate
    problems = []

    env.reset()
    spawn = (env.state["world_position_x"], env.state["world_position_y"])
    print(f"spawn at ({spawn[0]:.1f}, {spawn[1]:.1f})")

    info, done = {}, False
    for i in range(int(30 * hz)):
        steer = 0.0 if i < 3 * hz else 1.0  # build a little speed, then full lock
        env.set_actions(np.array([steer, 0.0, -1.0]))  # 50% throttle
        _, _, done, info = env.step(None)
        if done:
            break
    if not done:
        problems.append("episode did not terminate within 30 s of driving off track")
    else:
        print(f"terminated after {i / hz:.1f} s: reason={info.get('termination_reason')} "
              f"terminated={info.get('terminated')} tyres_out={env.state.get('numberOfTyresOut')} "
              f"lateral={info.get('lateral_error', float('nan')):+.1f} m")
    env.close()

    env.reset()  # sends the CSP reset (send_reset_at_start: True)
    st = env.state
    d = math.hypot(st["world_position_x"] - spawn[0], st["world_position_y"] - spawn[1])
    print(f"after reset: {d:.1f} m from spawn, speed {st['speed'] * 3.6:.1f} km/h")
    if d > 20:
        problems.append(f"car is {d:.0f} m from spawn after reset (is CSP installed?)")
    if st["speed"] * 3.6 > 5:
        problems.append("car still moving after reset")
    env.recover_car()
    env.close()
    print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
