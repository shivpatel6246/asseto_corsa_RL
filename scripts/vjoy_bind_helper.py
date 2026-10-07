"""Wiggle one vJoy axis so AC's control-binding screen can detect it.

In AC (or Content Manager) Controls, click the steering/throttle/brake bind box, then run e.g.:
    python scripts/vjoy_bind_helper.py steer
    python scripts/vjoy_bind_helper.py test      # sweeps all three to check the bindings
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.controls import VJoyController  # noqa: E402

AXES = ("steer", "throttle", "brake")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    if which not in AXES + ("test",):
        sys.exit(f"usage: {sys.argv[0]} [{'|'.join(AXES)}|test]")
    pad = VJoyController(1)
    pad.neutral(brake=0.0)
    for i in range(3, 0, -1):
        print(f"starting in {i}... (switch to AC)")
        time.sleep(1)

    targets = AXES if which == "test" else (which,)
    for axis in targets:
        print(f"moving {axis}")
        t0 = time.time()
        while time.time() - t0 < 4:
            x = math.sin((time.time() - t0) * math.pi)
            vals = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
            vals[axis] = x if axis == "steer" else abs(x)
            pad.set(vals["steer"], vals["throttle"], vals["brake"])
            time.sleep(0.02)
        pad.neutral(brake=0.0)
        time.sleep(0.5)
    pad.close()


if __name__ == "__main__":
    main()
