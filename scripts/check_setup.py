"""Sanity-check AC shared memory, vJoy and the reference line before training.

Start AC, load into a session on track, then run:  python scripts/check_setup.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.env import line_path_for  # noqa: E402
from ac_rl.shared_memory import ACSharedMemory, ACStatus, wstr  # noqa: E402

ok = True


def check(cond, msg, hint=""):
    global ok
    print(f"[{'OK' if cond else 'FAIL'}] {msg}" + (f"\n       -> {hint}" if not cond and hint else ""))
    ok &= bool(cond)


sm = ACSharedMemory()
st, gfx = sm.static(), sm.graphics()
track, layout = wstr(st.track), wstr(st.trackConfiguration)
print(f"AC {wstr(st.acVersion)} / SM {wstr(st.smVersion)}  car={wstr(st.carModel)}  track={track} {layout}")
print(f"maxRpm={st.maxRpm}  spline length={st.trackSPlineLength:.1f} m")

check(gfx.status == ACStatus.LIVE, "session is live", "drive out of the pits / unpause")

p1 = sm.physics()
time.sleep(0.5)
p2 = sm.physics()
check(p2.packetId != p1.packetId, "physics packets advancing")
check(p2.autoShifterOn == 1, "auto shifter on", "enable automatic gearbox in AC assists; the agent doesn't shift")

print("Drive at > 30 km/h for a few seconds to verify the struct layout...")
deadline = time.time() + 30
while time.time() < deadline:
    p = sm.physics()
    if p.speedKmh > 30:
        lv = math.sqrt(sum(v * v for v in p.localVelocity)) * 3.6
        wv = math.sqrt(sum(v * v for v in p.velocity)) * 3.6
        check(abs(wv - p.speedKmh) < 5, f"world velocity matches speed ({wv:.1f} vs {p.speedKmh:.1f} km/h)")
        check(abs(lv - p.speedKmh) < 5, f"local velocity matches speed ({lv:.1f} vs {p.speedKmh:.1f} km/h)",
              "physics struct offsets are wrong for your AC version")
        break
    time.sleep(0.1)
else:
    print("[SKIP] never exceeded 30 km/h")

try:
    from ac_rl.controls import VJoyController

    VJoyController(1).neutral(brake=0.0)
    check(True, "vJoy device 1 acquired")
except Exception as e:
    check(False, f"vJoy device 1 acquired ({e})", "open 'Configure vJoy' and enable device 1 with X, Y, Z axes")

line = line_path_for(track, layout)
check(line.exists(), f"reference line {line.name}", "run python scripts/record_line.py")

print("\nAll good." if ok else "\nFix the FAIL items above before training.")
