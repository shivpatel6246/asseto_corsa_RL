"""Record a reference line by driving one clean lap yourself (keyboard/wheel, normal controls).

Start AC, get on track, run this, then drive across the start/finish line. Recording begins at the
next line crossing and stops at the one after, then saves to data/lines/<track>__<layout>.npz.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.env import line_path_for  # noqa: E402
from ac_rl.shared_memory import ACSharedMemory, wstr  # noqa: E402
from ac_rl.track import ReferenceLine  # noqa: E402


def main():
    sm = ACSharedMemory()
    st = sm.static()
    track, layout = wstr(st.track), wstr(st.trackConfiguration)
    out = line_path_for(track, layout)
    print(f"Track: {track} {layout} -> {out}")

    laps0 = sm.graphics().completedLaps
    print("Cross the start/finish line to begin recording...")
    while sm.graphics().completedLaps == laps0:
        time.sleep(0.02)

    laps1 = laps0 + 1
    pts, offtrack = [], False
    print("Recording. Drive one clean lap (stay on track).")
    while True:
        g, p = sm.graphics(), sm.physics()
        if g.completedLaps > laps1:
            break
        xz = (g.carCoordinates[0], g.carCoordinates[2])
        if not pts or np.hypot(xz[0] - pts[-1][0], xz[1] - pts[-1][1]) > 0.5:
            pts.append(xz)
        if p.numberOfTyresOut > 2:
            offtrack = True
        time.sleep(0.02)

    line = ReferenceLine(np.array(pts))
    print(f"Recorded {len(pts)} points, line length {line.length:.0f} m (AC spline: {st.trackSPlineLength:.0f} m)")
    if offtrack:
        print("WARNING: you went off track during the lap; consider re-recording.")
    line.save(out, track=track, layout=layout)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
