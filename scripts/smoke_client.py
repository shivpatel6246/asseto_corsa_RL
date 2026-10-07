"""Phase 2 smoke test 1 (from upstream test_client.ipynb): connect and print telemetry for 10 s.

Needs AC running in the hotlap session with sensors_par enabled. Inputs stay neutral.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrl_indy.config import load_config  # noqa: E402
from acrl_indy.paths import add_upstream_to_path  # noqa: E402


def main():
    cfg = load_config(None, sys.argv[1:])
    add_upstream_to_path()
    import AssettoCorsaEnv.assettoCorsa as assettoCorsa

    client = assettoCorsa.make_client_only(cfg.AssettoCorsa)
    static = client.simulation_management.get_static_info()
    mod = client.simulation_management.get_config()
    print("static info:", {k: static[k] for k in ("TrackFullName", "CarName", "TrackLength") if k in static})
    print("plugin config: ego_sampling_freq =", mod.get("ego_sampling_freq"),
          " vjoy_executed_by_server =", mod.get("vjoy_executed_by_server"))

    problems = []
    if static.get("TrackFullName") != cfg.AssettoCorsa.track:
        problems.append(f"track is {static.get('TrackFullName')}, config expects {cfg.AssettoCorsa.track}")
    if static.get("CarName") != cfg.AssettoCorsa.car:
        problems.append(f"car is {static.get('CarName')}, config expects {cfg.AssettoCorsa.car}")
    if mod.get("ego_sampling_freq") != cfg.AssettoCorsa.ego_sampling_freq:
        problems.append("ego_sampling_freq differs between plugin and config")

    client.reset(send_reset=False)  # connect without moving the car
    hz = cfg.AssettoCorsa.ego_sampling_freq
    t0, n, first, last = time.time(), 0, None, None
    while time.time() - t0 < 10.0:
        client.controls.set_defaults()
        client.respond_to_server()
        st = client.step_sim()
        first = first or dict(st)
        last = st
        if n % hz == 0:
            print(f"t={time.time() - t0:5.1f}s speed={st['speed'] * 3.6:6.1f} km/h "
                  f"pos=({st['world_position_x']:8.1f},{st['world_position_y']:8.1f}) "
                  f"spline={st['NormalizedSplinePosition']:.4f} tyres_out={st['numberOfTyresOut']}")
        n += 1
    client.close()

    rate = n / 10.0
    print(f"\nreceived {n} packets in 10 s ({rate:.1f} Hz, expected ~{hz})")
    if abs(rate - hz) > 0.2 * hz:
        problems.append(f"packet rate {rate:.1f} Hz is far from {hz} Hz (check AC FPS limit = 50)")
    if first and last and first.get("steps") == last.get("steps"):
        problems.append("server step counter never advanced")
    print("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
