"""One-lap rollout shared by the baseline and eval: per-step trace + success metrics + plots."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd


def run_lap(env, policy, max_time_s, absolute_actions_fn=None, log_every=125):
    """policy(obs, state) -> action passed to env.set_actions. Runs until the lap is completed,
    the episode ends, or max_time_s of env steps. Returns (trace DataFrame, metrics dict)."""
    obs = env.reset()
    dt = 1.0 / env.ctrl_rate
    rows, info, done, steps = [], {}, False, 0
    max_steps = int(max_time_s / dt)
    t0 = time.time()
    while not done and steps < max_steps:
        action, extra = policy(obs, env.state)
        env.set_actions(action)
        obs, reward, done, info = env.step(None)
        st = env.state
        rows.append({
            "t": steps * dt, "progress_m": info["progress_m"], "s": info["centerline_s"],
            "lateral_error": info["lateral_error"], "speed_kmh": st["speed"] * 3.6,
            "steer_cmd": float(env.current_actions[0]), "pedal_cmd": float(env.current_actions[1]),
            "brake_cmd": float(env.current_actions[2]), "out_of_track": int(st.get("out_of_track", 0)),
            "x": st["world_position_x"], "y": st["world_position_y"], "reward": reward, **(extra or {}),
        })
        steps += 1
        if steps % log_every == 0:
            print(f"  t={steps * dt:6.1f}s  progress={info['progress_m']:7.1f} m  lat={info['lateral_error']:+.2f} m  "
                  f"v={st['speed'] * 3.6:5.1f} km/h")
        if info.get("lap_completed"):
            break
    metrics = env.tracker.metrics()
    metrics["termination_reason"] = info.get("termination_reason", "lap_completed" if info.get("lap_completed")
                                             else ("time_limit" if steps >= max_steps else "done"))
    metrics["wall_time_s"] = time.time() - t0
    try:
        env.recover_car()
    except Exception:
        pass
    env.close()
    return pd.DataFrame(rows), metrics


def save_outputs(out_dir, trace, metrics, title, compare=None):
    """trace.csv, metrics.json and a lateral-error/speed vs progress plot (optionally vs a baseline)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(out_dir / "trace.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: (v.item() if isinstance(v, np.generic) else v) for k, v in metrics.items()}, f, indent=2)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    series = [(title, trace)] + ([compare] if compare is not None else [])
    for name, tr in series:
        a1.plot(tr.progress_m, tr.lateral_error, label=name, lw=0.9)
        a2.plot(tr.progress_m, tr.speed_kmh, label=name, lw=0.9)
    for y in (-3, 3):
        a1.axhline(y, color="red", ls="--", lw=0.7)
    a1.axhline(0, color="0.5", lw=0.5)
    a1.set_ylabel("lateral error [m]")
    a2.set_ylabel("speed [km/h]")
    a2.set_xlabel("progress along centerline [m]")
    a1.legend()
    a1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "lateral_speed_vs_progress.png", dpi=120)
    plt.close(fig)
