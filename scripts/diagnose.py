"""Phase 5 diagnosis for a stalled run: reward breakdown and action histograms per episode.

    python scripts/diagnose.py runs/<run>
Reads the per-episode parquet files upstream writes to <run>/laps/ and writes <run>/diagnose.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    run = Path(sys.argv[1])
    files = sorted((run / "laps").glob("*_states.parquet"))
    if not files:
        sys.exit(f"no episode parquet files in {run / 'laps'}")
    cols = ["reward", "r_progress", "r_lat", "r_dsteer", "lateral_error", "progress_m", "speed",
            "actions_0", "actions_1", "actions_2", "current_action_abs_0", "current_action_abs_1",
            "current_action_abs_2", "out_of_track", "terminated"]
    eps = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=None)
        df = df[[c for c in cols if c in df.columns]].copy()
        df["episode"] = i
        eps.append(df)
    all_df = pd.concat(eps, ignore_index=True)

    per_ep = all_df.groupby("episode").agg(
        steps=("reward", "size"), ret=("reward", "sum"), r_progress=("r_progress", "sum"),
        r_lat=("r_lat", "sum"), r_dsteer=("r_dsteer", "sum"), progress_m=("progress_m", "max"),
        lat_abs_mean=("lateral_error", lambda x: x.abs().mean()), speed_mean=("speed", "mean"),
        oot_end=("out_of_track", "last"))
    pd.set_option("display.width", 200)
    print(per_ep.tail(30).round(2).to_string())
    print("\nlast 20% of episodes, mean:\n", per_ep.tail(max(1, len(per_ep) // 5)).mean().round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    per_ep[["r_progress", "r_lat", "r_dsteer"]].plot(ax=axes[0, 0], title="reward components per episode")
    per_ep["progress_m"].plot(ax=axes[0, 1], title="progress per episode [m]")
    per_ep["lat_abs_mean"].plot(ax=axes[0, 2], title="mean |lateral| per episode [m]")
    recent = all_df[all_df.episode >= all_df.episode.max() - max(1, len(per_ep) // 5)]
    for ax, i, name in zip(axes[1], range(3), ["steer", "pedal", "brake"]):
        ax.hist(recent[f"actions_{i}"], bins=50, alpha=0.6, label="raw action")
        if f"current_action_abs_{i}" in recent:
            ax.hist(recent[f"current_action_abs_{i}"], bins=50, alpha=0.6, label="applied (abs)")
        ax.set_title(f"{name} (recent episodes)")
        ax.legend()
    fig.tight_layout()
    out = run / "diagnose.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
