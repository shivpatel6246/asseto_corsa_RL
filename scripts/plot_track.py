"""Phase 3: plot track bounds, AC's racing line and our centerline; sanity-check the centerline.

    python scripts/plot_track.py     # -> runs/track_indianapolis_sp.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acrl_indy.assets import upstream_track_entry  # noqa: E402
from acrl_indy.centerline import check_inside_bounds, load_reference_csv, load_track_bounds  # noqa: E402
from acrl_indy.config import load_config, repo_path  # noqa: E402
from acrl_indy.paths import RUNS, UPSTREAM_TRACKS  # noqa: E402


def main():
    cfg = load_config()
    track = cfg.AssettoCorsa.track
    entry = upstream_track_entry(track)
    left, right, _ = load_track_bounds(UPSTREAM_TRACKS / entry["track_file"])
    racing = load_reference_csv(UPSTREAM_TRACKS / entry["ref_lap_file"])
    c = load_reference_csv(repo_path(cfg.centerline.file))
    chk = check_inside_bounds(c, left, right)
    margin = np.minimum(chk["d_left"], chk["d_right"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [2, 1]})
    ax.plot(left[:, 0], left[:, 1], color="0.3", lw=0.8, label="left bound")
    ax.plot(right[:, 0], right[:, 1], color="0.6", lw=0.8, label="right bound")
    ax.plot(racing[:, 0], racing[:, 1], color="tab:orange", lw=0.8, label="AC racing line")
    ax.plot(c[:, 0], c[:, 1], color="tab:blue", lw=1.0, label="centerline (ours)")
    bad = ~chk["inside"]
    if bad.any():
        ax.scatter(c[bad, 0], c[bad, 1], color="red", s=12, label="outside bounds", zorder=5)
    ax.scatter(*c[0], color="green", s=50, zorder=6, label="start (s=0)")
    ax.annotate("", xy=c[15], xytext=c[0], arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.set_title(f"{track}: bounds, AC racing line, centerline")

    s = np.arange(len(c)) * float(cfg.centerline.spacing_m)
    ax2.plot(s, chk["d_left"], label="dist to left bound")
    ax2.plot(s, chk["d_right"], label="dist to right bound")
    ax2.set_xlabel("centerline distance [m]")
    ax2.set_ylabel("m")
    ax2.legend()
    ax2.set_title("centering check (curves should overlap)")

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"track_{track}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"inside bounds: {chk['inside'].mean() * 100:.1f}% of points; min margin {margin.min():.2f} m; "
          f"max off-center {np.max(np.abs(chk['d_left'] - chk['d_right'])) / 2:.2f} m")
    sys.exit(0 if chk["inside"].all() else 1)


if __name__ == "__main__":
    main()
