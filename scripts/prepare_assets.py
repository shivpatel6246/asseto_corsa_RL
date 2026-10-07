"""Phase 2/3 assets: occupancy grid, centerline, and the AssettoCorsaConfigs overlay.

    python scripts/prepare_assets.py               # download grid if missing, build everything
    python scripts/prepare_assets.py --no-download # fail instead of downloading the grid

Checks that upstream ships what we need for indianapolis_sp + ks_mazda_miata, and stops if not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrl_indy.assets import (build_and_check_centerline, build_overlay, ensure_grid,  # noqa: E402
                              upstream_track_entry)
from acrl_indy.config import load_config  # noqa: E402
from acrl_indy.paths import UPSTREAM_TRACKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    entry = upstream_track_entry(cfg.AssettoCorsa.track)
    for key in ("track_file", "ref_lap_file"):
        p = UPSTREAM_TRACKS / entry[key]
        if not p.exists():
            sys.exit(f"Missing upstream asset {p}. Generate it with upstream generate_track.ipynb (AC running).")
        print(f"{key}: {p.name} ok")

    if args.no_download and not (UPSTREAM_TRACKS / entry["track_grid_file"]).exists():
        sys.exit(f"Missing {entry['track_grid_file']} and --no-download given.")
    ensure_grid(entry)
    build_and_check_centerline(cfg)
    build_overlay(cfg)
    print("\nAssets ready. Next: python scripts/plot_track.py")


if __name__ == "__main__":
    main()
