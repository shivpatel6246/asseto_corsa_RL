"""Track/car assets: fetch the occupancy grid, build the centerline, assemble the configs overlay.

The overlay (generated/AssettoCorsaConfigs) is a copy of what upstream needs for this track+car,
with tracks/config.yaml pointing `ref_lap_file` at our centerline. Upstream's env takes
`ac_configs_path`, so nothing inside the submodule is edited.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from .centerline import arc_length, build_centerline, check_inside_bounds, load_track_bounds, save_reference_csv
from .config import repo_path
from .paths import AC_CONFIGS_OVERLAY, UPSTREAM_CARS, UPSTREAM_TRACKS

HF_BASE = "https://huggingface.co/datasets/dasgringuen/assettoCorsaGym/resolve/main/AssettoCorsaConfigs/tracks/"


def upstream_track_entry(track):
    cfg = yaml.safe_load(open(UPSTREAM_TRACKS / "config.yaml"))
    if track not in cfg:
        raise KeyError(f"Track {track} not in upstream tracks/config.yaml. Available: {list(cfg)}")
    return cfg[track]


def ensure_grid(track_entry, log=print):
    """The occupancy grid is gitignored upstream; download it from the HF dataset into the upstream
    tracks folder (CLAUDE.md Phase 2). This adds an ignored file, it doesn't modify tracked files."""
    dst = UPSTREAM_TRACKS / track_entry["track_grid_file"]
    if dst.exists():
        log(f"grid present: {dst.name}")
        return dst
    url = HF_BASE + track_entry["track_grid_file"]
    log(f"downloading {url} (~117 MB)...")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    log(f"saved {dst}")
    return dst


def build_and_check_centerline(cfg, log=print):
    track = cfg.AssettoCorsa.track
    entry = upstream_track_entry(track)
    left, right, racing = load_track_bounds(UPSTREAM_TRACKS / entry["track_file"])
    # Align point 0 with the start of upstream's reference line (AC's spline origin) and its direction.
    c = build_centerline(left, right, start_near=racing[0], start_heading_ref=racing[5],
                         spacing=cfg.centerline.spacing_m, smooth_sigma_m=cfg.centerline.smooth_sigma_m)
    chk = check_inside_bounds(c, left, right)
    out = repo_path(cfg.centerline.file)
    save_reference_csv(c, out)
    margin = np.minimum(chk["d_left"], chk["d_right"])
    report = {
        "points": len(c),
        "length_m": float(arc_length(c)[-1]),
        "ac_track_length_m": float(entry["TrackLength"]),
        "inside_fraction": float(chk["inside"].mean()),
        "min_margin_m": float(margin.min()),
        "max_offcenter_m": float(np.max(np.abs(chk["d_left"] - chk["d_right"])) / 2),
    }
    log(f"centerline -> {out}")
    for k, v in report.items():
        log(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    if report["inside_fraction"] < 1.0:
        raise RuntimeError("Centerline leaves the track bounds somewhere; see scripts/plot_track.py")
    return out, report


def build_overlay(cfg, log=print):
    track, car = cfg.AssettoCorsa.track, cfg.AssettoCorsa.car
    entry = dict(upstream_track_entry(track))
    if not (UPSTREAM_CARS / car / "steer_map.csv").exists():
        supported = sorted(p.name for p in UPSTREAM_CARS.iterdir() if p.is_dir())
        raise FileNotFoundError(f"No upstream car config for {car}. Supported cars: {supported}")

    tracks = AC_CONFIGS_OVERLAY / "tracks"
    cars = AC_CONFIGS_OVERLAY / "cars"
    tracks.mkdir(parents=True, exist_ok=True)
    for name in (entry["track_file"], entry["ref_lap_file"]):
        shutil.copy2(UPSTREAM_TRACKS / name, tracks / name)
    grid = tracks / entry["track_grid_file"]
    if not grid.exists() or grid.stat().st_size != (UPSTREAM_TRACKS / entry["track_grid_file"]).stat().st_size:
        shutil.copy2(UPSTREAM_TRACKS / entry["track_grid_file"], grid)

    centerline = repo_path(cfg.centerline.file)
    shutil.copy2(centerline, tracks / centerline.name)
    entry["upstream_ref_lap_file"] = entry["ref_lap_file"]
    entry["ref_lap_file"] = centerline.name
    with open(tracks / "config.yaml", "w") as f:
        yaml.safe_dump({track: entry}, f, sort_keys=False)

    if (cars / car).exists():
        shutil.rmtree(cars / car)
    shutil.copytree(UPSTREAM_CARS / car, cars / car)
    log(f"overlay -> {AC_CONFIGS_OVERLAY} (ref_lap_file = {centerline.name})")
    return AC_CONFIGS_OVERLAY
