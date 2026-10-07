"""Repo paths and upstream import setup."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party" / "assetto_corsa_gym"
UPSTREAM_PKG = UPSTREAM / "assetto_corsa_gym"  # contains AssettoCorsaEnv/, AssettoCorsaConfigs/, AssettoCorsaPlugin/
UPSTREAM_CONFIGS = UPSTREAM_PKG / "AssettoCorsaConfigs"
UPSTREAM_TRACKS = UPSTREAM_CONFIGS / "tracks"
UPSTREAM_CARS = UPSTREAM_CONFIGS / "cars"
UPSTREAM_CONFIG_YML = UPSTREAM / "config.yml"

CONFIGS = REPO / "configs"
DATA = REPO / "data"
RUNS = REPO / "runs"
DOCS = REPO / "docs"
# Our copy of AssettoCorsaConfigs with the centerline swapped in as the reference line.
# Built by scripts/prepare_assets.py; gitignored because it holds the 117 MB occupancy grid.
AC_CONFIGS_OVERLAY = REPO / "generated" / "AssettoCorsaConfigs"


def add_upstream_to_path():
    """Same paths upstream's train.py adds, but absolute so cwd doesn't matter."""
    for p in (UPSTREAM, UPSTREAM_PKG, UPSTREAM / "algorithm" / "discor"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
