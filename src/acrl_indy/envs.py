"""Build upstream's AssettoCorsaEnv against our AssettoCorsaConfigs overlay (centerline as reference)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .centerline import Centerline
from .config import repo_path
from .paths import AC_CONFIGS_OVERLAY, add_upstream_to_path
from .reward import CenterlineRewardWrapper


def _check_overlay(cfg):
    track_cfg = AC_CONFIGS_OVERLAY / "tracks" / "config.yaml"
    car_dir = AC_CONFIGS_OVERLAY / "cars" / cfg.AssettoCorsa.car
    if not track_cfg.exists() or not car_dir.exists():
        raise FileNotFoundError(
            f"{AC_CONFIGS_OVERLAY} is missing or incomplete. Run: python scripts/prepare_assets.py")


def make_upstream_env(cfg, work_dir):
    add_upstream_to_path()
    import AssettoCorsaEnv.assettoCorsa as assettoCorsa

    _check_overlay(cfg)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return assettoCorsa.make_ac_env(cfg=cfg, work_dir=str(work_dir) + os.sep,
                                    ac_configs_path=str(AC_CONFIGS_OVERLAY))


def make_env(cfg, work_dir, summary_writer=None):
    env = make_upstream_env(cfg, work_dir)
    cl = Centerline.from_csv(repo_path(cfg.centerline.file))
    return CenterlineRewardWrapper(env, cl, cfg.reward, summary_writer=summary_writer)


def max_steer_deg(cfg):
    """Steering-wheel degrees at steer = +1, from the car's steer_map.csv (same read as upstream)."""
    df = pd.read_csv(AC_CONFIGS_OVERLAY / "cars" / cfg.AssettoCorsa.car / "steer_map.csv")
    return float(df.values[1, 1] / df.values[1, 0])
