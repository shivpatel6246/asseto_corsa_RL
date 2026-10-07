"""Config = upstream config.yml <- configs/indy.yml <- CLI dotlist overrides."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .paths import CONFIGS, REPO, UPSTREAM_CONFIG_YML


def load_config(path=None, overrides=()):
    base = OmegaConf.load(UPSTREAM_CONFIG_YML)
    ours = OmegaConf.load(path or CONFIGS / "indy.yml")
    return OmegaConf.merge(base, ours, OmegaConf.from_dotlist(list(overrides)))


def baseline_config(cfg):
    """Copy of cfg with the baseline's env overrides applied (absolute actions)."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.AssettoCorsa.use_relative_actions = bool(cfg.baseline.use_relative_actions)
    return cfg


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p
