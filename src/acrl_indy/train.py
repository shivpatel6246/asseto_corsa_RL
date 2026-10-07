"""Train upstream SAC (algorithm/discor) on the centerline task.

    python -m acrl_indy.train                         # configs/indy.yml
    python -m acrl_indy.train Agent.num_steps=50000   # dotlist overrides
    python -m acrl_indy.train --load_path runs/<run>/model/checkpoints/step_00100000

Mirrors upstream train.py, with these differences: our config/env factory (centerline reward
wrapper), CPU allowed, outputs under runs/, and a stall check (CLAUDE.md Phase 5): if the best
episode progress is still below `train.stall_min_progress_m` after `train.stall_check_steps`,
training stops so the run can be diagnosed with scripts/diagnose.py.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from omegaconf import OmegaConf

from .config import load_config
from .envs import make_env
from .paths import RUNS, add_upstream_to_path

logger = logging.getLogger(__name__)


def pick_device(name):
    import torch

    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--load_path", default=None, help="model dir to resume from (loads replay_buffer.pkl if present)")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    add_upstream_to_path()
    import common.logging_config as logging_config
    import common.misc as misc
    from discor.agent import Agent
    from discor.algorithm import SAC
    from torch.utils.tensorboard import SummaryWriter

    work_dir = RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_sac"
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir) + os.sep
    logging_config.create_logging(level=logging.DEBUG, file_name=str(work_dir / "log.log"))
    logging.getLogger().setLevel(logging.INFO)
    misc.get_system_info()
    OmegaConf.save(cfg, work_dir / "config.yml")
    logger.info("work_dir: %s", work_dir)

    writer = SummaryWriter(log_dir=str(work_dir / "summary_centerline"))
    env = make_env(cfg, work_dir, summary_writer=writer)
    device = pick_device(cfg.device)
    logger.info("device: %s", device)

    algo = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
               device=device, seed=cfg.seed, **OmegaConf.to_container(cfg.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir=cfg.work_dir, device=device, seed=cfg.seed,
                  **cfg.Agent, wandb_logger=None)

    if args.load_path:
        load_path = os.path.abspath(args.load_path) + os.sep
        agent.load(load_path, load_buffer=os.path.exists(load_path + "replay_buffer.pkl"))

    stall_checked = False
    try:
        while agent._steps <= cfg.Agent.num_steps:
            agent.train_episode()
            if not stall_checked and agent._steps >= cfg.train.stall_check_steps:
                stall_checked = True
                if env.best_progress_m < cfg.train.stall_min_progress_m:
                    logger.warning(
                        "STALL: best episode progress %.0f m after %d steps (< %.0f m). Stopping to diagnose: "
                        "python scripts/diagnose.py %s", env.best_progress_m, agent._steps,
                        cfg.train.stall_min_progress_m, work_dir)
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        agent.save(os.path.join(agent._model_dir, "final"), save_buffer=True)
        writer.close()
        logger.info("Saved %s", os.path.join(agent._model_dir, "final"))


if __name__ == "__main__":
    main()
