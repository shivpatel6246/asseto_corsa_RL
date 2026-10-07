"""Train SAC on Assetto Corsa.

    python train.py                                   # new run
    python train.py --resume runs/<run>/checkpoints/sac_100000_steps.zip

Gradient updates run at the end of each episode (while the car sits at the start), because the
game keeps running in real time and blocking mid-episode would leave the car uncontrolled.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from ac_rl.env import AssettoCorsaEnv


class LapLogger(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "lap_time_ms" in info:
                t = info["lap_time_ms"] / 1000.0
                valid = info.get("lap_valid", False)
                print(f"[lap] {t:.3f}s {'valid' if valid else 'INVALID'} at step {self.num_timesteps}")
                self.logger.record("laps/time_s", t)
                self.logger.record("laps/valid", float(valid))
            if "termination_reason" in info:
                self.logger.record(f"episode_end/{info['termination_reason']}", 1.0)
        return True


def replay_buffer_for(checkpoint: Path) -> Path:
    # CheckpointCallback names: sac_<N>_steps.zip <-> sac_replay_buffer_<N>_steps.pkl
    steps = checkpoint.stem.split("_")[-2]
    return checkpoint.with_name(f"sac_replay_buffer_{steps}_steps.pkl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", type=Path, help="checkpoint .zip to continue from")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sac_cfg, train_cfg = dict(cfg["sac"]), cfg["train"]

    env = AssettoCorsaEnv(cfg["env"])
    print(f"Car: {env.car_name}  Track: {env.track_name} {env.layout}  length: {env.track_length:.0f} m")

    if args.resume:
        run_dir = args.resume.parent.parent
        env = Monitor(env, str(run_dir / f"monitor_{int(time.time())}"))
        model = SAC.load(args.resume, env=env, device=sac_cfg["device"])
        buf = replay_buffer_for(args.resume)
        if buf.exists():
            model.load_replay_buffer(buf)
            print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")
        else:
            print(f"No replay buffer at {buf}; continuing with an empty buffer")
        reset_num_timesteps = False
    else:
        run_dir = Path(train_cfg["run_dir"]) / f"sac_{env.track_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        env = Monitor(env, str(run_dir / "monitor"))
        net_arch = sac_cfg.pop("net_arch")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": net_arch},
            train_freq=(1, "episode"),
            gradient_steps=-1,  # as many updates as env steps collected
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
            **sac_cfg,
        )
        reset_num_timesteps = True

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=train_cfg["checkpoint_every"],
            save_path=str(run_dir / "checkpoints"),
            name_prefix="sac",
            save_replay_buffer=True,
        ),
        LapLogger(),
    ])

    print(f"Run dir: {run_dir}\nClick into the AC window now; training starts in 5 s...")
    time.sleep(5)
    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="sac",
        )
    except KeyboardInterrupt:
        print("Interrupted; saving...")
    finally:
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save(ckpt / f"sac_{model.num_timesteps}_steps.zip")
        model.save_replay_buffer(ckpt / f"sac_replay_buffer_{model.num_timesteps}_steps.pkl")
        print(f"Saved {ckpt / f'sac_{model.num_timesteps}_steps.zip'}")
        env.close()


if __name__ == "__main__":
    main()
