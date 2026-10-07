"""Drive with a trained policy (deterministic) and print lap times.

    python evaluate.py runs/<run>/checkpoints/sac_200000_steps.zip --episodes 3
"""
from __future__ import annotations

import argparse
import time

import yaml
from stable_baselines3 import SAC

from ac_rl.env import AssettoCorsaEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    env = AssettoCorsaEnv(cfg["env"])
    model = SAC.load(args.checkpoint, device="auto")

    print("Click into the AC window; starting in 5 s...")
    time.sleep(5)
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                total += r
                done = term or trunc
                if "lap_time_ms" in info:
                    print(f"  lap {info['lap_time_ms'] / 1000:.3f}s {'valid' if info['lap_valid'] else 'INVALID'}")
            print(f"episode {ep}: return {total:.1f}, progress {info['progress_m']:.0f} m, "
                  f"end: {info.get('termination_reason', 'time limit')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
