"""Centerline reward wrapper around upstream's AssettoCorsaEnv.

Per step:
    r = w_progress * progress_m - w_lat * lateral^2 - w_dsteer * (d steer_cmd)^2
Terminates (on top of upstream's out-of-track and low-speed rules) when |lateral| > threshold.

Upstream's Agent drives the env as: set_actions(a); step(None); ...; close() at the end of every
episode (close() returns episode stats). It also reads env._max_episode_steps and env.states.
So this wrapper forwards every attribute, and writes its reward back into env.state/env.states so
upstream's parquet logs, summary.csv and "best_reward" checkpointing all see our reward.
"""
from __future__ import annotations

import numpy as np

from .centerline import Centerline
from .metrics import EpisodeTracker


class CenterlineRewardWrapper:
    def __init__(self, env, centerline: Centerline, reward_cfg, summary_writer=None):
        self.env = env
        self.cl = centerline
        self.w_progress = float(reward_cfg.w_progress)
        self.w_lat = float(reward_cfg.w_lat)
        self.w_dsteer = float(reward_cfg.w_dsteer)
        self.lat_limit = float(reward_cfg.lateral_termination_m)
        self.term_penalty = float(reward_cfg.get("termination_penalty", 0.0))
        self.writer = summary_writer
        self.tracker = None
        self.best_progress_m = 0.0
        self.best_lap_lat_rms = None
        self.episodes = 0

    # Everything not defined here (set_actions, states, observation_space, _max_episode_steps,
    # seed, recover_car, client, ...) goes to the upstream env.
    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        st = self.env.state
        s, lat, idx = self.cl.project((st["world_position_x"], st["world_position_y"]))
        self._idx, self._s = idx, s
        self._prev_steer = float(self.env.current_actions[0])
        self.tracker = EpisodeTracker(self.cl.length)
        return obs

    def step(self, action=None):
        obs, _, done, info = self.env.step(action)
        st = self.env.state

        s, lat, self._idx = self.cl.project((st["world_position_x"], st["world_position_y"]), hint=self._idx)
        progress = Centerline.progress_delta(s, self._s, self.cl.length)
        self._s = s
        steer = float(self.env.current_actions[0])
        dsteer = steer - self._prev_steer
        self._prev_steer = steer

        r_prog = self.w_progress * progress
        r_lat = -self.w_lat * lat * lat
        r_steer = -self.w_dsteer * dsteer * dsteer
        reward = r_prog + r_lat + r_steer

        if abs(lat) > self.lat_limit and not done:
            done = 1
            info["terminated"] = True
            info["termination_reason"] = "lateral_error"
        if done and info.get("terminated"):
            reward -= self.term_penalty
            info.setdefault("termination_reason", "out_of_track" if st.get("out_of_track") else "upstream")

        self.tracker.add(progress, lat, bool(st.get("out_of_track", 0)), st.get("speed", 0.0))
        info["lateral_error"] = lat
        info["centerline_s"] = s
        info["progress_m"] = self.tracker.progress_m
        info["lap_completed"] = self.tracker.lap_completed

        # Make upstream's own logging see our reward and components.
        rec = {"reward": reward, "r_progress": r_prog, "r_lat": r_lat, "r_dsteer": r_steer,
               "lateral_error": lat, "centerline_s": s, "progress_m": self.tracker.progress_m}
        st.update(rec)
        st["done"] = done
        if self.env.states:
            self.env.states[-1].update(rec)
            self.env.states[-1]["done"] = done
        return obs, reward, done, info

    def close(self):
        stats = self.env.close()
        if self.tracker is None or not self.tracker.steps:
            return stats
        m = self.tracker.metrics()
        self.episodes += 1
        self.best_progress_m = max(self.best_progress_m, m["progress_m"])
        if m["lap_completed"]:
            rms = m["lap_lat_rms"]
            self.best_lap_lat_rms = rms if self.best_lap_lat_rms is None else min(self.best_lap_lat_rms, rms)
        m["best_progress_m"] = self.best_progress_m
        if isinstance(stats, dict):
            stats.update(m)
        if self.writer is not None:
            step = getattr(self.env, "total_steps", self.episodes)
            for k in ("progress_m", "lat_rms", "lat_max", "oot_events", "lap_completed", "speed_mean"):
                self.writer.add_scalar(f"centerline/{k}", float(m[k]), step)
            self.writer.add_scalar("centerline/best_progress_m", self.best_progress_m, step)
            if self.best_lap_lat_rms is not None:
                self.writer.add_scalar("centerline/best_lap_lat_rms", self.best_lap_lat_rms, step)
            self.writer.flush()
        self.tracker = None
        return stats
