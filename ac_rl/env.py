"""Real-time Gymnasium environment for Assetto Corsa.

State comes from AC shared memory, actions go out through vJoy, resets are a keyboard hotkey.
The game is never paused between steps, so the env paces itself at a fixed control rate.

Action: [steer, pedal] in [-1, 1]. pedal > 0 is throttle, pedal < 0 is brake.
"""
from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .shared_memory import ACStatus, wstr
from .track import ReferenceLine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_OBS_DIM = 22


class RealClock:
    now = staticmethod(time.perf_counter)
    sleep = staticmethod(time.sleep)


def line_path_for(track: str, layout: str) -> Path:
    name = f"{track}__{layout}" if layout else track
    return PROJECT_ROOT / "data" / "lines" / f"{name}.npz"


class AssettoCorsaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, sm=None, pad=None, resetter=None, clock=None):
        super().__init__()
        self.cfg = cfg
        self.clock = clock or RealClock()

        if sm is None:
            from .shared_memory import ACSharedMemory

            sm = ACSharedMemory()
        if pad is None:
            from .controls import VJoyController

            pad = VJoyController(cfg.get("vjoy_device_id", 1))
        if resetter is None:
            from .controls import HotkeyResetter

            resetter = HotkeyResetter(cfg["reset"].get("hotkey") or [])
        self.sm, self.pad, self.resetter = sm, pad, resetter

        static = self.sm.static()
        self.track_name = wstr(static.track)
        self.layout = wstr(static.trackConfiguration)
        self.car_name = wstr(static.carModel)
        self.max_rpm = static.maxRpm if static.maxRpm > 0 else 8000

        self.line = self._load_line(cfg.get("reference_line", "auto"))
        self.lookahead = np.asarray(cfg.get("lookahead_m", []), dtype=np.float64)

        self.track_length = (
            cfg.get("track_length_m")
            or (static.trackSPlineLength if static.trackSPlineLength > 0 else None)
            or (self.line.length if self.line is not None else None)
        )
        if not self.track_length:
            raise RuntimeError("Unknown track length: set env.track_length_m in config.yaml.")

        self.dt = 1.0 / cfg.get("control_hz", 20)
        self.max_steps = int(cfg.get("max_episode_seconds", 180) / self.dt)

        obs_dim = BASE_OBS_DIM + (2 + len(self.lookahead) if self.line is not None else 0)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._next_tick = 0.0

    # ------------------------------------------------------------------ setup helpers
    def _load_line(self, setting):
        if setting is None:
            return None
        path = line_path_for(self.track_name, self.layout) if setting == "auto" else Path(setting)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"No reference line at {path}. Drive one clean lap with `python scripts/record_line.py`, "
                "or set env.reference_line: null to train without track features."
            )
        return ReferenceLine.load(path)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pad.neutral()
        self.resetter.reset()
        phys, gfx = self._wait_for_start()

        self._steps = 0
        self._prev_action = np.array([0.0, -1.0], dtype=np.float32)
        self._prev_pos = gfx.normalizedCarPosition
        self._progress = 0.0
        self._best_progress = 0.0
        self._offtrack_t = 0.0
        self._stuck_t = 0.0
        self._laps = gfx.completedLaps
        self._lap_valid = True
        self._damage0 = float(sum(phys.carDamage))
        self._next_tick = self.clock.now() + self.dt
        return self._obs(phys, gfx), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._apply(a)
        self._sleep_until_tick()
        phys, gfx = self._read_live(a)
        self._steps += 1

        c = self.cfg
        rc, tc = c["reward"], c["termination"]
        elapsed = self._steps * self.dt

        # Progress along the track spline, in metres, handling the start/finish wrap.
        d = gfx.normalizedCarPosition - self._prev_pos
        if d < -0.5:
            d += 1.0
        elif d > 0.5:
            d -= 1.0
        self._prev_pos = gfx.normalizedCarPosition
        progress_m = d * self.track_length
        self._progress += progress_m
        self._best_progress = max(self._best_progress, self._progress)

        offtrack = phys.numberOfTyresOut > tc["max_tyres_out"]
        self._offtrack_t = self._offtrack_t + self.dt if offtrack else 0.0
        if offtrack:
            self._lap_valid = False
        stuck = elapsed > tc["start_grace_s"] and phys.speedKmh < tc["stuck_speed_kmh"]
        self._stuck_t = self._stuck_t + self.dt if stuck else 0.0

        reward = rc["progress_per_m"] * progress_m
        if offtrack:
            reward -= rc["offtrack_per_step"]
        reward -= rc["steer_rate"] * abs(float(a[0] - self._prev_action[0]))

        reason = None
        if self._offtrack_t > tc["offtrack_grace_s"]:
            reason = "offtrack"
        elif self._stuck_t > tc["stuck_grace_s"]:
            reason = "stuck"
        elif self._best_progress - self._progress > tc["max_backwards_m"]:
            reason = "backwards"
        elif tc.get("damage_threshold") and sum(phys.carDamage) - self._damage0 > tc["damage_threshold"]:
            reason = "crash"
        terminated = reason is not None
        if terminated:
            reward -= rc["termination_penalty"]
        truncated = not terminated and self._steps >= self.max_steps

        info = {
            "speed_kmh": phys.speedKmh,
            "progress_m": self._progress,
            "tyres_out": phys.numberOfTyresOut,
        }
        if reason:
            info["termination_reason"] = reason
        if gfx.completedLaps > self._laps:
            self._laps = gfx.completedLaps
            info["lap_time_ms"] = gfx.iLastTime
            info["lap_valid"] = self._lap_valid
            self._lap_valid = True

        self._prev_action = a
        obs = self._obs(phys, gfx)
        if terminated or truncated:
            self.pad.neutral()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.pad.close()
        finally:
            if hasattr(self.sm, "close"):
                self.sm.close()

    # ------------------------------------------------------------------ internals
    def _apply(self, a):
        steer, pedal = float(a[0]), float(a[1])
        self.pad.set(steer, max(pedal, 0.0), max(-pedal, 0.0))

    def _sleep_until_tick(self):
        now = self.clock.now()
        if self._next_tick > now:
            self.clock.sleep(self._next_tick - now)
            self._next_tick += self.dt
        else:
            # Fell behind (e.g. slow step); resync instead of bursting to catch up.
            self._next_tick = now + self.dt

    def _read_live(self, a):
        """Read state; if the game is paused or in a menu, hold neutral and wait."""
        gfx = self.sm.graphics()
        if gfx.status != ACStatus.LIVE:
            self.pad.neutral()
            print("[env] AC not live (paused?). Waiting...")
            while gfx.status != ACStatus.LIVE:
                self.clock.sleep(0.2)
                gfx = self.sm.graphics()
            self._apply(a)
            self.clock.sleep(self.dt)
            self._next_tick = self.clock.now() + self.dt
            gfx = self.sm.graphics()
        return self.sm.physics(), gfx

    def _wait_for_start(self):
        rc = self.cfg["reset"]
        self.clock.sleep(rc.get("settle_s", 2.0))
        deadline = self.clock.now() + rc.get("timeout_s", 20.0)
        last_packet = None
        while self.clock.now() < deadline:
            phys, gfx = self.sm.physics(), self.sm.graphics()
            advancing = last_packet is not None and phys.packetId != last_packet
            last_packet = phys.packetId
            if gfx.status == ACStatus.LIVE and advancing and phys.speedKmh < 3.0:
                return phys, gfx
            self.clock.sleep(0.1)
        raise RuntimeError(
            "Reset timed out: the car never came to rest in a live session. Check that the reset hotkey "
            "in config.yaml matches what is bound in AC and that the AC window has focus."
        )

    def _obs(self, phys, gfx):
        o = [
            phys.speedKmh / 300.0,
            *(v / 80.0 for v in phys.localVelocity),
            *(v / 3.0 for v in phys.localAngularVel),
            *(v / 3.0 for v in phys.accG),
            *(min(v, 20.0) / 5.0 for v in phys.wheelSlip),
            phys.rpms / self.max_rpm,
            phys.gear / 8.0,
            phys.steerAngle,
            phys.gas,
            phys.brake,
            phys.numberOfTyresOut / 4.0,
            *self._prev_action,
        ]
        if self.line is not None:
            pos = (gfx.carCoordinates[0], gfx.carCoordinates[2])
            vel = (phys.velocity[0], phys.velocity[2])
            lateral, heading_err, turn = self.line.features(pos, vel, self.lookahead)
            o += [lateral / 10.0, heading_err / np.pi, *(turn / np.pi)]
        return np.clip(np.asarray(o, dtype=np.float32), -10.0, 10.0)
