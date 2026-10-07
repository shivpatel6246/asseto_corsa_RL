"""A tiny stand-in for Assetto Corsa: kinematic car on a circular track of radius R, driven CCW.

Implements the same interfaces the env uses (shared memory reads, pad, resetter, clock) so the env
and training loop can be exercised off Windows.
"""
from __future__ import annotations

import math

from ac_rl.shared_memory import ACStatus, SPageFileGraphic, SPageFilePhysics, SPageFileStatic

R = 100.0
HALF_WIDTH = 6.0


def _set_wstr(arr, s):
    for i, ch in enumerate(s):
        arr[i] = ord(ch)


class FakeAC:
    def __init__(self):
        self.t = 0.0
        self.steer = self.throttle = self.brake = 0.0
        self.packet = 0
        self.laps = 0
        self.last_lap_ms = 0
        self.lap_start = 0.0
        self.resets = 0
        self.place_at_start()

    # --- sim
    def place_at_start(self):
        self.x, self.z = R, 0.0
        self.psi = math.pi / 2  # heading along +z = CCW tangent at (R, 0)
        self.v = 0.0
        self.phi_prev = 0.0

    def advance(self, dt):
        n = max(1, int(round(dt / 0.005)))
        h = dt / n
        for _ in range(n):
            acc = 6.0 * self.throttle - 12.0 * self.brake - 0.002 * self.v * self.v
            self.v = max(0.0, self.v + acc * h)
            self.psi += self.v * self.steer * 0.05 * h  # max curvature 0.05 1/m
            self.x += self.v * math.cos(self.psi) * h
            self.z += self.v * math.sin(self.psi) * h
            self.t += h
            phi = math.atan2(self.z, self.x) % (2 * math.pi)
            if self.phi_prev > 5.5 and phi < 0.8:  # crossed start/finish going forward
                self.laps += 1
                self.last_lap_ms = int((self.t - self.lap_start) * 1000)
                self.lap_start = self.t
            self.phi_prev = phi
        self.packet += 1

    # --- shared memory interface
    def static(self):
        s = SPageFileStatic()
        _set_wstr(s.track, "fake_ring")
        _set_wstr(s.carModel, "fake_car")
        s.maxRpm = 8000
        s.trackSPlineLength = 2 * math.pi * R
        return s

    def physics(self):
        p = SPageFilePhysics()
        p.packetId = self.packet
        p.speedKmh = self.v * 3.6
        p.velocity[0] = self.v * math.cos(self.psi)
        p.velocity[2] = self.v * math.sin(self.psi)
        p.localVelocity[2] = self.v
        p.gas, p.brake, p.steerAngle = self.throttle, self.brake, self.steer
        p.rpms = 3000
        p.gear = 3
        p.autoShifterOn = 1
        p.numberOfTyresOut = 4 if abs(math.hypot(self.x, self.z) - R) > HALF_WIDTH else 0
        return p

    def graphics(self):
        g = SPageFileGraphic()
        g.status = ACStatus.LIVE
        g.completedLaps = self.laps
        g.iLastTime = self.last_lap_ms
        g.normalizedCarPosition = (math.atan2(self.z, self.x) % (2 * math.pi)) / (2 * math.pi)
        g.carCoordinates[0], g.carCoordinates[2] = self.x, self.z
        return g

    # --- pad / resetter / clock interfaces
    def set(self, steer, throttle, brake):
        self.steer, self.throttle, self.brake = steer, throttle, brake

    def neutral(self, brake=1.0):
        self.set(0.0, 0.0, brake)

    def close(self):
        pass

    def reset(self):
        self.resets += 1
        self.place_at_start()

    def now(self):
        return self.t

    def sleep(self, s):
        self.advance(s)


def make_env(cfg_env=None, with_line=True, tmp_path=None):
    import numpy as np
    import yaml
    from pathlib import Path

    from ac_rl.env import AssettoCorsaEnv
    from ac_rl.track import ReferenceLine

    cfg = cfg_env or yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))["env"]
    cfg = dict(cfg)
    cfg["reset"] = dict(cfg["reset"], settle_s=0.2)
    if with_line:
        a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        path = Path(tmp_path) / "ring.npz"
        ReferenceLine(np.stack([R * np.cos(a), R * np.sin(a)], axis=1)).save(path)
        cfg["reference_line"] = str(path)
    else:
        cfg["reference_line"] = None
    sim = FakeAC()
    return AssettoCorsaEnv(cfg, sm=sim, pad=sim, resetter=sim, clock=sim), sim
