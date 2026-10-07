"""Stand-in for upstream's ac_client.Client: a kinematic bicycle car on the real Indy centerline.

Lets the real upstream AssettoCorsaEnv (and our wrapper, baseline, smoke scripts, SAC agent) run
without Windows/AC. STEER_SIGN_TRUE is deliberately -1 so tests prove the steer-sign probe works.
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np

from acrl_indy.centerline import Centerline

HZ = 25
STEER_SIGN_TRUE = -1  # +steer turns the car clockwise in upstream's frame (unknown in the real game)


def install_import_stubs():
    """upstream car_control imports vjoy_linux on posix, which needs evdev (Linux only)."""
    if "evdev" not in sys.modules:
        m = types.ModuleType("evdev")
        m.InputDevice = object
        m.list_devices = lambda: []
        sys.modules["evdev"] = m


class Sim:
    def __init__(self, cl: Centerline, max_steer_deg=302.4, steer_ratio=15.0, wheelbase=2.265, half_width=6.5):
        self.cl, self.max_steer_deg, self.ratio, self.L, self.half_width = cl, max_steer_deg, steer_ratio, wheelbase, half_width
        self.cmd = (0.0, -1.0, -1.0)
        self.steps = 0
        self.t = 0.0
        self.resets = 0
        self.place()

    def place(self):
        s0 = self.cl.length - 40.0  # hotlap spawn: just before the start/finish line
        self.x, self.y = self.cl.point_at(s0)
        self.psi = self.cl.heading_at(s0)
        self.v = 0.0
        self.idx = None
        self.resets += 1

    def advance(self, dt):
        steer, acc, brake = self.cmd
        thr, brk = (acc + 1) / 2, (brake + 1) / 2
        self.v = max(0.0, self.v + (5.0 * thr - 10.0 * brk - 0.0008 * self.v ** 2 - 0.1) * dt)
        delta = math.radians(STEER_SIGN_TRUE * steer * self.max_steer_deg / self.ratio)
        self.psi += self.v / self.L * math.tan(delta) * dt
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt
        self.t += dt
        self.steps += 1

    def state(self):
        s, lat, self.idx = self.cl.project((self.x, self.y), hint=self.idx)
        steer, acc, brake = self.cmd
        return {
            "steps": self.steps, "packetId": self.steps, "currentTime": int(self.t * 1000), "done": False,
            "world_position_x": self.x, "world_position_y": self.y, "world_position_z": 0.0,
            "speed": self.v, "velocity_z": self.v * math.cos(self.psi), "velocity_x": self.v * math.sin(self.psi),
            "velocity_y": 0.0, "local_velocity_x": self.v, "local_velocity_y": 0.0, "local_velocity_z": 0.0,
            "yaw": self.psi, "pitch": 0.0, "roll": 0.0, "accelX": 0.0, "accelY": 0.0, "RPM": 3000.0,
            "actualGear": 3, "LastFF": 0.0, "angular_velocity_x": 0.0, "angular_velocity_y": 0.0,
            "angular_velocity_z": 0.0, "SlipAngle_fl": 0.0, "SlipAngle_fr": 0.0, "SlipAngle_rl": 0.0,
            "SlipAngle_rr": 0.0, "steerAngle": steer * self.max_steer_deg, "accStatus": (acc + 1) / 2,
            "brakeStatus": (brake + 1) / 2, "NormalizedSplinePosition": s / self.cl.length,
            "numberOfTyresOut": 4 if abs(lat) > self.half_width else 0, "LapCount": 0, "BestLap": 0,
            "iLastTime": 0, "completedLaps": 0, "LapInvalidated": 0,
        }


SIM: Sim | None = None  # set by the test before the env is built


class _Controls(dict):
    def set_defaults(self):
        SIM.cmd = (0.0, -1.0, -1.0)

    def set_controls(self, steer, acc, brake, enable_gear_shift=False, shift_up=False, shift_down=False):
        SIM.cmd = (float(steer), float(acc), float(brake))


class _SimManagement:
    def send_reset(self):
        SIM.place()

    def get_static_info(self):
        return {"TrackName": "indianapolis_sp", "TrackConfiguration": "", "TrackFullName": "indianapolis_sp",
                "CarName": "ks_mazda_miata", "TrackLength": 4066.004150390625}

    def get_config(self):
        return {"ego_sampling_freq": HZ, "vjoy_executed_by_server": False}


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.controls = _Controls()
        self.simulation_management = _SimManagement()

    def respond_to_server(self):
        pass

    def step_sim(self):
        SIM.advance(1.0 / HZ)
        return SIM.state()

    def reset(self, send_reset=True):
        if send_reset:
            SIM.place()
        self.controls.set_defaults()

    def close(self):
        pass
