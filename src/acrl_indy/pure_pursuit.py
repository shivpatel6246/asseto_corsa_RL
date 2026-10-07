"""Classical baseline: pure-pursuit steering on the centerline + PI speed control.

Run with absolute actions (use_relative_actions: False, see configs/indy.yml `baseline`), so the
returned [steer, pedal, brake] go straight to vJoy. Upstream's action convention: all in [-1, 1],
steer -1..1 maps to +-max_steer_deg of steering wheel (car steer_map.csv), pedal/brake -1 = released.

Heading comes from the velocity vector (frame-convention free); at standstill we assume the car
is aligned with the centerline, which holds at the hotlap spawn.
"""
from __future__ import annotations

import math

import numpy as np

from .centerline import Centerline, wrap_angle


class PurePursuitController:
    def __init__(self, centerline: Centerline, cfg, max_steer_deg: float):
        self.cl = centerline
        self.c = cfg
        self.max_steer_deg = float(max_steer_deg)
        self.reset()

    def reset(self):
        self._idx = None
        self._integ = 0.0

    def act(self, st, dt):
        c = self.c
        x, y = st["world_position_x"], st["world_position_y"]
        s, lat, self._idx = self.cl.project((x, y), hint=self._idx)
        v = float(st["speed"])

        # --- steering
        ld = float(np.clip(c.lookahead_base_m + c.lookahead_gain_s * v, c.lookahead_min_m, c.lookahead_max_m))
        tx, ty = self.cl.point_at(s + ld)
        vx, vy = st["velocity_z"], st["velocity_x"]  # upstream frame: x = AC z, y = AC x
        heading = math.atan2(vy, vx) if v > 2.0 else self.cl.heading_at(s)
        alpha = wrap_angle(math.atan2(ty - y, tx - x) - heading)
        delta = math.atan2(2.0 * c.wheelbase_m * math.sin(alpha), ld)  # road-wheel angle, rad
        steer = c.steer_sign * math.degrees(delta) * c.steer_ratio / self.max_steer_deg
        steer = float(np.clip(steer, -1.0, 1.0))

        # --- speed: constant target, capped before corners by v <= sqrt(a_lat / curvature)
        kappa = self.cl.max_curvature_ahead(s, c.curvature_preview_m, step=5.0)
        v_cap = math.sqrt(c.max_lat_accel / max(kappa, 1e-4))
        v_target = min(c.target_speed_kmh / 3.6, v_cap)
        e = v_target - v
        self._integ = float(np.clip(self._integ + e * dt, -c.speed_i_max / max(c.speed_ki, 1e-9),
                                    c.speed_i_max / max(c.speed_ki, 1e-9)))
        u = c.speed_kp * e + c.speed_ki * self._integ
        throttle = float(np.clip(u, 0.0, 1.0))
        brake = float(np.clip(-u, 0.0, 1.0))

        action = np.array([steer, 2.0 * throttle - 1.0, 2.0 * brake - 1.0])
        diag = {"s": s, "lateral": lat, "lookahead_m": ld, "alpha": alpha, "v_target": v_target,
                "kappa_ahead": kappa}
        return action, diag
