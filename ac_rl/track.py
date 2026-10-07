"""Reference line: a closed loop recorded from one clean lap, used for track-relative observations.

All features are frame-independent (lateral offset, heading error relative to the line, and how much
the line turns over the next N metres), so we never depend on AC's yaw sign convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ReferenceLine:
    def __init__(self, points: np.ndarray, spacing: float = 2.0, smooth_window: int = 5):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 10:
            raise ValueError("points must be an (N>=10, 2) array of (x, z) world coordinates")

        # Resample the closed loop at uniform spacing.
        closed = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        length = s[-1]
        n = max(int(round(length / spacing)), 10)
        s_new = np.linspace(0.0, length, n, endpoint=False)
        res = np.stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])], axis=1)

        # Circular moving average to remove recording jitter.
        if smooth_window > 1:
            k = np.ones(smooth_window) / smooth_window
            pad = smooth_window // 2
            ext = np.vstack([res[-pad:], res, res[:pad]]) if pad else res
            res = np.stack([np.convolve(ext[:, i], k, mode="valid")[: len(res)] for i in range(2)], axis=1)

        self.points = res
        self.spacing = length / n
        self.length = float(length)
        nxt = np.roll(res, -1, axis=0)
        self.theta = np.arctan2(nxt[:, 1] - res[:, 1], nxt[:, 0] - res[:, 0])

    def nearest(self, xz) -> int:
        d2 = np.sum((self.points - np.asarray(xz)) ** 2, axis=1)
        return int(np.argmin(d2))

    def features(self, xz, vel_xz, lookahead_m, min_speed: float = 2.0):
        """Returns (lateral_offset_m, heading_error_rad, line_turn_rad[len(lookahead_m)])."""
        i = self.nearest(xz)
        th = self.theta[i]
        rel = np.asarray(xz) - self.points[i]
        lateral = float(np.cos(th) * rel[1] - np.sin(th) * rel[0])

        vx, vz = vel_xz
        if np.hypot(vx, vz) > min_speed:
            heading_err = float(wrap_angle(np.arctan2(vz, vx) - th))
        else:
            heading_err = 0.0

        idx = (i + np.round(np.asarray(lookahead_m) / self.spacing).astype(int)) % len(self.points)
        turn = wrap_angle(self.theta[idx] - th)
        return lateral, heading_err, turn

    def save(self, path, **meta):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, points=self.points, **meta)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        # Saved points are already resampled/smoothed.
        pts = data["points"]
        closed = np.vstack([pts, pts[:1]])
        spacing = float(np.mean(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
        return cls(pts, spacing=spacing, smooth_window=1)
