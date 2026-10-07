"""Geometric centerline (midpoint of left/right track bounds) and fast projection onto it.

Coordinates are upstream's frame: x = AC world z, y = AC world x (see sensors_par/structures.py).
The saved CSV uses upstream's reference-line format (columns pos_x, pos_y; the last point precedes
the first, no duplicated closing point) so it can replace `ref_lap_file` directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- polyline helpers
def _closed(pts):
    return np.vstack([pts, pts[:1]])


def arc_length(pts, closed=True):
    p = _closed(pts) if closed else pts
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample_closed(pts, n):
    """Resample a closed polyline to n points equally spaced in arc length (no repeated end point)."""
    p = _closed(np.asarray(pts, dtype=np.float64))
    s = arc_length(pts)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(s_new, s, p[:, 0]), np.interp(s_new, s, p[:, 1])], axis=1)


def smooth_closed(pts, sigma_pts):
    """Circular Gaussian smoothing, sigma in samples."""
    if sigma_pts <= 0:
        return pts
    r = int(np.ceil(3 * sigma_pts))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_pts) ** 2)
    k /= k.sum()
    ext = np.vstack([pts[-r:], pts, pts[:r]])
    return np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], axis=1)


def _point_to_polyline(p, poly):
    """Distance from each point in p (N,2) to closed polyline poly (M,2). O(N*M); fine for checks."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    d = b - a
    dd = np.maximum(np.sum(d * d, axis=1), 1e-12)
    out = np.empty(len(p))
    for i, q in enumerate(p):
        t = np.clip(np.sum((q - a) * d, axis=1) / dd, 0.0, 1.0)
        proj = a + t[:, None] * d
        out[i] = np.min(np.linalg.norm(proj - q, axis=1))
    return out


# ----------------------------------------------------------------------------- building
def load_track_bounds(track_csv):
    df = pd.read_csv(track_csv)
    left = df[["left_border_x", "left_border_y"]].to_numpy(np.float64)
    right = df[["right_border_x", "right_border_y"]].to_numpy(np.float64)
    racing = df[["pos_x", "pos_y"]].to_numpy(np.float64) if "pos_x" in df else None
    return left, right, racing


def build_centerline(left, right, start_near=None, start_heading_ref=None, spacing=1.0,
                     bound_spacing=0.5, pair_window_m=40.0, smooth_sigma_m=3.0):
    """Midpoints of the left/right bounds -> smoothed closed loop at `spacing` metres.

    Both bounds are resampled to equal arc length (bound_spacing). Pairing purely by arc-length
    fraction drifts through corners (the inside bound is shorter), so each left sample is matched to
    the nearest right sample within +-pair_window_m of its arc-length-proportional partner.

    start_near: rotate the loop so point 0 is the closest to this (x, y) (AC's start/finish).
    start_heading_ref: a point slightly after start_near on the reference line; used to make the
                       loop run in the driving direction.
    """
    len_l = arc_length(left)[-1]
    len_r = arc_length(right)[-1]
    n_l = int(round(len_l / bound_spacing))
    n_r = int(round(len_r / bound_spacing))
    L = resample_closed(left, n_l)
    R = resample_closed(right, n_r)

    w = int(np.ceil(pair_window_m / bound_spacing))
    offsets = np.arange(-w, w + 1)
    expected = np.round(np.arange(n_l) * n_r / n_l).astype(int)
    cand = (expected[:, None] + offsets[None, :]) % n_r  # (n_l, 2w+1)
    d2 = np.sum((R[cand] - L[:, None, :]) ** 2, axis=2)
    j = cand[np.arange(n_l), np.argmin(d2, axis=1)]
    mid = 0.5 * (L + R[j])

    mid = smooth_closed(mid, smooth_sigma_m / bound_spacing)
    n = int(round(arc_length(mid)[-1] / spacing))
    c = resample_closed(mid, n)

    if start_near is not None:
        i0 = int(np.argmin(np.sum((c - np.asarray(start_near)) ** 2, axis=1)))
        c = np.roll(c, -i0, axis=0)
        if start_heading_ref is not None:
            fwd = np.asarray(start_heading_ref) - np.asarray(start_near)
            if np.dot(c[1] - c[0], fwd) < 0:
                c = np.vstack([c[:1], c[1:][::-1]])
    return c


def check_inside_bounds(c, left, right, chunk=256):
    """Is each centerline point inside the track ribbon? The ribbon is the union of quads
    (L_i, L_i+1, R_i+1, R_i) built from the row-aligned bounds in upstream's track CSV.
    Also returns the distance to each bound (margin) for reporting."""
    L, R = np.asarray(left), np.asarray(right)
    q = np.stack([L, np.roll(L, -1, axis=0), np.roll(R, -1, axis=0), R], axis=1)  # (M, 4, 2)
    inside = np.zeros(len(c), dtype=bool)
    for k in range(0, len(c), chunk):
        p = c[k:k + chunk]  # (P, 2)
        crosses = []
        for e in range(4):
            a, b = q[:, e], q[:, (e + 1) % 4]  # (M, 2)
            ab = b - a
            ap = p[:, None, :] - a[None, :, :]  # (P, M, 2)
            crosses.append(ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0])
        cr = np.stack(crosses, axis=-1)  # (P, M, 4)
        in_quad = np.all(cr >= -1e-9, axis=-1) | np.all(cr <= 1e-9, axis=-1)
        inside[k:k + chunk] = in_quad.any(axis=1)
    return {"d_left": _point_to_polyline(c, L), "d_right": _point_to_polyline(c, R), "inside": inside}


def save_reference_csv(c, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pos_x": c[:, 0], "pos_y": c[:, 1]}).to_csv(path, index=False)


def load_reference_csv(path):
    return pd.read_csv(path)[["pos_x", "pos_y"]].to_numpy(np.float64)


# ----------------------------------------------------------------------------- runtime projection
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Centerline:
    """Closed polyline with arc-length projection. lateral > 0 means left of the line
    (counter-clockwise in the x/y plane of upstream's frame)."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.n = len(self.pts)
        self.seg = np.roll(self.pts, -1, axis=0) - self.pts
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])[:-1]
        self.length = float(self.s[-1] + self.seg_len[-1])
        self.heading = np.arctan2(self.seg[:, 1], self.seg[:, 0])

    @classmethod
    def from_csv(cls, path):
        return cls(load_reference_csv(path))

    def project(self, xy, hint=None, window=80):
        """Returns (s, lateral, idx). With `hint` (previous idx) only nearby segments are searched,
        which keeps the projection on the right part of the track where sections run close."""
        p = np.asarray(xy, dtype=np.float64)
        if hint is None:
            idx = np.arange(self.n)
        else:
            idx = (hint + np.arange(-window, window + 1)) % self.n
        a = self.pts[idx]
        d = self.seg[idx]
        t = np.clip(np.sum((p - a) * d, axis=1) / np.maximum(self.seg_len[idx] ** 2, 1e-12), 0.0, 1.0)
        proj = a + t[:, None] * d
        k = int(np.argmin(np.sum((proj - p) ** 2, axis=1)))
        i = int(idx[k])
        rel = p - self.pts[i]
        u = self.seg[i] / self.seg_len[i]
        lateral = float(u[0] * rel[1] - u[1] * rel[0])
        s = float(self.s[i] + t[k] * self.seg_len[i])
        return s, lateral, i

    def point_at(self, s):
        s = np.mod(s, self.length)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        t = (s - self.s[i]) / self.seg_len[i]
        return self.pts[i] + t * self.seg[i]

    def heading_at(self, s):
        s = np.mod(s, self.length)
        return float(self.heading[int(np.searchsorted(self.s, s, side="right") - 1)])

    def max_curvature_ahead(self, s, distance, step=2.0):
        """Largest |curvature| (1/m) between s and s+distance, from heading change per metre."""
        ss = s + np.arange(0.0, distance + step, step)
        h = np.array([self.heading_at(x) for x in ss])
        k = np.abs(wrap_angle(np.diff(h))) / step
        return float(k.max()) if len(k) else 0.0

    @staticmethod
    def progress_delta(s_new, s_old, length):
        d = s_new - s_old
        if d < -length / 2:
            d += length
        elif d > length / 2:
            d -= length
        return d
