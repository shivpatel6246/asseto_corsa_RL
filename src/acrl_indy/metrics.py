"""Success metrics from CLAUDE.md: lap completed, lateral RMS < 1.0 m, max < 3.0 m, no out-of-track."""
from __future__ import annotations

import numpy as np

LAT_RMS_LIMIT = 1.0
LAT_MAX_LIMIT = 3.0


class EpisodeTracker:
    """Accumulates per-step centerline stats. A lap counts as completed once the car has covered
    one full centerline length of forward progress from its spawn point."""

    def __init__(self, lap_length_m):
        self.lap_length = lap_length_m
        self.progress_m = 0.0
        self.lat = []
        self.lat_in_lap = []
        self.speed = []
        self.oot_events = 0
        self._was_oot = False
        self.lap_completed = False
        self.steps = 0

    def add(self, progress_m, lateral, out_of_track, speed):
        self.steps += 1
        self.progress_m += progress_m
        self.lat.append(lateral)
        self.speed.append(speed)
        if not self.lap_completed:
            self.lat_in_lap.append(lateral)
            if self.progress_m >= self.lap_length:
                self.lap_completed = True
        if out_of_track and not self._was_oot:
            self.oot_events += 1
        self._was_oot = out_of_track

    def metrics(self):
        lat = np.asarray(self.lat or [0.0])
        lap = np.asarray(self.lat_in_lap or [0.0])
        m = {
            "progress_m": self.progress_m,
            "lap_completed": int(self.lap_completed),
            "lat_rms": float(np.sqrt(np.mean(lat ** 2))),
            "lat_max": float(np.max(np.abs(lat))),
            "lap_lat_rms": float(np.sqrt(np.mean(lap ** 2))),
            "lap_lat_max": float(np.max(np.abs(lap))),
            "oot_events": self.oot_events,
            "speed_mean": float(np.mean(self.speed)) if self.speed else 0.0,
        }
        m["success"] = int(m["lap_completed"] and m["lap_lat_rms"] < LAT_RMS_LIMIT
                           and m["lap_lat_max"] < LAT_MAX_LIMIT and m["oot_events"] == 0)
        return m


def format_metrics(m):
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    return "\n".join([
        f"  lap completed : {'yes' if m['lap_completed'] else 'no'} ({m['progress_m']:.0f} m)  {ok(m['lap_completed'])}",
        f"  lateral RMS   : {m['lap_lat_rms']:.2f} m (< {LAT_RMS_LIMIT})  {ok(m['lap_lat_rms'] < LAT_RMS_LIMIT)}",
        f"  lateral max   : {m['lap_lat_max']:.2f} m (< {LAT_MAX_LIMIT})  {ok(m['lap_lat_max'] < LAT_MAX_LIMIT)}",
        f"  out of track  : {m['oot_events']} events  {ok(m['oot_events'] == 0)}",
        f"  mean speed    : {m['speed_mean'] * 3.6:.1f} km/h",
        f"  OVERALL       : {ok(m['success'])}",
    ])
