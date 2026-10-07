"""Outputs to the game: vJoy axes for driving, keyboard hotkey for resetting the car."""
from __future__ import annotations

import time

AXIS_MIN = 0x1
AXIS_MAX = 0x8000


def _to_axis(v: float) -> int:
    """Map [0, 1] to the vJoy axis range."""
    v = min(max(v, 0.0), 1.0)
    return int(AXIS_MIN + v * (AXIS_MAX - AXIS_MIN))


class VJoyController:
    """Steering on X, throttle on Y, brake on Z. Bind these in AC's controls (see README)."""

    def __init__(self, device_id: int = 1):
        try:
            import pyvjoy
        except Exception as e:  # missing package or vJoyInterface.dll
            raise RuntimeError(
                "Could not load pyvjoy. Install it (pip install pyvjoy) and, if it complains about the DLL, "
                "copy C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll into the pyvjoy package folder."
            ) from e
        self._dev = pyvjoy.VJoyDevice(device_id)

    def set(self, steer: float, throttle: float, brake: float):
        """steer in [-1, 1] (left negative), throttle/brake in [0, 1]."""
        d = self._dev.data
        d.wAxisX = _to_axis((steer + 1.0) / 2.0)
        d.wAxisY = _to_axis(throttle)
        d.wAxisZ = _to_axis(brake)
        self._dev.update()

    def neutral(self, brake: float = 1.0):
        self.set(0.0, 0.0, brake)

    def close(self):
        self.neutral(brake=0.0)


class HotkeyResetter:
    """Resets the car by sending a key combo to the focused AC window.

    `keys` is a list of pydirectinput key names, e.g. ["ctrl", "r"]. Bind the same combo to
    "Restart session" (or a teleport-to-start action) in Content Manager / CSP. With an empty list,
    falls back to asking you to reset by hand, which is handy for a first test.
    """

    def __init__(self, keys: list[str], hold_s: float = 0.1):
        self.keys = list(keys or [])
        self.hold_s = hold_s
        self._pdi = None
        if self.keys:
            try:
                import pydirectinput
            except Exception as e:
                raise RuntimeError("pydirectinput is required for hotkey resets: pip install pydirectinput") from e
            pydirectinput.PAUSE = 0.0
            self._pdi = pydirectinput

    def reset(self):
        if not self.keys:
            input("[reset] No reset hotkey configured. Reset the car in AC, then press Enter here...")
            return
        for k in self.keys:
            self._pdi.keyDown(k)
        time.sleep(self.hold_s)
        for k in reversed(self.keys):
            self._pdi.keyUp(k)
