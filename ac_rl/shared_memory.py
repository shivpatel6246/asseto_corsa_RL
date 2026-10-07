"""Read-only access to Assetto Corsa's shared memory pages (acpmf_physics / graphics / static).

Struct layouts follow the official AC shared memory reference (v1.7). Only the leading
fields we use are declared; reading a prefix of each page is safe.
Strings are UTF-16 on Windows, so they are declared as uint16 arrays (keeps the layout
identical on every platform, which lets the tests import this module off Windows).
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import c_float, c_int32, c_uint16


class ACStatus:
    OFF = 0
    REPLAY = 1
    LIVE = 2
    PAUSE = 3


def wstr(arr) -> str:
    """Decode a uint16 (wchar) array into a Python string."""
    out = []
    for c in arr:
        if c == 0:
            break
        out.append(chr(c))
    return "".join(out)


class _Struct(ctypes.Structure):
    _pack_ = 4


class SPageFilePhysics(_Struct):
    _fields_ = [
        ("packetId", c_int32),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),
        ("gear", c_int32),  # 0 = R, 1 = N, 2 = 1st, ...
        ("rpms", c_int32),
        ("steerAngle", c_float),
        ("speedKmh", c_float),
        ("velocity", c_float * 3),  # world frame (x, y-up, z)
        ("accG", c_float * 3),
        ("wheelSlip", c_float * 4),
        ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4),
        ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4),
        ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4),
        ("drs", c_float),
        ("tc", c_float),
        ("heading", c_float),
        ("pitch", c_float),
        ("roll", c_float),
        ("cgHeight", c_float),
        ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int32),
        ("pitLimiterOn", c_int32),
        ("abs", c_float),
        ("kersCharge", c_float),
        ("kersInput", c_float),
        ("autoShifterOn", c_int32),
        ("rideHeight", c_float * 2),
        ("turboBoost", c_float),
        ("ballast", c_float),
        ("airDensity", c_float),
        ("airTemp", c_float),
        ("roadTemp", c_float),
        ("localAngularVel", c_float * 3),
        ("finalFF", c_float),
        ("performanceMeter", c_float),
        ("engineBrake", c_int32),
        ("ersRecoveryLevel", c_int32),
        ("ersPowerLevel", c_int32),
        ("ersHeatCharging", c_int32),
        ("ersIsCharging", c_int32),
        ("kersCurrentKJ", c_float),
        ("drsAvailable", c_int32),
        ("drsEnabled", c_int32),
        ("brakeTemp", c_float * 4),
        ("clutch", c_float),
        ("tyreTempI", c_float * 4),
        ("tyreTempM", c_float * 4),
        ("tyreTempO", c_float * 4),
        ("isAIControlled", c_int32),
        ("tyreContactPoint", (c_float * 3) * 4),
        ("tyreContactNormal", (c_float * 3) * 4),
        ("tyreContactHeading", (c_float * 3) * 4),
        ("brakeBias", c_float),
        ("localVelocity", c_float * 3),  # car frame
    ]


class SPageFileGraphic(_Struct):
    _fields_ = [
        ("packetId", c_int32),
        ("status", c_int32),  # ACStatus
        ("session", c_int32),
        ("currentTime", c_uint16 * 15),
        ("lastTime", c_uint16 * 15),
        ("bestTime", c_uint16 * 15),
        ("split", c_uint16 * 15),
        ("completedLaps", c_int32),
        ("position", c_int32),
        ("iCurrentTime", c_int32),
        ("iLastTime", c_int32),
        ("iBestTime", c_int32),
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int32),
        ("currentSectorIndex", c_int32),
        ("lastSectorTime", c_int32),
        ("numberOfLaps", c_int32),
        ("tyreCompound", c_uint16 * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),  # 0..1 along the track spline
        ("carCoordinates", c_float * 3),  # world frame (x, y-up, z)
        ("penaltyTime", c_float),
        ("flag", c_int32),
        ("idealLineOn", c_int32),
        ("isInPitLane", c_int32),
        ("surfaceGrip", c_float),
        ("mandatoryPitDone", c_int32),
        ("windSpeed", c_float),
        ("windDirection", c_float),
    ]


class SPageFileStatic(_Struct):
    _fields_ = [
        ("smVersion", c_uint16 * 15),
        ("acVersion", c_uint16 * 15),
        ("numberOfSessions", c_int32),
        ("numCars", c_int32),
        ("carModel", c_uint16 * 33),
        ("track", c_uint16 * 33),
        ("playerName", c_uint16 * 33),
        ("playerSurname", c_uint16 * 33),
        ("playerNick", c_uint16 * 33),
        ("sectorCount", c_int32),
        ("maxTorque", c_float),
        ("maxPower", c_float),
        ("maxRpm", c_int32),
        ("maxFuel", c_float),
        ("suspensionMaxTravel", c_float * 4),
        ("tyreRadius", c_float * 4),
        ("maxTurboBoost", c_float),
        ("deprecated_1", c_float),
        ("deprecated_2", c_float),
        ("penaltiesEnabled", c_int32),
        ("aidFuelRate", c_float),
        ("aidTireRate", c_float),
        ("aidMechanicalDamage", c_float),
        ("aidAllowTyreBlankets", c_int32),
        ("aidStability", c_float),
        ("aidAutoClutch", c_int32),
        ("aidAutoBlip", c_int32),
        ("hasDRS", c_int32),
        ("hasERS", c_int32),
        ("hasKERS", c_int32),
        ("kersMaxJ", c_float),
        ("engineBrakeSettingsCount", c_int32),
        ("ersPowerControllerCount", c_int32),
        ("trackSPlineLength", c_float),
        ("trackConfiguration", c_uint16 * 33),
    ]


def _mapping_exists(name: str) -> bool:
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenFileMappingW.restype = ctypes.c_void_p
    FILE_MAP_READ = 0x0004
    handle = kernel32.OpenFileMappingW(FILE_MAP_READ, False, name)
    if not handle:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return True


class ACSharedMemory:
    """Maps the three AC pages. AC must be running and inside a session."""

    PAGES = {
        "physics": ("Local\\acpmf_physics", SPageFilePhysics),
        "graphics": ("Local\\acpmf_graphics", SPageFileGraphic),
        "static": ("Local\\acpmf_static", SPageFileStatic),
    }

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("AC shared memory is only available on Windows.")
        import mmap

        self._maps = {}
        for key, (name, struct) in self.PAGES.items():
            # Never create the mapping ourselves: if we did it with our (smaller) struct
            # size before AC starts, AC would reuse an undersized mapping.
            if not _mapping_exists(name):
                raise RuntimeError(
                    f"Shared memory '{name}' not found. Start Assetto Corsa and load into a session first."
                )
            self._maps[key] = (mmap.mmap(-1, ctypes.sizeof(struct), name), struct)

    def _read(self, key):
        mm, struct = self._maps[key]
        return struct.from_buffer_copy(mm[: ctypes.sizeof(struct)])

    def physics(self) -> SPageFilePhysics:
        return self._read("physics")

    def graphics(self) -> SPageFileGraphic:
        return self._read("graphics")

    def static(self) -> SPageFileStatic:
        return self._read("static")

    def close(self):
        for mm, _ in self._maps.values():
            mm.close()
        self._maps.clear()
