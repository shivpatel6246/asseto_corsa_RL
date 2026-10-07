"""Phase 1: print the HUMAN checklist for Assetto Corsa setup, with paths resolved on this machine.

Copy commands are printed, not executed: Program Files needs an elevated shell (CLAUDE.md rule 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrl_indy import winenv as w  # noqa: E402
from acrl_indy.paths import UPSTREAM_PKG  # noqa: E402


def main():
    acs = w.find_ac()
    ac = acs[0] if acs else Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
    docs = w.ac_documents()
    plugin_src = UPSTREAM_PKG / "AssettoCorsaPlugin" / "plugins" / "sensors_par"
    libs = UPSTREAM_PKG / "AssettoCorsaPlugin" / "windows-libs"
    saved = docs / "cfg" / "controllers" / "savedsetups"
    note = "" if acs else "  (AC not detected; using the default path; fix it if yours differs)"

    print(f"""
PHASE 1: Assetto Corsa setup. These are HUMAN steps; tick each one and report back.
AC install: {ac}{note}
AC documents: {docs}

[ ] 1. Install vJoy: https://sourceforge.net/projects/vjoystick/
       Then open "Configure vJoy" and make sure device 1 is enabled (X, Y, Z axes).

[ ] 2. Copy the plugin. Upstream's INSTALL.md spells the destination "sensor_par", but the folder
       and its main file are "sensors_par" and AC loads apps/python/<name>/<name>.py, so keep
       "sensors_par". In an ADMIN PowerShell:
         robocopy "{plugin_src}" "{ac}\\apps\\python\\sensors_par" /E

[ ] 3. Copy controller presets and the Python socket libs (ADMIN PowerShell):
         New-Item -ItemType Directory -Force "{saved}"
         Copy-Item "{libs}\\Vjoy.ini" "{saved}\\"
         Copy-Item "{libs}\\WASD.ini" "{saved}\\"
         robocopy "{libs}\\DLLs" "{ac}\\system\\x64\\DLLs" /E
         robocopy "{libs}\\Lib"  "{ac}\\system\\x64\\Lib"  /E

[ ] 4. Install Content Manager (https://acstuff.ru/app/), then in CM:
       Settings > Custom Shaders Patch > Install. (Required: resets use CSP's ext_resetCar.)

[ ] 5. In AC/CM:
       - Options > General > UI Modules: enable sensors_par
       - Options > Controls: check that Vjoy and WASD are listed, then load "Vjoy" as the active input

[ ] 6. Options > Video > Display > Framerate limit = 50 FPS (the plugin asserts 50 Hz when vJoy runs on the client).
       Display mode: windowed. Keep the AC window visible during runs.

[ ] 7. Session: Challenge > Hotlap, track indianapolis_sp, car ks_mazda_miata.
       Assists: auto gearbox ON, auto clutch ON, auto blip OFF, TC OFF, ESC/stability OFF, ABS OFF,
       damage OFF, fuel OFF, tyre wear OFF, tyre blankets ON, slipstream 1x, penalties ON.
       Conditions: optimum surface, time multiplier 1x (upstream: 10:00, Mid Clear, 26 C).

[ ] 8. Start the session and drive onto track. Debug log, if anything misbehaves:
         {docs}\\logs\\py_log.txt

When all eight are done, run:  python scripts/audit_env.py --log
""")


if __name__ == "__main__":
    main()
