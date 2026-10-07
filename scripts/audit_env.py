"""Phase 0: environment audit. Run on the Windows machine (any Python works; no deps needed).

    python scripts/audit_env.py            # print the report
    python scripts/audit_env.py --log      # also append it to docs/SETUP_LOG.md

Stops short of anything GUI-related; see scripts/phase1_checklist.py for the human steps.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrl_indy import winenv as w  # noqa: E402
from acrl_indy.paths import DOCS, UPSTREAM, UPSTREAM_CARS, UPSTREAM_TRACKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="append report to docs/SETUP_LOG.md")
    args = ap.parse_args()

    lines, blockers = [], []
    add = lines.append

    def check(ok, msg, blocker=None):
        add(f"- [{'x' if ok else ' '}] {msg}")
        if not ok and blocker:
            blockers.append(blocker)

    add(f"## Phase 0 audit ({datetime.now():%Y-%m-%d %H:%M})")
    add(f"- Windows: {platform.platform()} ({platform.version()})")
    add(f"- Python running this script: {sys.version.split()[0]} at {sys.executable}")

    gpu = w.nvidia_smi() if w.is_windows() else None
    if gpu:
        add(f"- GPU(s) (name, driver, memory, compute cap): {gpu['gpus']}")
        add(f"- Driver max CUDA: {gpu['driver_cuda']}")
    else:
        add("- GPU: nvidia-smi not found (no NVIDIA GPU or driver)")
    add(f"- Torch recommendation: {w.torch_recommendation(gpu['gpus'].splitlines()[0] if gpu else None)}")

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            (torch.ones(256, 256, device="cuda") @ torch.ones(256, 256, device="cuda")).sum().item()
        add(f"- Installed torch: {torch.__version__}, CUDA build {torch.version.cuda}, cuda usable: {cuda_ok}")
    except Exception as e:  # not installed yet, or a kernel-image error on a too-new GPU
        add(f"- Installed torch: not usable in this interpreter ({type(e).__name__}: {e})")

    acs = w.find_ac()
    check(bool(acs), f"Assetto Corsa install: {acs[0] if acs else 'NOT FOUND'}"
          + (f" (also found: {acs[1:]})" if len(acs) > 1 else ""),
          "Assetto Corsa not found in any Steam library")
    if acs:
        ac = acs[0]
        check((ac / "content" / "tracks" / w.TRACK).is_dir(), f"track folder content/tracks/{w.TRACK}",
              f"Track {w.TRACK} is missing from content/tracks. STOP: it may not be base content.")
        check((ac / "content" / "cars" / w.CAR).is_dir(), f"car folder content/cars/{w.CAR}",
              f"Car {w.CAR} missing from content/cars")
        check(w.csp_installed(ac), "Custom Shaders Patch (extension/ + dwrite.dll)", "CSP not installed (needed for resets)")
        check(w.plugin_installed(ac), "sensors_par plugin in apps/python/sensors_par (Phase 1 step)")
        check((ac / "system" / "x64" / "Lib").is_dir() and (ac / "system" / "x64" / "DLLs").is_dir(),
              "windows-libs DLLs/ and Lib/ copied to system/x64 (Phase 1 step)")
    cm, cm_path = w.content_manager_installed()
    check(cm, f"Content Manager data folder {cm_path}", "Content Manager not detected")
    check(w.VJOY_DLL.exists(), f"vJoy DLL at {w.VJOY_DLL} (upstream hardcodes this path)", "vJoy not installed")

    docs = w.ac_documents()
    check(docs.exists(), f"AC Documents folder {docs}")
    saved = docs / "cfg" / "controllers" / "savedsetups"
    check((saved / "Vjoy.ini").exists(), f"Vjoy.ini in {saved} (Phase 1 step)")

    check(bool(shutil.which("conda")), "conda on PATH (needed for the `acrl` env)")
    vc = w.vc_build_tools() if w.is_windows() else None
    check(bool(vc), f"Visual C++ Build Tools: {vc or 'not found (may be needed if a pip package builds from source)'}")

    check((UPSTREAM / "config.yml").exists(), f"upstream submodule checked out at {UPSTREAM}",
          "Submodule missing: git submodule update --init")
    check((UPSTREAM_TRACKS / f"{w.TRACK}.csv").exists(), f"upstream track bounds {w.TRACK}.csv")
    check((UPSTREAM_TRACKS / f"{w.TRACK}_0.1m.pkl").exists(),
          f"occupancy grid {w.TRACK}_0.1m.pkl (scripts/prepare_assets.py downloads it)")
    check((UPSTREAM_CARS / w.CAR / "steer_map.csv").exists(), f"upstream car config cars/{w.CAR}",
          f"Upstream has no car config for {w.CAR}")

    if blockers:
        add("\n**Blockers:**")
        lines.extend(f"- {b}" for b in blockers)

    report = "\n".join(lines)
    print(report)
    if args.log:
        log = DOCS / "SETUP_LOG.md"
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n\n" + report + "\n")
        print(f"\nAppended to {log}")
    sys.exit(1 if blockers else 0)


if __name__ == "__main__":
    main()
