"""Locate Assetto Corsa, Documents, vJoy, CSP, Content Manager on Windows (filesystem checks only)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_STEAM = Path(r"C:\Program Files (x86)\Steam")
VJOY_DLL = Path(r"C:\Program Files\vJoy\x64\vJoyInterface.dll")  # path hardcoded in upstream vjoy.py
TRACK = "indianapolis_sp"
CAR = "ks_mazda_miata"


def is_windows():
    return sys.platform == "win32"


def steam_root():
    if is_windows():
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
                return Path(winreg.QueryValueEx(k, "SteamPath")[0])
        except OSError:
            pass
    return DEFAULT_STEAM


def steam_libraries():
    root = steam_root()
    libs = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore")):
            p = Path(m.group(1).replace("\\\\", "\\"))
            if p not in libs:
                libs.append(p)
    return libs


def find_ac():
    """All candidate AC install dirs that exist (first is the one we use)."""
    found = []
    for lib in steam_libraries():
        p = lib / "steamapps" / "common" / "assettocorsa"
        if p.exists():
            found.append(p)
    return found


def documents_dir():
    """Real Documents folder (handles OneDrive redirection)."""
    if is_windows():
        import ctypes
        from ctypes import wintypes

        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # CSIDL_PERSONAL
            return Path(buf.value)
    return Path.home() / "Documents"


def ac_documents():
    return documents_dir() / "Assetto Corsa"


def content_manager_installed():
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "AcTools Content Manager"
    return local.exists(), local


def csp_installed(ac: Path):
    return (ac / "extension").is_dir() and (ac / "dwrite.dll").exists()


def plugin_installed(ac: Path):
    return (ac / "apps" / "python" / "sensors_par" / "sensors_par.py").exists()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    q = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
    head = run(["nvidia-smi"]).splitlines()
    cuda = next((re.search(r"CUDA Version:\s*([\d.]+)", l).group(1) for l in head if "CUDA Version" in l), "?")
    return {"gpus": q, "driver_cuda": cuda}


def vc_build_tools():
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return None
    out = run([str(vswhere), "-products", "*", "-requires",
               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
    return out or None


def torch_recommendation(gpu_line: str | None):
    """Advice only; record the actual choice in docs/SETUP_LOG.md."""
    if not gpu_line:
        return "No NVIDIA GPU found: CPU build. pip install torch==1.12.1 (small MLPs, CPU is acceptable)"
    name = gpu_line.split(",")[0]
    cc = gpu_line.split(",")[-1].strip()
    try:
        major = float(cc)
    except ValueError:
        major = 0
    if major >= 12:  # RTX 50-series (Blackwell, sm_120) is not supported by torch 1.12/cu116
        return (f"{name} (sm_{cc}) is too new for torch 1.12/cu116. Try the newest torch with a cp39 cu128 wheel, "
                "e.g. pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 ; "
                "if no cp39 wheel resolves, use a CPU build.")
    if major >= 8.9:  # RTX 40-series
        return (f"{name} (sm_{cc}): torch 1.12/cu116 has no sm_89 kernels. Use e.g. "
                "pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118")
    return (f"{name} (sm_{cc}): upstream pin works: pip install torch==1.12.1+cu116 "
            "--extra-index-url https://download.pytorch.org/whl/cu116")
