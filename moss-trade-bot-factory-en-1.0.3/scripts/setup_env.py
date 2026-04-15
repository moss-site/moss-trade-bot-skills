#!/usr/bin/env python3
"""Create a local virtualenv and install the bundled Python dependencies."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def main() -> int:
    scripts_dir = pathlib.Path(__file__).resolve().parent
    skill_root = scripts_dir.parent
    venv_dir = skill_root / ".venv"
    requirements_path = scripts_dir / "requirements.txt"
    python_bin = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(requirements_path)])

    print(f"Local virtualenv ready: {venv_dir}")
    print(f"Use this Python for bundled scripts: {python_bin}")
    print("Only packages from scripts/requirements.txt were installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
