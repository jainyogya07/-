#!/usr/bin/env python3
"""
Entrypoint script for AI Weather Anomaly & Tracking Core.
Ministry of Earth Sciences (MoES) - Problem Statement 26078.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python3"

# 1. Seamless auto-activation: if .venv exists and user didn't activate it, re-exec inside .venv
if VENV_PYTHON.exists() and sys.prefix != str(VENV_DIR):
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["PATH"] = f"{VENV_DIR / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{BACKEND_DIR}:{env.get('PYTHONPATH', '')}"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv, env)

# 2. Ensure BACKEND_DIR is in sys.path and os.environ (inherited by reloader subprocesses)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

current_pythonpath = os.environ.get("PYTHONPATH", "")
if str(BACKEND_DIR) not in current_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{current_pythonpath}".strip(os.pathsep)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
