#!/usr/bin/env python3
"""
Entrypoint script for AI Weather Anomaly & Tracking Core.
Ministry of Earth Sciences (MoES) - Problem Statement 26078.
"""

import sys
from pathlib import Path
import uvicorn

# Ensure 'backend' directory is on Python search path
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
