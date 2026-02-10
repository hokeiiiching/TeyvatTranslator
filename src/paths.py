# -*- coding: utf-8 -*-
"""
Path Utilities for PyInstaller Compatibility

Provides centralized path resolution that works both in development
(running via `py main.py`) and in the bundled PyInstaller executable.

In development:
    BASE_DIR = project root (parent of src/)

In PyInstaller bundle:
    BASE_DIR = sys._MEIPASS (the _internal directory where data files live)
"""

import sys
import os


def get_base_dir() -> str:
    """
    Get the application base directory.
    
    Returns the project root in dev mode, or sys._MEIPASS in frozen mode.
    Data files are always at BASE_DIR/src/data/, BASE_DIR/src/ui/, BASE_DIR/assets/.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller onedir: data files extracted to sys._MEIPASS (_internal/)
        return sys._MEIPASS
    # Development: src/paths.py is in src/, so parent = project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, 'src', 'data')
UI_DIR = os.path.join(BASE_DIR, 'src', 'ui')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Log resolved paths at import time for debugging
print(f"[paths] frozen={getattr(sys, 'frozen', False)}")
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] DATA_DIR={DATA_DIR}")
print(f"[paths] ASSETS_DIR={ASSETS_DIR}")
