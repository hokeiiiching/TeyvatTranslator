#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genshin Translator - Learn Mandarin through Genshin Impact

A context-aware OCR translation desktop application that helps players 
learn Mandarin Chinese while playing Genshin Impact. Features live screen 
capture, intelligent vocabulary matching, and enhanced translation display.

Author: hokeiiiching
License: MIT
Version: 1.1.0
"""

import os
import sys

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

try:
    import importlib.util
    import importlib.machinery
    
    # Find paddlex package location without importing it
    paddlex_spec = importlib.util.find_spec("paddlex")
    if paddlex_spec and paddlex_spec.submodule_search_locations:
        paddlex_path = paddlex_spec.submodule_search_locations[0]
        flags_path = os.path.join(paddlex_path, "utils", "flags.py")
        
        if os.path.exists(flags_path):
            # Load ONLY the flags module directly (bypass paddlex __init__.py)
            loader = importlib.machinery.SourceFileLoader("paddlex.utils.flags", flags_path)
            spec = importlib.util.spec_from_loader("paddlex.utils.flags", loader)
            flags_module = importlib.util.module_from_spec(spec)
            
            # Pre-populate sys.modules so paddlex uses our patched version
            sys.modules["paddlex.utils.flags"] = flags_module
            
            # Execute the module (this reads env vars)
            loader.exec_module(flags_module)
            
            # Force the flag to True
            flags_module.DISABLE_MODEL_SOURCE_CHECK = True
except Exception:
    pass  # If anything fails, continue without the patch

# Ensure the application can find its modules
if getattr(sys, 'frozen', False):
    # PyInstaller: add _internal dir to path
    APP_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# === Log file for frozen mode (captures errors even with console=False) ===
if getattr(sys, 'frozen', False):
    try:
        _log_path = os.path.join(os.path.dirname(sys.executable), 'TeyvatTranslator.log')
        _log_file = open(_log_path, 'w', encoding='utf-8')
        sys.stdout = _log_file
        sys.stderr = _log_file
        print(f"[main] Log started — exe={sys.executable}")
        print(f"[main] APP_DIR={APP_DIR}")
    except Exception:
        pass  # If logging fails, continue without it

# Set Windows AppUserModelID so the taskbar shows our icon (not the Python icon)
try:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('hokeiiiching.TeyvatTranslator.1.0')
except Exception:
    pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from src.ui.main_window import MainWindow


def load_stylesheet(app: QApplication) -> None:
    """
    Load the application stylesheet from the styles directory.
    
    Args:
        app: The QApplication instance to apply styles to.
    """
    style_path = os.path.join(APP_DIR, "src", "ui", "styles.qss")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            print("Stylesheet loaded successfully")
    else:
        print("Stylesheet not found, using default theme")


def set_app_icon(app: QApplication) -> None:
    """
    Set the application window icon.
    
    Args:
        app: The QApplication instance.
    """
    # Try PNG first, then ICO as fallback
    icon_path = os.path.join(APP_DIR, "assets", "icon.png")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(APP_DIR, "assets", "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))


def main() -> None:
    """
    Application entry point.
    
    Initializes the PyQt6 application, loads styling, and displays the main window.
    """
    # Create application instance
    app = QApplication(sys.argv)
    app.setApplicationName("Genshin Translator")
    app.setApplicationVersion("1.1.0")
    app.setOrganizationName("GenshinTranslator")
    
    # Start OCR pre-initialization in background (reduces wait time on first translation)
    from src.engine.ocr import preload_ocr
    from src.engine.rag import preload_rag
    from src.engine.translator import preload_translator
    preload_ocr()
    preload_rag()
    preload_translator()
    
    # Configure application
    set_app_icon(app)
    load_stylesheet(app)
    
    # Create and show main window directly
    main_window = MainWindow()
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
