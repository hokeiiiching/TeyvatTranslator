# -*- coding: utf-8 -*-
"""
UI Package - PyQt6 User Interface Components

This package contains all graphical user interface components:
- MainWindow: Primary application window with tabs
- SplashScreen: Startup branding display
- RegionSelector: Screen area selection overlay
- TranslateWindow: Translation result overlay
"""

from .main_window import MainWindow
from .splash_screen import SplashScreen
from .region_selector import RegionSelector
from .translate_window import TranslateWindow

__all__ = ['MainWindow', 'SplashScreen', 'RegionSelector', 'TranslateWindow']
