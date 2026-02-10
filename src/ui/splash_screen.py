# -*- coding: utf-8 -*-
"""
Splash Screen Module

Displays an animated splash screen with Genshin-inspired branding
during application startup.
"""

import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap


class SplashScreen(QWidget):
    """
    Animated splash screen shown on application startup.
    
    Displays the application logo, name, and tagline while the
    main window initializes in the background. Automatically
    transitions to the main window after a brief delay.
    
    Attributes:
        main_window: Reference to the main application window.
    """
    
    def __init__(self, main_window: QWidget) -> None:
        """
        Initialize the splash screen.
        
        Args:
            main_window: The main window to show after splash closes.
        """
        super().__init__()
        self.main_window = main_window
        self._setup_window()
        self._create_layout()
        
    def _setup_window(self) -> None:
        """Configure window properties for splash screen appearance."""
        # Frameless, always-on-top window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 320)
        
        # Center on screen
        screen = self.screen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
    def _create_layout(self) -> None:
        """Create and configure the splash screen layout."""
        # Container widget with styled background (Genshin theme)
        container = QWidget(self)
        container.setObjectName("splash_container")
        container.setStyleSheet("""
            #splash_container {
                background-color: #0f1419;
                border: 2px solid rgba(201, 169, 98, 0.6);
                border-radius: 16px;
            }
        """)
        container.setGeometry(0, 0, 400, 320)
        
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)
        
        # Logo image
        logo_label = QLabel()
        from src.paths import ASSETS_DIR
        logo_path = os.path.join(ASSETS_DIR, "icon.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale to reasonable size while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(180, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback to text if image not found
            logo_label.setText("原神")
            logo_label.setFont(QFont("HYWenHei-85W", 48, QFont.Weight.Bold))
            logo_label.setStyleSheet("color: #c9a962;")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)
        
        # English subtitle - Mondstadt Blue
        subtitle = QLabel("Genshin Translator")
        subtitle.setFont(QFont("HYWenHei-85W", 16))
        subtitle.setStyleSheet("color: #7eb8c9;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Tagline
        tagline = QLabel("Learn Mandarin through Genshin Impact")
        tagline.setFont(QFont("HYWenHei-85W", 10))
        tagline.setStyleSheet("color: #a0a8b3;")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tagline)
        
        # Loading indicator
        self.loading_label = QLabel("Loading...")
        self.loading_label.setFont(QFont("HYWenHei-85W", 9))
        self.loading_label.setStyleSheet("color: #6b7280;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        
    def finish(self) -> None:
        """Close splash screen and display the main window."""
        self.close()
        self.main_window.show()

