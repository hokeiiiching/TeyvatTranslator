# -*- coding: utf-8 -*-
"""
Main Window Module

The primary application window featuring a tabbed interface with:
- Translate: Language selection and capture controls
- Settings: Display and behavior customization
- Guide: Usage instructions
- About: Application information
"""

import ctypes
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox,
    QSlider, QCheckBox, QMessageBox, QSizePolicy, QButtonGroup
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .region_selector import RegionSelector
from .translate_window import TranslateWindow
from src.engine.language_config import (
    SOURCE_LANGUAGE_OPTIONS,
    TraditionalChineseSupportError,
    get_source_label,
    validate_source_language_support,
)
from src.engine.ocr import OCRWorker


# Language code mappings for OCR and translation
OCR_LANGUAGE_CODES = {
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Japanese": "jpn",
    "Korean": "kor",
    "English": "eng",
}

TRANSLATE_LANGUAGE_CODES = {
    "English": "eng",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Japanese": "jpn",
    "Korean": "kor",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
}


class MainWindow(QMainWindow):
    """
    Main application window with tabbed interface.
    
    Provides the primary user interface for configuring and starting
    translation sessions. Contains tabs for translation control,
    settings, usage guide, and application info.
    
    Attributes:
        region_selector: Widget for screen region selection
        translate_window: Overlay window for displaying translations
        ocr_worker: Background thread for OCR processing
        selected_region: Tuple of (x1, y1, x2, y2) coordinates
    """
    
    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        
        # State variables
        self.region_selector: Optional[RegionSelector] = None
        self.translate_window: Optional[TranslateWindow] = None
        self.ocr_worker: Optional[OCRWorker] = None
        self.selected_region: Optional[Tuple[int, int, int, int]] = None
        
        self._setup_window()
        self._create_ui()
        self._setup_region_selector()
        
    def _setup_window(self) -> None:
        """Configure window properties."""
        # Set Windows app ID for taskbar grouping
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'GenshinTranslator.v1.0'
            )
        except AttributeError:
            pass  # Not on Windows
            
        self.setWindowTitle("Genshin Translator")
        self.setMinimumSize(680, 620)
        self.resize(720, 650)
        
    def _create_ui(self) -> None:
        """Create the main user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Add stretch to center content
        main_layout.addStretch(1)
        
        # Create container with max width for content
        content_container = QWidget()
        content_container.setMaximumWidth(800)
        content_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabbed interface
        tabs = QTabWidget()
        tabs.addTab(self._create_translate_tab(), "Translate")
        tabs.addTab(self._create_settings_tab(), "Settings")
        tabs.addTab(self._create_guide_tab(), "Guide")
        tabs.addTab(self._create_about_tab(), "About")
        content_layout.addWidget(tabs)
        
        main_layout.addWidget(content_container, 0)
        
        # Add stretch to center content
        main_layout.addStretch(1)
        
    def _setup_region_selector(self) -> None:
        """Initialize the screen region selector."""
        # Create without parent so it's independent of main window visibility
        self.region_selector = RegionSelector()
        self.region_selector.main_window = self  # Store reference for callbacks
        self.region_selector.region_selected.connect(self._on_region_selected)
        
    def _create_translate_tab(self) -> QWidget:
        """
        Create the translation control tab.
        
        Returns:
            QWidget containing translation controls.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)
        
        # Title
        title = QLabel("Language")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("HYWenHei-85W", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(10)
        source_label = QLabel("Chinese Script")
        source_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        source_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        lang_layout.addWidget(source_label)

        self.source_lang_group = QButtonGroup(self)
        self.source_lang_group.setExclusive(True)
        self.source_lang_buttons = {}

        for index, (label, code) in enumerate(SOURCE_LANGUAGE_OPTIONS):
            button = QPushButton(label.replace("Chinese (", "").replace(")", ""))
            button.setCheckable(True)
            button.setProperty("source_lang", code)
            button.setMinimumHeight(34)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(f"Use {label} OCR and translation support")
            button.setStyleSheet(
                "QPushButton {"
                "border: 1px solid #4a4a68;"
                "border-radius: 6px;"
                "padding: 6px 14px;"
                "color: #d8d8e8;"
                "background: #252536;"
                "}"
                "QPushButton:checked {"
                "border-color: #c9a962;"
                "color: #ffffff;"
                "background: #3a3424;"
                "}"
            )
            button.clicked.connect(self._update_language_info)
            self.source_lang_group.addButton(button)
            self.source_lang_buttons[code] = button
            lang_layout.addWidget(button)
            if index == 0:
                button.setChecked(True)

        layout.addLayout(lang_layout)
        
        # Language info
        lang_info = QLabel("")
        lang_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lang_info.setFont(QFont("HYWenHei-85W", 12))
        lang_info.setStyleSheet("color: #c9a962; padding: 2px 0;")
        layout.addWidget(lang_info)
        self.lang_info = lang_info
        self._update_language_info()
        
        # ===== CAPTURE METHOD SECTION =====
        method_label = QLabel("Step 1: Choose Capture Method")
        method_label.setFont(QFont("HYWenHei-85W", 12, QFont.Weight.Bold))
        method_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(method_label)
        
        help_text = QLabel(
            "Option A (Recommended): Select Genshin from dropdown below\n"
            "Option B (Advanced): Use 'Select Region' to draw a box on screen"
        )
        help_text.setMinimumHeight(46)
        help_text.setStyleSheet("color: #8080a0; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(help_text)
        
        # Window selection section - responsive layout
        window_layout = QHBoxLayout()
        window_layout.setSpacing(12)
        
        # Window dropdown - fills available space
        self.window_combo = QComboBox()
        self.window_combo.addItem("-- Select a window --")
        self.window_combo.setMinimumHeight(44)
        self.window_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        window_layout.addWidget(self.window_combo, 3)  # Takes 3 parts of space
        
        # Auto-detect Genshin button
        self.detect_btn = QPushButton("Find Genshin")
        self.detect_btn.setMinimumHeight(44)
        self.detect_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.detect_btn.clicked.connect(self._on_find_genshin)
        window_layout.addWidget(self.detect_btn, 1)  # Takes 1 part
        
        # Refresh windows button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMinimumHeight(44)
        self.refresh_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.refresh_btn.setToolTip("Refresh window list")
        self.refresh_btn.clicked.connect(self._refresh_window_list)
        window_layout.addWidget(self.refresh_btn)  # Minimum space
        
        layout.addLayout(window_layout)
        
        # Dialogue-only option
        self.dialogue_checkbox = QCheckBox("Dialogue Only (Recommended)")
        self.dialogue_checkbox.setChecked(True)
        self.dialogue_checkbox.setToolTip(
            "When checked, only captures the dialogue/subtitle area at the bottom of the screen.\n"
            "Recommended for translating story dialogue. Uncheck to capture full window."
        )
        self.dialogue_checkbox.setStyleSheet("margin-top: 8px; color: #7dd3fc;")
        layout.addWidget(self.dialogue_checkbox)
        
        # Description for dialogue mode
        dialogue_desc = QLabel("Captures bottom subtitle area only")
        dialogue_desc.setStyleSheet("color: #8080a0; font-size: 11px; margin-left: 28px;")
        layout.addWidget(dialogue_desc)
        
        # Populate window list
        self._refresh_window_list()
        
        # Spacer before action buttons
        layout.addSpacing(16)
        
        # Action buttons - equal width, fill available space
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)
        
        # Region selection button
        self.select_btn = QPushButton("Select Region")
        self.select_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.select_btn.setMinimumHeight(48)
        self.select_btn.setToolTip("Alternative: Draw a box around text on screen")
        self.select_btn.clicked.connect(self._on_select_region)
        btn_layout.addWidget(self.select_btn, 1)  # Equal stretch
        
        # Start translation button
        self.translate_btn = QPushButton("Start Translation")
        self.translate_btn.setObjectName("primary")
        self.translate_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.translate_btn.setMinimumHeight(48)
        self.translate_btn.clicked.connect(self._on_start_translation)
        btn_layout.addWidget(self.translate_btn, 2)  # Larger - primary action
        
        layout.addLayout(btn_layout)
        
        # Status indicator
        self.status_label = QLabel("Click 'Find Genshin' or select a window to begin")
        self.status_label.setObjectName("subtitle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("margin-top: 12px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch(1)
        return widget

    def _selected_source_lang(self) -> str:
        """Return the selected source language code."""
        if not hasattr(self, "source_lang_group"):
            return "chi_sim"
        checked_button = self.source_lang_group.checkedButton()
        if checked_button is None:
            return "chi_sim"
        return checked_button.property("source_lang") or "chi_sim"

    def _update_language_info(self) -> None:
        """Update the displayed language direction."""
        if not hasattr(self, "lang_info"):
            return
        source_lang = self._selected_source_lang()
        self.lang_info.setText(f"{get_source_label(source_lang)}  ->  English")
        if source_lang == "chi_tra":
            from src.engine.ocr import preload_ocr

            preload_ocr(source_lang)
    
    def _refresh_window_list(self) -> None:
        """Refresh the list of available windows."""
        try:
            from src.engine.window_capture import get_window_list
            
            self.window_combo.clear()
            self.window_combo.addItem("(Screen Region - select below)")
            
            windows = get_window_list()
            self._windows_cache = {w['title']: w for w in windows}
            
            for w in windows:
                # Truncate long titles
                title = w['title'][:60] + "..." if len(w['title']) > 60 else w['title']
                self.window_combo.addItem(f"{title} ({w['size'][0]}x{w['size'][1]})")
                
        except Exception as e:
            print(f"Could not list windows: {e}")
            self._windows_cache = {}
    
    def _on_find_genshin(self) -> None:
        """Auto-detect and select Genshin Impact window."""
        try:
            from src.engine.window_capture import find_genshin_window
            
            genshin = find_genshin_window()
            if genshin:
                # First refresh the window list to ensure it's up to date
                self._refresh_window_list()
                
                # Find and select in dropdown
                print(f"DEBUG: Looking for '{genshin['title']}' in dropdown...")
                for i in range(self.window_combo.count()):
                    item_text = self.window_combo.itemText(i)
                    print(f"  [{i}] '{item_text}'")
                    if genshin['title'] in item_text:
                        self.window_combo.setCurrentIndex(i)
                        self.status_label.setText(f"Found: {genshin['title']}")
                        self.status_label.setStyleSheet("color: #4ade80;")
                        return
            
            QMessageBox.information(
                self,
                "Genshin Not Found",
                "Could not find Genshin Impact window.\n\n"
                "Make sure the game is running and try again."
            )
        except Exception as e:
            print(f"Error finding Genshin: {e}")
        
    def _create_settings_tab(self) -> QWidget:
        """
        Create the settings configuration tab.
        
        Returns:
            QWidget containing settings controls.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Section title
        title = QLabel("Display Settings")
        title.setObjectName("title")
        title.setFont(QFont("HYWenHei-85W", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Customize how translations are displayed in the overlay window.")
        desc.setStyleSheet("color: #8080a0; margin-bottom: 16px;")
        layout.addWidget(desc)
        
        # Font size selector
        size_layout = QHBoxLayout()
        size_layout.setSpacing(16)
        size_label = QLabel("Translation Font Size:")
        size_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.size_combo = QComboBox()
        self.size_combo.addItems([str(i) for i in range(10, 36)])
        self.size_combo.setCurrentText("14")
        self.size_combo.setToolTip("Font size for English translation text")
        self.size_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_combo, 1)
        layout.addLayout(size_layout)
        
        # Overlay opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.setSpacing(16)
        opacity_label = QLabel("Overlay Transparency:")
        opacity_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.setToolTip("How transparent the translation overlay appears (lower = more transparent)")
        self.opacity_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.opacity_value_label = QLabel("70%")
        self.opacity_value_label.setMinimumWidth(45)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_value_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider, 1)
        opacity_layout.addWidget(self.opacity_value_label)
        layout.addLayout(opacity_layout)
        
        layout.addStretch(1)
        return widget
        
    def _create_guide_tab(self) -> QWidget:
        """
        Create the usage guide tab.
        
        Returns:
            QWidget containing usage instructions.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        title = QLabel("Quick Start Guide")
        title.setObjectName("title")
        title.setFont(QFont("HYWenHei-85W", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        guide_text = QLabel("""
<div style="color: #e0e0f0; line-height: 1.7;">

<p style="color: #c9a962; font-weight: bold; font-size: 14px;">Getting Started</p>
<ol>
<li><b>Launch Genshin Impact</b> in windowed or borderless mode</li>
<li>Click <span style="color: #4ade80;">Find Genshin</span> to auto-detect the game</li>
<li>Keep <b>"Dialogue Only"</b> checked for best performance</li>
<li>Click <span style="color: #7dd3fc;">Start Translation</span></li>
</ol>

<p style="color: #c9a962; font-weight: bold; font-size: 14px; margin-top: 15px;">What You'll See</p>
<ul>
<li><b style="color: #c9a962;">Speaker Name</b> — Character speaking (in Chinese, Pinyin, English)</li>
<li><b style="color: #7eb8c9;">Pinyin</b> — Pronunciation guide above each character</li>
<li><b style="color: #e0e0e0;">Translation</b> — English translation of the dialogue</li>
</ul>

<p style="color: #c9a962; font-weight: bold; font-size: 14px; margin-top: 15px;">Tips</p>
<ul>
<li>Drag the overlay window to reposition it</li>
<li>Click on Chinese text to copy to clipboard</li>
<li>The overlay updates automatically when dialogue changes</li>
</ul>

</div>
        """)
        guide_text.setWordWrap(True)
        guide_text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(guide_text)
        
        layout.addStretch(1)
        return widget
        
    def _create_about_tab(self) -> QWidget:
        """
        Create the about information tab.
        
        Returns:
            QWidget containing application info.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)
        
        # Chinese logo
        logo = QLabel("原神翻译器")
        logo.setFont(QFont("HYWenHei-85W", 32, QFont.Weight.Bold))
        logo.setStyleSheet("color: #c9a962;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)
        
        # Application name in English
        name = QLabel("Genshin Translator")
        name.setFont(QFont("Segoe UI", 16))
        name.setStyleSheet("color: #e0e0e0;")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name)
        
        # Version
        version = QLabel("v1.0.0")
        version.setObjectName("subtitle")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)
        
        # Separator
        sep = QLabel("─────────────")
        sep.setStyleSheet("color: rgba(201, 169, 98, 0.3);")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sep)
        
        # Description
        desc = QLabel("""
<div style="color: #a0a0b0; line-height: 1.8;">
<p style="text-align: center;">Learn Mandarin Chinese while playing Genshin Impact!</p>
<p style="margin-top: 16px;"><b style="color: #c9a962;">Features:</b></p>
<ul style="margin-left: 0; padding-left: 20px;">
<li>Live OCR screen capture with PaddleOCR</li>
<li>150+ Genshin-specific vocabulary terms</li>
<li>Pinyin pronunciation for every character</li>
<li>Character name recognition and translation</li>
<li>Google Translate fallback for unknown text</li>
</ul>
</div>
        """)
        desc.setTextFormat(Qt.TextFormat.RichText)
        desc.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(desc)
        
        # Credits
        credits = QLabel("""
<div style="margin-top: 20px;">
<p style="color: #505060; font-size: 11px; text-align: center;">Powered by PaddleOCR • Sentence Transformers • ChromaDB</p>
</div>
        """)
        credits.setTextFormat(Qt.TextFormat.RichText)
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(credits)
        
        layout.addStretch(1)
        return widget
        
    # ===== Event Handlers =====
    
    def _on_select_region(self) -> None:
        """Handle region selection button click."""
        self.hide()
        self.region_selector.start_selection()
        
    def _on_region_selected(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Handle successful region selection.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
        """
        self.selected_region = (x1, y1, x2, y2)
        self.status_label.setText(f"Region selected: {x2-x1}×{y2-y1} pixels")
        self.status_label.setStyleSheet("color: #4ade80;")
        self.show()
        
    def _on_start_translation(self) -> None:
        """Handle start translation button click."""
        # Check if window selected or region selected
        selected_window = None
        window_idx = self.window_combo.currentIndex()
        
        if window_idx > 0:  # A window is selected (not first item)
            # Get window from cache
            window_text = self.window_combo.currentText()
            for title, info in getattr(self, '_windows_cache', {}).items():
                if title in window_text:
                    selected_window = info
                    break
        
        # Require either window or region selection
        if not selected_window and not self.selected_region:
            QMessageBox.warning(
                self, 
                "No Target Selected",
                "Please either:\n\n"
                "1. Select a window from the dropdown, or\n"
                "2. Click 'Select Region' to select a screen area"
            )
            return
            
        # Source language comes from the selected Chinese script.
        from_lang = self._selected_source_lang()
        to_lang = "eng"

        try:
            validate_source_language_support(from_lang)
        except TraditionalChineseSupportError as exc:
            QMessageBox.critical(
                self,
                "Traditional Chinese Support Unavailable",
                str(exc),
            )
            return
        
        # Close existing translate window if open
        if self.translate_window:
            self.translate_window.close()
            
        # Create new translate window
        self.translate_window = TranslateWindow(
            opacity=self.opacity_slider.value() / 100,
            font_size=int(self.size_combo.currentText()),
            enable_tts=False,
            enable_context=True
        )
        self.translate_window.show()
        
        # Start OCR worker thread
        if selected_window:
            # Window-based capture mode
            # For window capture, we pass the window handle and use full window
            # The OCR worker will need to support window capture mode
            hwnd = selected_window['hwnd']
            rect = selected_window['rect']
            
            print(f"\nUsing WINDOW capture mode")
            print(f"  Window: {selected_window['title']}")
            print(f"  Handle: {hwnd}")
            print(f"  Size: {selected_window['size']}")
            print(f"  Source language: {get_source_label(from_lang)}")
            
            # Check if dialogue-only mode
            dialogue_only = self.dialogue_checkbox.isChecked()
            if dialogue_only:
                print(f"  Mode: DIALOGUE ONLY (bottom subtitle area)")
            else:
                print(f"  Mode: FULL WINDOW")
            
            # Create OCR worker with window handle
            self.ocr_worker = OCRWorker(
                0, 0, selected_window['size'][0], selected_window['size'][1],
                from_lang, to_lang,
                self.translate_window,
                False,
                True,
                window_hwnd=hwnd,  # Pass window handle
                dialogue_only=dialogue_only  # Pass dialogue mode
            )
            
            mode_str = "dialogue" if dialogue_only else "full window"
            self.status_label.setText(
                f"Translating {get_source_label(from_lang)} {mode_str}: "
                f"{selected_window['title'][:25]}..."
            )
        else:
            # Screen region capture mode
            x1, y1, x2, y2 = self.selected_region
            
            print(f"\nUsing SCREEN REGION capture mode")
            print(f"  Region: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Source language: {get_source_label(from_lang)}")
            
            self.ocr_worker = OCRWorker(
                x1, y1, x2, y2,
                from_lang, to_lang,
                self.translate_window,
                False,
                True
            )
            
            self.status_label.setText(f"Translation active: {get_source_label(from_lang)}")
        
        self.translate_window.set_worker(self.ocr_worker)
        self.ocr_worker.start()
        self.status_label.setStyleSheet("color: #7dd3fc;")
        
    def closeEvent(self, event) -> None:
        """
        Clean up resources when window is closed.
        
        Args:
            event: The close event.
        """
        if self.ocr_worker:
            self.ocr_worker.stop()
        if self.translate_window:
            self.translate_window.close()
        event.accept()
