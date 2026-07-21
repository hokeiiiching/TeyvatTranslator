# -*- coding: utf-8 -*-
"""
Translation Window Module

Provides the floating overlay window that displays OCR results
and translations with context-aware vocabulary information.
"""

from typing import Dict, List, Optional, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QScrollArea, QPushButton, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QMetaObject, Q_ARG, Qt as QtCore, QPoint
from PyQt6.QtGui import QFont, QCursor
import pyperclip


CHINESE_HTML_FONT_FAMILY = (
    "HYWenHei-85W, Microsoft JhengHei, Microsoft YaHei, Noto Sans CJK TC"
)


class TranslateWindow(QMainWindow):
    """
    Floating overlay window for displaying translations.
    
    Shows the captured Chinese text, pinyin pronunciation,
    English translation, and optional context information
    including character breakdowns and usage examples.
    
    Attributes:
        worker: Reference to the OCR worker thread
        font_size: Display font size for translations
        enable_tts: Whether text-to-speech is enabled
        enable_context: Whether to show vocabulary context
    """
    
    def __init__(
        self, 
        opacity: float = 0.95, 
        font_size: int = 14, 
        enable_tts: bool = False, 
        enable_context: bool = True
    ) -> None:
        """
        Initialize the translation window.
        
        Args:
            opacity: Window opacity (0.0 to 1.0)
            font_size: Font size for English translation
            enable_tts: Enable text-to-speech
            enable_context: Enable vocabulary context display
        """
        super().__init__()
        self.worker = None
        self.font_size = font_size
        self.enable_tts = enable_tts
        self.enable_context = enable_context
        self._drag_pos = None  # For frameless window dragging
        
        self._setup_window()
        self._create_ui()
        self.setWindowOpacity(opacity)
        
    def _setup_window(self) -> None:
        """Configure window properties with responsive sizing and top-right positioning."""
        self.setWindowTitle("Genshin Translator")
        self.setObjectName("translateWindow")
        
        # Frameless, always-on-top, tool window (doesn't show in taskbar)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool  # Prevents showing in taskbar
        )
        
        # Make background translucent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Responsive sizing based on screen dimensions
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.size()
            screen_geometry = screen.availableGeometry()
            
            # Window is 50% of screen width to fit long sentences, height is flexible
            # The window will auto-expand up to this max width
            width = max(600, int(screen_size.width() * 0.50))
            height = max(280, int(screen_size.height() * 0.35))  # Taller for stacked pinyin
            self.resize(width, height)
            self.setMinimumSize(500, 250)  # Increased min height for pinyin+char stacking
            self.setMaximumWidth(int(screen_size.width() * 0.65))  # Allow wider expansion
            self.setMaximumHeight(int(screen_size.height() * 0.60))  # Allow taller for long dialogues
            
            # Position in top-right corner with padding
            padding = 20
            x = screen_geometry.right() - width - padding
            y = screen_geometry.top() + padding
            self.move(x, y)
        else:
            # Fallback for when screen info unavailable
            self.setMinimumSize(500, 250)
            self.resize(700, 300)
        
    def _create_ui(self) -> None:
        """Create compact, translucent translation overlay with dynamic height."""
        central = QWidget()
        self.setCentralWidget(central)
        # Rounded corners with semi-transparent dark background
        central.setStyleSheet("""
            background-color: rgba(15, 20, 25, 0.85);
            border-radius: 12px;
            border: 1px solid rgba(201, 169, 98, 0.3);
        """)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(0)
        
        # === HEADER with close button ===
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 4, 0)  # Minimal margins, button near top-right
        
        # Drag hint (invisible but allows drag)
        drag_spacer = QWidget()
        drag_spacer.setSizePolicy(drag_spacer.sizePolicy().horizontalPolicy(), drag_spacer.sizePolicy().verticalPolicy())
        header_layout.addWidget(drag_spacer, 1)
        
        # Close button - perfect circle
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(239, 68, 68, 0.3);
                border: 1px solid rgba(239, 68, 68, 0.5);
                border-radius: 10px;
                color: #ef4444;
                font-size: 10px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.6);
                color: #ffffff;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        header_layout.addWidget(self.close_btn)
        
        layout.addLayout(header_layout)
        
        # === SCROLLABLE CONTENT AREA ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 4, 0)  # Right margin for scrollbar
        content_layout.setSpacing(6)
        
        # === SPEAKER SECTION ===
        self.speaker_label = QLabel("")
        self.speaker_label.setTextFormat(Qt.TextFormat.RichText)
        self.speaker_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.speaker_label.setWordWrap(True)
        self.speaker_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.speaker_label.setStyleSheet("margin-bottom: 4px; background: transparent;")
        content_layout.addWidget(self.speaker_label)
        
        # Thin separator
        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet("background-color: rgba(201, 169, 98, 0.2);")
        content_layout.addWidget(sep1)
        
        # === DIALOGUE SECTION (Pinyin + Chinese) ===
        # This section displays stacked pinyin above each Chinese character
        self.dialogue_label = QLabel("")
        self.dialogue_label.setTextFormat(Qt.TextFormat.RichText)
        self.dialogue_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.dialogue_label.setWordWrap(True)
        self.dialogue_label.setMinimumHeight(80)  # Minimum height for pinyin + character stacking
        self.dialogue_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.dialogue_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dialogue_label.setStyleSheet("background: transparent; padding: 8px 0;")
        self.dialogue_label.mousePressEvent = self._copy_chinese
        content_layout.addWidget(self.dialogue_label, 1)  # Give stretch factor of 1
        
        # Separator
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: rgba(201, 169, 98, 0.3);")
        content_layout.addWidget(sep2)
        
        # === ENGLISH TRANSLATION ===
        self.english_label = QLabel("Waiting for translation...")
        self.english_label.setFont(QFont("Segoe UI", self.font_size + 4))  # Much larger font
        self.english_label.setStyleSheet("color: #e0e0e0; background: transparent; padding: 20px 0; letter-spacing: 0.3px; line-height: 1.5;")
        self.english_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.english_label.setWordWrap(True)
        self.english_label.setMinimumHeight(80)  # Larger minimum height
        self.english_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.english_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.english_label.mousePressEvent = self._copy_english
        content_layout.addWidget(self.english_label, 1)  # Give it stretch factor too
        
        # No stretch at end - dialogue_label already has stretch factor of 1
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)  # Give scroll area stretch priority
        
        # Hidden storage labels for copy functions
        self.chinese_label = QLabel("")
        self.chinese_label.hide()
        self.pinyin_label = QLabel("")
        self.pinyin_label.hide()
        
        # Legacy compatibility
        self.combined_label = self.dialogue_label
        
    def _create_context_card(self) -> None:
        """Create the context information display card."""
        self.context_frame = QFrame()
        self.context_frame.setObjectName("card")
        self.context_frame.setStyleSheet("""
            #card {
                background-color: rgba(26, 32, 41, 0.9);
                border: 1px solid rgba(201, 169, 98, 0.2);
                border-radius: 10px;
                padding: 12px;
            }
        """)
        context_layout = QVBoxLayout(self.context_frame)
        context_layout.setSpacing(8)
        
        # Character breakdown
        self.breakdown_label = QLabel("")
        self.breakdown_label.setFont(QFont("HYWenHei-85W", 10))
        self.breakdown_label.setStyleSheet("color: #a0a0b0;")
        self.breakdown_label.setWordWrap(True)
        context_layout.addWidget(self.breakdown_label)
        
        # Game-specific meaning
        self.game_label = QLabel("")
        self.game_label.setFont(QFont("HYWenHei-85W", 10))
        self.game_label.setStyleSheet("color: #8080a0;")
        self.game_label.setWordWrap(True)
        context_layout.addWidget(self.game_label)
        
        # Real-world usage
        self.real_label = QLabel("")
        self.real_label.setFont(QFont("HYWenHei-85W", 10))
        self.real_label.setStyleSheet("color: #6080a0;")
        self.real_label.setWordWrap(True)
        context_layout.addWidget(self.real_label)
        
        self.context_frame.hide()  # Hidden until we have context
        self.content_layout.addWidget(self.context_frame)
        
    def set_worker(self, worker) -> None:
        """
        Set the OCR worker reference.
        
        Args:
            worker: The OCRWorker instance.
        """
        self.worker = worker

    def update_status(self, message: str) -> None:
        """Display OCR startup, ready, or failure state from the worker thread."""
        QMetaObject.invokeMethod(
            self.english_label, "setText",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, message)
        )
        
    def update_translation(
        self, 
        chinese: str, 
        english: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the displayed translation with speaker name and aligned pinyin.
        Thread-safe: can be called from worker thread.
        
        Args:
            chinese: The original Chinese text (may include speaker name as first line)
            english: The English translation
            context: Optional dict with 'pinyin', 'speaker', 'speaker_pinyin', 'speaker_english'
        """
        pinyin_text = context.get('pinyin', '') if context else ''
        speaker = context.get('speaker', '') if context else ''
        speaker_pinyin = context.get('speaker_pinyin', '') if context else ''
        speaker_english = context.get('speaker_english', '') if context else ''
        descriptor = context.get('descriptor', '') if context else ''
        descriptor_english = context.get('descriptor_english', '') if context else ''
        
        # === BUILD SPEAKER SECTION HTML ===
        # Display speaker info in stacked format: Mandarin, Pinyin, English, Descriptor (4 lines)
        if speaker:
            speaker_html = (
                f'<div style="margin-bottom: 10px;">'
                # Line 1: Mandarin name (large, golden, Genshin font)
                f'<div style="margin-bottom: 2px;">'
                f'<span style="color: #c9a962; font-size: 20px; font-family: {CHINESE_HTML_FONT_FAMILY}; font-weight: bold;">{speaker}</span>'
                f'</div>'
                # Line 2: Pinyin
                f'<div style="margin-bottom: 2px;">'
                f'<span style="color: #7eb8c9; font-size: 13px; font-family: Consolas;">{speaker_pinyin}</span>'
                f'</div>'
                # Line 3: English translation (on its own line)
                f'<div style="margin-bottom: 2px;">'
                f'<span style="color: #e0e0e0; font-size: 14px; font-style: italic;">{speaker_english if speaker_english else ""}</span>'
                f'</div>'
            )
            # Line 4-5: NPC Descriptor with translation (if present)
            if descriptor:
                speaker_html += (
                    f'<div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(201, 169, 98, 0.15);">'
                    # Descriptor Chinese (same size as dialogue)
                    f'<div style="margin-bottom: 2px;">'
                    f'<span style="color: #c9a962; font-size: 18px; font-family: {CHINESE_HTML_FONT_FAMILY};">{descriptor}</span>'
                    f'</div>'
                    # Descriptor English translation
                    f'<div>'
                    f'<span style="color: #a0a0a0; font-size: 13px; font-style: italic;">{descriptor_english if descriptor_english else ""}</span>'
                    f'</div>'
                    f'</div>'
                )
            speaker_html += '</div>'
        else:
            speaker_html = ''
        
        # === BUILD DIALOGUE HTML using TABLE layout (Qt properly supports tables) ===
        # Row 1: Pinyin syllables | Row 2: Chinese characters
        if chinese and pinyin_text:
            pinyin_parts = pinyin_text.split()
            pinyin_cells = []
            char_cells = []
            pinyin_idx = 0
            
            for char in chinese:
                if '\u4e00' <= char <= '\u9fff':  # Chinese character
                    py = pinyin_parts[pinyin_idx] if pinyin_idx < len(pinyin_parts) else ''
                    pinyin_idx += 1
                    pinyin_cells.append(f'<td align="center" style="padding: 0 2px;"><span style="font-size:13px; color:#7eb8c9; font-family:Consolas;">{py}</span></td>')
                    char_cells.append(f'<td align="center" style="padding: 0 2px;"><span style="font-size:24px; color:#c9a962; font-family:{CHINESE_HTML_FONT_FAMILY};">{char}</span></td>')
                else:
                    # Punctuation - empty pinyin cell, punctuation in char cell
                    pinyin_cells.append(f'<td align="center" style="padding: 0 1px;"><span style="font-size:13px;">&nbsp;</span></td>')
                    char_cells.append(f'<td align="center" style="padding: 0 1px;"><span style="font-size:24px; color:#c9a962;">{char}</span></td>')
            
            # Build table with 2 rows: pinyin on top, characters below
            dialogue_html = (
                f'<div style="padding: 10px 0;">'
                f'<table cellspacing="0" cellpadding="0" style="border-collapse:collapse;">'
                f'<tr style="line-height:1.3;">{"".join(pinyin_cells)}</tr>'
                f'<tr style="line-height:1.4;">{"".join(char_cells)}</tr>'
                f'</table>'
                f'</div>'
            )
        else:
            dialogue_html = f'<div style="color: #c9a962; font-size: 24px; font-family: {CHINESE_HTML_FONT_FAMILY}; padding: 10px 0;">{chinese or ""}</div>'
        
        # Store for copy function
        self.chinese_label.setText(chinese or "")
        self.pinyin_label.setText(pinyin_text)
        
        # Thread-safe UI updates
        QMetaObject.invokeMethod(
            self.speaker_label, "setText",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, speaker_html)
        )
        QMetaObject.invokeMethod(
            self.dialogue_label, "setText",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, dialogue_html)
        )
        QMetaObject.invokeMethod(
            self.english_label, "setText",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, english or "")
        )
    
    def _generate_ruby_html(self, chinese: str, pinyin: str) -> str:
        """
        Generate HTML with pinyin stacked above each character.
        Uses inline-block CSS since PyQt doesn't support ruby tags.
        
        Args:
            chinese: Chinese text
            pinyin: Space-separated pinyin syllables
            
        Returns:
            HTML string with stacked pinyin above characters
        """
        # Split pinyin into syllables
        pinyin_parts = pinyin.split()
        
        html_parts = ['<div style="line-height: 2.2;">']
        pinyin_idx = 0
        
        for char in chinese:
            if '\u4e00' <= char <= '\u9fff':  # Is Chinese character
                if pinyin_idx < len(pinyin_parts):
                    py = pinyin_parts[pinyin_idx]
                    # Stack: pinyin on top, character below
                    html_parts.append(
                        f'<span style="display:inline-block; text-align:center; margin:0 1px;">'
                        f'<span style="display:block; font-size:10px; color:#7eb8c9; line-height:1;">{py}</span>'
                        f'<span style="display:block; font-size:22px; color:#c9a962; line-height:1.2;">{char}</span>'
                        f'</span>'
                    )
                    pinyin_idx += 1
                else:
                    html_parts.append(f'<span style="font-size:22px; color:#c9a962;">{char}</span>')
            else:
                # Non-Chinese character (punctuation, etc.)
                html_parts.append(f'<span style="font-size:22px; color:#c9a962;">{char}</span>')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
            
    def _add_tag(self, tag_text: str) -> None:
        """
        Add a context tag to the display.
        
        Args:
            tag_text: The tag label text.
        """
        tag = QLabel(tag_text)
        tag.setFont(QFont("HYWenHei-85W", 9))
        # Apply tag-specific styling based on tag type
        tag_colors = {
            'combat': ('rgba(239, 122, 60, 0.15)', '#ef7a3c', 'rgba(239, 122, 60, 0.3)'),
            'dialogue': ('rgba(126, 184, 201, 0.15)', '#7eb8c9', 'rgba(126, 184, 201, 0.3)'),
            'ui': ('rgba(201, 169, 98, 0.15)', '#c9a962', 'rgba(201, 169, 98, 0.3)'),
            'location': ('rgba(116, 194, 168, 0.15)', '#74c2a8', 'rgba(116, 194, 168, 0.3)'),
            'character': ('rgba(184, 119, 217, 0.15)', '#b877d9', 'rgba(184, 119, 217, 0.3)'),
        }
        bg, color, border = tag_colors.get(tag_text.lower(), ('rgba(201, 169, 98, 0.15)', '#c9a962', 'rgba(201, 169, 98, 0.3)'))
        tag.setStyleSheet(f"""
            background-color: {bg};
            color: {color};
            padding: 4px 12px;
            border-radius: 12px;
            border: 1px solid {border};
        """)
        self.tags_layout.addWidget(tag)
        
    def _clear_tags(self) -> None:
        """Remove all displayed tags."""
        while self.tags_layout.count():
            child = self.tags_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _copy_chinese(self, event) -> None:
        """Copy Chinese text to clipboard when clicked."""
        text = self.chinese_label.text()
        if text and text != "等待翻译...":
            pyperclip.copy(text)
            
    def _copy_english(self, event) -> None:
        """Copy English text to clipboard when clicked."""
        text = self.english_label.text()
        if text and text != "Waiting for translation...":
            pyperclip.copy(text)
    
    # === FRAMELESS WINDOW DRAG SUPPORT ===
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for window dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for window dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release to end dragging."""
        self._drag_pos = None
        event.accept()
            
    def closeEvent(self, event) -> None:
        """
        Clean up when window is closed.
        
        Args:
            event: The close event.
        """
        if self.worker:
            self.worker.stop()
        event.accept()
