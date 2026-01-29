# -*- coding: utf-8 -*-
"""
Region Selector Module

Provides a fullscreen overlay for selecting a rectangular screen region.
The selected coordinates are used for OCR capture.
"""

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QCursor


class RegionSelector(QWidget):
    """
    Fullscreen overlay for selecting a screen region.
    
    Displays a semi-transparent overlay across the entire screen.
    Users can click and drag to select a rectangular area.
    
    Signals:
        region_selected: Emitted when a valid region is selected.
            Arguments: (x1, y1, x2, y2) coordinates.
    
    Attributes:
        begin: Starting point of selection
        end: Ending point of selection
        is_selecting: Whether user is currently dragging
    """
    
    region_selected = pyqtSignal(int, int, int, int)
    
    def __init__(self, parent=None) -> None:
        """
        Initialize the region selector.
        
        Args:
            parent: Parent widget (usually MainWindow).
        """
        super().__init__(parent)
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        self.main_window = None  # Reference to main window for callbacks
        
        self._setup_window()
        
    def _setup_window(self) -> None:
        """Configure window for fullscreen transparent overlay."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Cover entire primary screen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
    def start_selection(self) -> None:
        """Begin the region selection process."""
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        
        # Change cursor to crosshair
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.show()
        
    def paintEvent(self, event) -> None:
        """
        Draw the selection overlay.
        
        Args:
            event: Paint event.
        """
        painter = QPainter(self)
        
        # Dark semi-transparent background
        painter.fillRect(self.rect(), QColor(10, 10, 18, 180))
        
        if self.is_selecting and not self.begin.isNull() and not self.end.isNull():
            rect = QRect(self.begin, self.end).normalized()
            
            # Clear the selected area to show screen content
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear
            )
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            
            # Draw selection border
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            pen = QPen(QColor(106, 74, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            
            # Draw corner indicators for visual polish
            self._draw_corners(painter, rect)
            
            # Draw dimension label
            width = abs(rect.width())
            height = abs(rect.height())
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect.left() + 5, rect.top() - 8, f"{width} × {height}")
        
        # Instructions text
        painter.setPen(QColor(160, 160, 176))
        painter.drawText(20, 30, "Click and drag to select a region. Press ESC to cancel.")
        
    def _draw_corners(self, painter: QPainter, rect: QRect) -> None:
        """
        Draw decorative corner indicators.
        
        Args:
            painter: QPainter instance
            rect: The selection rectangle
        """
        corner_size = 10
        pen = QPen(QColor(125, 211, 252), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        # Top-left corner
        painter.drawLine(rect.left(), rect.top(), 
                        rect.left() + corner_size, rect.top())
        painter.drawLine(rect.left(), rect.top(), 
                        rect.left(), rect.top() + corner_size)
        
        # Top-right corner
        painter.drawLine(rect.right(), rect.top(), 
                        rect.right() - corner_size, rect.top())
        painter.drawLine(rect.right(), rect.top(), 
                        rect.right(), rect.top() + corner_size)
        
        # Bottom-left corner
        painter.drawLine(rect.left(), rect.bottom(), 
                        rect.left() + corner_size, rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), 
                        rect.left(), rect.bottom() - corner_size)
        
        # Bottom-right corner
        painter.drawLine(rect.right(), rect.bottom(), 
                        rect.right() - corner_size, rect.bottom())
        painter.drawLine(rect.right(), rect.bottom(), 
                        rect.right(), rect.bottom() - corner_size)
        
    def mousePressEvent(self, event) -> None:
        """Handle mouse button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.pos()
            self.end = self.begin
            self.is_selecting = True
            self.update()
            
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement during drag."""
        if self.is_selecting:
            self.end = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse button release."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.end = event.pos()
            self.is_selecting = False
            
            # Get screen geometry offset (important for multi-monitor setups)
            screen = QApplication.primaryScreen().geometry()
            screen_offset_x = screen.x()
            screen_offset_y = screen.y()
            
            # Convert widget-relative coordinates to global screen coordinates
            # The overlay widget starts at screen position (0,0) relative to primary screen
            x1 = min(self.begin.x(), self.end.x()) + screen_offset_x
            y1 = min(self.begin.y(), self.end.y()) + screen_offset_y
            x2 = max(self.begin.x(), self.end.x()) + screen_offset_x
            y2 = max(self.begin.y(), self.end.y()) + screen_offset_y
            
            # Debug output
            print(f"\n{'='*50}")
            print(f"REGION SELECTED")
            print(f"{'='*50}")
            print(f"  Screen offset: ({screen_offset_x}, {screen_offset_y})")
            print(f"  Widget coords: ({self.begin.x()}, {self.begin.y()}) to ({self.end.x()}, {self.end.y()})")
            print(f"  Global coords: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Size: {x2-x1}x{y2-y1} pixels")
            
            # Only emit signal if region is large enough
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.region_selected.emit(x1, y1, x2, y2)
            
            self._finish_selection()
            
    def keyPressEvent(self, event) -> None:
        """Handle keyboard input."""
        if event.key() == Qt.Key.Key_Escape:
            self._finish_selection()
            # Show main window if available (stored as attribute)
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.show()
                
    def _finish_selection(self) -> None:
        """Clean up after selection completes or is cancelled."""
        QApplication.restoreOverrideCursor()
        self.hide()
