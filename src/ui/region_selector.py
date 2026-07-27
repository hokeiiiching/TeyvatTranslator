# -*- coding: utf-8 -*-
"""
Region Selector Module

Provides a fullscreen overlay for selecting a rectangular screen region.
The selected coordinates are used for OCR capture.
"""

import logging

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QCursor

from src.diagnostics import record_pipeline_event


logger = logging.getLogger("RegionSelector")


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
    selection_cancelled = pyqtSignal()
    
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
        
        self._apply_virtual_desktop_geometry()

    def _apply_virtual_desktop_geometry(self) -> QRect:
        """Cover the union of every screen, including negative monitor offsets."""
        screens = QApplication.screens()
        if not screens:
            geometry = QApplication.primaryScreen().geometry()
        else:
            geometry = QRect(screens[0].geometry())
            for screen in screens[1:]:
                geometry = geometry.united(screen.geometry())
        self.setGeometry(geometry)
        logger.info(
            "region_selector_geometry virtual_desktop=%r screens=%r",
            (geometry.x(), geometry.y(), geometry.width(), geometry.height()),
            [
                (
                    screen.name(),
                    screen.geometry().x(),
                    screen.geometry().y(),
                    screen.geometry().width(),
                    screen.geometry().height(),
                    screen.devicePixelRatio(),
                )
                for screen in screens
            ],
        )
        return geometry
        
    def start_selection(self) -> None:
        """Begin the region selection process."""
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        
        geometry = self._apply_virtual_desktop_geometry()
        logger.info(
            "region_selection_started overlay_geometry=%r",
            (geometry.x(), geometry.y(), geometry.width(), geometry.height()),
        )
        record_pipeline_event(
            "region_selector",
            "selection_started",
            overlay_geometry=[
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            ],
        )

        # Change cursor to crosshair
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.show()
        self.raise_()
        self.activateWindow()
        
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
            global_begin = self.mapToGlobal(self.begin)
            logger.debug(
                "region_drag_started widget=%r global=%r",
                (self.begin.x(), self.begin.y()),
                (global_begin.x(), global_begin.y()),
            )
            self.update()
            
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement during drag."""
        if self.is_selecting:
            self.end = event.pos()
            self.update()
            
    def _to_physical_coordinates(self, lx1: int, ly1: int, lx2: int, ly2: int) -> tuple[int, int, int, int]:
        """Convert Qt logical screen coordinates to Windows physical screen pixels based on screen DPI scaling."""
        screen = QApplication.screenAt(QPoint(lx1, ly1)) or QApplication.primaryScreen()
        if not screen:
            return lx1, ly1, lx2, ly2
            
        dpr = screen.devicePixelRatio()
        sg = screen.geometry()  # Qt logical geometry of screen
        
        # Primary screen (origin 0,0)
        if sg.x() == 0 and sg.y() == 0:
            px1 = int(round(lx1 * dpr))
            py1 = int(round(ly1 * dpr))
            px2 = int(round(lx2 * dpr))
            py2 = int(round(ly2 * dpr))
        else:
            # Secondary screen
            rel_x1 = lx1 - sg.x()
            rel_y1 = ly1 - sg.y()
            rel_x2 = lx2 - sg.x()
            rel_y2 = ly2 - sg.y()
            
            primary = QApplication.primaryScreen()
            primary_phys_w = int(round(primary.geometry().width() * primary.devicePixelRatio())) if primary else 0
            phys_origin_x = primary_phys_w if sg.x() >= (primary.geometry().width() if primary else 0) else int(round(sg.x() * dpr))
            phys_origin_y = int(round(sg.y() * dpr))
            
            px1 = phys_origin_x + int(round(rel_x1 * dpr))
            py1 = phys_origin_y + int(round(rel_y1 * dpr))
            px2 = phys_origin_x + int(round(rel_x2 * dpr))
            py2 = phys_origin_y + int(round(rel_y2 * dpr))

        return px1, py1, px2, py2

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse button release."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.end = event.pos()
            self.is_selecting = False
            
            global_begin = self.mapToGlobal(self.begin)
            global_end = self.mapToGlobal(self.end)
            lx1 = min(global_begin.x(), global_end.x())
            ly1 = min(global_begin.y(), global_end.y())
            lx2 = max(global_begin.x(), global_end.x())
            ly2 = max(global_begin.y(), global_end.y())

            px1, py1, px2, py2 = self._to_physical_coordinates(lx1, ly1, lx2, ly2)

            logger.info(
                "region_drag_finished widget_start=%r widget_end=%r "
                "logical_bbox=%r physical_bbox=%r size=%sx%s valid=%s",
                (self.begin.x(), self.begin.y()),
                (self.end.x(), self.end.y()),
                (lx1, ly1, lx2, ly2),
                (px1, py1, px2, py2),
                px2 - px1,
                py2 - py1,
                px2 - px1 > 10 and py2 - py1 > 10,
            )
            record_pipeline_event(
                "region_selector",
                "selection_finished",
                widget_start=[self.begin.x(), self.begin.y()],
                widget_end=[self.end.x(), self.end.y()],
                logical_bbox=[lx1, ly1, lx2, ly2],
                global_bbox=[px1, py1, px2, py2],
                width=px2 - px1,
                height=py2 - py1,
                valid=px2 - px1 > 10 and py2 - py1 > 10,
            )
            
            # Only emit signal if region is large enough
            if px2 - px1 > 10 and py2 - py1 > 10:
                self.region_selected.emit(px1, py1, px2, py2)
            else:
                logger.warning("region_selection_rejected reason=too-small")
                self.selection_cancelled.emit()
            
            self._finish_selection()
            
    def keyPressEvent(self, event) -> None:
        """Handle keyboard input."""
        if event.key() == Qt.Key.Key_Escape:
            logger.info("region_selection_cancelled reason=escape")
            record_pipeline_event(
                "region_selector",
                "selection_cancelled",
                reason="escape",
            )
            self._finish_selection()
            self.selection_cancelled.emit()
                
    def _finish_selection(self) -> None:
        """Clean up after selection completes or is cancelled."""
        QApplication.restoreOverrideCursor()
        self.hide()
