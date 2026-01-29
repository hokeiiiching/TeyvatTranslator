# -*- coding: utf-8 -*-
"""
Engine Package - OCR and Translation Logic

This package contains the core processing components:
- OCRWorker: Background thread for screen capture and text extraction
- ContextEngine: Vocabulary matching and lookup
"""

from .ocr import OCRWorker
from .context import ContextEngine

__all__ = ['OCRWorker', 'ContextEngine']
