# -*- coding: utf-8 -*-
"""
Engine Package - OCR and Translation Logic

This package contains the core processing components:
- OCRWorker: Background thread for screen capture and text extraction
- ContextEngine: Vocabulary matching and lookup
"""

__all__ = ['OCRWorker', 'ContextEngine']


def __getattr__(name):
    """Lazy-load heavy engine components only when requested."""
    if name == 'OCRWorker':
        from .ocr import OCRWorker

        return OCRWorker
    if name == 'ContextEngine':
        from .context import ContextEngine

        return ContextEngine
    raise AttributeError(f"module 'src.engine' has no attribute {name!r}")
