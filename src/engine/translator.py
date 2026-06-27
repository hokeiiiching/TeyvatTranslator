# -*- coding: utf-8 -*-
"""
Translator Module - MarianMT with Google Translate Fallback

Provides fast, offline-capable translation using Helsinki-NLP/opus-mt-zh-en.
Falls back to Google Translate API on errors.
"""

import os
import logging
from typing import Optional
from threading import Thread, Lock

# Configure logging
logger = logging.getLogger("Translator")

# =============================================================================
# MARIANMT SETUP
# =============================================================================

MARIAN_AVAILABLE = False
MARIAN_ERROR = None
_model = None
_tokenizer = None
_init_lock = Lock()
_init_thread: Optional[Thread] = None
_marian_ready = False

try:
    from transformers import MarianMTModel, MarianTokenizer
    MARIAN_AVAILABLE = True
    logger.info("MarianMT (transformers) available")
except ImportError as e:
    MARIAN_ERROR = str(e)
    logger.warning(f"MarianMT not available: {e}")

# Google Translate fallback
try:
    from deep_translator import GoogleTranslator
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google Translate not available")


MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"


def _init_marian_sync():
    """Initialize MarianMT model synchronously (called in background thread)."""
    global _model, _tokenizer, _marian_ready
    
    if not MARIAN_AVAILABLE:
        logger.error("Cannot initialize: MarianMT not available")
        return
    
    try:
        logger.info(f"Loading MarianMT model: {MODEL_NAME}")
        
        _tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        _model = MarianMTModel.from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                _model = _model.cuda()
                logger.info("MarianMT using GPU acceleration")
            else:
                logger.info("MarianMT using CPU")
        except Exception:
            pass
        
        # Warmup pass - do directly to avoid threading issues
        logger.info("Warming up MarianMT...")
        try:
            warmup_inputs = _tokenizer("测试", return_tensors="pt", padding=True, truncation=True)
            if next(_model.parameters()).is_cuda:
                warmup_inputs = {k: v.cuda() for k, v in warmup_inputs.items()}
            _ = _model.generate(**warmup_inputs, max_length=50)
            logger.info("MarianMT warmup complete")
        except Exception as warmup_err:
            logger.warning(f"Warmup failed (non-critical): {warmup_err}")
        
        _marian_ready = True
        logger.info("MarianMT ready")
        
    except Exception as e:
        logger.error(f"MarianMT init failed: {e}")


def preload_translator():
    """
    Start MarianMT initialization in background thread.
    Call this at app startup to reduce wait time.
    """
    global _init_thread
    
    if not MARIAN_AVAILABLE:
        return
    
    if _marian_ready or _model is not None:
        return  # Already initialized
    
    _init_thread = Thread(target=_init_marian_sync, daemon=True)
    _init_thread.start()
    logger.info("MarianMT background initialization started")


def _get_marian():
    """Get the MarianMT model, waiting for background init if needed."""
    global _init_thread
    
    if not MARIAN_AVAILABLE:
        return None, None
    
    # Wait for background init if in progress
    if _init_thread is not None and _init_thread.is_alive():
        logger.info("Waiting for MarianMT init...")
        _init_thread.join()
    
    if _model is not None:
        return _model, _tokenizer
    
    # Fallback: init synchronously
    _init_marian_sync()
    return _model, _tokenizer


def _translate_marian(text: str) -> Optional[str]:
    """Translate using MarianMT (local model)."""
    model, tokenizer = _get_marian()
    
    if model is None or tokenizer is None:
        return None
    
    try:
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move to same device as model
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        translated_ids = model.generate(**inputs, max_length=512)
        result = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        return result
        
    except Exception as e:
        logger.error(f"MarianMT translation error: {e}")
        return None


def _translate_google(text: str, target: str = 'en') -> Optional[str]:
    """Translate using Google Translate API (fallback)."""
    if not GOOGLE_AVAILABLE:
        return None
    
    try:
        translator = GoogleTranslator(source='auto', target=target)
        return translator.translate(text)
    except Exception as e:
        logger.error(f"Google Translate error: {e}")
        return None


class Translator:
    """
    Translation engine with MarianMT primary and Google fallback.
    
    Usage:
        translator = Translator()
        result = translator.translate("你好世界")
    """
    
    def __init__(self, target_lang: str = 'en'):
        self.target_lang = target_lang
        self._google_translator = None
        
        # Reusable Google translator instance
        if GOOGLE_AVAILABLE:
            try:
                self._google_translator = GoogleTranslator(source='auto', target=target_lang)
            except Exception:
                pass
    
    def translate(self, text: str, marian_text: Optional[str] = None) -> str:
        """
        Translate text. Tries MarianMT first, falls back to Google.
        
        Args:
            text: Chinese text to translate
            marian_text: Optional text to send to MarianMT only. This lets
                callers normalize text for the local model while preserving
                the original text for Google Translate fallback.
            
        Returns:
            English translation, or error message if all methods fail
        """
        if not text or not text.strip():
            return ""

        offline_text = marian_text if marian_text is not None else text
        
        # Try MarianMT first (fast, offline)
        if MARIAN_AVAILABLE and _marian_ready:
            result = _translate_marian(offline_text)
            if result:
                logger.debug(f"MarianMT: {text[:20]}... → {result[:30]}...")
                return result
        
        # Fallback to Google Translate
        if self._google_translator:
            try:
                result = self._google_translator.translate(text)
                if result:
                    logger.debug(f"Google (fallback): {text[:20]}... → {result[:30]}...")
                    return result
            except Exception as e:
                logger.warning(f"Google fallback failed: {e}")
        
        # Last resort: try fresh Google instance
        result = _translate_google(text, self.target_lang)
        if result:
            return result
        
        return "[Translation unavailable]"
    
    @property
    def is_offline_capable(self) -> bool:
        """Returns True if MarianMT is ready for offline translation."""
        return MARIAN_AVAILABLE and _marian_ready
    
    @property
    def backend(self) -> str:
        """Returns the current active backend name."""
        if MARIAN_AVAILABLE and _marian_ready:
            return "MarianMT"
        elif GOOGLE_AVAILABLE:
            return "Google Translate"
        return "None"


# Shared instance for convenience
_shared_translator: Optional[Translator] = None

def get_translator() -> Translator:
    """Get the shared Translator instance."""
    global _shared_translator
    if _shared_translator is None:
        _shared_translator = Translator()
    return _shared_translator
