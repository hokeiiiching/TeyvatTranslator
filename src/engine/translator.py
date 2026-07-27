# -*- coding: utf-8 -*-
"""
Translator Module - MarianMT with Google Translate Fallback

Provides fast, offline-capable translation using Helsinki-NLP/opus-mt-zh-en.
Falls back to Google Translate API on errors.
"""

import os
import logging
import time
from typing import Optional
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logger = logging.getLogger("Translator")

TRANSLATION_UNAVAILABLE = "[Translation unavailable]"

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
        started = time.perf_counter()
        logger.info("marian_initialization_started model=%s", MODEL_NAME)
        
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
        logger.info(
            "marian_initialization_completed elapsed_ms=%.1f",
            (time.perf_counter() - started) * 1000,
        )
        
    except Exception as e:
        logger.exception("MarianMT init failed: %s", e)


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
    
    _init_thread = Thread(
        target=_init_marian_sync,
        daemon=True,
        name="MarianPreload",
    )
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
        import torch
        # Optimize CPU threads for PyTorch matrix operations
        try:
            if torch.get_num_threads() < 4:
                torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass

        started = time.perf_counter()
        with torch.no_grad():
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Move to same device as model
            if next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate translation with repetition penalty and max_new_tokens to prevent 27s generation loops
            translated_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            result = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        logger.info(
            "marian_translation_completed elapsed_ms=%.1f input_chars=%s output_chars=%s",
            (time.perf_counter() - started) * 1000,
            len(text),
            len(result),
        )
        return result
        
    except Exception as e:
        logger.exception("MarianMT translation error: %s", e)
        return None


def is_valid_english_translation(input_text: str, result_text: Optional[str]) -> bool:
    """Validate that a translation output is non-empty, non-identical to input, not untranslated Chinese, and not a repetitive loop."""
    if not result_text or not result_text.strip():
        return False
    clean_result = result_text.strip()
    clean_input = input_text.strip()
    if clean_result == clean_input or clean_result in (TRANSLATION_UNAVAILABLE, "[Translation unavailable]"):
        return False
    # If the output consists mostly of Chinese characters, it is untranslated
    chinese_count = sum(1 for c in clean_result if '\u4e00' <= c <= '\u9fff')
    if chinese_count > len(clean_result) * 0.3:
        return False
    # Check for repetitive word hallucination loops (e.g. "Wait, wait, wait, wait...")
    import re
    if re.search(r'(\b\w+\b)(?:\s*,\s*\1){3,}', clean_result, re.IGNORECASE):
        logger.warning(f"Translation rejected due to repetitive word loop: {clean_result!r}")
        return False
    return True


def _translate_google(text: str, target: str = 'en', source: str = 'auto') -> Optional[str]:
    """Translate using Google Translate API (fallback)."""
    if not GOOGLE_AVAILABLE:
        return None
    
    try:
        started = time.perf_counter()
        # Try explicit Chinese source languages before falling back to auto
        sources_to_try = [source] if source != 'auto' else ['zh-TW', 'zh-CN', 'zh', 'auto']
        for src in sources_to_try:
            try:
                translator = GoogleTranslator(source=src, target=target)
                result = translator.translate(text)
                if is_valid_english_translation(text, result):
                    logger.info(
                        "google_translation_completed elapsed_ms=%.1f input_chars=%s output_chars=%s src=%s",
                        (time.perf_counter() - started) * 1000,
                        len(text),
                        len(result or ""),
                        src,
                    )
                    return result
            except Exception as google_err:
                logger.debug(f"Google Translate with src={src} failed: {google_err}")
                continue
        return None
    except Exception as e:
        logger.exception("Google Translate error: %s", e)
        return None


_marian_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MarianMTWorker")


class Translator:
    """
    Translation engine with MarianMT primary and Google fallback.
    
    Usage:
        translator = Translator()
        result = translator.translate("你好世界")
    """
    
    def __init__(self, from_lang: str = "chi_tra", target_lang: str = "en") -> None:
        """
        Initialize the translator.
        
        Args:
            from_lang: Source language code ('chi_tra' or 'chi_sim')
            target_lang: Target language code ('en' or 'eng')
        """
        self.from_lang = from_lang
        self.target_lang = target_lang
        self._google_translator = None
        
        if GOOGLE_AVAILABLE:
            try:
                # Map source language to Google Translate language codes
                google_src = "zh-TW" if "tra" in from_lang else "zh-CN"
                self._google_translator = GoogleTranslator(source=google_src, target=target_lang)
            except Exception as e:
                logger.warning("Failed to initialize GoogleTranslator: %s", e)
    
    @property
    def backend(self) -> str:
        """Returns name of current active backend."""
        if MARIAN_AVAILABLE and _marian_ready:
            return "MarianMT"
        elif self._google_translator or GOOGLE_AVAILABLE:
            return "Google Translate"
        return "None"
    
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
            logger.debug("Translation skipped because input is blank")
            return ""

        offline_text = marian_text if marian_text is not None else text
        logger.info(
            "translate_request target=%s backend=%s input_chars=%s "
            "marian_input_normalized=%s text=%r",
            self.target_lang,
            self.backend,
            len(text),
            offline_text != text,
            text,
        )
        
        # Try MarianMT first (fast, offline) with a 1.5s timeout on global thread pool
        if MARIAN_AVAILABLE and _marian_ready:
            from concurrent.futures import TimeoutError
            try:
                future = _marian_executor.submit(_translate_marian, offline_text)
                result = future.result(timeout=1.5)
                if is_valid_english_translation(text, result):
                    logger.debug(f"MarianMT: {text[:20]}... → {result[:30]}...")
                    return result
                logger.warning("MarianMT result invalid or untranslated: %r", result)
            except TimeoutError:
                logger.warning(
                    "MarianMT inference timed out (>1.5s) for text=%r; falling back immediately to fast Google Translate",
                    text[:30],
                )
            except Exception as e:
                logger.exception("MarianMT translation submission error: %s", e)
        
        # Fallback to Google Translate
        if self._google_translator:
            try:
                started = time.perf_counter()
                result = self._google_translator.translate(text)
                if is_valid_english_translation(text, result):
                    logger.info(
                        "google_fallback_completed elapsed_ms=%.1f "
                        "input_chars=%s output_chars=%s",
                        (time.perf_counter() - started) * 1000,
                        len(text),
                        len(result),
                    )
                    return result
            except Exception as e:
                logger.exception("Google fallback failed: %s", e)
        
        # Last resort: try fresh Google instance with explicit source languages
        google_src = "zh-TW" if "tra" in self.from_lang else "zh-CN"
        result = _translate_google(text, self.target_lang, source=google_src)
        if is_valid_english_translation(text, result):
            return result
        
        logger.error("All translation backends failed for text=%r", text)
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
