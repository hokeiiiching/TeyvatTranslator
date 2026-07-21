"""
OCR Worker Module

Provides background thread processing for continuous screen capture,
OCR text extraction, and translation with context matching.


Supports PaddleOCR (preferred for Chinese).
"""

# Skip PaddleOCR connectivity check if models already cached (faster startup for returning users)
import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# PaddlePaddle 3.x on Windows CPU can fail in oneDNN/MKLDNN graph execution with
# ConvertPirAttribute2RuntimeAttribute errors. Disable it before Paddle imports.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"


import time
import os
import logging
from typing import Optional, Dict, Any, List
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager

import cv2
import numpy as np

from PIL import ImageGrab, Image
from .translator import Translator, get_translator
from .language_config import get_paddle_lang, make_cache_key, normalize_for_lookup

# Auto-pinyin generation
try:
    from pypinyin import pinyin, Style
    PINYIN_AVAILABLE = True
except ImportError:
    PINYIN_AVAILABLE = False
    print("pypinyin not installed, pinyin will not be auto-generated")
import pyttsx3

from .rag import RAGEngine
from .ocr_config import ocr_config, ocr_diagnostics, OCRConfig

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Create debug directory (writable location — use exe's parent for frozen mode)
import sys as _sys
if getattr(_sys, 'frozen', False):
    # PyInstaller: put debug output next to the exe, not inside _internal
    DEBUG_DIR = os.path.join(os.path.dirname(_sys.executable), "debug_output")
    # Ensure bundled DLLs (mklml.dll, mkldnn.dll, etc.) are discoverable
    _exe_dir = os.path.dirname(_sys.executable)
    _internal_dir = os.path.join(_exe_dir, '_internal')
    for _dll_dir in [_exe_dir, _internal_dir]:
        if os.path.isdir(_dll_dir) and _dll_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(_dll_dir)
        except (OSError, AttributeError):
            pass
else:
    from src.paths import BASE_DIR
    DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configure structured logging
LOG_FILE = os.path.join(DEBUG_DIR, "ocr_debug.log")

# Create logger
logger = logging.getLogger("OCR")
logger.setLevel(logging.DEBUG)

# File handler - detailed logs
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_format)

# Console handler - summary logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)

# Add handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingStats:
    """Performance timing statistics for a single OCR frame."""
    frame_id: int
    capture_ms: float = 0.0
    ocr_ms: float = 0.0
    translation_ms: float = 0.0
    total_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'capture_ms': round(self.capture_ms, 2),
            'ocr_ms': round(self.ocr_ms, 2),
            'translation_ms': round(self.translation_ms, 2),
            'total_ms': round(self.total_ms, 2)
        }


@dataclass 
class OCRResult:
    """Detailed OCR result with metadata for debugging."""
    text: str
    confidence: float = 0.0
    num_lines: int = 0
    chinese_chars: int = 0
    total_chars: int = 0
    bounding_boxes: List[Dict] = field(default_factory=list)
    raw_result: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': round(self.confidence, 3),
            'num_lines': self.num_lines,
            'chinese_chars': self.chinese_chars,
            'total_chars': self.total_chars,
            'bounding_boxes': self.bounding_boxes
        }


@contextmanager
def timed_operation(name: str, stats: TimingStats = None, attr: str = None):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if stats and attr:
            setattr(stats, attr, elapsed_ms)
        logger.debug(f"TIMING {name}: {elapsed_ms:.2f}ms")


# =============================================================================
# PADDLEOCR SETUP
# =============================================================================

# PaddleOCR is REQUIRED for Chinese OCR
PADDLE_AVAILABLE = False
PADDLE_ERROR = None
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError as e:
    PADDLE_ERROR = str(e)
    logger.error("=" * 60)
    logger.error("ERROR: PaddleOCR is NOT installed!")
    logger.error("=" * 60)
    logger.error("PaddleOCR is required for Chinese text recognition.")
    logger.error("To install, run these commands:")
    logger.error("  py -m pip install paddlepaddle")
    logger.error("  py -m pip install paddleocr")
    logger.error("Then restart the application.")
    logger.error("=" * 60)
except Exception as e:
    PADDLE_ERROR = f"{type(e).__name__}: {e}"
    logger.warning(f"PaddleOCR import error: {PADDLE_ERROR}")

# =============================================================================
# GLOBAL OCR INSTANCES (for background pre-initialization)
# =============================================================================
class OCRInitializationError(RuntimeError):
    """Raised when the shared Chinese OCR profile cannot become ready."""


OCR_INIT_TIMEOUT_SECONDS = 180
_global_paddle_ocr: Dict[str, Any] = {}
_ocr_init_lock = Lock()
_ocr_init_threads: Dict[str, Thread] = {}
_ocr_ready: Dict[str, bool] = {}
_ocr_init_errors: Dict[str, str] = {}

def _init_paddle_ocr_sync(source_lang: str = "chi_sim"):
    """Initialize PaddleOCR synchronously (called in background thread)."""
    paddle_lang = get_paddle_lang(source_lang)
    
    if not PADDLE_AVAILABLE:
        logger.error("Cannot initialize: PaddleOCR not available")
        return
    
    with _ocr_init_lock:
        if paddle_lang in _global_paddle_ocr:
            _ocr_ready[paddle_lang] = True
            return
        _ocr_init_errors.pop(paddle_lang, None)
    
    try:
        import paddle
        paddle.device.set_device('cpu')
        logger.info("Using CPU mode")
        
        logger.info(f"Initializing PaddleOCR model '{paddle_lang}'...")
        
        # === PyInstaller fix: bypass PaddleX dependency checks ===
        # In frozen mode, importlib.metadata can't find .dist-info directories,
        # so PaddleX thinks dependencies are missing even though they're bundled.
        # Monkey-patch the checks to no-ops since we know packages are present.
        import sys
        if getattr(sys, 'frozen', False):
            # Fix NameErrors: PaddleX's lazy-import decorators fail in frozen mode,
            # so packages like cv2, pyclipper, shapely never get injected into module
            # namespaces. Adding them to builtins makes them resolvable everywhere.
            import builtins
            for _pkg in ['cv2', 'pyclipper', 'shapely']:
                try:
                    _mod = __import__(_pkg)
                    setattr(builtins, _pkg, _mod)
                except ImportError:
                    logger.warning(f"{_pkg} not available in frozen mode")
            logger.info("Injected lazy-import packages into builtins (frozen mode)")
            try:
                import paddlex.utils.deps as _px_deps
                _px_deps.require_deps = lambda *a, **kw: None
                _px_deps.require_extra = lambda *a, **kw: None
                logger.info("Bypassed PaddleX dependency checks (frozen mode)")
            except Exception as patch_err:
                logger.warning(f"Could not patch PaddleX deps: {patch_err}")
        
        # PaddleOCR 3.x: Disable document preprocessing features for game dialogue
        # These features (orientation, unwarping, textline) add significant latency
        # but are unnecessary for horizontal game subtitles
        ocr = PaddleOCR(
            lang=paddle_lang,
            use_doc_orientation_classify=False,  # Skip document orientation detection
            use_doc_unwarping=False,             # Skip document dewarping
            use_textline_orientation=False       # Skip textline orientation (conflicts with use_angle_cls)
        )
        
        # Warmup pass
        logger.info(f"Warming up OCR model '{paddle_lang}'...")
        try:
            from PIL import Image, ImageDraw, ImageFont
            warmup_pil = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(warmup_pil)
            try:
                font = ImageFont.truetype("msyh.ttc", 32)
            except:
                font = ImageFont.load_default()
            draw.text((20, 30), "测试文字 測試文字", fill='black', font=font)
            warmup_img = np.array(warmup_pil)
        except Exception:
            warmup_img = np.zeros((100, 300, 3), dtype=np.uint8)
            warmup_img.fill(255)
            warmup_img[40:60, 50:250] = 0
        
        _ = ocr.predict(warmup_img)
        logger.info(f"PaddleOCR warmup complete for '{paddle_lang}'")
        logger.info(f"PaddleOCR ready: {paddle_lang}")
        with _ocr_init_lock:
            _global_paddle_ocr[paddle_lang] = ocr
            _ocr_ready[paddle_lang] = True
            _ocr_init_errors.pop(paddle_lang, None)
        
    except Exception as e:
        import traceback
        with _ocr_init_lock:
            _ocr_ready[paddle_lang] = False
            _ocr_init_errors[paddle_lang] = f"{type(e).__name__}: {e}"
        logger.error(f"OCR init failed for '{paddle_lang}': {e}")
        traceback.print_exc()


def preload_ocr(source_lang: str = "chi_sim"):
    """
    Start PaddleOCR initialization in a background thread.
    Call this at app startup to reduce wait time when OCR is first needed.
    """
    paddle_lang = get_paddle_lang(source_lang)
    
    if not PADDLE_AVAILABLE:
        return
    
    if paddle_lang in _global_paddle_ocr or _ocr_ready.get(paddle_lang):
        return  # Already initialized
    
    thread = _ocr_init_threads.get(paddle_lang)
    if thread is not None and thread.is_alive():
        return
    
    thread = Thread(target=_init_paddle_ocr_sync, args=(source_lang,), daemon=True)
    _ocr_init_threads[paddle_lang] = thread
    thread.start()
    logger.info(f"OCR background initialization started for '{paddle_lang}'")


def get_paddle_ocr(source_lang: str = "chi_sim"):
    """
    Get the shared PaddleOCR instance.
    If preload_ocr() was called, returns the pre-initialized instance.
    Otherwise, initializes on demand (lazy loading).
    """
    paddle_lang = get_paddle_lang(source_lang)
    
    if not PADDLE_AVAILABLE:
        raise OCRInitializationError(
            f"PaddleOCR is unavailable: {PADDLE_ERROR or 'unknown import error'}"
        )
    
    # Wait for background init to complete if in progress
    thread = _ocr_init_threads.get(paddle_lang)
    if thread is not None and thread.is_alive():
        logger.info(f"Waiting for background OCR init to complete for '{paddle_lang}'...")
        thread.join(timeout=OCR_INIT_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise OCRInitializationError(
                "Chinese OCR model setup timed out. Check the internet connection "
                "and restart TeyvatTranslator."
            )
    
    # Return cached instance if available
    if paddle_lang in _global_paddle_ocr:
        return _global_paddle_ocr[paddle_lang]

    init_error = _ocr_init_errors.get(paddle_lang)
    if init_error:
        raise OCRInitializationError(
            f"Chinese OCR model setup failed ({init_error}). Restart "
            f"TeyvatTranslator and check {LOG_FILE}."
        )
    
    # Fallback: initialize synchronously
    _init_paddle_ocr_sync(source_lang)
    if paddle_lang in _global_paddle_ocr:
        return _global_paddle_ocr[paddle_lang]

    init_error = _ocr_init_errors.get(paddle_lang, "unknown initialization error")
    raise OCRInitializationError(
        f"Chinese OCR model setup failed ({init_error}). Restart "
        f"TeyvatTranslator and check {LOG_FILE}."
    )


# Debug settings - set to False for production to reduce overhead
DEBUG_MODE = False  # Set to True for verbose logging
SAVE_DEBUG_IMAGES = False  # Save captured/processed images for inspection


# =============================================================================
# NPC DESCRIPTOR PATTERNS
# =============================================================================
# These patterns identify NPC title/affiliation lines that appear between
# the speaker name and actual dialogue. They should be filtered out.

# Role/job suffixes that commonly appear in NPC descriptors
NPC_ROLE_SUFFIXES = {
    # Management/Leadership
    '主管', '店长', '店主', '老板', '掌柜', '会长', '团长', '长老',
    '队长', '组长', '首领', '领袖', '领主', '门主', '总管', '管家',
    
    # Military/Guard roles
    '守卫', '卫兵', '士兵', '骑士', '武士', '侍卫', '巡逻', '警卫',
    '将军', '统领', '千户', '百户', '旗官', '军官',
    
    # Professional roles
    '商人', '学者', '研究员', '医生', '护士', '厨师', '铁匠', '工匠',
    '猎人', '渔夫', '农夫', '船长', '水手', '向导', '记者', '画家',
    '诗人', '歌手', '舞者', '乐师', '演员', '冒险家', '旅行者',
    
    # Service roles
    '助手', '侍者', '服务员', '接待', '大厅', '前台', '职员', '伙计',
    
    # Religious/Spiritual
    '祭司', '巫女', '神官', '住持', '修士', '信徒',
    
    # Government/Diplomatic
    '使节', '大使', '官员', '执事', '奉行', '评议', '议员',
    
    # Generic descriptors
    '成员', '人员', '负责人', '代表', '居民', '市民', '村民',
}

# Organization/faction keywords that appear in descriptors
NPC_ORGANIZATION_KEYWORDS = {
    # Mondstadt
    '骑士团', '西风骑士团', '冒险家协会', '天使分享', '歌德大酒店',
    '晨曦酒庄', '猫尾酒馆',
    
    # Liyue
    '璃月港', '北国银行', '往生堂', '群玉阁', '和裕茶馆', '万民堂',
    '千岩军', '层岩巨渊', '琉璃亭',
    
    # Inazuma
    '稻妻城', '天领奉行', '社奉行', '勘定奉行', '海祇岛', '�的鬼岛',
    '八重堂', '木漏茶室', '神里家', '九条家', '柊家',
    
    # Sumeru
    '须弥城', '教令院', '妙论派', '知论派', '沙漠', '雨林',
    '大巴扎', '化城郭',
    
    # Fontaine
    '枫丹', '歌剧院', '梅洛彼得堡', '审判庭', '玛梅尔', '伊黎耶',
    '堡垒', '研究所', '特许', '工坊',
    
    # General
    '总部', '分部', '支部', '本部', '总店', '分店',
}

def is_descriptor_line(line: str) -> bool:
    """
    Check if a line is an NPC descriptor (title/affiliation) rather than dialogue.
    
    Descriptor lines typically appear between the speaker name and dialogue,
    indicating the NPC's role or organizational affiliation.
    
    Examples:
        - "「特许食堂」主管" (Concession Supervisor)
        - "骑士团成员" (Knights of Favonius Member)
        - "璃月港商人" (Liyue Harbor Merchant)
    
    Args:
        line: A single line of OCR text
        
    Returns:
        True if the line appears to be a descriptor, False otherwise
    """
    line = line.strip()
    
    # Skip empty lines
    if not line:
        return False
    
    # Pattern 1: Contains 「」brackets (organization/place name)
    # These are almost always descriptors like 「特许食堂」主管
    if '「' in line and '」' in line:
        logger.debug(f"  Descriptor detected (bracket pattern): {line}")
        return True
    
    # Pattern 2: Ends with a known role suffix
    for suffix in NPC_ROLE_SUFFIXES:
        if line.endswith(suffix):
            logger.debug(f"  Descriptor detected (role suffix '{suffix}'): {line}")
            return True
    
    # Pattern 3: Contains organization/faction keyword
    for keyword in NPC_ORGANIZATION_KEYWORDS:
        if keyword in line:
            # Make sure it's not a long dialogue line that just mentions the org
            # Descriptors are typically short (< 15 chars)
            if len(line) <= 15:
                logger.debug(f"  Descriptor detected (org keyword '{keyword}'): {line}")
                return True
    
    # Pattern 4: Short line with only Chinese chars and no punctuation
    # Descriptors don't usually have dialogue punctuation (。？！)
    chinese_chars = [c for c in line if '\u4e00' <= c <= '\u9fff']
    has_dialogue_punct = any(p in line for p in '。？！，、')
    if 3 <= len(chinese_chars) <= 10 and not has_dialogue_punct:
        # Could be a descriptor, but be conservative - only flag if mostly Chinese
        if len(chinese_chars) / max(len(line), 1) > 0.8:
            # Additional check: descriptors often contain certain structural chars
            if any(c in line for c in '「」·'):
                logger.debug(f"  Descriptor detected (short title pattern): {line}")
                return True
    
    return False


def _chinese_chars(line: str) -> List[str]:
    """Return CJK characters from a text line."""
    return [c for c in line if '\u4e00' <= c <= '\u9fff']


class OCRWorker:
    """
    Background worker for continuous OCR and translation.
    
    Captures a specified screen region at regular intervals,
    extracts text using PaddleOCR, matches against the
    vocabulary database, and updates the translation window.
    """
    
    # Mapping from OCR codes to Google Translate codes
    OCR_TO_TRANSLATE = {
        'chi_sim': 'zh-CN',
        'chi_tra': 'zh-TW',
        'jpn': 'ja',
        'kor': 'ko',
        'eng': 'en',
        'spa': 'es',
        'fra': 'fr',
        'deu': 'de'
    }
    
    def __init__(
        self,
        x1: int, y1: int, x2: int, y2: int,
        from_lang: str,
        to_lang: str,
        translate_window,
        enable_tts: bool = False,
        enable_context: bool = True,
        window_hwnd: int = None,  # Optional window handle for window capture mode
        dialogue_only: bool = True  # If True, capture only dialogue region (bottom of screen)
    ) -> None:
        """Initialize the OCR worker."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.translate_window = translate_window
        self.enable_tts = enable_tts
        self._enable_context = enable_context  # Private var, accessed via property
        self.window_hwnd = window_hwnd  # If set, capture from this window
        self.dialogue_only = dialogue_only  # If True, focus on dialogue area only
        
        # State
        self.running = False
        self.thread: Optional[Thread] = None
        self.current_text: Optional[str] = None
        self.frame_count = 0
        self._ocr_status_ready = False
        
        # Thread pool for async translation (prevents blocking OCR loop)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
        self._pending_translation: Optional[Future] = None
        self._pending_text: Optional[str] = None  # Text being translated
        
        # OCR temporal voting buffer - stores last N results for stability
        self.ocr_result_buffer: List[OCRResult] = []
        self.OCR_BUFFER_SIZE = 3  # Number of frames to consider for voting
        self.OCR_CONSENSUS_THRESHOLD = 2  # Minimum frames that must agree
        
        # Timer-based text stability - wait for text to stabilize before translating
        self._candidate_text: Optional[str] = None  # Text waiting for stability
        self._candidate_timestamp: float = 0.0  # When candidate was first seen
        self.TEXT_STABILITY_DELAY = 0.8  # Seconds text must remain stable (800ms)
        
        # Debug info
        if DEBUG_MODE:
            print(f"\n{'='*50}")
            print(f"OCR WORKER INITIALIZATION")
            print(f"{'='*50}")
            if window_hwnd:
                mode = "DIALOGUE ONLY" if dialogue_only else "FULL WINDOW"
                print(f"  Mode: WINDOW CAPTURE ({mode})")
                print(f"  Window handle: {window_hwnd}")
            else:
                print(f"  Mode: SCREEN REGION")
                print(f"  Region: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Size: {x2-x1}x{y2-y1} pixels")
            print(f"  From language: {from_lang}")
            print(f"  To language: {to_lang}")
            print(f"  Context matching: {enable_context}")
            print(f"  TTS: {enable_tts}")
            
            # Create debug directory
            if SAVE_DEBUG_IMAGES:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                print(f"  Debug images: {DEBUG_DIR}")
        
        # RAG engine for context matching (lazy-loaded to prevent UI freeze)
        self._rag_engine = None  # Lazy-loaded, accessed via property
        
        # Translation cache - persistent across sessions using shelve
        # Cache file stored in debug_output directory
        import shelve
        from threading import Lock
        self.CACHE_MAX_SIZE = 500  # Max entries before LRU eviction
        self._cache_path = os.path.join(DEBUG_DIR, "translation_cache")
        self._shelve_cache = None  # Lazy-loaded
        self._cache_lock = Lock()  # Thread-safe cache access
        
        # Reusable translator instance (MarianMT with Google fallback)
        self._translator = Translator(target_lang='en')
        print(f"  Translator initialized (backend: {self._translator.backend})")
        
        # PaddleOCR will be lazily initialized on first use (faster startup)
        self.paddle_ocr = None
        if PADDLE_AVAILABLE:
            print(f"  OCR Engine: PaddleOCR ({get_paddle_lang(from_lang)}, lazy-loaded)")
        else:
            print("  OCR Engine: NONE (PaddleOCR missing)")
        
        # Initialize text-to-speech engine
        self.tts_engine = None
        if enable_tts:
            try:
                self.tts_engine = pyttsx3.init()
            except Exception as e:
                print(f"TTS initialization failed: {e}")
    
    @property
    def enable_context(self) -> bool:
        """Whether context matching is enabled."""
        return self._enable_context
    
    @property
    def rag_engine(self):
        """Lazy-load RAG engine on first access (prevents UI freeze)."""
        if self._rag_engine is None and self._enable_context:
            from .rag import get_rag_engine
            self._rag_engine = get_rag_engine()
        return self._rag_engine
    
    @property
    def translation_cache(self):
        """Lazy-load persistent shelve cache on first access (thread-safe)."""
        with self._cache_lock:
            if self._shelve_cache is None:
                import shelve
                self._shelve_cache = shelve.open(self._cache_path, writeback=True)
                logger.info(f"Loaded persistent cache from {self._cache_path}")
            return self._shelve_cache
    
    def _cache_get(self, key: str) -> str:
        """Get a value from the persistent cache (thread-safe)."""
        with self._cache_lock:
            # Initialize cache if needed
            if self._shelve_cache is None:
                import shelve
                self._shelve_cache = shelve.open(self._cache_path, writeback=True)
                logger.info(f"Loaded persistent cache from {self._cache_path}")
            if key in self._shelve_cache:
                return self._shelve_cache[key]
            return None
    
    def _cache_set(self, key: str, value: str) -> None:
        """Set a value in the persistent cache with LRU eviction (thread-safe)."""
        with self._cache_lock:
            # Initialize cache if needed
            if self._shelve_cache is None:
                import shelve
                self._shelve_cache = shelve.open(self._cache_path, writeback=True)
                logger.info(f"Loaded persistent cache from {self._cache_path}")
            # LRU eviction: remove oldest if full
            if len(self._shelve_cache) >= self.CACHE_MAX_SIZE:
                keys_to_remove = list(self._shelve_cache.keys())[:self.CACHE_MAX_SIZE // 10]
                for k in keys_to_remove:
                    del self._shelve_cache[k]
            self._shelve_cache[key] = value
            self._shelve_cache.sync()
                
    def start(self) -> None:
        """Start the OCR processing thread."""
        self._update_status(
            "Preparing Chinese OCR... Keep Genshin focused. First launch may download the OCR model."
        )
        self.running = True
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("OCR worker started")

    def _update_status(self, message: str) -> None:
        """Show worker progress in the overlay when supported."""
        update_status = getattr(getattr(self, "translate_window", None), "update_status", None)
        if callable(update_status):
            update_status(message)
        
    def stop(self) -> None:
        """Stop the OCR processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        # Shutdown thread pool executor
        if self._executor:
            self._executor.shutdown(wait=False)
        # Close persistent cache (may fail if opened from different thread)
        if self._shelve_cache is not None:
            try:
                self._shelve_cache.close()
                print("Translation cache saved")
            except Exception as e:
                # SQLite threading issue - cache was synced during writes anyway
                print(f"Cache close skipped (thread issue): {e}")
            finally:
                self._shelve_cache = None
        print("OCR worker stopped")
        
    def _run_loop(self) -> None:
        """Main OCR processing loop with comprehensive timing and logging."""
        consecutive_errors = 0
        
        logger.info("=" * 50)
        logger.info("OCR PROCESSING LOOP STARTED")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("=" * 50)
        
        while self.running:
            self.frame_count += 1
            frame_start = time.perf_counter()
            
            # Create timing stats for this frame
            timing = TimingStats(frame_id=self.frame_count)
            
            try:
                # === CHECK FOR COMPLETED ASYNC TRANSLATION ===
                if self._pending_translation is not None:
                    if self._pending_translation.done():
                        try:
                            # Get the result (this won't block since it's done)
                            self._pending_translation.result()
                            logger.debug(f"[Frame {self.frame_count}] Async translation completed")
                        except Exception as e:
                            logger.error(f"Async translation error: {e}")
                        finally:
                            self._pending_translation = None
                            self._pending_text = None
                
                # === STEP 1: Capture image from game ===
                with timed_operation("Capture", timing, "capture_ms"):
                    image = self._capture_region()
                
                if image is None:
                    time.sleep(0.1)  # Fast response - check 10x per second
                    continue
                
                # === STEP 2: Extract text using PaddleOCR ===
                with timed_operation("OCR", timing, "ocr_ms"):
                    ocr_result = self._extract_text(image)
                
                # Skip if no meaningful text (less than 2 Chinese chars)
                if ocr_result.chinese_chars < 2:
                    logger.debug(
                        f"[Frame {self.frame_count}] Skipped: only {ocr_result.chinese_chars} Chinese chars"
                    )
                    time.sleep(0.1)  # Fast response
                    continue
                
                # === Simple duplicate check - display immediately if different ===
                # With image preprocessing, OCR should be more stable
                # Skip if same as previous text (prevent redundant translations)
                # Use normalized comparison to handle minor punctuation variations
                def normalize_for_comparison(text: str) -> str:
                    """Normalize punctuation for comparison to avoid re-translating due to OCR variations."""
                    if not text:
                        return ""
                    # Normalize halfwidth to fullwidth punctuation
                    replacements = [
                        (',', '，'),  # Comma
                        ('.', '。'),  # Period
                        ('!', '！'),  # Exclamation
                        ('?', '？'),  # Question mark
                        (':', '：'),  # Colon
                        (';', '；'),  # Semicolon
                    ]
                    for half, full in replacements:
                        text = text.replace(half, full)
                    return text
                
                normalized_new = normalize_for_comparison(ocr_result.text)
                normalized_current = normalize_for_comparison(self.current_text)
                # Skip if same as already-translated text
                if normalized_new == normalized_current:
                    logger.debug(f"[Frame {self.frame_count}] Same as current, skipping")
                    time.sleep(0.1)
                    continue
                
                # Skip if this text is already being translated
                if normalized_new == normalize_for_comparison(self._pending_text):
                    logger.debug(f"[Frame {self.frame_count}] Already translating, skipping")
                    time.sleep(0.1)
                    continue

                # === TIMER-BASED TEXT STABILITY CHECK ===
                # Both Chinese scripts use this exact same gate.
                normalized_candidate = normalize_for_comparison(self._candidate_text)
                current_time = time.perf_counter()

                if normalized_new != normalized_candidate:
                    # New candidate - start timer
                    self._candidate_text = ocr_result.text
                    self._candidate_timestamp = current_time
                    logger.debug(f"[Frame {self.frame_count}] New candidate text, starting stability timer")
                    time.sleep(0.1)
                    continue

                # Same as candidate - check if stable long enough
                elapsed = current_time - self._candidate_timestamp
                if elapsed < self.TEXT_STABILITY_DELAY:
                    logger.debug(f"[Frame {self.frame_count}] Waiting for stability ({elapsed:.2f}s / {self.TEXT_STABILITY_DELAY}s)")
                    time.sleep(0.1)
                    continue
                
                # Text is stable! Clear candidate and proceed to translate
                logger.debug(f"[Frame {self.frame_count}] Text stable for {elapsed:.2f}s, translating")
                self._candidate_text = None
                self._candidate_timestamp = 0.0
                
                # === STEP 3: Update current text ===
                self.current_text = ocr_result.text
                logger.info(f"\nDetected: {ocr_result.text}")
                logger.info(f"   Confidence: {ocr_result.confidence:.2f} | Lines: {ocr_result.num_lines}")
                
                # === STEP 4: Submit translation ASYNCHRONOUSLY ===
                # This doesn't block - OCR loop continues immediately
                self._pending_text = ocr_result.text
                self._pending_translation = self._executor.submit(
                    self._process_translation, ocr_result.text
                )
                logger.debug(f"[Frame {self.frame_count}] Translation submitted to thread pool")
                
                # Note: timing.translation_ms will be 0 since we're not waiting
                timing.translation_ms = 0.0
                
                # Calculate total time
                timing.total_ms = (time.perf_counter() - frame_start) * 1000
                
                # Log performance summary
                logger.debug(
                    f"[Frame {self.frame_count}] TIMING: "
                    f"capture={timing.capture_ms:.0f}ms, "
                    f"ocr={timing.ocr_ms:.0f}ms, "
                    f"translate={timing.translation_ms:.0f}ms, "
                    f"total={timing.total_ms:.0f}ms"
                )
                
                # Reset error counter on success
                consecutive_errors = 0
                
            except OCRInitializationError as e:
                logger.error(f"Fatal OCR initialization error: {e}")
                self._update_status(f"OCR could not start: {e}")
                self.running = False
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in frame {self.frame_count}: {type(e).__name__}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Exponential backoff on repeated errors
                if consecutive_errors >= 3:
                    wait_time = min(10, 2 ** (consecutive_errors - 2))
                    logger.warning(f"   {consecutive_errors} consecutive errors, waiting {wait_time}s...")
                    time.sleep(wait_time)
                
            time.sleep(0.1)  # Fast response - check 10x per second
            
    def _capture_region(self) -> Optional[np.ndarray]:
        """Capture the specified screen region or window."""
        try:
            # Use window capture if window handle is set
            if self.window_hwnd:
                # Choose capture method based on dialogue_only setting
                if self.dialogue_only:
                    from .window_capture import capture_window_dialogue
                    
                    # Capture dialogue region only (bottom of screen)
                    img_np = capture_window_dialogue(self.window_hwnd)
                    
                    if DEBUG_MODE and self.frame_count <= 3:
                        print(f"  Using DIALOGUE capture mode")
                else:
                    from .window_capture import capture_window
                    
                    # Capture full window
                    img_np = capture_window(self.window_hwnd)
                    
                    if DEBUG_MODE and self.frame_count <= 3:
                        print(f"  Using FULL WINDOW capture mode")
                
                if img_np is None:
                    if DEBUG_MODE and self.frame_count <= 3:
                        print(f"  Window capture returned None")
                    return None
                
                if DEBUG_MODE and self.frame_count <= 3:
                    print(f"  Window capture shape: {img_np.shape}")
                
                if SAVE_DEBUG_IMAGES and self.frame_count <= 5:
                    path = os.path.join(DEBUG_DIR, f"capture_{self.frame_count}.png")
                    Image.fromarray(img_np).save(path)
                    print(f"  Saved debug image: {path}")
                
                return img_np
            
            # Screen region capture (original method)
            img = ImageGrab.grab(bbox=(self.x1, self.y1, self.x2, self.y2))
            
            if img is None:
                print(f"  ImageGrab returned None")
                return None
                
            img_np = np.array(img)
            
            if DEBUG_MODE and self.frame_count <= 3:
                print(f"  Screen capture shape: {img_np.shape}")
                print(f"  Image dtype: {img_np.dtype}")
                print(f"  Image min/max: {img_np.min()}/{img_np.max()}")
            
            if SAVE_DEBUG_IMAGES and self.frame_count <= 5:
                path = os.path.join(DEBUG_DIR, f"capture_{self.frame_count}.png")
                img.save(path)
                print(f"  Saved debug image: {path}")
            
            return img_np
            
        except Exception as e:
            print(f"  Capture error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract Chinese text from image using PaddleOCR.
        
        Returns an OCRResult object with detailed metadata for debugging:
        - text: The extracted text
        - confidence: Average confidence score (0-1)
        - num_lines: Number of text lines detected
        - chinese_chars: Count of Chinese characters
        - bounding_boxes: List of bounding box coordinates
        
        Args:
            image: RGB numpy array of the captured region
            
        Returns:
            OCRResult with text and metadata, empty result if no text found
        """
        if not PADDLE_AVAILABLE:
            raise RuntimeError(
                "PaddleOCR is not installed!\n"
                "Run: py -m pip install paddlepaddle paddleocr"
            )
        
        # Get PaddleOCR instance (uses pre-initialized global instance if available)
        if self.paddle_ocr is None:
            self.paddle_ocr = get_paddle_ocr(self.from_lang)
        if not getattr(self, "_ocr_status_ready", False):
            self._update_status("OCR ready - waiting for Chinese dialogue...")
            self._ocr_status_ready = True
        
        # === Convert RGB to BGR if needed (PaddleOCR uses OpenCV convention) ===
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if image looks like RGB (from PIL) - convert to BGR for PaddleOCR
            # Most capture methods give RGB, but PaddleOCR expects BGR
            import cv2
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.debug(f"  Converted RGB to BGR for OCR")
        else:
            image_bgr = image
        
        # === IMAGE PREPROCESSING for OCR stability ===
        import cv2
        
        h, w = image_bgr.shape[:2]
        
        # Moderate upscaling for better text recognition (2000px balances speed and accuracy)
        MAX_SIDE = 2000
        max_dim = max(w, h)
        if max_dim < MAX_SIDE:
            scale_factor = MAX_SIDE / max_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast, making text stand out and reducing garbage detections
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        ocr_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Save preprocessed image for debugging (first few frames only)
        if SAVE_DEBUG_IMAGES and self.frame_count <= 3:
            from PIL import Image
            debug_path = os.path.join(DEBUG_DIR, f"preprocessed_{self.frame_count}.png")
            # Convert BGR back to RGB for saving
            preprocessed_rgb = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(preprocessed_rgb).save(debug_path)
            logger.debug(f"  Saved preprocessed image: {debug_path}")
        
        # === Run OCR on the preprocessed image ===
        # Note: cls argument removed - deprecated in newer PaddleOCR versions
        result = self.paddle_ocr.predict(ocr_image)
        
        # === Handle empty results ===
        if not result or not result[0]:
            logger.debug(f"[Frame {self.frame_count}] No text detected")
            return OCRResult(text="", confidence=0.0, num_lines=0)
        
        # === Extract detailed results ===
        lines = []
        confidences = []
        bounding_boxes = []
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"[Frame {self.frame_count}] PaddleOCR RESULTS")
        logger.debug(f"{'='*60}")
        
        # Handle new PaddleX OCRResult object format (PaddleOCR 3.x)
        ocr_result_obj = result[0]
        
        # Debug: Log the actual result type and structure
        logger.debug(f"  Result type: {type(ocr_result_obj)}")
        if hasattr(ocr_result_obj, 'keys'):
            logger.debug(f"  Dict keys: {list(ocr_result_obj.keys())}")
        elif isinstance(ocr_result_obj, list) and len(ocr_result_obj) > 0:
            logger.debug(f"  List length: {len(ocr_result_obj)}, first item type: {type(ocr_result_obj[0])}")
        
        # Try different ways to access the text based on the result type
        try:
            texts = []
            scores = []
            
            # Method 1: PaddleX 3.x format - dict-like access
            if hasattr(ocr_result_obj, 'get'):
                texts = ocr_result_obj.get('rec_texts', []) or []
                scores = ocr_result_obj.get('rec_scores', []) or []
                logger.debug(f"  Tried dict access: {len(texts)} texts")
                # Debug: Log the actual values
                logger.debug(f"  rec_texts value: {ocr_result_obj.get('rec_texts')}")
                logger.debug(f"  rec_scores value: {ocr_result_obj.get('rec_scores')}")
            
            # Method 1b: Try attribute access if dict access failed
            if not texts and hasattr(ocr_result_obj, 'rec_texts'):
                texts = getattr(ocr_result_obj, 'rec_texts', []) or []
                scores = getattr(ocr_result_obj, 'rec_scores', []) or []
                logger.debug(f"  Tried attribute access: {len(texts)} texts")
                logger.debug(f"  rec_texts attr: {texts}")
            
            # Method 1c: Try subscript access
            if not texts:
                try:
                    texts = ocr_result_obj['rec_texts'] or []
                    scores = ocr_result_obj['rec_scores'] or []
                    logger.debug(f"  Tried subscript access: {len(texts)} texts")
                except (KeyError, TypeError) as e:
                    logger.debug(f"  Subscript access failed: {e}")
            
            # Method 2: Legacy PaddleOCR 2.x format - list of [box, (text, score)]
            if not texts and isinstance(ocr_result_obj, list):
                for item in ocr_result_obj:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        # item = [box_coords, (text, confidence)]
                        text_conf = item[-1]  # Last element is (text, conf)
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            texts.append(str(text_conf[0]))
                            scores.append(float(text_conf[1]))
                logger.debug(f"  Tried legacy list format: {len(texts)} texts")
            
            # Method 3: Check if result[0] IS the list of items directly
            if not texts and isinstance(result, list):
                for item in result:
                    if isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                                text_conf = sub_item[-1]
                                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                    texts.append(str(text_conf[0]))
                                    scores.append(float(text_conf[1]))
                logger.debug(f"  Tried nested list format: {len(texts)} texts")
            
            logger.debug(f"  Found {len(texts)} text entries")
            
            for i, text in enumerate(texts):
                if text:
                    score = scores[i] if i < len(scores) else 0.0
                    # Filter out low-confidence lines (likely OCR noise/garbage)
                    # Using 0.6 threshold to filter garbage while keeping real dialogue
                    if score < 0.6:
                        logger.debug(f"  Line {i+1}: \"{text}\" (conf: {score:.3f}) - SKIPPED (low confidence)")
                        continue
                    lines.append(str(text))
                    confidences.append(float(score) if score else 0.0)
                    logger.debug(f"  Line {i+1}: \"{text}\" (conf: {score:.3f})")
                    
        except Exception as e:
            logger.error(f"  Failed to parse OCR result: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # === OCR Text Cleanup ===
        # Fix common misrecognitions of Chinese quotation marks 「」
        # PaddleOCR sometimes confuses them with various bracket-like characters
        def cleanup_ocr_text(text: str) -> str:
            # Common role suffixes that appear AFTER closing brackets in NPC descriptors
            role_suffixes = ['主管', '店长', '店主', '老板', '掌柜', '守卫', '队长',
                            '商人', '学者', '成员', '侍卫', '士兵', '骑士', '住持',
                            '助手', '研究员', '负责人', '管家', '执事', '奉行']
            
            # Define all bracket pairs to check for unbalanced state
            # (open_char, close_char) - check these BEFORE normalization
            bracket_pairs = [
                ('「', '」'),   # Corner brackets (target format)
                ('【', '】'),   # Black lenticular brackets
                ('［', '］'),   # Fullwidth square brackets
                ('〔', '〕'),   # Tortoise shell brackets
                ('〖', '〗'),   # White lenticular brackets
                ('《', '》'),   # Double angle brackets
                ('[', ']'),     # ASCII square brackets
            ]
            
            # Fix unbalanced brackets BEFORE normalization
            # Handle both cases:
            # 1. Has opening but no closing: 「XXX主管 -> 「XXX」主管
            # 2. Has closing but no opening: XXX」 -> 「XXX」
            for open_char, close_char in bracket_pairs:
                has_open = open_char in text
                has_close = close_char in text
                
                if has_open and not has_close:
                    # Missing closing bracket - try to insert before role suffix
                    inserted = False
                    for suffix in role_suffixes:
                        if text.endswith(suffix):
                            insert_pos = len(text) - len(suffix)
                            text = text[:insert_pos] + close_char + text[insert_pos:]
                            logger.debug(f"  Fixed missing close {open_char}{close_char}: {text}")
                            inserted = True
                            break
                    # If no suffix match, just append closing bracket at end
                    if not inserted:
                        text = text + close_char
                        logger.debug(f"  Appended missing close bracket: {text}")
                        
                elif has_close and not has_open:
                    # Missing opening bracket - prepend at start
                    text = open_char + text
                    logger.debug(f"  Prepended missing open bracket: {text}")
            
            # Now normalize all bracket types to 「」
            replacements = [
                # Fullwidth brackets (most common misdetection)
                ('【', '「'),   # Fullwidth left black lenticular bracket
                ('】', '」'),   # Fullwidth right black lenticular bracket
                ('［', '「'),   # Fullwidth left square bracket
                ('］', '」'),   # Fullwidth right square bracket
                ('〔', '「'),   # Left tortoise shell bracket
                ('〕', '」'),   # Right tortoise shell bracket
                ('〖', '「'),   # Left white lenticular bracket
                ('〗', '」'),   # Right white lenticular bracket
                ('《', '「'),   # Left double angle bracket (if misread as quote)
                ('》', '」'),   # Right double angle bracket
                
                # Halfwidth ASCII brackets
                ('[', '「'),    # Left square bracket
                (']', '」'),    # Right square bracket
                
                # Chinese radicals that look like brackets
                ('厂', '「'),   # Factory radical (CJK)
                ('广', '「'),   # Wide radical (CJK)
                ('丿', '」'),   # Slash radical sometimes misread
            ]
            for old, new in replacements:
                text = text.replace(old, new)
            
            # === Safeguard: Remove duplicate consecutive brackets ===
            # Prevent cases like 「「XXX」」 from ever occurring
            while '「「' in text:
                text = text.replace('「「', '「')
            while '」」' in text:
                text = text.replace('」」', '」')
            
            return text
        
        # Clean up each line
        lines = [cleanup_ocr_text(line) for line in lines]
        
        # === Merge split dialogue lines ===
        # OCR sometimes splits a single dialogue line into multiple lines
        # This is incorrect - we need to merge them back together
        # Structure should be: [speaker] [optional descriptor] [dialogue]
        if len(lines) >= 3:
            # Check if first line is a speaker (contains corner brackets)
            is_speaker_first = '」' in lines[0] or '「' in lines[0]
            
            if is_speaker_first:
                # Check if line 2 is a descriptor (short, no dialogue punctuation)
                line2 = lines[1]
                is_descriptor = is_descriptor_line(line2)
                
                if is_descriptor:
                    # Structure: [speaker] [descriptor] [dialogue fragments...]
                    # Merge lines 3+ into a single dialogue line
                    merged_dialogue = ''.join(lines[2:])
                    lines = [lines[0], lines[1], merged_dialogue]
                    logger.debug(f"  Merged dialogue lines (after descriptor): {merged_dialogue}")
                else:
                    # Structure: [speaker] [dialogue fragments...]
                    # Merge lines 2+ into a single dialogue line
                    merged_dialogue = ''.join(lines[1:])
                    lines = [lines[0], merged_dialogue]
                    logger.debug(f"  Merged dialogue lines: {merged_dialogue}")
        
        # === Calculate aggregate metrics ===
        # Use newline to preserve line structure for speaker detection
        full_text = "\n".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        chinese_chars = sum(1 for c in full_text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(full_text.replace(' ', '').replace('\n', ''))
        
        # === Create result object ===
        ocr_result = OCRResult(
            text=full_text,
            confidence=avg_confidence,
            num_lines=len(lines),
            chinese_chars=chinese_chars,
            total_chars=total_chars,
            bounding_boxes=bounding_boxes,
            raw_result=result
        )
        
        # === Log summary ===
        logger.debug(f"\n  SUMMARY:")
        logger.debug(f"     Total lines: {len(lines)}")
        logger.debug(f"     Avg confidence: {avg_confidence:.3f}")
        logger.debug(f"     Chinese chars: {chinese_chars}/{total_chars}")
        logger.debug(f"     Full text: \"{full_text[:100]}{'...' if len(full_text) > 100 else ''}\"")
        logger.debug(f"{'='*60}")
        
        # === Quality warnings ===
        if avg_confidence < 0.5:
            logger.warning(f"Low confidence ({avg_confidence:.2f}) - OCR may be inaccurate")
        if chinese_chars == 0 and total_chars > 0:
            logger.warning(f"No Chinese characters detected in: {full_text[:50]}")
        
        return ocr_result
        
    def _process_translation(self, text: str) -> None:
        """Process text and update translation window with speaker info."""
        if DEBUG_MODE:
            print(f"\n{'='*50}")
            print(f"TRANSLATING: '{text}'")
            print(f"{'='*50}")
        
        text_original = text
        text_lookup = normalize_for_lookup(text_original, self.from_lang)
        context = {}
        translated = ""
        speaker = ""
        dialogue = text_original
        
        # Parse speaker name if OCR captured multiple lines
        # Speaker name is typically the first shorter line (< 10 chars)
        # Descriptor lines (NPC title/affiliation) may appear between speaker and dialogue
        lines = text_original.split('\n')
        descriptor = ""  # NPC title/affiliation line
        
        if len(lines) >= 2:
            potential_speaker = lines[0].strip()
            # Speaker name is usually short (2-5 Chinese characters) and centered
            chinese_chars = _chinese_chars(potential_speaker)
            
            # Skip garbage lines with no Chinese characters (e.g., 'AAAA' from UI artifacts)
            start_idx = 0
            if len(chinese_chars) == 0 and len(lines) >= 3:
                # First line is garbage, try line 2 as speaker
                start_idx = 1
                potential_speaker = lines[1].strip()
                chinese_chars = _chinese_chars(potential_speaker)
                logger.debug(f"  Skipped garbage line: '{lines[0].strip()}', using line 2 as speaker")
            
            if 1 <= len(chinese_chars) <= 6 and len(potential_speaker) <= 10:
                speaker = potential_speaker
                
                # Position-based descriptor detection (accounting for skipped garbage lines):
                # - Speaker is at start_idx
                # - Descriptor is at start_idx+1 IF there are enough lines AND it's short
                # - Remaining lines: Dialogue
                dialogue_lines = []
                remaining_lines = lines[start_idx + 1:]  # Lines after speaker
                
                if len(remaining_lines) >= 2:
                    # Check if first remaining line looks like a descriptor (short, no dialogue punctuation)
                    line2 = remaining_lines[0].strip()
                    has_dialogue_punct = any(p in line2 for p in '。？！')
                    
                    if len(line2) <= 15 and not has_dialogue_punct:
                        # This is the descriptor
                        descriptor = line2
                        dialogue_lines = [l.strip() for l in remaining_lines[1:] if l.strip()]
                        logger.info(f"  Captured descriptor: {line2}")
                    else:
                        # No descriptor, all remaining lines are dialogue
                        dialogue_lines = [l.strip() for l in remaining_lines if l.strip()]
                else:
                    # Only speaker + dialogue, no descriptor
                    dialogue_lines = [l.strip() for l in remaining_lines if l.strip()]
                
                dialogue = '\n'.join(dialogue_lines).strip()
                print(f"  Detected speaker: {speaker}")
                if descriptor:
                    print(f"  NPC descriptor: {descriptor}")
        
        speaker_lookup = normalize_for_lookup(speaker, self.from_lang)
        descriptor_lookup = normalize_for_lookup(descriptor, self.from_lang)
        dialogue_lookup = normalize_for_lookup(dialogue, self.from_lang)
        logger.debug(
            "  Lookup normalization: "
            f"source={self.from_lang}, text_changed={text_lookup != text_original}, "
            f"dialogue_changed={dialogue_lookup != dialogue}, "
            f"descriptor_changed={descriptor_lookup != descriptor}"
        )
        
        # Generate pinyin for speaker
        if speaker and PINYIN_AVAILABLE:
            try:
                py_list = pinyin(speaker, style=Style.TONE)
                context['speaker'] = speaker
                context['speaker_pinyin'] = ' '.join([p[0] for p in py_list])
                
                # Add descriptor (NPC title/affiliation) if found
                if descriptor:
                    context['descriptor'] = descriptor
                
                # Try to find English name from vocabulary
                if self.rag_engine:
                    matches = self.rag_engine.get_context(speaker_lookup)
                    logger.debug(f"  RAG returned {len(matches) if matches else 0} matches for speaker '{speaker_lookup}'")
                    # Search ALL matches for exact mandarin match (not just first)
                    for match in (matches or []):
                        if match.get('mandarin', '') == speaker_lookup:
                            context['speaker_english'] = match.get('english', '')
                            # Also use vocabulary pinyin if available (more accurate)
                            vocab_pinyin = match.get('pinyin', '')
                            if vocab_pinyin:
                                context['speaker_pinyin'] = vocab_pinyin
                            logger.debug(f"  Found speaker: {context.get('speaker_english')} ({vocab_pinyin})")
                            break
                    else:
                        # Log what we found if no exact match
                        if matches:
                            logger.debug(f"  No exact match. First result mandarin: '{matches[0].get('mandarin', '')}' vs speaker lookup: '{speaker_lookup}'")
            except Exception as e:
                context['speaker'] = speaker
                logger.debug(f"  Speaker pinyin failed: {e}")
        
        # Generate pinyin for dialogue - ONLY for Chinese characters
        # This ensures pinyin list aligns with display code that only assigns pinyin to Chinese chars
        if dialogue and PINYIN_AVAILABLE:
            try:
                # Extract only Chinese characters for pinyin generation
                chinese_only = ''.join(c for c in dialogue if '\u4e00' <= c <= '\u9fff')
                if chinese_only:
                    py_list = pinyin(chinese_only, style=Style.TONE)
                    context['pinyin'] = ' '.join([p[0] for p in py_list])
                    logger.debug(f"  Dialogue pinyin generated for {len(chinese_only)} Chinese chars")
                else:
                    context['pinyin'] = ''
            except Exception as e:
                logger.debug(f"  Pinyin generation failed: {e}")
        
        # Translate descriptor (NPC title/affiliation) if present
        if descriptor:
            try:
                # Strip 「」 brackets before translation - they confuse Google Translate
                descriptor_clean = descriptor.replace('「', '').replace('」', '')
                descriptor_translation_lookup = normalize_for_lookup(
                    descriptor_clean, self.from_lang
                )
                descriptor_marian_text = (
                    descriptor_translation_lookup
                    if descriptor_translation_lookup != descriptor_clean
                    else None
                )
                
                # Check cache first (descriptors repeat frequently)
                # Check cache first
                descriptor_cache_key = make_cache_key(
                    self.from_lang, "descriptor", descriptor_clean
                )
                cached = self._cache_get(descriptor_cache_key)
                if cached:
                    descriptor_english = cached
                    logger.debug(f"  Descriptor cache hit: {descriptor_english}")
                else:
                    descriptor_english = self._translate_with_retry(
                        descriptor_clean,
                        max_retries=2,
                        marian_text=descriptor_marian_text,
                    )
                    # Cache descriptor translation
                    self._cache_set(descriptor_cache_key, descriptor_english)
                
                context['descriptor_english'] = descriptor_english
            except Exception as e:
                context['descriptor_english'] = ''
                logger.debug(f"  Descriptor translation failed: {e}")
        
        # RAG for exact vocabulary matches only
        if self.enable_context and self.rag_engine and len(dialogue_lookup) <= 10:
            matches = self.rag_engine.get_context(dialogue_lookup)
            if matches and matches[0].get('mandarin', '') == dialogue_lookup:
                translated = matches[0].get('english', '')
                print(f"  RAG exact match: {translated}")
        
        # Fall back to translation (check cache first)
        if not translated:
            dialogue_cache_key = make_cache_key(self.from_lang, "dialogue", dialogue)
            cached = self._cache_get(dialogue_cache_key)
            if cached:
                translated = cached
                logger.debug(f"  Cache hit: {translated}")
            else:
                translated = self._translate_with_retry(
                    dialogue,
                    max_retries=3,
                    marian_text=dialogue_lookup if dialogue_lookup != dialogue else None,
                )
                # Persist to cache
                self._cache_set(dialogue_cache_key, translated)
        
        # Update the display window
        self.translate_window.update_translation(dialogue, translated, context)
        
        # Text-to-speech output
        if self.enable_tts and self.tts_engine and translated:
            try:
                self.tts_engine.say(translated)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def _translate_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        marian_text: Optional[str] = None,
    ) -> str:
        """Translate text using MarianMT (primary) or Google Translate (fallback)."""
        if DEBUG_MODE:
            print(f"  Translating: '{text}' (backend: {self._translator.backend})")
            if marian_text is not None and marian_text != text:
                print("  MarianMT input normalized for Traditional Chinese")
        
        for attempt in range(max_retries):
            try:
                translated = self._translator.translate(text, marian_text=marian_text)
                
                if translated and translated != "[Translation unavailable]":
                    print(f"  🌐 Result: {translated}")
                    return translated
                    
            except Exception as e:
                error_msg = str(e)
                print(f"  Attempt {attempt + 1} failed: {error_msg}")
                
                # Exponential backoff for rate limiting
                if "429" in error_msg or "500" in error_msg:
                    wait_time = (2 ** attempt) * 0.5
                    print(f"     Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        return "[Translation unavailable]"
