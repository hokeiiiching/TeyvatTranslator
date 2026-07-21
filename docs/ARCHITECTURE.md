# Architecture Overview

This document explains the Genshin Translator codebase architecture for developers.

## High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Game Screen   │────▶│   OCR Worker    │────▶│   Translation   │
│   (Genshin)     │     │   (PaddleOCR)   │     │   (Google API)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Window Capture  │     │   RAG Engine    │────▶│  Overlay Window │
│  (pywin32)      │     │ (Vocabulary)    │     │   (PyQt6)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Module Responsibilities

### `src/engine/ocr.py` - OCR Worker
**Purpose**: Continuous screen capture and text extraction.

**Key Classes**:
- `OCRWorker`: Background thread that captures game screen, extracts Chinese text, and triggers translation.
- `OCRResult`: Data class holding OCR output with confidence scores.
- `TimingStats`: Performance profiling for each frame.

**Flow**:
1. Read the selected source script (`chi_sim` or `chi_tra`)
2. Capture the dialogue region from the game window
3. Run the shared, Traditional-capable PaddleOCR `ch` profile
4. Keep the captured text unchanged for display and pinyin
5. Normalize a private Traditional-to-Simplified copy with OpenCC for vocabulary lookup
6. Detect speaker name vs dialogue text
7. Trigger translation and UI update with a source-aware cache key

**Important Implementation Details**:
- PaddleOCR is lazily loaded on first use (faster startup)
- Simplified and Traditional reuse one verified PP-OCRv6 OCR instance
- Capture, preprocessing, confidence filtering, stability, and line parsing are script-agnostic
- A warmup pass runs on initialization to ensure first-frame accuracy
- Dialogue detection compares lines to avoid redundant translations
- Missing Traditional conversion data is caught before capture starts rather than silently degrading lookup accuracy

---

### `src/engine/rag.py` - RAG Engine
**Purpose**: Semantic vocabulary search for context-aware translations.

**Key Class**: `RAGEngine`

**How it works**:
1. Indexes 700+ Genshin-specific vocabulary terms
2. Uses semantic similarity to find relevant matches
3. Prioritizes exact matches over fuzzy matches
4. Returns vocabulary entries with pinyin, English, breakdowns

**Use Case**: When OCR detects "希格雯", RAG finds the exact match and returns "Sigewinne" instead of a generic translation.

---

### `src/engine/window_capture.py` - Window Capture
**Purpose**: Capture Genshin Impact window content on Windows.

**Key Functions**:
- `find_genshin_window()`: Auto-detect game window
- `capture_window_dialogue(hwnd)`: Capture dialogue region only
- `get_dialogue_region()`: Calculate dialogue box coordinates

**Why screen grab?**: DirectX games return black with PrintWindow, so we use screen coordinates.

---

### `src/engine/context.py` - Context Engine  
**Purpose**: Simple keyword-based vocabulary matching (faster than RAG).

**Key Class**: `ContextEngine`

**Use Case**: Quick lookups for short terms without semantic search overhead.

---

### `src/ui/translate_window.py` - Translation Overlay
**Purpose**: Floating transparent window showing translations.

**Key Features**:
- Always-on-top, frameless window
- Click-through transparency
- Displays: Chinese text, pinyin, English translation
- Speaker name with separate styling

---

### `src/ui/main_window.py` - Main Window
**Purpose**: Settings panel and translation controls.

**Tabs**:
- Language selection (source/target)
- Region selection for manual capture
- Settings (font size, opacity, TTS toggle)

---

### `src/data/vocabulary.py` - Vocabulary Database
**Purpose**: 700+ curated Genshin Impact terms.

**Entry Format**:
```python
{
    'id': 'char_paimon',
    'mandarin': '派蒙',
    'pinyin': 'Pài Méng',
    'english': 'Paimon',
    'breakdown': [{'char': '派', 'pinyin': 'pài', 'meaning': 'dispatch'}],
    'game_context': 'Traveler companion and emergency food',
    'tags': ['character', 'dialogue']
}
```

## Key Design Decisions

### 1. PaddleOCR over Tesseract
PaddleOCR has significantly better Chinese character recognition accuracy compared to Tesseract, especially for stylized game fonts.

### 2. Lazy Loading
PaddleOCR model loading takes 5-10 seconds. We defer this until first capture to show the UI faster.

### 3. Warmup Pass
The first PaddleOCR inference after loading has reduced accuracy. A dummy warmup image ensures consistent accuracy from frame 1.

### 4. Dialogue-Only Capture
Capturing the full game screen is expensive. We calculate the dialogue region (bottom 22% of screen) and capture only that area.

### 5. Speaker Detection
Genshin dialogue shows speaker name on a separate line. We detect this by checking if the first line is short (2-6 Chinese chars).

## Adding New Features

### Adding Vocabulary Terms
Edit `src/data/vocabulary.py`:
```python
VOCABULARY.append({
    'id': 'unique_id',
    'mandarin': '中文',
    'pinyin': 'zhōng wén',
    'english': 'Chinese',
    'tags': ['language']
})
```

### Adjusting Dialogue Region
Edit `src/engine/window_capture.py` → `get_dialogue_region()`:
```python
region_y = int(window_height * 0.73)  # Start at 73% down
region_height = int(window_height * 0.22)  # 22% of height
```

### Debug Mode
Set in `src/engine/ocr.py`:
```python
DEBUG_MODE = True  # Verbose logging
SAVE_DEBUG_IMAGES = True  # Save captures to debug_output/
```

## File Dependencies

```
main.py
└── src/ui/main_window.py
    ├── src/ui/translate_window.py
    ├── src/ui/region_selector.py
    ├── src/ui/splash_screen.py
    └── src/engine/ocr.py
        ├── src/engine/rag.py
        │   └── src/data/vocabulary.py
        ├── src/engine/window_capture.py
        └── src/engine/ocr_config.py
            └── src/data/ocr_config.json
```
