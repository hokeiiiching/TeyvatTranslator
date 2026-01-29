# OCR Text Recognition Guide

A beginner-friendly guide to understanding and improving the Genshin Translator's OCR system.

---

## How It Works: The Big Picture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CAPTURE    │───▶│  PREPROCESS  │───▶│  PADDLEOCR   │───▶│  TRANSLATE   │
│  (Screen/    │    │  (Clean up   │    │  (Extract    │    │  (RAG or     │
│   Window)    │    │   image)     │    │   text)      │    │   Google)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| [`ocr.py`](file:///d:/cs/Work/Genshin-Translater/src/engine/ocr.py) | Main OCR logic |
| [`window_capture.py`](file:///d:/cs/Work/Genshin-Translater/src/engine/window_capture.py) | Window-based capture |
| `debug_output/` | Saved debug images |

---

## Step 1: Capture

**Where:** `_capture_region()` method (lines 204-253)

**Two modes:**
1. **Window capture** - Uses win32 API to capture specific window
2. **Screen region** - Uses `ImageGrab.grab()` for screen coordinates

**Current code:**
```python
# Window capture mode
if self.window_hwnd:
    from .window_capture import capture_window
    img_np = capture_window(self.window_hwnd)

# Screen region mode  
else:
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    img_np = np.array(img)
```

**Common issues:**
- Wrong coordinates → Captures wrong area
- Window not in focus → May capture blank/stale image

---

## Step 2: Preprocessing

**Where:** `_preprocess_image()` method (lines 255-295)

**What it does:**
1. Convert RGB → Grayscale
2. Check brightness (dark bg = game text)
3. Invert if dark background
4. Apply thresholding
5. Denoise

**Current code:**
```python
# Convert to grayscale
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Check if dark background (game UI)
mean_brightness = np.mean(gray)

if mean_brightness < 128:
    # Dark bg with light text - invert for OCR
    processed = cv2.bitwise_not(gray)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(
    processed, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2  # ← These numbers affect quality!
)
```

### 🔧 How to Improve Preprocessing

**Try different threshold values:**
```python
# Current: blockSize=11, C=2
cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                      cv2.THRESH_BINARY, 11, 2)

# Try larger block for bigger text:
cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                      cv2.THRESH_BINARY, 21, 5)
```

**Add scaling for small text:**
```python
# Scale up small text for better recognition
scale = 2.0
scaled = cv2.resize(gray, None, fx=scale, fy=scale, 
                    interpolation=cv2.INTER_CUBIC)
```

---

## Step 3: PaddleOCR

**Where:** `_extract_text()` method

**What it does:**
1. Uses PaddleOCR (PP-OCR model) to detect and recognize text
2. Optimized for Chinese characters by default
3. Handles text angle detection automatically

**Why PaddleOCR?**
- Significantly better accuracy for Chinese than Tesseract
- Better at handling complex game backgrounds
- Built-in support for vertical and angled text

---

## Step 4: Translation

**Where:** `_process_translation()` method (lines 374-414)

**Flow:**
1. Search RAG vocabulary for exact match
2. If found → Use vocabulary translation
3. If not → Call Google Translate

---

## Debugging Tips

### Enable Debug Mode
Already enabled! Check `debug_output/` folder:
- `capture_X.png` - What was captured
- `processed_X.png` - After preprocessing

### Check Terminal Output
```
OCR RESULTS:
  [PaddleOCR] Detected: '钟离使用元素爆发'
```

---

## Common Problems & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Garbage text | Wrong language | Check PaddleOCR has correct language model |
| Partial text | Region too small | Select larger area |
| No text found | Poor contrast | Adjust threshold values |
| Slow/laggy | Too many OCR attempts | Reduce configs list |
| Wrong window | Stale hwnd | Refresh window list |

---

## Quick Experiments to Try

### 1. Better preprocessing
In `_preprocess_image()`, try:
```python
# Add Gaussian blur before threshold
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
```

### 2. Scale up small text
```python
# Before thresholding
if img.shape[0] < 50:  # If text is small
    img = cv2.resize(img, None, fx=2, fy=2)
```


---

## Files to Edit

1. **Preprocessing:** `src/engine/ocr.py` line 255-295
2. **OCR configs:** `src/engine/ocr.py` line 310-315
3. **Debug images:** Check `debug_output/` folder
