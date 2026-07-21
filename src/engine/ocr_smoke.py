"""Real-model smoke check shared by tests and the frozen Windows build."""

from pathlib import Path


TRADITIONAL_SMOKE_TEXT = "鍾離：風與龍的冒險。"


def run_real_ocr_parity_smoke() -> str:
    """Recognize Traditional text through the one shared Chinese OCR profile."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from .language_config import get_paddle_lang
    from . import ocr

    if get_paddle_lang("chi_sim") != "ch" or get_paddle_lang("chi_tra") != "ch":
        raise AssertionError("Simplified and Traditional must share the 'ch' OCR profile")

    simplified_engine = ocr.get_paddle_ocr("chi_sim")
    traditional_engine = ocr.get_paddle_ocr("chi_tra")
    if simplified_engine is not traditional_engine:
        raise AssertionError("Simplified and Traditional created different OCR instances")

    font_candidates = (
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    )
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path is None:
        raise RuntimeError("No Traditional Chinese smoke-test font found")

    image = Image.new("RGB", (1200, 220), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 72)
    draw.text((30, 55), TRADITIONAL_SMOKE_TEXT, fill="black", font=font)

    results = traditional_engine.predict(np.array(image))
    recognized = "".join(results[0].get("rec_texts", []))
    if recognized != TRADITIONAL_SMOKE_TEXT:
        raise AssertionError(
            f"Traditional OCR mismatch: expected {TRADITIONAL_SMOKE_TEXT!r}, "
            f"got {recognized!r}"
        )

    return recognized
