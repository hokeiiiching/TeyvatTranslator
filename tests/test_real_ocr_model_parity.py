import os
import unittest


RUN_REAL_OCR_SMOKE = os.environ.get("TEYVAT_REAL_OCR_SMOKE") == "1"


@unittest.skipUnless(
    RUN_REAL_OCR_SMOKE,
    "set TEYVAT_REAL_OCR_SMOKE=1 to download and run the real OCR model",
)
class RealOCRModelParityTests(unittest.TestCase):
    def test_working_chinese_profile_recognizes_traditional_text_for_both_modes(self):
        from src.engine.ocr_smoke import (
            TRADITIONAL_SMOKE_TEXT,
            run_real_ocr_parity_smoke,
        )

        self.assertEqual(run_real_ocr_parity_smoke(), TRADITIONAL_SMOKE_TEXT)


if __name__ == "__main__":
    unittest.main()
