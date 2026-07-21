from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TraditionalPackagingTests(unittest.TestCase):
    def test_frozen_build_collects_opencc_conversion_data(self):
        spec_text = (ROOT / "TeyvatTranslator.spec").read_text(encoding="utf-8")

        self.assertIn("collect_data_files('opencc')", spec_text)
        self.assertIn("'opencc'", spec_text)

    def test_declared_ocr_versions_match_the_3x_api_used_by_the_app(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("paddlepaddle>=3.0.0,<4.0.0", requirements)
        self.assertIn("paddleocr>=3.0.0,<4.0.0", requirements)

    def test_windows_only_capture_dependency_has_a_platform_marker(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn('pywin32>=306; sys_platform == "win32"', requirements)


if __name__ == "__main__":
    unittest.main()
