from importlib.util import find_spec
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


PYQT_AVAILABLE = find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    if find_spec("pyperclip") is None:
        sys.modules["pyperclip"] = types.SimpleNamespace(copy=lambda text: None)
    if find_spec("pyttsx3") is None:
        sys.modules["pyttsx3"] = types.SimpleNamespace(init=lambda: None)

    from PyQt6.QtWidgets import QApplication

    from src.engine.language_config import TraditionalChineseSupportError
    from src.ui import main_window
    from src.ui.translate_window import CHINESE_HTML_FONT_FAMILY


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 dependency is not installed")
class MainWindowLanguageSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        refresh_patch = patch.object(main_window.MainWindow, "_refresh_window_list")
        refresh_patch.start()
        self.addCleanup(refresh_patch.stop)
        self.window = main_window.MainWindow()
        self.addCleanup(self.window.close)

    def test_simplified_is_default_and_traditional_is_selectable(self):
        self.assertEqual(self.window._selected_source_lang(), "chi_sim")

        with patch("src.engine.ocr.preload_ocr") as preload_ocr:
            self.window.source_lang_buttons["chi_tra"].click()

        self.assertEqual(self.window._selected_source_lang(), "chi_tra")
        self.assertIn("Chinese (Traditional)", self.window.lang_info.text())
        preload_ocr.assert_called_once_with("chi_tra")

    def test_selected_traditional_code_reaches_the_ocr_worker(self):
        self.window.source_lang_buttons["chi_tra"].setChecked(True)
        self.window.selected_region = (10, 20, 310, 220)
        overlay = MagicMock()
        worker = MagicMock()

        with (
            patch.object(main_window, "validate_source_language_support") as validate,
            patch.object(main_window, "TranslateWindow", return_value=overlay),
            patch.object(main_window, "OCRWorker", return_value=worker) as worker_class,
        ):
            self.window._on_start_translation()

        validate.assert_called_once_with("chi_tra")
        self.assertEqual(worker_class.call_args.args[4], "chi_tra")
        overlay.set_worker.assert_called_once_with(worker)
        worker.start.assert_called_once_with()

    def test_failed_traditional_preflight_does_not_start_ocr(self):
        self.window.source_lang_buttons["chi_tra"].setChecked(True)
        self.window.selected_region = (10, 20, 310, 220)

        with (
            patch.object(
                main_window,
                "validate_source_language_support",
                side_effect=TraditionalChineseSupportError("OpenCC data missing"),
            ),
            patch.object(main_window.QMessageBox, "critical") as critical,
            patch.object(main_window, "OCRWorker") as worker_class,
        ):
            self.window._on_start_translation()

        worker_class.assert_not_called()
        critical.assert_called_once()

    def test_overlay_font_stack_includes_a_traditional_chinese_font(self):
        self.assertIn("Microsoft JhengHei", CHINESE_HTML_FONT_FAMILY)


if __name__ == "__main__":
    unittest.main()
