import inspect
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# Text-to-speech is optional for these worker-level tests. Stub it only when the
# local test environment has not installed the full desktop requirements.
try:
    import pyttsx3  # noqa: F401
except ImportError:
    sys.modules["pyttsx3"] = types.SimpleNamespace(init=lambda: None)

from src.engine import language_config
from src.engine import ocr


class FakeConverter:
    _translation = str.maketrans(
        {
            "\u937e": "\u949f",
            "\u96e2": "\u79bb",
            "\u5143": "\u5143",
            "\u7d20": "\u7d20",
            "\u7206": "\u7206",
            "\u767c": "\u53d1",
            "\u98a8": "\u98ce",
            "\u8207": "\u4e0e",
            "\u9f8d": "\u9f99",
            "\u7684": "\u7684",
            "\u5192": "\u5192",
            "\u96aa": "\u9669",
        }
    )

    def convert(self, text):
        return text.translate(self._translation)


class FakeRag:
    terms = {
        "\u949f\u79bb": {
            "mandarin": "\u949f\u79bb",
            "pinyin": "zh\u014dng l\u00ed",
            "english": "Zhongli",
        },
        "\u5143\u7d20\u7206\u53d1": {
            "mandarin": "\u5143\u7d20\u7206\u53d1",
            "pinyin": "yu\u00e1n s\u00f9 b\u00e0o f\u0101",
            "english": "Elemental Burst",
        },
    }

    def get_context(self, text):
        term = self.terms.get(text)
        return [term] if term else []


class FakeTranslateWindow:
    def __init__(self):
        self.updates = []

    def update_translation(self, chinese, english, context):
        self.updates.append((chinese, english, context))


class OCRLanguageParityTests(unittest.TestCase):
    def setUp(self):
        self.converter_patch = patch.object(
            language_config,
            "_get_t2s_converter",
            return_value=FakeConverter(),
        )
        self.converter_patch.start()

    def tearDown(self):
        self.converter_patch.stop()

    def _worker(self, source_lang):
        worker = object.__new__(ocr.OCRWorker)
        worker.from_lang = source_lang
        worker._enable_context = True
        worker._rag_engine = FakeRag()
        worker.translate_window = FakeTranslateWindow()
        worker.enable_tts = False
        worker.tts_engine = None
        worker._cache_get = MagicMock(return_value=None)
        worker._cache_set = MagicMock()
        worker._translate_with_retry = MagicMock(return_value="Translated dialogue")
        return worker

    def test_both_scripts_reuse_the_same_working_ocr_instance(self):
        shared_chinese_ocr = object()

        with (
            patch.object(ocr, "PADDLE_AVAILABLE", True),
            patch.dict(
                ocr._global_paddle_ocr,
                {"ch": shared_chinese_ocr},
                clear=True,
            ),
        ):
            self.assertIs(ocr.get_paddle_ocr("chi_sim"), shared_chinese_ocr)
            self.assertIs(ocr.get_paddle_ocr("chi_tra"), shared_chinese_ocr)

    def test_preloading_both_scripts_starts_only_one_ocr_initialization(self):
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True

        with (
            patch.object(ocr, "PADDLE_AVAILABLE", True),
            patch.dict(ocr._global_paddle_ocr, {}, clear=True),
            patch.dict(ocr._ocr_ready, {}, clear=True),
            patch.dict(ocr._ocr_init_threads, {}, clear=True),
            patch.object(ocr, "Thread", return_value=fake_thread) as thread_class,
        ):
            ocr.preload_ocr("chi_sim")
            ocr.preload_ocr("chi_tra")

        thread_class.assert_called_once()
        fake_thread.start.assert_called_once_with()

    def test_unavailable_ocr_reports_an_error_instead_of_returning_none(self):
        with (
            patch.object(ocr, "PADDLE_AVAILABLE", False),
            patch.object(ocr, "PADDLE_ERROR", "simulated import failure"),
        ):
            with self.assertRaisesRegex(
                ocr.OCRInitializationError,
                "simulated import failure",
            ):
                ocr.get_paddle_ocr("chi_tra")

    def test_dialogue_capture_is_identical_for_both_scripts(self):
        captured = ocr.np.arange(120 * 200 * 3, dtype=ocr.np.uint8).reshape(
            (120, 200, 3)
        )
        outputs = {}

        for source_lang in ("chi_sim", "chi_tra"):
            worker = object.__new__(ocr.OCRWorker)
            worker.from_lang = source_lang
            worker.window_hwnd = 123
            worker.dialogue_only = True
            worker.frame_count = 1

            with patch(
                "src.engine.window_capture.capture_window_dialogue",
                return_value=captured.copy(),
            ):
                outputs[source_lang] = worker._capture_region()

        ocr.np.testing.assert_array_equal(outputs["chi_sim"], captured)
        ocr.np.testing.assert_array_equal(outputs["chi_tra"], captured)

    def test_core_pipeline_has_no_script_specific_branches(self):
        for method in (
            ocr.OCRWorker._run_loop,
            ocr.OCRWorker._capture_region,
            ocr.OCRWorker._process_translation,
        ):
            with self.subTest(method=method.__name__):
                self.assertNotIn("chi_tra", inspect.getsource(method))

    def test_ocr_inference_uses_the_paddleocr_3x_prediction_api(self):
        worker = object.__new__(ocr.OCRWorker)
        worker.from_lang = "chi_tra"
        worker.frame_count = 1
        worker.paddle_ocr = MagicMock()
        worker.paddle_ocr.predict.return_value = [
            {"rec_texts": ["\u937e\u96e2"], "rec_scores": [0.99]}
        ]
        image = ocr.np.zeros((100, 200, 3), dtype=ocr.np.uint8)

        with patch.object(ocr, "PADDLE_AVAILABLE", True):
            result = worker._extract_text(image)

        self.assertEqual(result.text, "\u937e\u96e2")
        worker.paddle_ocr.predict.assert_called_once()

    def test_traditional_lookup_preserves_original_display_text(self):
        worker = self._worker("chi_tra")

        with (
            patch.object(ocr, "PINYIN_AVAILABLE", True),
            patch.object(
                ocr,
                "pinyin",
                side_effect=lambda text, style: [[f"py-{char}"] for char in text],
                create=True,
            ),
            patch.object(ocr, "Style", types.SimpleNamespace(TONE="tone"), create=True),
        ):
            worker._process_translation("\u937e\u96e2\n\u5143\u7d20\u7206\u767c")

        chinese, english, context = worker.translate_window.updates[-1]
        self.assertEqual(chinese, "\u5143\u7d20\u7206\u767c")
        self.assertEqual(english, "Elemental Burst")
        self.assertEqual(context["speaker"], "\u937e\u96e2")
        self.assertEqual(context["speaker_english"], "Zhongli")
        worker._translate_with_retry.assert_not_called()

    def test_each_script_uses_its_own_cache_and_marian_input(self):
        cases = (
            (
                "chi_sim",
                "\u949f\u79bb\n\u98ce\u4e0e\u9f99\u7684\u5192\u9669\u3002",
                "\u98ce\u4e0e\u9f99\u7684\u5192\u9669\u3002",
                None,
            ),
            (
                "chi_tra",
                "\u937e\u96e2\n\u98a8\u8207\u9f8d\u7684\u5192\u96aa\u3002",
                "\u98a8\u8207\u9f8d\u7684\u5192\u96aa\u3002",
                "\u98ce\u4e0e\u9f99\u7684\u5192\u9669\u3002",
            ),
        )

        for source_lang, captured, dialogue, marian_text in cases:
            with self.subTest(source_lang=source_lang):
                worker = self._worker(source_lang)
                with patch.object(ocr, "PINYIN_AVAILABLE", False):
                    worker._process_translation(captured)

                worker._cache_get.assert_called_once_with(
                    f"{source_lang}:dialogue:{dialogue}"
                )
                worker._translate_with_retry.assert_called_once_with(
                    dialogue,
                    max_retries=3,
                    marian_text=marian_text,
                )
                self.assertEqual(worker.translate_window.updates[-1][0], dialogue)


if __name__ == "__main__":
    unittest.main()
