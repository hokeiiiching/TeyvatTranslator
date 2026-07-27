import inspect
import sys
import types
import unittest
from threading import Lock
from unittest.mock import ANY, MagicMock, patch


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


class FakeShelf(dict):
    def sync(self):
        pass


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

    def test_minimized_window_skip_logs_target_window(self):
        from src.engine import window_capture

        fake_win32gui = MagicMock()
        fake_win32gui.IsIconic.return_value = True
        fake_win32gui.GetWindowText.side_effect = lambda hwnd: {
            123: "Genshin Impact",
        }.get(hwnd, "")

        with (
            patch.object(window_capture, "HAS_WIN32", True),
            patch.object(window_capture, "win32gui", fake_win32gui, create=True),
            patch.object(window_capture, "_last_dialogue_capture_state", None),
            patch.object(window_capture, "_last_dialogue_capture_log", 0.0),
            self.assertLogs("WindowCapture", level="INFO") as captured_logs,
        ):
            result = window_capture.capture_window_dialogue(123)

        self.assertIsNone(result)
        log_text = "\n".join(captured_logs.output)
        self.assertIn("window-minimized", log_text)
        self.assertIn("Genshin Impact", log_text)

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
                    job_id=ANY,
                    segment="dialogue",
                )
                self.assertEqual(worker.translate_window.updates[-1][0], dialogue)

    def test_aligned_traveller_choices_translate_individually(self):
        worker = self._worker("chi_sim")
        worker._translate_with_retry.side_effect = [
            "Let's set out together.",
            "I want to stay here.",
        ]
        text = "一起出发吧\n我想留在这里"
        result = ocr.OCRResult(
            text=text,
            confidence=0.96,
            num_lines=2,
            chinese_chars=11,
            total_chars=11,
            bounding_boxes=[
                {
                    "text": "一起出发吧",
                    "confidence": 0.97,
                    "box": [900.0, 300.0, 1250.0, 350.0],
                },
                {
                    "text": "我想留在这里",
                    "confidence": 0.95,
                    "box": [906.0, 390.0, 1320.0, 440.0],
                },
            ],
            image_size=(2000, 1000),
        )

        outcome = worker._process_translation(
            text,
            ocr_result=result,
            job_id="choice-job",
        )

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.layout, "choices")
        chinese, english, context = worker.translate_window.updates[-1]
        self.assertEqual(chinese, text)
        self.assertEqual(
            english,
            "Let's set out together.\nI want to stay here.",
        )
        self.assertEqual(context["layout"], "choices")
        self.assertEqual(context["choices"], text.splitlines())
        self.assertEqual(
            [call.kwargs["segment"] for call in worker._translate_with_retry.call_args_list],
            ["choice-1", "choice-2"],
        )
        self.assertEqual(
            [call.args[0] for call in worker._translate_with_retry.call_args_list],
            text.splitlines(),
        )

    def test_choice_markers_do_not_destabilize_or_pollute_translation(self):
        worker = self._worker("chi_tra")
        parsed = worker._parse_translation_input(
            "◆ 一起出發吧\n◇ 我想留在這裡"
        )

        self.assertEqual(parsed.layout, "choices")
        self.assertEqual(parsed.choices, ["一起出發吧", "感想留在這裡".replace("感想", "我想")])
        self.assertTrue(any("choice-markers" in item for item in parsed.evidence))

    def test_three_line_dialogue_is_not_misclassified_as_choices(self):
        worker = self._worker("chi_tra")
        parsed1 = worker._parse_translation_input(
            "樂平波琳\n嗯…場景到位之後效果比我預想得好多了\n就把片子剪出來。"
        )
        self.assertEqual(parsed1.layout, "dialogue")
        self.assertEqual(parsed1.speaker, "樂平波琳")
        self.assertEqual(
            parsed1.dialogue,
            "嗯…場景到位之後效果比我預想得好多了\n就把片子剪出來。",
        )

        parsed2 = worker._parse_translation_input(
            "澤維爾\n映影製片人\n那就這麼說定了！"
        )
        self.assertEqual(parsed2.layout, "dialogue")
        self.assertEqual(parsed2.speaker, "澤維爾")
        self.assertEqual(parsed2.descriptor, "映影製片人")
        self.assertEqual(parsed2.dialogue, "那就這麼說定了！")

    def test_unavailable_translation_is_not_cached_or_displayed(self):
        worker = self._worker("chi_sim")
        worker._translate_with_retry.return_value = None

        failed = worker._process_translation(
            "这个选项暂时失败。",
            job_id="failed-job",
        )

        self.assertFalse(failed.success)
        worker._cache_set.assert_not_called()
        self.assertEqual(worker.translate_window.updates, [])

        worker._translate_with_retry.return_value = "This option now works."
        succeeded = worker._process_translation(
            "这个选项暂时失败。",
            job_id="retry-job",
        )

        self.assertTrue(succeeded.success)
        self.assertEqual(
            worker.translate_window.updates[-1][1],
            "This option now works.",
        )

    def test_stale_unavailable_cache_entry_is_removed(self):
        worker = object.__new__(ocr.OCRWorker)
        worker.worker_id = "cache-test"
        worker.from_lang = "chi_tra"
        worker.frame_count = 0
        worker._cache_lock = Lock()
        worker._shelve_cache = FakeShelf(
            {"chi_tra:choice:一起出發吧": ocr.TRANSLATION_UNAVAILABLE}
        )
        worker._pipeline_event = MagicMock()

        cached = worker._cache_get("chi_tra:choice:一起出發吧")

        self.assertIsNone(cached)
        self.assertNotIn(
            "chi_tra:choice:一起出發吧",
            worker._shelve_cache,
        )
        worker._pipeline_event.assert_called_once_with(
            "cache_entry_discarded",
            cache_key="chi_tra:choice:一起出發吧",
            cached_value=ocr.TRANSLATION_UNAVAILABLE,
        )

    def test_current_text_commits_only_after_successful_translation(self):
        worker = object.__new__(ocr.OCRWorker)
        worker.worker_id = "worker-test"
        worker.frame_count = 10
        worker.current_text = None
        worker._failed_text = None
        worker._failed_text_attempts = 0
        worker._translation_retry_after = 0.0
        worker._pending_job_started = ocr.time.perf_counter()
        worker._last_capture_image = None
        worker._update_status = MagicMock()
        worker._pipeline_event = MagicMock()
        worker._save_diagnostic_snapshot = MagicMock()
        text = "我们走吧。"

        worker._handle_translation_outcome(
            ocr.TranslationOutcome(
                job_id="job-1",
                source_text=text,
                display_text=text,
                translated_text="",
                success=False,
                layout="dialogue",
                error="temporary backend failure",
            )
        )
        self.assertIsNone(worker.current_text)
        self.assertGreater(worker._translation_retry_after, ocr.time.perf_counter())

        worker._handle_translation_outcome(
            ocr.TranslationOutcome(
                job_id="job-2",
                source_text=text,
                display_text=text,
                translated_text="Let's go.",
                success=True,
                layout="dialogue",
            )
        )
        self.assertEqual(worker.current_text, text)
        self.assertEqual(worker._translation_retry_after, 0.0)

    def test_pipeline_event_handles_duplicate_source_language_kwarg(self):
        worker = object.__new__(ocr.OCRWorker)
        worker.worker_id = "worker-kwarg-test"
        worker.frame_count = 5
        worker.from_lang = "chi_tra"

        with patch("src.engine.ocr.record_pipeline_event") as mock_record:
            # Calling _pipeline_event with explicit source_language in details
            # should not raise TypeError from duplicate kwargs
            worker._pipeline_event(
                "worker_created",
                source_language="chi_tra",
                target_language="eng",
            )
            mock_record.assert_called_once_with(
                "ocr_worker",
                "worker_created",
                worker_id="worker-kwarg-test",
                frame=5,
                source_language="chi_tra",
                target_language="eng",
            )


if __name__ == "__main__":
    unittest.main()
