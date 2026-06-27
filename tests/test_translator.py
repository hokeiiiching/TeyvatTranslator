import unittest
from unittest.mock import patch

from src.engine import translator as translator_module
from src.engine.translator import Translator


class TranslatorTests(unittest.TestCase):
    def test_marian_uses_optional_normalized_input(self):
        translator = Translator()
        translator._google_translator = None

        with (
            patch.object(translator_module, "MARIAN_AVAILABLE", True),
            patch.object(translator_module, "_marian_ready", True),
            patch.object(
                translator_module,
                "_translate_marian",
                return_value="Zhongli",
            ) as translate_marian,
        ):
            result = translator.translate("\u937e\u96e2", marian_text="\u949f\u79bb")

        self.assertEqual(result, "Zhongli")
        translate_marian.assert_called_once_with("\u949f\u79bb")

    def test_google_fallback_keeps_original_text(self):
        class FakeGoogleTranslator:
            def __init__(self):
                self.seen_text = None

            def translate(self, text):
                self.seen_text = text
                return f"google:{text}"

        google = FakeGoogleTranslator()
        translator = Translator()
        translator._google_translator = google

        with (
            patch.object(translator_module, "MARIAN_AVAILABLE", True),
            patch.object(translator_module, "_marian_ready", True),
            patch.object(translator_module, "_translate_marian", return_value=None),
        ):
            result = translator.translate("\u937e\u96e2", marian_text="\u949f\u79bb")

        self.assertEqual(result, "google:\u937e\u96e2")
        self.assertEqual(google.seen_text, "\u937e\u96e2")


if __name__ == "__main__":
    unittest.main()
