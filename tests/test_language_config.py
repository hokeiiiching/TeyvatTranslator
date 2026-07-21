import builtins
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from src.engine.language_config import (
    TraditionalChineseSupportError,
    _get_t2s_converter,
    get_paddle_lang,
    make_cache_key,
    normalize_for_lookup,
    validate_source_language_support,
)


OPENCC_AVAILABLE = find_spec("opencc") is not None


class LanguageConfigTests(unittest.TestCase):
    def test_empty_traditional_text_normalizes_to_empty(self):
        self.assertEqual(normalize_for_lookup("", "chi_tra"), "")

    def test_simplified_text_is_unchanged(self):
        text = "\u949f\u79bb"
        self.assertEqual(normalize_for_lookup(text, "chi_sim"), text)

    @unittest.skipUnless(OPENCC_AVAILABLE, "OpenCC dependency is not installed")
    def test_traditional_term_normalizes_to_vocabulary_form(self):
        self.assertEqual(
            normalize_for_lookup("\u937e\u96e2", "chi_tra"),
            "\u949f\u79bb",
        )

    @unittest.skipUnless(OPENCC_AVAILABLE, "OpenCC dependency is not installed")
    def test_traditional_phrase_keeps_line_structure(self):
        text = "\u937e\u96e2\uff1a\u300c\u98a8\u8207\u9f8d\u7684\u5192\u96aa\u3002\u300d\n\u5192\u96aa\u5bb6\u5354\u6703"
        normalized = normalize_for_lookup(text, "chi_tra")

        self.assertIn("\n", normalized)
        self.assertEqual(
            normalized,
            "\u949f\u79bb\uff1a\u300c\u98ce\u4e0e\u9f99\u7684\u5192\u9669\u3002\u300d\n\u5192\u9669\u5bb6\u534f\u4f1a",
        )

    @unittest.skipUnless(OPENCC_AVAILABLE, "OpenCC dependency is not installed")
    def test_traditional_support_preflight_succeeds_with_opencc_data(self):
        validate_source_language_support("chi_tra")

    def test_missing_opencc_fails_instead_of_partially_normalizing(self):
        real_import = builtins.__import__

        def import_without_opencc(name, *args, **kwargs):
            if name == "opencc":
                raise ImportError("simulated missing OpenCC")
            return real_import(name, *args, **kwargs)

        _get_t2s_converter.cache_clear()
        validate_source_language_support.cache_clear()
        try:
            with patch("builtins.__import__", side_effect=import_without_opencc):
                with self.assertRaisesRegex(
                    TraditionalChineseSupportError,
                    "opencc-python-reimplemented",
                ):
                    normalize_for_lookup("\u937e\u96e2", "chi_tra")
        finally:
            _get_t2s_converter.cache_clear()
            validate_source_language_support.cache_clear()

    def test_paddle_language_mapping(self):
        self.assertEqual(get_paddle_lang("chi_sim"), "ch")
        self.assertEqual(get_paddle_lang("chi_tra"), "ch")

    def test_cache_key_separates_source_languages(self):
        text = "\u937e\u96e2"
        self.assertNotEqual(
            make_cache_key("chi_sim", "dialogue", text),
            make_cache_key("chi_tra", "dialogue", text),
        )


if __name__ == "__main__":
    unittest.main()
