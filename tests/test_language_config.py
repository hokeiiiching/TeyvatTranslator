import unittest

from src.engine.language_config import (
    get_paddle_lang,
    make_cache_key,
    normalize_for_lookup,
)


class LanguageConfigTests(unittest.TestCase):
    def test_empty_traditional_text_normalizes_to_empty(self):
        self.assertEqual(normalize_for_lookup("", "chi_tra"), "")

    def test_simplified_text_is_unchanged(self):
        text = "\u949f\u79bb"
        self.assertEqual(normalize_for_lookup(text, "chi_sim"), text)

    def test_traditional_term_normalizes_to_vocabulary_form(self):
        self.assertEqual(
            normalize_for_lookup("\u937e\u96e2", "chi_tra"),
            "\u949f\u79bb",
        )

    def test_traditional_phrase_keeps_line_structure(self):
        text = "\u937e\u96e2\n\u5192\u96aa\u5bb6\u5354\u6703"
        normalized = normalize_for_lookup(text, "chi_tra")

        self.assertIn("\n", normalized)
        self.assertEqual(normalized.splitlines()[0], "\u949f\u79bb")

    def test_paddle_language_mapping(self):
        self.assertEqual(get_paddle_lang("chi_sim"), "ch")
        self.assertEqual(get_paddle_lang("chi_tra"), "chinese_cht")

    def test_cache_key_separates_source_languages(self):
        text = "\u937e\u96e2"
        self.assertNotEqual(
            make_cache_key("chi_sim", "dialogue", text),
            make_cache_key("chi_tra", "dialogue", text),
        )


if __name__ == "__main__":
    unittest.main()
