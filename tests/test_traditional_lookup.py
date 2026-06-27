import unittest

from src.engine.context import ContextEngine
from src.engine.language_config import normalize_for_lookup


class TraditionalLookupTests(unittest.TestCase):
    def test_normalized_traditional_finds_simplified_vocabulary_entry(self):
        engine = ContextEngine()

        lookup_text = normalize_for_lookup("\u937e\u96e2", "chi_tra")
        matches = engine.find_matches(lookup_text)

        self.assertTrue(matches)
        self.assertEqual(matches[0].get("mandarin"), "\u949f\u79bb")
        self.assertEqual(matches[0].get("english"), "Zhongli")


if __name__ == "__main__":
    unittest.main()
