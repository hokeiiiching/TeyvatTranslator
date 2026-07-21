from importlib.util import find_spec
import unittest

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None

from src.engine.context import ContextEngine
from src.engine.language_config import normalize_for_lookup
from src.data.vocabulary import VOCABULARY


OPENCC_AVAILABLE = find_spec("opencc") is not None


@unittest.skipUnless(OPENCC_AVAILABLE, "OpenCC dependency is not installed")
class TraditionalLookupTests(unittest.TestCase):
    def test_normalized_traditional_finds_simplified_vocabulary_entry(self):
        engine = ContextEngine()

        lookup_text = normalize_for_lookup("\u937e\u96e2", "chi_tra")
        matches = engine.find_matches(lookup_text)

        self.assertTrue(matches)
        self.assertEqual(matches[0].get("mandarin"), "\u949f\u79bb")
        self.assertEqual(matches[0].get("english"), "Zhongli")

    def test_every_vocabulary_term_survives_traditional_round_trip(self):
        to_traditional = OpenCC("s2t")

        failures = []
        for term in VOCABULARY:
            simplified = term.get("mandarin", "")
            traditional = to_traditional.convert(simplified)
            normalized = normalize_for_lookup(traditional, "chi_tra")
            if normalized != simplified:
                failures.append((simplified, traditional, normalized))

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
