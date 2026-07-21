from importlib.util import find_spec
import unittest


PYPINYIN_AVAILABLE = find_spec("pypinyin") is not None

if PYPINYIN_AVAILABLE:
    from pypinyin import Style, pinyin


@unittest.skipUnless(PYPINYIN_AVAILABLE, "pypinyin dependency is not installed")
class TraditionalPinyinTests(unittest.TestCase):
    def test_traditional_and_simplified_names_have_the_same_pronunciation(self):
        simplified = pinyin("\u949f\u79bb", style=Style.TONE)
        traditional = pinyin("\u937e\u96e2", style=Style.TONE)

        self.assertEqual(traditional, simplified)
        self.assertEqual(traditional, [["zh\u014dng"], ["l\u00ed"]])


if __name__ == "__main__":
    unittest.main()
