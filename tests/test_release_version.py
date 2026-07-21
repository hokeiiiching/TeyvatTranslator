import re
from pathlib import Path
import unittest

import build


ROOT = Path(__file__).resolve().parents[1]


class ReleaseVersionTests(unittest.TestCase):
    def test_release_versions_stay_in_sync(self):
        version = build.VERSION
        installer = (ROOT / "TeyvatTranslator.iss").read_text(encoding="utf-8")
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        main_window = (ROOT / "src/ui/main_window.py").read_text(encoding="utf-8")

        installer_version = re.search(
            r'#define MyAppVersion "([^"]+)"', installer
        ).group(1)
        app_version = re.search(
            r'app\.setApplicationVersion\("([^"]+)"\)', main
        ).group(1)
        about_version = re.search(
            r'version = QLabel\("v([^"]+)"\)', main_window
        ).group(1)

        self.assertEqual(installer_version, version)
        self.assertEqual(app_version, version)
        self.assertEqual(about_version, version)


if __name__ == "__main__":
    unittest.main()
