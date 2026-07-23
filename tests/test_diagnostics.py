import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticsTests(unittest.TestCase):
    def test_session_log_is_user_writable_and_preserves_traditional_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script = """
import logging
from pathlib import Path
from src.diagnostics import (
    configure_diagnostics,
    get_capture_directory,
    get_log_file,
    LATEST_SESSION_FILE,
)

configure_diagnostics("test-version")
logging.getLogger("DiagnosticsTest").info(
    "source=chi_tra shared_profile=ch raw_text=%r confidence=%.3f",
    "鍾離：風與龍的冒險。",
    0.987,
)
for handler in logging.getLogger().handlers:
    handler.flush()
log_file = get_log_file()
assert log_file.is_file(), log_file
assert get_capture_directory().is_dir()
assert LATEST_SESSION_FILE.read_text(encoding="utf-8").strip() == str(log_file.parent)
text = log_file.read_text(encoding="utf-8")
assert "app_version=test-version" in text
assert "source=chi_tra shared_profile=ch" in text
assert "鍾離：風與龍的冒險。" in text
print(log_file)
"""
            env = os.environ.copy()
            env["TEYVAT_DIAGNOSTICS_DIR"] = temp_dir
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_path = Path(result.stdout.strip().splitlines()[-1])
            self.assertTrue(log_path.is_relative_to(Path(temp_dir)))

    def test_release_smoke_reads_the_same_latest_session_log(self):
        workflow = (
            ROOT / ".github" / "workflows" / "release-windows.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("latest-session.txt", workflow)
        self.assertIn("diagnostics.log", workflow)
        self.assertNotIn(
            r".\dist\TeyvatTranslator\TeyvatTranslator.log",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
