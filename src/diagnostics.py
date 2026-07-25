"""Central diagnostics for the desktop application.

Every launch writes to a new, user-writable session directory.  Keeping the
logs per session makes it possible to inspect startup failures without a later
launch overwriting the evidence.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path


APP_NAME = "TeyvatTranslator"
SESSION_ID = uuid.uuid4().hex[:8]
SESSION_STARTED = datetime.now()


def _local_app_data() -> Path:
    override = os.environ.get("TEYVAT_DIAGNOSTICS_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))


def _app_data_root() -> Path:
    if os.environ.get("TEYVAT_DIAGNOSTICS_DIR"):
        return _local_app_data()
    return _local_app_data() / APP_NAME


DIAGNOSTICS_ROOT = _app_data_root() / "diagnostics"
STATE_ROOT = _app_data_root() / "state"
SESSION_DIRECTORY = DIAGNOSTICS_ROOT / (
    f"{SESSION_STARTED:%Y%m%d-%H%M%S}-{SESSION_ID}"
)
LOG_FILE = SESSION_DIRECTORY / "diagnostics.log"
EVENTS_FILE = SESSION_DIRECTORY / "pipeline-events.jsonl"
CAPTURE_DIRECTORY = SESSION_DIRECTORY / "captures"
LATEST_SESSION_FILE = DIAGNOSTICS_ROOT / "latest-session.txt"

_configured = False
_stdio_file = None
_original_excepthook = sys.excepthook
_event_lock = threading.Lock()


def configure_diagnostics(app_version: str) -> Path:
    """Configure the shared file logger and return this session's log path."""
    global _configured, _stdio_file
    if _configured:
        return LOG_FILE

    SESSION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_SESSION_FILE.write_text(str(SESSION_DIRECTORY), encoding="utf-8")
    EVENTS_FILE.touch(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    handler._teyvat_diagnostics_handler = True  # type: ignore[attr-defined]
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
            "%(threadName)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)

    # A windowed PyInstaller executable has no console streams.  Point plain
    # print output from dependencies at the same session log so it is not lost.
    if getattr(sys, "frozen", False):
        _stdio_file = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        sys.stdout = _stdio_file
        sys.stderr = _stdio_file

    _install_exception_hooks()
    _configured = True

    logger = logging.getLogger("Diagnostics")
    logger.info("=" * 72)
    logger.info("DIAGNOSTIC SESSION STARTED")
    logger.info("session_id=%s app_version=%s", SESSION_ID, app_version)
    logger.info("session_directory=%s", SESSION_DIRECTORY)
    logger.info("log_file=%s", LOG_FILE)
    logger.info("pipeline_events_file=%s", EVENTS_FILE)
    log_runtime_environment(app_version)
    return LOG_FILE


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback):
        logging.getLogger("Diagnostics").critical(
            "UNHANDLED MAIN-THREAD EXCEPTION",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if _original_excepthook:
            _original_excepthook(exc_type, exc_value, exc_traceback)

    def handle_thread_exception(args):
        logging.getLogger("Diagnostics").critical(
            "UNHANDLED THREAD EXCEPTION thread=%s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = handle_thread_exception


def log_runtime_environment(app_version: str) -> None:
    """Record safe runtime and dependency details without dumping environment secrets."""
    logger = logging.getLogger("Diagnostics")
    logger.info(
        "runtime app_version=%s frozen=%s python=%s platform=%s",
        app_version,
        bool(getattr(sys, "frozen", False)),
        sys.version.replace("\n", " "),
        platform.platform(),
    )
    logger.info(
        "process executable=%s cwd=%s argv=%r",
        sys.executable,
        os.getcwd(),
        sys.argv,
    )
    logger.info(
        "model_flags source_check=%s pdx_source_check=%s mkldnn=%s flags_mkldnn=%s",
        os.environ.get("DISABLE_MODEL_SOURCE_CHECK"),
        os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"),
        os.environ.get("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"),
        os.environ.get("FLAGS_use_mkldnn"),
    )
    versions = {}
    for package in (
        "paddleocr",
        "paddlepaddle",
        "paddlex",
        "opencv-python",
        "numpy",
        "Pillow",
        "PyQt6",
        "opencc-python-reimplemented",
        "transformers",
        "torch",
        "deep-translator",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    logger.info(
        "dependency_versions %s",
        " ".join(f"{name}={version}" for name, version in versions.items()),
    )


def get_diagnostics_root() -> Path:
    return DIAGNOSTICS_ROOT


def get_session_directory() -> Path:
    return SESSION_DIRECTORY


def get_log_file() -> Path:
    return LOG_FILE


def get_events_file() -> Path:
    return EVENTS_FILE


def record_pipeline_event(component: str, event: str, **details) -> None:
    """Append a machine-readable pipeline event without risking app execution."""
    try:
        SESSION_DIRECTORY.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "session_id": SESSION_ID,
            "component": component,
            "event": event,
            **details,
        }
        with _event_lock:
            with EVENTS_FILE.open("a", encoding="utf-8") as event_file:
                event_file.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=lambda value: repr(value),
                    )
                    + "\n"
                )
    except Exception:
        logging.getLogger("Diagnostics").exception(
            "Could not record pipeline event component=%s event=%s",
            component,
            event,
        )


def get_capture_directory() -> Path:
    CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return CAPTURE_DIRECTORY


def get_state_directory() -> Path:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return STATE_ROOT


def open_diagnostics_folder() -> None:
    """Open the diagnostics root in the platform file manager."""
    DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(DIAGNOSTICS_ROOT))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(DIAGNOSTICS_ROOT)])
    else:
        subprocess.Popen(["xdg-open", str(DIAGNOSTICS_ROOT)])
