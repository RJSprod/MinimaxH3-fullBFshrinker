"""Conversion logging.

One timestamped log file per run under ``logs/``, plus console output. Nothing
user-identifying beyond the checkpoint paths the user selected is recorded.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from h3converter.paths import logs_dir

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-24s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_active_file: Path | None = None


def start_run_log(suffix: str = "conversion") -> Path:
    """Attach a fresh file handler for this run and return its path."""
    global _active_file

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = logs_dir() / f"{stamp}_{suffix}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        # Keep the console quiet enough for the CLI progress bar; the file
        # handler still records everything at DEBUG.
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        root.addHandler(console)

    _active_file = path
    return path


def active_log_path() -> Path | None:
    return _active_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
