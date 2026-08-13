"""Conversion logging.

One timestamped log file per run under ``logs/``, plus console output. Nothing
user-identifying beyond the checkpoint paths the user selected is recorded.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from h3converter.paths import logs_dir

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-24s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_active_file: Path | None = None
# Kept alive for the process lifetime: faulthandler writes through this raw
# handle, so it must not be garbage collected.
_crash_stream = None


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
    _install_crash_handlers(path)
    return path


def _install_crash_handlers(path: Path) -> None:
    """Make a hard crash leave evidence.

    A native fault -- an access violation inside a CUDA driver, a LAPACK
    worker, or a memory-mapped page fault -- kills the process without ever
    reaching a Python ``except`` block, so the log simply stops mid-run with no
    indication of where. ``faulthandler`` writes the C-level and Python-level
    frames of every thread straight to a raw file descriptor when that happens,
    which survives the death of the interpreter.

    The two excepthooks cover the other silent path: an exception raised
    outside a guarded call, or inside a worker thread's ``run``.
    """
    global _crash_stream

    try:
        if _crash_stream is not None:
            faulthandler.disable()
            _crash_stream.close()
        # A second, separately buffered handle to the same log file. Logging's
        # own handler flushes per record, so the two never interleave a
        # half-written line.
        _crash_stream = open(path, "a", encoding="utf-8", errors="replace")
        faulthandler.enable(file=_crash_stream, all_threads=True)
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        _crash_stream = None

    root = logging.getLogger()

    def handle_exception(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        root.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    def handle_thread_exception(args) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        root.critical(
            "Unhandled exception in thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception


def active_log_path() -> Path | None:
    return _active_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
