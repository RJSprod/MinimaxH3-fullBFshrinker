"""Application entry point: ``python -m h3converter.app``.

Starts the run log, then opens the native GUI. If PySide6 is unavailable the
command line path is still fully functional, so the user is told how to use it
rather than left with a traceback.
"""

from __future__ import annotations

import sys

from h3converter import constants as C
from h3converter.logging_setup import get_logger, start_run_log

log = get_logger("h3converter.app")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    log_path = start_run_log("session")
    log.info("%s %s starting; log at %s", C.APP_NAME, C.APP_VERSION, log_path)

    try:
        from h3converter.gui import run_gui
    except ImportError as exc:
        log.error("PySide6 is unavailable: %s", exc)
        print(f"{C.APP_NAME} {C.APP_VERSION}")
        print()
        print(f"The desktop interface could not start: {exc}")
        print()
        print("Run update_windows.bat to repair the environment, or use the command line:")
        print("    uv run h3convert <checkpoint.safetensors> --format a")
        print("    uv run h3convert <checkpoint.safetensors> --format b")
        print(f"\nLog: {log_path}")
        return 1

    try:
        return run_gui(argv)
    except Exception:  # noqa: BLE001
        log.exception("The application terminated with an unhandled error")
        print(f"The application stopped unexpectedly. See the log: {log_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
