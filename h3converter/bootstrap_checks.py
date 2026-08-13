"""Startup self-tests, run by ``start_windows.bat`` before the GUI opens.

Fatal failures -- a missing or broken dependency -- stop the launcher with a
readable explanation. A missing or unsuitable GPU is *not* fatal: the
application still opens so the user can see the diagnosis on the validation
screen, and the affected conversion button stays disabled with a reason.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from h3converter import constants as C
from h3converter.logging_setup import get_logger, start_run_log

log = get_logger("h3converter.bootstrap")

_OK = "  [ ok ]"
_WARN = "  [warn]"
_FAIL = "  [FAIL]"


def _print_versions() -> int:
    from h3converter.hardware import system_info

    info = system_info()
    print(f"{C.APP_NAME} {C.APP_VERSION}")
    print(f"  Python           {info.python_version}  ({info.python_executable})")
    for name, value in info.package_versions.items():
        print(f"  {name:<16} {value}")
    if info.gpu.available:
        print(f"  GPU              {info.gpu.name} ({info.gpu.capability_str})")
    return 0


def run(verbose: bool = True) -> int:
    """Return 0 if the application can start, 1 if it cannot."""
    log_path = start_run_log("startup")
    fatal: list[str] = []
    warnings: list[str] = []

    if verbose:
        print(f"{C.APP_NAME} {C.APP_VERSION}")
        print(f"Log: {log_path}")
        print()

    # --- imports -----------------------------------------------------------
    for module, label, required in (
        ("torch", "PyTorch", True),
        ("safetensors", "safetensors", True),
        ("comfy_kitchen", "comfy-kitchen", True),
        ("numpy", "numpy", True),
        ("psutil", "psutil", False),
        ("PySide6.QtWidgets", "PySide6", False),
    ):
        try:
            __import__(module)
            if verbose:
                print(f"{_OK} import {label}")
        except Exception as exc:  # noqa: BLE001
            message = f"{label} could not be imported: {type(exc).__name__}: {exc}"
            log.error(message)
            if required:
                fatal.append(message)
                if verbose:
                    print(f"{_FAIL} import {label}: {exc}")
            else:
                warnings.append(message)
                if verbose:
                    print(f"{_WARN} import {label}: {exc}")

    if fatal:
        _print_failure(fatal, log_path, verbose)
        return 1

    # --- environment -------------------------------------------------------
    from h3converter.hardware import system_info

    info = system_info()
    log.info("System: %s", info.as_dict())
    if verbose:
        print()
        print(f"  Python {info.python_version}, project venv: {'yes' if info.in_project_venv else 'no'}")
        print(f"  RAM    {info.total_ram_bytes / 1000**3:.0f} GB total, "
              f"{info.available_ram_bytes / 1000**3:.0f} GB available")
        if info.gpu.available:
            print(f"  GPU    {info.gpu.name}, {info.gpu.capability_str}, "
                  f"{info.gpu.total_vram_bytes / 1000**3:.0f} GB VRAM, "
                  f"driver {info.gpu.driver_version or 'unknown'}")
        else:
            print(f"  GPU    not available - {info.gpu.detail}")
        print(f"  torch  {info.package_versions.get('torch')} (CUDA {info.gpu.torch_cuda_version})")
        print()

    if not info.gpu.available:
        warnings.append(
            "No CUDA GPU is available. Conversion will fall back to CPU, which is much slower; "
            "checkpoints produced this way are byte-identical but cannot be runtime-tested here."
        )
    elif info.gpu.torch_cuda_version:
        try:
            major = int(str(info.gpu.torch_cuda_version).split(".")[0])
        except ValueError:
            major = 0
        if major < C.REQUIRED_TORCH_CUDA_MAJOR:
            warnings.append(
                f"torch is built against CUDA {info.gpu.torch_cuda_version}; comfy-kitchen's "
                f"accelerated CUDA backend needs CUDA {C.REQUIRED_TORCH_CUDA_MAJOR}+ and will stay "
                "disabled. Update the NVIDIA driver and reinstall the pinned cu130 wheels."
            )

    # --- quantization self-tests ------------------------------------------
    from h3converter.quant import capability

    try:
        report = capability.probe()
    except Exception as exc:  # noqa: BLE001
        log.exception("Capability probe crashed")
        _print_failure([f"quantization self-test crashed: {exc}"], log_path, verbose)
        return 1

    log.info("Capability report: %s", report.as_dict())
    if verbose:
        for check in report.checks:
            marker = _OK if check.ok else _FAIL
            print(f"{marker} {check.name}: {check.detail}")
        print()

    if not report.group_ok("runtime."):
        _print_failure(
            [f"{c.name}: {c.detail}" for c in report.failures() if c.name.startswith("runtime.")],
            log_path, verbose,
        )
        return 1

    for output_format, label in C.FORMAT_LABELS.items():
        if report.ok_for(output_format):
            if verbose:
                print(f"{_OK} {label}: available")
        else:
            reason = report.blocking_reason(output_format)
            warnings.append(f"{label} is unavailable: {reason}")
            if verbose:
                print(f"{_WARN} {label}: unavailable - {reason}")

    if not any(report.ok_for(f) for f in C.FORMAT_LABELS):
        _print_failure(
            ["neither output format passed its self-test; there is nothing this build can convert"],
            log_path, verbose,
        )
        return 1

    if verbose:
        print()
        for warning in warnings:
            print(f"{_WARN} {warning}")
        print()
        print("Environment OK.")
    return 0


def _print_failure(messages: list[str], log_path, verbose: bool) -> None:
    if not verbose:
        return
    print()
    print("=" * 74)
    print(" STARTUP CHECKS FAILED")
    print("=" * 74)
    for message in messages:
        print(f"  - {message}")
    print()
    print(f" Full diagnostic log: {log_path}")
    print(" Try running update_windows.bat to re-sync the locked dependencies.")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniMax H3 Converter environment self-test")
    parser.add_argument("--versions-only", action="store_true", help="print installed versions and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress output; use the exit code")
    args = parser.parse_args(argv)

    if args.versions_only:
        return _print_versions()
    try:
        return run(verbose=not args.quiet)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
