"""System, GPU, memory and disk diagnostics.

Collected at startup for the validation screen and the conversion report, and
used to refuse a conversion that cannot finish -- insufficient free disk, or a
driver too old for the pinned CUDA build -- before anything is written.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from h3converter import constants as C


@dataclass
class GPUInfo:
    available: bool = False
    name: str | None = None
    compute_capability: tuple[int, int] | None = None
    total_vram_bytes: int = 0
    free_vram_bytes: int = 0
    driver_version: str | None = None
    torch_cuda_version: str | None = None
    detail: str = ""

    @property
    def capability_str(self) -> str:
        if not self.compute_capability:
            return "unknown"
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"

    def supports(self, output_format: str) -> bool:
        if not self.available or self.compute_capability is None:
            return False
        needed = (
            C.MIN_COMPUTE_CAPABILITY_W4A8
            if output_format == C.FORMAT_W4A8
            else C.MIN_COMPUTE_CAPABILITY_NVFP4
        )
        return self.compute_capability >= needed

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "name": self.name,
            "compute_capability": self.capability_str,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "driver_version": self.driver_version,
            "torch_cuda_version": self.torch_cuda_version,
            "detail": self.detail,
        }


@dataclass
class SystemInfo:
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    python_version: str = ""
    python_executable: str = ""
    in_project_venv: bool = False
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    package_versions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "os": f"{self.os_name} {self.os_version}",
            "architecture": self.architecture,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "in_project_venv": self.in_project_venv,
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "gpu": self.gpu.as_dict(),
            "package_versions": dict(self.package_versions),
        }


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in ("torch", "comfy-kitchen", "safetensors", "numpy", "psutil", "PySide6"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


def gpu_info() -> GPUInfo:
    info = GPUInfo()
    try:
        import torch
    except ImportError as exc:
        info.detail = f"torch is not importable: {exc}"
        return info

    info.torch_cuda_version = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        info.detail = (
            "torch.cuda.is_available() is False - no NVIDIA driver, or a CPU-only torch build"
        )
        return info

    try:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.available = True
        info.name = properties.name
        info.compute_capability = (properties.major, properties.minor)
        free, total = torch.cuda.mem_get_info(index)
        info.total_vram_bytes = int(total)
        info.free_vram_bytes = int(free)
    except Exception as exc:  # noqa: BLE001
        info.detail = f"{type(exc).__name__}: {exc}"
        return info

    # The driver version is informational; we never install or update drivers.
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info.driver_version = result.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        pass

    return info


def system_info() -> SystemInfo:
    info = SystemInfo()
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.architecture = platform.machine()
    info.python_version = platform.python_version()
    info.python_executable = sys.executable
    info.in_project_venv = "\\.venv\\" in sys.executable or "/.venv/" in sys.executable

    try:
        import psutil

        memory = psutil.virtual_memory()
        info.total_ram_bytes = int(memory.total)
        info.available_ram_bytes = int(memory.available)
    except ImportError:
        pass

    info.gpu = gpu_info()
    info.package_versions = _package_versions()
    return info


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

@dataclass
class DiskCheck:
    ok: bool
    free_bytes: int
    required_bytes: int
    preferred_bytes: int
    volume: str
    message: str = ""


def free_space(path: Path) -> int:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def required_free_space(output_format: str, planned_output_bytes: int) -> tuple[int, int]:
    """(required, preferred) free bytes for an output of this planned size."""
    planned = max(int(planned_output_bytes), 0)
    required = max(
        int(planned * C.DISK_REQUIRED_MULTIPLIER[output_format]),
        planned + C.DISK_WORKING_RESERVE_BYTES,
    )
    preferred = max(
        int(planned * C.DISK_PREFERRED_MULTIPLIER[output_format]),
        required,
    )
    return required, preferred


def check_disk(output_path: Path, output_format: str, planned_output_bytes: int) -> DiskCheck:
    """Refuse to start when the output volume cannot hold the result.

    The source is read in place and never duplicated, and no large intermediate
    is written, so only the ``.partial`` output plus bounded working data needs
    to fit.
    """
    required, preferred = required_free_space(output_format, planned_output_bytes)
    free = free_space(output_path.parent)
    volume = str(Path(os.path.abspath(output_path)).anchor or output_path.parent)

    if free < required:
        return DiskCheck(
            ok=False,
            free_bytes=free,
            required_bytes=required,
            preferred_bytes=preferred,
            volume=volume,
            message=(
                f"{volume} has {free / 1000**3:.1f} GB free; "
                f"{required / 1000**3:.1f} GB is required "
                f"({preferred / 1000**3:.1f} GB recommended) for a "
                f"{planned_output_bytes / 1000**3:.1f} GB output"
            ),
        )

    message = ""
    if free < preferred:
        message = (
            f"{free / 1000**3:.1f} GB free is above the {required / 1000**3:.1f} GB minimum "
            f"but below the {preferred / 1000**3:.1f} GB recommendation"
        )
    return DiskCheck(True, free, required, preferred, volume, message)


# ---------------------------------------------------------------------------
# Peak usage instrumentation
# ---------------------------------------------------------------------------

class PeakUsage:
    """Samples process RSS and CUDA allocation to report real peaks."""

    def __init__(self) -> None:
        self.peak_ram_bytes = 0
        self.peak_vram_bytes = 0
        self._process = None
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:  # noqa: BLE001
            self._process = None

    def sample(self) -> None:
        if self._process is not None:
            try:
                self.peak_ram_bytes = max(self.peak_ram_bytes, int(self._process.memory_info().rss))
            except Exception:  # noqa: BLE001
                pass
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_vram_bytes = max(
                    self.peak_vram_bytes, int(torch.cuda.max_memory_allocated())
                )
        except Exception:  # noqa: BLE001
            pass

    def reset_cuda_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
