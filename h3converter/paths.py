"""Project-root and output-path resolution.

Kept separate from ``constants`` so that path logic can be imported by the
bootstrap checks without pulling in torch.
"""

from __future__ import annotations

import re
from pathlib import Path

from h3converter.constants import FORMAT_FILENAME_INFIX


def project_root() -> Path:
    """The directory containing start_windows.bat / pyproject.toml."""
    return Path(__file__).resolve().parent.parent


def logs_dir() -> Path:
    d = project_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def calibration_assets_dir() -> Path:
    return project_root() / "calibration_assets"


# Recognised precision/pruning markers in a source filename. These are stripped
# before the output infix is inserted so we do not produce names like
# "..._bf16_pruned_w4a8_convrot...".
# "unpruned" precedes "pruned" so the longer alternative wins; a source already
# named "..._pruned" would otherwise produce "..._pruned_pruned_w4a8_convrot",
# which is exactly what an already-pruned TenStrip checkpoint is called.
_SOURCE_MARKERS = re.compile(
    r"(?i)(?:^|[_-])(bf16|fp16|f16|fp32|f32|full|unpruned|pruned)(?=[_-]|$)"
)


def derive_output_path(source: Path, output_format: str) -> Path:
    """Name the output beside the source, following the reference convention.

        PinkCherry_MiniMax_H3_bf16_beta-0.6.safetensors
     -> PinkCherry_MiniMax_H3_pruned_w4a8_convrot_beta-0.6.safetensors

    The source precision marker is replaced in place when present, so trailing
    version suffixes stay at the end where a human expects them.
    """
    infix = FORMAT_FILENAME_INFIX[output_format]
    stem = source.stem

    replaced, count = _SOURCE_MARKERS.subn(lambda m: m.group(0)[:-len(m.group(1))] + infix, stem, count=1)
    new_stem = replaced if count else f"{stem}_{infix}"
    return source.with_name(f"{new_stem}{source.suffix}")


def unique_output_path(path: Path) -> Path:
    """First free ``name.safetensors`` / ``name_2.safetensors`` / ... variant.

    Used instead of overwriting when the caller has not explicitly authorised
    replacing an existing file.
    """
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free output filename next to {path}")


def partial_path(final: Path) -> Path:
    return final.with_name(final.name + ".partial")


def report_path(final: Path) -> Path:
    return final.with_name(final.name + ".report.json")
