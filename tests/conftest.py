"""Shared fixtures.

Conversions of the miniature fixture take a few seconds, so the converted
checkpoints are session-scoped and reused by every test that inspects one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.synthetic import SMALL_GEOMETRY, build_synthetic_h3  # noqa: E402

from h3converter import constants as C  # noqa: E402
from h3converter.pipeline import ConversionRequest, convert  # noqa: E402


@pytest.fixture(scope="session")
def geometry():
    return SMALL_GEOMETRY


@pytest.fixture(scope="session")
def synthetic_source(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("source")
    return build_synthetic_h3(root / "Synthetic_MiniMax_H3_bf16_beta-0.1.safetensors")


def _convert(source: Path, output_format: str, tmp_path_factory):
    destination = tmp_path_factory.mktemp(f"out_{output_format}")
    outcome = convert(
        ConversionRequest(
            source=source,
            output_format=output_format,
            output_path=destination / f"converted_{output_format}.safetensors",
            measure_every=40,
        )
    )
    assert outcome.ok, f"conversion failed: {outcome.error}"
    return outcome


@pytest.fixture(scope="session")
def converted_w4a8(synthetic_source, tmp_path_factory):
    return _convert(synthetic_source, C.FORMAT_W4A8, tmp_path_factory)


@pytest.fixture(scope="session")
def converted_nvfp4(synthetic_source, tmp_path_factory):
    return _convert(synthetic_source, C.FORMAT_NVFP4, tmp_path_factory)
