"""A pre-pruned source, converted end to end.

The load-bearing assertion here is byte-identity: the AdaLN curve the source
arrived with must be the exact curve the output ships, down to the bytes. Shape
and dtype agreement is not enough -- a rebuilt table would pass that and still
be a different model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from h3converter import constants as C
from h3converter.h3_detect import detect
from h3converter.pipeline import ConversionRequest, analyze_source, convert
from h3converter.safetensor_io import read_header

from fixtures.synthetic import SMALL_GEOMETRY, build_prepruned_h3

pytestmark = pytest.mark.slow


def _raw(handle, key: str) -> torch.Tensor:
    return handle.get_tensor(key).reshape(-1).view(torch.uint8)


def test_prepruned_conversion_succeeds(converted_from_prepruned):
    outcome = converted_from_prepruned
    assert outcome.ok, outcome.error
    assert outcome.output_path.is_file()


def test_report_records_the_source_form(converted_from_prepruned):
    report = converted_from_prepruned.report
    assert report.architecture["source_form"] == C.SOURCE_FORM_PREPRUNED
    assert report.pruning["mode"] == "preserved_from_source"
    assert report.pruning["projections_preserved"] == SMALL_GEOMETRY.num_layers + 1
    assert report.quantization["dropped_tensors"] == 0


def test_curve_tensors_are_byte_identical(prepruned_source, converted_from_prepruned):
    """The whole promise of this path, asserted at the byte level."""
    output = converted_from_prepruned.output_path

    keys = [C.ADALN_TABLE_KEY]
    for index in range(SMALL_GEOMETRY.num_layers):
        keys.append(f"blocks.{index}.adaln_proj.linear.weight")
        keys.append(f"blocks.{index}.adaln_proj.linear.bias")
    keys.append(f"{C.KEY_FINAL_ADALN}.weight")
    keys.append(f"{C.KEY_FINAL_ADALN}.bias")

    with safe_open(str(prepruned_source), framework="pt", device="cpu") as src, \
            safe_open(str(output), framework="pt", device="cpu") as out:
        for key in keys:
            assert torch.equal(_raw(src, key), _raw(out, key)), f"{key} was modified"


def test_output_has_no_time_embedder_and_keeps_the_table(converted_from_prepruned):
    header = read_header(converted_from_prepruned.output_path)
    assert not [k for k in header.tensors if k.startswith(f"{C.KEY_TIME_EMBEDDER}.")]
    table = header.get(C.ADALN_TABLE_KEY)
    assert table.dtype == "F32"
    assert table.shape == (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK)


def test_output_validates_and_is_quantized(converted_from_prepruned):
    validation = converted_from_prepruned.report.validation
    assert validation["ok"], validation["summary"]
    expected = SMALL_GEOMETRY.num_layers * len(C.W4A8_BLOCK_LINEARS)
    assert converted_from_prepruned.report.quantization["quantized_layers"] == expected


def test_output_is_refused_as_a_source(converted_from_prepruned):
    detection = detect(read_header(converted_from_prepruned.output_path))
    assert detection.already_quantized
    assert not detection.convertible


def test_source_is_untouched(prepruned_source, converted_from_prepruned):
    detection = detect(read_header(prepruned_source))
    assert detection.convertible
    assert detection.source_form == C.SOURCE_FORM_PREPRUNED


def test_progress_reaches_one_hundred_percent(prepruned_source, tmp_path):
    """The pre-pruned weighting must not strand the bar where the AdaLN phases were."""
    seen: list[float] = []
    outcome = convert(
        ConversionRequest(
            source=prepruned_source,
            output_format=C.FORMAT_W4A8,
            output_path=tmp_path / "progress.safetensors",
            measure_every=0,
        ),
        progress_callback=lambda event: seen.append(event.overall),
    )
    assert outcome.ok, outcome.error
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:])), "progress went backwards"
    # Nothing should sit at the 0.70 ceiling the full-path weighting would impose.
    assert max(seen[:-1]) > 0.70


def test_f32_curve_source_converts_and_validates(tmp_path):
    """An F32-curve source is copied through and must pass output validation."""
    source = build_prepruned_h3(
        tmp_path / "f32curve.safetensors", geometry=SMALL_GEOMETRY, adaln_dtype="F32"
    )
    outcome = convert(
        ConversionRequest(
            source=source,
            output_format=C.FORMAT_W4A8,
            output_path=tmp_path / "f32curve_out.safetensors",
            measure_every=0,
        )
    )
    assert outcome.ok, outcome.error
    assert outcome.report.validation["ok"], outcome.report.validation["summary"]

    header = read_header(outcome.output_path)
    assert header.get("blocks.0.adaln_proj.linear.weight").dtype == "F32"


def test_analysis_offers_both_formats_for_a_prepruned_source(prepruned_source):
    """The screen that refused this file must now plan both outputs for it."""
    analysis = analyze_source(prepruned_source)
    assert analysis.ok, analysis.errors + analysis.detection.errors
    assert set(analysis.plan_preview) == {C.FORMAT_W4A8, C.FORMAT_NVFP4}
    for output_format in (C.FORMAT_W4A8, C.FORMAT_NVFP4):
        assert analysis.estimated_output_bytes(output_format) < analysis.size_bytes
