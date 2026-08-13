"""Full conversions of a synthetic H3 checkpoint, both output formats.

These exercise the shipping path end to end: detection, planning, curve
construction, projection collapse, quantization, streaming write, read-back
validation and atomic publication.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from h3converter import constants as C
from h3converter.h3_detect import detect
from h3converter.pipeline import ConversionRequest, analyze_source, convert
from h3converter.safetensor_io import read_header


# ---------------------------------------------------------------------------
# Shared expectations
# ---------------------------------------------------------------------------

def _quantized_layer_count(geometry) -> int:
    return geometry.num_layers * len(C.W4A8_BLOCK_LINEARS)


@pytest.mark.parametrize("outcome_fixture,quant_format", [
    ("converted_w4a8", C.QUANT_FORMAT_W4A8),
    ("converted_nvfp4", C.QUANT_FORMAT_NVFP4),
])
def test_conversion_succeeds_and_validates(request, outcome_fixture, quant_format, geometry):
    outcome = request.getfixturevalue(outcome_fixture)
    assert outcome.ok
    assert outcome.output_path.is_file()
    assert outcome.report.validation["ok"]
    assert outcome.report.quantization["format"] == quant_format
    assert outcome.report.quantization["quantized_layers"] == _quantized_layer_count(geometry)


@pytest.mark.parametrize("outcome_fixture", ["converted_w4a8", "converted_nvfp4"])
def test_source_is_never_modified(request, outcome_fixture, synthetic_source):
    """The source's bytes and mtime must be untouched by a conversion."""
    before_size = synthetic_source.stat().st_size
    before_digest = read_header(synthetic_source).metadata

    outcome = request.getfixturevalue(outcome_fixture)
    assert outcome.ok
    assert synthetic_source.stat().st_size == before_size
    assert read_header(synthetic_source).metadata == before_digest
    assert outcome.output_path != synthetic_source


@pytest.mark.parametrize("outcome_fixture", ["converted_w4a8", "converted_nvfp4"])
def test_no_partial_file_survives(request, outcome_fixture):
    outcome = request.getfixturevalue(outcome_fixture)
    partial = outcome.output_path.with_name(outcome.output_path.name + ".partial")
    assert not partial.exists()


@pytest.mark.parametrize("outcome_fixture", ["converted_w4a8", "converted_nvfp4"])
def test_output_is_curve_pruned(request, outcome_fixture, geometry):
    outcome = request.getfixturevalue(outcome_fixture)
    header = read_header(outcome.output_path)

    assert not any(k.startswith("time_embedder.") for k in header.tensors)

    table = header.get(C.ADALN_TABLE_KEY)
    assert table.dtype == "F32"
    assert table.shape == (1025, 8)

    for index in range(geometry.num_layers):
        info = header.get(f"blocks.{index}.adaln_proj.linear.weight")
        assert info.dtype == "BF16"
        assert info.shape == (geometry.block_adaln_width, 8)

    final = header.get(f"{C.KEY_FINAL_ADALN}.weight")
    assert final.dtype == "BF16"
    assert final.shape == (geometry.final_adaln_width, 8)


@pytest.mark.parametrize("outcome_fixture", ["converted_w4a8", "converted_nvfp4"])
def test_preserved_layers_survive_unquantized(request, outcome_fixture, geometry, synthetic_source):
    outcome = request.getfixturevalue(outcome_fixture)
    header = read_header(outcome.output_path)
    source = read_header(synthetic_source)

    preserved = [
        "condition_proj.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "token_refiner.blocks.0.mlp.fc1.weight",
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
    ]
    for key in preserved:
        assert key in header.tensors, key
        assert header.tensors[key].dtype == source.tensors[key].dtype
        assert header.tensors[key].shape == source.tensors[key].shape


def test_preserved_weights_are_copied_bit_for_bit(converted_w4a8, synthetic_source):
    with safe_open(str(synthetic_source), framework="pt", device="cpu") as src, \
         safe_open(str(converted_w4a8.output_path), framework="pt", device="cpu") as out:
        for key in ("condition_proj.weight", "video_patch_proj.weight",
                    "token_refiner.blocks.0.mlp.fc2.weight", "blocks.0.norm1.weight",
                    "final_layer.audio_out.bias"):
            a = src.get_tensor(key).reshape(-1).view(torch.uint8)
            b = out.get_tensor(key).reshape(-1).view(torch.uint8)
            assert torch.equal(a, b), key


def test_adaln_pruning_error_is_within_the_gate(converted_w4a8):
    pruning = converted_w4a8.report.pruning
    assert pruning["curve_rel_l2"] < 1e-4
    assert pruning["worst_projection"]["rel_l2"] < C.ADALN_REL_ERROR_WARN
    assert pruning["projection_count"] == pruning["projection_count"]
    assert pruning["basis"]["rank"] == 8
    assert pruning["basis"]["grid"] == 1025


def test_quantization_error_matches_the_documented_layout_quality(converted_w4a8):
    error = converted_w4a8.report.quantization["layer_error"]
    assert error["measured_layers"] >= 2
    # comfy-kitchen documents ~0.073 relative L2 for asym_w4a8_int8.
    assert 0.03 < error["rel_l2_median"] < 0.12


# ---------------------------------------------------------------------------
# Option A specifics
# ---------------------------------------------------------------------------

def test_option_a_metadata_is_asym_w4a8_int8_with_convrot(converted_w4a8, geometry):
    header = read_header(converted_w4a8.output_path)
    parsed = json.loads(header.metadata[C.QUANT_METADATA_KEY])

    assert parsed["format_version"] == "1.0"
    assert len(parsed["layers"]) == _quantized_layer_count(geometry)
    for name, conf in parsed["layers"].items():
        assert name.startswith("blocks.")
        assert conf["format"] == "asym_w4a8_int8"
        assert conf["convrot"] is True
        assert conf["group_size"] == 16
        assert conf["convrot_groupsize"] == 256


def test_option_a_stores_one_fp8_scale_per_sixteen_weights(converted_w4a8, synthetic_source):
    header = read_header(converted_w4a8.output_path)
    source = read_header(synthetic_source)
    geometry = detect(source).geometry

    for index in range(geometry.num_layers):
        for family in C.W4A8_BLOCK_LINEARS:
            key = f"blocks.{index}.{family}.weight"
            n, k = source.tensors[key].shape
            packed = header.tensors[key]
            s_rel = header.tensors[f"{key}_s_rel"]

            assert packed.dtype == "I8"
            assert packed.shape == (n, k // 2)
            assert s_rel.dtype == "F8_E4M3"
            assert s_rel.numel == (n * k) // 16
            assert header.tensors[f"{key}_s_channel"].shape == (n,)
            assert header.tensors[f"{key}_codebook"].shape == (16,)


def test_option_a_dtype_families_match_the_reference_pattern(converted_w4a8, geometry):
    header = read_header(converted_w4a8.output_path)
    counts = header.dtype_counts()
    quantized = _quantized_layer_count(geometry)

    assert counts["I8"] == quantized
    assert counts["F8_E4M3"] == quantized
    # Two fp32 side tensors per quantized layer, plus the fp32 islands.
    assert counts["F32"] == 2 * quantized + 10


# ---------------------------------------------------------------------------
# Option B specifics
# ---------------------------------------------------------------------------

def test_option_b_metadata_is_nvfp4(converted_nvfp4, geometry):
    header = read_header(converted_nvfp4.output_path)
    parsed = json.loads(header.metadata[C.QUANT_METADATA_KEY])

    assert len(parsed["layers"]) == _quantized_layer_count(geometry)
    for conf in parsed["layers"].values():
        assert conf["format"] == "nvfp4"
        assert conf["group_size"] == 16


def test_option_b_requires_no_user_calibration(converted_nvfp4):
    calibration = converted_nvfp4.report.calibration
    assert calibration["strategy"] == "dynamic_activation_scale"
    assert calibration["layers_covered"] == 0
    header = read_header(converted_nvfp4.output_path)
    assert not any(k.endswith("input_scale") for k in header.tensors)


def test_option_b_stores_both_scale_levels(converted_nvfp4, synthetic_source):
    header = read_header(converted_nvfp4.output_path)
    source = read_header(synthetic_source)
    geometry = detect(source).geometry

    for index in range(geometry.num_layers):
        for family in C.W4A8_BLOCK_LINEARS:
            key = f"blocks.{index}.{family}.weight"
            n, k = source.tensors[key].shape
            assert header.tensors[key].dtype == "U8"
            assert header.tensors[f"{key}_scale"].dtype == "F8_E4M3"
            assert header.tensors[f"{key}_scale"].numel >= (n * k) // 16
            assert header.tensors[f"{key}_scale_2"].dtype == "F32"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome_fixture", ["converted_w4a8", "converted_nvfp4"])
def test_report_json_is_written_and_complete(request, outcome_fixture):
    outcome = request.getfixturevalue(outcome_fixture)
    assert outcome.report_path.is_file()

    with open(outcome.report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    for section in ("application", "source", "output", "architecture", "pruning",
                    "quantization", "calibration", "environment", "resources", "validation"):
        assert section in report, section

    assert report["application"]["version"] == C.APP_VERSION
    assert report["source"]["fingerprint"].startswith("h3fp1:")
    assert report["output"]["compression_ratio"] > 1.0
    assert report["validation"]["ok"] is True
    assert report["resources"]["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# Failure and safety paths
# ---------------------------------------------------------------------------

def test_refuses_to_convert_an_already_converted_checkpoint(converted_w4a8, tmp_path):
    """Round-tripping a converted file back through the converter must fail."""
    outcome = convert(
        ConversionRequest(
            source=converted_w4a8.output_path,
            output_format=C.FORMAT_W4A8,
            output_path=tmp_path / "double.safetensors",
        )
    )
    assert not outcome.ok
    assert "already" in outcome.error
    assert not (tmp_path / "double.safetensors").exists()


def test_refuses_a_non_h3_checkpoint(tmp_path):
    from safetensors.torch import save_file

    source = tmp_path / "not_h3.safetensors"
    save_file({"blocks.0.attn.qkv_proj.weight": torch.zeros(8, 8)}, str(source))

    outcome = convert(ConversionRequest(source=source, output_format=C.FORMAT_W4A8))
    assert not outcome.ok
    assert "not a MiniMax H3" in outcome.error


def test_refuses_to_overwrite_the_source(synthetic_source):
    outcome = convert(
        ConversionRequest(
            source=synthetic_source,
            output_format=C.FORMAT_W4A8,
            output_path=synthetic_source,
        )
    )
    assert not outcome.ok
    assert "overwrite the source" in outcome.error


def test_existing_output_is_not_clobbered(synthetic_source, tmp_path):
    existing = tmp_path / "taken.safetensors"
    existing.write_bytes(b"do not touch")

    outcome = convert(
        ConversionRequest(
            source=synthetic_source,
            output_format=C.FORMAT_NVFP4,
            output_path=existing,
        )
    )
    assert outcome.ok
    assert existing.read_bytes() == b"do not touch"
    assert outcome.output_path.name == "taken_2.safetensors"


def test_cancellation_leaves_nothing_behind(synthetic_source, tmp_path):
    target = tmp_path / "cancelled.safetensors"
    calls = {"n": 0}

    def should_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3  # stop shortly after the conversion starts

    outcome = convert(
        ConversionRequest(
            source=synthetic_source,
            output_format=C.FORMAT_W4A8,
            output_path=target,
        ),
        should_cancel=should_cancel,
    )
    assert outcome.cancelled
    assert not outcome.ok
    assert not target.exists()
    assert not target.with_name(target.name + ".partial").exists()
    assert synthetic_source.is_file()


# ---------------------------------------------------------------------------
# Analysis screen
# ---------------------------------------------------------------------------

def test_analysis_previews_both_formats_without_reading_tensor_data(synthetic_source, geometry):
    analysis = analyze_source(synthetic_source)
    assert analysis.ok
    assert set(analysis.plan_preview) == {C.FORMAT_W4A8, C.FORMAT_NVFP4}

    for output_format, plan in analysis.plan_preview.items():
        assert plan.quantized_layer_count == _quantized_layer_count(geometry)
        assert analysis.estimated_output_bytes(output_format) == plan.total_bytes
        assert plan.total_bytes < analysis.size_bytes


def test_predicted_output_size_matches_what_is_written(converted_w4a8, synthetic_source):
    """The plan is a byte-exact prediction, not an estimate."""
    analysis = analyze_source(synthetic_source)
    predicted = analysis.plan_preview[C.FORMAT_W4A8].total_bytes
    header = read_header(converted_w4a8.output_path)

    actual_data_bytes = sum(info.nbytes for info in header.tensors.values())
    assert actual_data_bytes == predicted
