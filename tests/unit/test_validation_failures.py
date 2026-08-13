"""Validation must reject deliberately damaged output.

Each test takes a real converted checkpoint, breaks exactly one thing, and
asserts that ``validate_output`` catches it. A validator that only ever sees
correct files proves nothing.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3converter import constants as C
from h3converter.h3_detect import detect
from h3converter.h3_policy import build_output_plan
from h3converter.safetensor_io import read_header
from h3converter.validate import validate_output


@pytest.fixture(scope="module")
def converted(converted_w4a8):
    return converted_w4a8


@pytest.fixture(scope="module")
def plan_and_geometry(synthetic_source):
    header = read_header(synthetic_source)
    geometry = detect(header).geometry
    return build_output_plan(header, geometry, C.FORMAT_W4A8), geometry


def _load(path):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def _rewrite(tmp_path, tensors, metadata, name="tampered.safetensors"):
    path = tmp_path / name
    save_file(tensors, str(path), metadata=metadata)
    return path


def _validate(path, plan_and_geometry):
    plan, geometry = plan_and_geometry
    return validate_output(path, plan, geometry, C.FORMAT_W4A8, sample_layers=2)


def test_an_untouched_conversion_passes(converted, plan_and_geometry):
    report = _validate(converted.output_path, plan_and_geometry)
    assert report.ok, report.summary()
    assert len(report.findings) >= 12


def test_missing_group_scale_tensor_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    removed = "blocks.3.mlp.fc1.weight_s_rel"
    assert removed in tensors
    del tensors[removed]

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "structure.inventory_matches_plan" for f in report.failures())
    assert any(f.check == "layers.tensor_group_complete" for f in report.failures())


def test_missing_codebook_tensor_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    del tensors["blocks.0.attn.qkv_proj.weight_codebook"]

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "layers.tensor_group_complete" for f in report.failures())


def test_wrong_scale_count_fails(converted, plan_and_geometry, tmp_path):
    """A scale tensor sized for group 32 instead of group 16."""
    tensors, metadata = _load(converted.output_path)
    key = "blocks.1.attn.out_proj.weight_s_rel"
    original = tensors[key]
    tensors[key] = original[:, ::2].contiguous()

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    failed = {f.check for f in report.failures()}
    assert "layers.scale_count_matches_group_size" in failed or "structure.dtypes_and_shapes" in failed


def test_missing_quantization_metadata_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    metadata.pop(C.QUANT_METADATA_KEY)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "metadata.present" for f in report.failures())


def test_wrong_layer_format_in_metadata_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    parsed = json.loads(metadata[C.QUANT_METADATA_KEY])
    parsed["layers"]["blocks.5.mlp.fc2"]["format"] = "convrot_w4a4"
    metadata[C.QUANT_METADATA_KEY] = json.dumps(parsed)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "metadata.layer_format" for f in report.failures())


def test_wrong_group_size_in_metadata_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    parsed = json.loads(metadata[C.QUANT_METADATA_KEY])
    parsed["layers"]["blocks.2.attn.qkv_proj"]["group_size"] = 32
    metadata[C.QUANT_METADATA_KEY] = json.dumps(parsed)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "metadata.w4a8_group_size" for f in report.failures())


def test_convrot_disabled_in_metadata_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    parsed = json.loads(metadata[C.QUANT_METADATA_KEY])
    parsed["layers"]["blocks.9.mlp.fc1"]["convrot"] = False
    metadata[C.QUANT_METADATA_KEY] = json.dumps(parsed)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "metadata.w4a8_convrot_true" for f in report.failures())


def test_dropped_metadata_layer_entry_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    parsed = json.loads(metadata[C.QUANT_METADATA_KEY])
    parsed["layers"].pop("blocks.11.attn.out_proj")
    metadata[C.QUANT_METADATA_KEY] = json.dumps(parsed)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "metadata.layers" for f in report.failures())


def test_leftover_time_embedder_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    tensors["time_embedder.proj_in.weight"] = torch.zeros(8, 8, dtype=torch.float32)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "pruning.time_embedder_removed" for f in report.failures())


def test_wrong_adaln_table_shape_fails(converted, plan_and_geometry, tmp_path):
    tensors, metadata = _load(converted.output_path)
    tensors[C.ADALN_TABLE_KEY] = torch.zeros(512, 8, dtype=torch.float32)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    assert any(f.check == "pruning.adaln_table" for f in report.failures())


def test_downcast_fp32_island_fails(converted, plan_and_geometry, tmp_path):
    """The patch projections and output heads must stay fp32."""
    tensors, metadata = _load(converted.output_path)
    tensors["final_layer.video_out.weight"] = tensors["final_layer.video_out.weight"].to(torch.bfloat16)

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    failed = {f.check for f in report.failures()}
    assert "policy.fp32_islands_preserved" in failed


def test_corrupt_scale_data_fails_dequantization(converted, plan_and_geometry, tmp_path):
    """NaN group scales must be caught, not written out as a usable checkpoint."""
    tensors, metadata = _load(converted.output_path)
    for target in ("blocks.0.attn.qkv_proj", "blocks.20.mlp.fc1"):
        key = f"{target}.weight_s_channel"
        broken = tensors[key].clone()
        broken[0] = float("nan")
        tensors[key] = broken

    report = _validate(_rewrite(tmp_path, tensors, metadata), plan_and_geometry)
    assert not report.ok
    # Caught on every layer, not just the dequantization sample.
    failed = {f.check for f in report.failures()}
    assert "layers.scales_finite" in failed
    detail = next(f.detail for f in report.failures() if f.check == "layers.scales_finite")
    assert "NaN or Inf" in detail


def test_unreadable_file_fails_cleanly(plan_and_geometry, tmp_path):
    path = tmp_path / "garbage.safetensors"
    path.write_bytes(b"\x00" * 64)

    report = _validate(path, plan_and_geometry)
    assert not report.ok
    assert report.failures()[0].check == "file.readable"
