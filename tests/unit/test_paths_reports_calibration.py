"""Output naming, checkpoint metadata, progress weighting, and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from h3converter import calibration as calib
from h3converter import constants as C
from h3converter.h3_policy import build_output_plan
from h3converter.h3_reference import full_source_inventory, reference_geometry
from h3converter.paths import derive_output_path, partial_path, report_path, unique_output_path
from h3converter.progress import ProgressEvent, ProgressTracker
from h3converter.quant.w4a8 import layer_config as w4a8_config
from h3converter.reports import (
    build_output_metadata,
    build_quantization_metadata,
    fingerprint,
    sha256_file,
    summarise_layer_stats,
)
from h3converter.validate import header_from_specs


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,output_format,expected", [
    ("PinkCherry_MiniMax_H3_bf16_beta-0.6.safetensors", C.FORMAT_W4A8,
     "PinkCherry_MiniMax_H3_pruned_w4a8_convrot_beta-0.6.safetensors"),
    ("PinkCherry_MiniMax_H3_bf16_beta-0.6.safetensors", C.FORMAT_NVFP4,
     "PinkCherry_MiniMax_H3_pruned_nvfp4_beta-0.6.safetensors"),
    ("some_h3_model.safetensors", C.FORMAT_W4A8,
     "some_h3_model_pruned_w4a8_convrot.safetensors"),
    ("model-fp32.safetensors", C.FORMAT_NVFP4, "model-pruned_nvfp4.safetensors"),
])
def test_output_is_named_beside_the_source(source, output_format, expected):
    result = derive_output_path(Path("/models") / source, output_format)
    assert result.name == expected
    assert result.parent == Path("/models")


def test_output_never_collides_with_the_source():
    source = Path("/models/PinkCherry_MiniMax_H3_bf16_beta-0.6.safetensors")
    for output_format in (C.FORMAT_W4A8, C.FORMAT_NVFP4):
        assert derive_output_path(source, output_format) != source


def test_existing_output_gets_a_numbered_name(tmp_path):
    first = tmp_path / "out.safetensors"
    assert unique_output_path(first) == first

    first.write_bytes(b"x")
    second = unique_output_path(first)
    assert second.name == "out_2.safetensors"

    second.write_bytes(b"x")
    assert unique_output_path(first).name == "out_3.safetensors"


def test_partial_and_report_names(tmp_path):
    final = tmp_path / "out.safetensors"
    assert partial_path(final).name == "out.safetensors.partial"
    assert report_path(final).name == "out.safetensors.report.json"


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_plan():
    header = header_from_specs(full_source_inventory(reference_geometry()))
    return build_output_plan(header, reference_geometry(), C.FORMAT_W4A8)


def test_quantization_metadata_uses_the_loader_schema(reference_plan):
    configs = {t.layer: w4a8_config() for t in reference_plan.quant_targets}
    parsed = json.loads(build_quantization_metadata(reference_plan, configs))

    assert parsed["format_version"] == "1.0"
    assert len(parsed["layers"]) == 200
    entry = parsed["layers"]["blocks.0.attn.qkv_proj"]
    assert entry["format"] == "asym_w4a8_int8"
    assert entry["group_size"] == 16
    assert entry["convrot_groupsize"] == 256
    assert entry["convrot"] is True


def test_quantization_metadata_refuses_a_layer_with_no_config(reference_plan):
    configs = {t.layer: w4a8_config() for t in reference_plan.quant_targets}
    configs.pop("blocks.4.mlp.fc1")
    with pytest.raises(ValueError, match="blocks.4.mlp.fc1"):
        build_quantization_metadata(reference_plan, configs)


def test_output_metadata_records_provenance(reference_plan):
    configs = {t.layer: w4a8_config() for t in reference_plan.quant_targets}
    metadata = build_output_metadata(
        source_metadata={"author": "someone", "format": "bf16"},
        source_path=Path("/models/Source_bf16.safetensors"),
        output_format=C.FORMAT_W4A8,
        plan=reference_plan,
        layer_configs=configs,
    )

    assert metadata["converter_id"] == C.CONVERTER_ID
    assert metadata["source_filename"] == "Source_bf16.safetensors"
    assert metadata["source_architecture"] == "minimax_h3"
    assert metadata["adaln_curve_grid"] == "1025"
    assert metadata["adaln_curve_rank"] == "8"
    assert metadata["quantization_policy"] == C.W4A8_POLICY_VERSION
    assert metadata["quantized_layer_count"] == "200"
    assert all(isinstance(v, str) for v in metadata.values())


def test_output_metadata_quarantines_stale_source_fields(reference_plan):
    configs = {t.layer: w4a8_config() for t in reference_plan.quant_targets}
    metadata = build_output_metadata(
        source_metadata={
            "author": "someone",
            "format": "bf16",
            C.QUANT_METADATA_KEY: '{"layers": {"stale": {}}}',
        },
        source_path=Path("/models/Source_bf16.safetensors"),
        output_format=C.FORMAT_W4A8,
        plan=reference_plan,
        layer_configs=configs,
    )

    # The source's own quantization descriptor must not survive as ours.
    assert "stale" not in metadata[C.QUANT_METADATA_KEY]
    assert metadata["source.author"] == "someone"
    assert "source.format" not in metadata
    assert "format" not in metadata


# ---------------------------------------------------------------------------
# Source identification
# ---------------------------------------------------------------------------

def test_fingerprint_distinguishes_files_without_reading_them_whole(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"\x01" * 4096)
    b.write_bytes(b"\x02" * 4096)

    assert fingerprint(a) == fingerprint(a)
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a).startswith("h3fp1:")


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    payload = b"minimax h3" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_layer_stat_summary():
    stats = [{"rel_l2": 0.07, "max_abs_error": 0.01},
             {"rel_l2": 0.08, "max_abs_error": 0.02},
             {"rel_l2": 0.06, "max_abs_error": 0.005}]
    summary = summarise_layer_stats(stats)
    assert summary["measured_layers"] == 3
    assert summary["rel_l2_min"] == pytest.approx(0.06)
    assert summary["rel_l2_max"] == pytest.approx(0.08)
    assert summary["max_abs_error"] == pytest.approx(0.02)
    assert summarise_layer_stats([]) == {"measured_layers": 0}


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_progress_is_weighted_monotonic_and_reaches_one():
    seen: list[ProgressEvent] = []
    tracker = ProgressTracker.for_format(C.FORMAT_W4A8, seen.append)

    tracker.start_phase("inspect", 2)
    tracker.advance(2)
    after_inspect = tracker.overall

    tracker.start_phase("adaln_basis", 1)
    tracker.advance(1)
    tracker.start_phase("adaln_collapse", 51)
    for _ in range(51):
        tracker.advance(1)
    tracker.start_phase("quantize", 200)
    for _ in range(200):
        tracker.advance(1)
    tracker.start_phase("finalize", 1)
    tracker.finish()

    assert after_inspect == pytest.approx(C.PHASE_WEIGHTS[C.FORMAT_W4A8]["inspect"])
    assert tracker.overall == pytest.approx(1.0)
    values = [event.overall for event in seen]
    assert values == sorted(values), "progress must never move backwards"


def test_phase_weights_sum_to_one():
    for output_format, weights in C.PHASE_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), output_format


def test_unknown_phase_is_rejected():
    tracker = ProgressTracker.for_format(C.FORMAT_W4A8)
    with pytest.raises(KeyError):
        tracker.start_phase("calibrate")  # W4A8 has no calibration phase


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_default_is_dynamic_activation_scaling_with_no_pack(tmp_path):
    plan = calib.resolve(["blocks.0.attn.qkv_proj"], assets_dir=tmp_path)
    assert plan.strategy == calib.STRATEGY_DYNAMIC
    assert not plan.is_static
    assert plan.input_scale_for("blocks.0.attn.qkv_proj") is None
    assert "no user input" in plan.describe() or "derives" in plan.describe()


def test_shipped_manifest_declares_the_pack_format():
    manifest = calib.load_manifest()
    assert manifest is not None
    assert manifest["calibration_set"] == "h3-nvfp4-activation-amax"
    assert "input_scale_formula" in manifest["data_format"]
    # No pack is shipped, so conversions are calibration-free by default.
    assert manifest["files"] == []
    assert calib.load_pack() is None


def _install_pack(assets_dir: Path, amax: dict[str, float], version: str = "1") -> None:
    import hashlib

    data_path = assets_dir / "activation_amax.json"
    data_path.write_text(json.dumps({"activation_amax": amax}), encoding="utf-8")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (assets_dir / "manifest.json").write_text(json.dumps({
        "manifest_version": 1,
        "calibration_set": "test-pack",
        "version": version,
        "cases": 12,
        "coverage": {"timestep_regions": ["low", "mid", "high"]},
        "files": [{"name": "activation_amax.json", "sha256": digest}],
    }), encoding="utf-8")


def test_static_pack_supplies_input_scales(tmp_path):
    _install_pack(tmp_path, {"blocks.0.attn.qkv_proj": 12.0, "blocks.1.mlp.fc1": 6.0})
    plan = calib.resolve(["blocks.0.attn.qkv_proj", "blocks.1.mlp.fc1"], assets_dir=tmp_path)

    assert plan.is_static
    assert plan.covered_layers == 2
    assert plan.pack.case_count == 12
    expected = 12.0 / calib.NVFP4_ACT_SCALE_DENOMINATOR
    assert plan.input_scale_for("blocks.0.attn.qkv_proj") == pytest.approx(expected)


def test_partial_pack_coverage_is_reported(tmp_path):
    _install_pack(tmp_path, {"blocks.0.attn.qkv_proj": 12.0})
    plan = calib.resolve(["blocks.0.attn.qkv_proj", "blocks.1.mlp.fc1"], assets_dir=tmp_path)

    assert plan.is_static
    assert plan.covered_layers == 1
    assert plan.input_scale_for("blocks.1.mlp.fc1") is None
    assert any("keep dynamic scaling" in note for note in plan.notes)


def test_pack_covering_nothing_falls_back_and_says_so(tmp_path):
    _install_pack(tmp_path, {"some.other.model.layer": 3.0})
    plan = calib.resolve(["blocks.0.attn.qkv_proj"], assets_dir=tmp_path)

    assert plan.strategy == calib.STRATEGY_DYNAMIC
    assert any("covers none of this source's layers" in note for note in plan.notes)


def test_tampered_pack_is_rejected_not_silently_used(tmp_path):
    _install_pack(tmp_path, {"blocks.0.attn.qkv_proj": 12.0})
    (tmp_path / "activation_amax.json").write_text(
        json.dumps({"activation_amax": {"blocks.0.attn.qkv_proj": 999.0}}), encoding="utf-8"
    )

    plan = calib.resolve(["blocks.0.attn.qkv_proj"], assets_dir=tmp_path)
    assert plan.strategy == calib.STRATEGY_DYNAMIC
    assert any("failed its checksum" in note for note in plan.notes)


def test_manifest_cannot_name_a_path_outside_the_assets_folder(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": 1,
        "calibration_set": "evil",
        "version": "1",
        "files": [{"name": "../../etc/passwd"}],
    }), encoding="utf-8")

    with pytest.raises(calib.CalibrationError, match="not a plain filename"):
        calib.load_pack(tmp_path)


def test_non_positive_activation_maximum_is_rejected(tmp_path):
    _install_pack(tmp_path, {"blocks.0.attn.qkv_proj": 12.0})
    import hashlib

    data_path = tmp_path / "activation_amax.json"
    data_path.write_text(json.dumps({"activation_amax": {"blocks.0.attn.qkv_proj": -1.0}}),
                         encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(calib.CalibrationError, match="not a usable maximum"):
        calib.load_pack(tmp_path)
