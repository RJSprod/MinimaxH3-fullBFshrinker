"""The pre-pruned (PREPRUNED_H3_FLOAT) input path.

A TenStrip/10Eros-Max checkpoint arrives already in the compact AdaLN curve
form. The contract these tests hold the converter to is that it *recognises*
that form, *never rebuilds* it, and copies it out bit-for-bit -- including at
whatever dtype the source happened to use.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from h3converter import constants as C
from h3converter.h3_detect import detect
from h3converter.h3_policy import build_output_plan
from h3converter.h3_reference import curve_geometry, prepruned_source_inventory
from h3converter.safetensor_io import read_header

from fixtures.synthetic import SMALL_GEOMETRY, build_prepruned_h3, build_synthetic_h3


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_prepruned_source_is_detected_and_convertible(prepruned_source):
    detection = detect(read_header(prepruned_source))

    assert detection.is_h3
    assert detection.convertible, detection.errors
    assert detection.source_form == C.SOURCE_FORM_PREPRUNED
    assert detection.already_curve_pruned
    assert not detection.has_time_embedder
    assert not detection.needs_pruning
    assert not detection.already_quantized


def test_full_source_still_classifies_as_full(synthetic_source):
    detection = detect(read_header(synthetic_source))

    assert detection.convertible, detection.errors
    assert detection.source_form == C.SOURCE_FORM_FULL
    assert detection.needs_pruning


def test_curve_form_geometry_is_read_from_the_table(prepruned_source):
    """`time embed 8` on the inspection screen comes from the table's rank."""
    detection = detect(read_header(prepruned_source))

    assert detection.geometry.time_embed_dim == C.ADALN_CURVE_RANK
    assert detection.geometry.adaln_curve_grid == C.ADALN_CURVE_GRID
    assert detection.geometry.num_layers == SMALL_GEOMETRY.num_layers


def test_both_timestep_paths_present_is_rejected_as_ambiguous(tmp_path):
    """A checkpoint that could modulate either way must not be guessed at."""
    source = tmp_path / "both.safetensors"
    build_synthetic_h3(source, geometry=SMALL_GEOMETRY)

    from safetensors.torch import load_file

    tensors = load_file(str(source))
    tensors[C.ADALN_TABLE_KEY] = torch.zeros(
        (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK), dtype=torch.float32
    )
    save_file(tensors, str(source))

    detection = detect(read_header(source))
    assert not detection.convertible
    assert detection.source_form is None
    assert any("ambiguous" in e for e in detection.errors)


def test_no_timestep_path_at_all_is_rejected(tmp_path):
    source = tmp_path / "neither.safetensors"
    build_synthetic_h3(source, geometry=SMALL_GEOMETRY)

    from safetensors.torch import load_file

    tensors = {k: v for k, v in load_file(str(source)).items()
               if not k.startswith(f"{C.KEY_TIME_EMBEDDER}.")}
    save_file(tensors, str(source))

    detection = detect(read_header(source))
    assert not detection.convertible
    assert detection.source_form is None
    assert any("no timestep path" in e for e in detection.errors)


def test_wrong_table_rank_is_refused(tmp_path):
    """The table's rank is the AdaLN input width; a different one is a different model."""
    source = tmp_path / "badrank.safetensors"
    build_prepruned_h3(source, geometry=SMALL_GEOMETRY)

    from safetensors.torch import load_file

    tensors = load_file(str(source))
    tensors[C.ADALN_TABLE_KEY] = torch.zeros((C.ADALN_CURVE_GRID, 7), dtype=torch.float32)
    save_file(tensors, str(source))

    detection = detect(read_header(source))
    assert not detection.convertible
    assert any(C.ADALN_TABLE_KEY in e for e in detection.errors)


def test_bf16_table_is_refused(tmp_path):
    source = tmp_path / "bf16table.safetensors"
    build_prepruned_h3(source, geometry=SMALL_GEOMETRY)

    from safetensors.torch import load_file

    tensors = load_file(str(source))
    tensors[C.ADALN_TABLE_KEY] = tensors[C.ADALN_TABLE_KEY].to(torch.bfloat16)
    save_file(tensors, str(source))

    detection = detect(read_header(source))
    assert not detection.convertible
    assert any("expected F32" in e for e in detection.errors)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_collapses_nothing_and_drops_nothing(prepruned_source):
    header = read_header(prepruned_source)
    geometry = detect(header).geometry
    plan = build_output_plan(header, geometry, C.FORMAT_W4A8, C.SOURCE_FORM_PREPRUNED)

    assert plan.adaln_targets == []
    assert plan.dropped_keys == []
    assert plan.source_form == C.SOURCE_FORM_PREPRUNED
    # The layer selection is unchanged by the source form.
    assert plan.quantized_layer_count == geometry.num_layers * len(C.W4A8_BLOCK_LINEARS)


def test_plan_copies_the_curve_tensors_through(prepruned_source):
    header = read_header(prepruned_source)
    geometry = detect(header).geometry
    plan = build_output_plan(header, geometry, C.FORMAT_W4A8, C.SOURCE_FORM_PREPRUNED)

    assert C.ADALN_TABLE_KEY in plan.passthrough_keys
    planned = {t.name: t for t in plan.tensors}
    for name in (C.ADALN_TABLE_KEY, "blocks.0.adaln_proj.linear.weight",
                 f"{C.KEY_FINAL_ADALN}.weight"):
        assert planned[name].dtype == header.tensors[name].dtype
        assert planned[name].shape == header.tensors[name].shape


def test_plan_preserves_an_f32_curve(tmp_path):
    """An F32-curve source is legitimate and must not be silently downcast."""
    source = tmp_path / "f32curve.safetensors"
    build_prepruned_h3(source, geometry=SMALL_GEOMETRY, adaln_dtype="F32")

    header = read_header(source)
    geometry = detect(header).geometry
    plan = build_output_plan(header, geometry, C.FORMAT_W4A8, C.SOURCE_FORM_PREPRUNED)

    planned = {t.name: t for t in plan.tensors}
    assert planned["blocks.0.adaln_proj.linear.weight"].dtype == "F32"
    assert planned[f"{C.KEY_FINAL_ADALN}.weight"].dtype == "F32"


def test_prepruned_inventory_matches_a_curve_pruned_source(prepruned_source):
    header = read_header(prepruned_source)
    expected = {s.name: (s.dtype, s.shape)
                for s in prepruned_source_inventory(SMALL_GEOMETRY)}
    actual = {name: (info.dtype, info.shape) for name, info in header.tensors.items()}
    assert actual == expected


def test_curve_geometry_drops_the_embedder_widths():
    reduced = curve_geometry(SMALL_GEOMETRY)
    assert reduced.time_embed_dim == C.ADALN_CURVE_RANK
    assert reduced.timestep_input_dim is None
    assert reduced.time_embed_hidden_size is None
    # The AdaLN fan-out is a function of hidden size only, so it is untouched.
    assert reduced.block_adaln_width == SMALL_GEOMETRY.block_adaln_width


# ---------------------------------------------------------------------------
# Progress weighting
# ---------------------------------------------------------------------------

def test_phase_weights_sum_to_one_for_every_combination():
    for source_form in (C.SOURCE_FORM_FULL, C.SOURCE_FORM_PREPRUNED):
        for output_format in (C.FORMAT_W4A8, C.FORMAT_NVFP4):
            weights = C.phase_weights(output_format, source_form)
            assert abs(sum(weights.values()) - 1.0) < 1e-9, (source_form, output_format)


def test_prepruned_weighting_has_no_adaln_phases():
    weights = C.phase_weights(C.FORMAT_W4A8, C.SOURCE_FORM_PREPRUNED)
    assert "adaln_basis" not in weights
    assert "adaln_collapse" not in weights


def test_curve_rank_is_not_reported_as_a_geometry_anomaly(prepruned_source):
    """time_embed_dim == 8 is the curve form working, not a deviation."""
    detection = detect(read_header(prepruned_source))
    assert not any("time_embed_dim" in w for w in detection.warnings)


def test_full_source_still_warns_on_a_deviant_time_embed_dim(synthetic_source):
    detection = detect(read_header(synthetic_source))
    assert any("time_embed_dim" in w for w in detection.warnings)
