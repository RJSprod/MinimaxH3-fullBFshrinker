"""Source identification and the exact H3 layer policy."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from h3converter import constants as C
from h3converter.h3_detect import detect
from h3converter.h3_policy import (
    PolicyError,
    adaln_prefixes,
    build_output_plan,
    is_quantizable,
    nvfp4_storage,
    quantized_layer_names,
    w4a8_storage,
)
from h3converter.h3_reference import (
    TensorSpec,
    full_source_inventory,
    prepruned_source_inventory,
    reference_geometry,
)
from h3converter.safetensor_io import read_header
from h3converter.validate import header_from_specs


@pytest.fixture(scope="module")
def reference_header():
    return header_from_specs(full_source_inventory(reference_geometry()))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detects_reference_h3_geometry(reference_header):
    result = detect(reference_header)
    assert result.is_h3 and result.convertible
    geometry = result.geometry
    assert geometry.num_layers == 50
    assert geometry.hidden_size == 5376
    assert geometry.num_attention_heads == 56
    assert geometry.attention_head_dim == 128
    assert geometry.ffn_hidden_size == 14336
    assert geometry.time_embed_dim == 2688
    assert geometry.latents_dim == 24
    assert geometry.audio_latents_dim == 32
    assert not result.warnings, result.warnings


def test_detects_the_synthetic_fixture(synthetic_source, geometry):
    result = detect(read_header(synthetic_source))
    assert result.is_h3 and result.convertible
    assert result.geometry.num_layers == geometry.num_layers
    assert result.float_dtype == "BF16"
    assert not result.already_curve_pruned
    assert not result.already_quantized


def test_filename_is_never_trusted(tmp_path):
    """A file named like an H3 checkpoint but containing something else."""
    path = tmp_path / "PinkCherry_MiniMax_H3_bf16_beta-0.6.safetensors"
    save_file({"model.diffusion_model.weight": torch.zeros(4, 4)}, str(path))

    result = detect(read_header(path))
    assert not result.is_h3
    assert "not a MiniMax H3" in result.errors[0]


def test_accepts_a_well_formed_curve_pruned_source():
    """The compact curve form is a supported input, not a refusal."""
    header = header_from_specs(prepruned_source_inventory(reference_geometry()))

    result = detect(header)
    assert result.is_h3
    assert result.already_curve_pruned
    assert result.source_form == C.SOURCE_FORM_PREPRUNED
    assert result.convertible, result.errors


def test_rejects_a_curve_pruned_source_whose_projections_disagree_with_the_table():
    """Table rank and AdaLN input width are the same number; they must agree."""
    specs = [s for s in full_source_inventory(reference_geometry())
             if not s.name.startswith("time_embedder.")]
    specs.append(TensorSpec(C.ADALN_TABLE_KEY, "F32", (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK)))
    header = header_from_specs(specs)

    result = detect(header)
    assert result.is_h3
    assert result.source_form == C.SOURCE_FORM_PREPRUNED
    assert not result.convertible
    # The projections are still at the full time-embed width, so they cannot
    # consume a rank-8 table row.
    assert any("adaln_proj.linear.weight has shape" in e for e in result.errors)


def test_rejects_a_source_that_is_already_quantized():
    header = header_from_specs(full_source_inventory(reference_geometry()))
    header.metadata[C.QUANT_METADATA_KEY] = '{"layers":{}}'

    result = detect(header)
    assert result.already_quantized
    assert not result.convertible
    assert any("already quantized" in e for e in result.errors)


def test_rejects_a_materially_different_block_count():
    header = header_from_specs(full_source_inventory(reference_geometry(num_layers=8)))
    result = detect(header)
    assert result.is_h3
    assert not result.convertible
    assert any("outside the supported H3 range" in e for e in result.errors)


def test_accepts_a_fine_tune_that_differs_only_in_metadata():
    header = header_from_specs(
        full_source_inventory(reference_geometry()),
        metadata={"author": "somebody", "note": "fine-tune"},
    )
    assert detect(header).convertible


def test_warns_but_proceeds_on_a_variant_geometry():
    header = header_from_specs(full_source_inventory(reference_geometry(num_layers=48)))
    result = detect(header)
    assert result.convertible
    assert any("num_layers is 48" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Layer policy
# ---------------------------------------------------------------------------

def test_reference_policy_selects_exactly_200_layers():
    names = quantized_layer_names(reference_geometry())
    assert len(names) == 200
    assert len(set(names)) == 200
    for family in C.W4A8_BLOCK_LINEARS:
        assert sum(1 for n in names if n.endswith(family)) == 50


def test_plan_quantizes_no_adaln_refiner_or_head(reference_header):
    plan = build_output_plan(reference_header, reference_geometry(), C.FORMAT_W4A8)
    for target in plan.quant_targets:
        assert target.layer.startswith("blocks.")
        assert "adaln" not in target.layer
        assert "token_refiner" not in target.layer
        assert "condition_proj" not in target.layer
        assert "patch_proj" not in target.layer
        assert "final_layer" not in target.layer


@pytest.mark.parametrize("layer", [
    "blocks.0.adaln_proj.linear",
    "token_refiner.blocks.0.attn.qkv_proj",
    "token_refiner.blocks.1.mlp.fc1",
    "condition_proj",
    "final_layer.video_out",
    "final_layer.audio_out",
    "video_patch_proj",
    "time_embedder.proj_in",
])
def test_preserved_layers_are_never_quantizable(layer):
    assert not is_quantizable(layer)


@pytest.mark.parametrize("layer", [
    "blocks.0.attn.qkv_proj",
    "blocks.49.attn.out_proj",
    "blocks.7.mlp.fc1",
    "blocks.7.mlp.fc2",
])
def test_target_layers_are_quantizable(layer):
    assert is_quantizable(layer)


def test_token_refiner_shares_submodule_names_but_is_excluded(reference_header):
    """The refiner has attn.qkv_proj and mlp.fc1 too; only blocks.* may match."""
    plan = build_output_plan(reference_header, reference_geometry(), C.FORMAT_W4A8)
    refiner_weights = [
        k for k in reference_header.tensors
        if k.startswith("token_refiner.") and k.endswith(".weight")
    ]
    assert refiner_weights, "fixture should contain token refiner linears"
    for key in refiner_weights:
        assert key in plan.passthrough_keys


def test_plan_drops_the_time_embedder_and_adds_the_table(reference_header):
    plan = build_output_plan(reference_header, reference_geometry(), C.FORMAT_W4A8)
    assert sorted(plan.dropped_keys) == sorted(
        k for k in reference_header.tensors if k.startswith("time_embedder.")
    )
    names = {t.name for t in plan.tensors}
    assert C.ADALN_TABLE_KEY in names
    assert not any(n.startswith("time_embedder.") for n in names)


def test_plan_reduces_every_adaln_projection(reference_header):
    geometry = reference_geometry()
    plan = build_output_plan(reference_header, geometry, C.FORMAT_W4A8)
    assert len(plan.adaln_targets) == geometry.num_layers + 1
    assert {t.prefix for t in plan.adaln_targets} == set(adaln_prefixes(geometry))

    by_name = {t.name: t for t in plan.tensors}
    for target in plan.adaln_targets:
        planned = by_name[target.weight_key]
        assert planned.dtype == "BF16"
        assert planned.shape == (target.source_shape[0], C.ADALN_CURVE_RANK)


def test_plan_preserves_fp32_islands(reference_header):
    plan = build_output_plan(reference_header, reference_geometry(), C.FORMAT_W4A8)
    by_name = {t.name: t for t in plan.tensors}
    for key in ("video_patch_proj.weight", "audio_patch_proj.weight",
                "final_layer.video_out.weight", "final_layer.audio_out.weight"):
        assert by_name[key].dtype == "F32"


def test_plan_rejects_a_missing_target_layer():
    specs = [s for s in full_source_inventory(reference_geometry())
             if s.name != "blocks.13.mlp.fc2.weight"]
    with pytest.raises(PolicyError, match="blocks.13.mlp.fc2"):
        build_output_plan(header_from_specs(specs), reference_geometry(), C.FORMAT_W4A8)


def test_plan_rejects_an_unknown_format(reference_header):
    with pytest.raises(PolicyError, match="unknown output format"):
        build_output_plan(reference_header, reference_geometry(), "int4_something")


# ---------------------------------------------------------------------------
# Storage shape arithmetic
# ---------------------------------------------------------------------------

def test_w4a8_storage_shapes():
    storage = w4a8_storage(21504, 5376)
    assert storage[C.W4A8_SUFFIX_WEIGHT] == ("I8", (21504, 2688))
    assert storage[C.W4A8_SUFFIX_S_REL] == ("F8_E4M3", (21504, 336))
    assert storage[C.W4A8_SUFFIX_S_CHANNEL] == ("F32", (21504,))
    assert storage[C.W4A8_SUFFIX_CODEBOOK] == ("F32", (16,))


def test_w4a8_group_scale_arithmetic_holds_for_every_reference_layer():
    geometry = reference_geometry()
    header = header_from_specs(full_source_inventory(geometry))
    plan = build_output_plan(header, geometry, C.FORMAT_W4A8)

    for target in plan.quant_targets:
        n, k = target.logical_shape
        _, packed_shape = target.outputs[C.W4A8_SUFFIX_WEIGHT]
        _, scale_shape = target.outputs[C.W4A8_SUFFIX_S_REL]
        assert packed_shape[1] * 2 == k, "packed storage must recover the logical width"
        assert scale_shape[0] * scale_shape[1] == (n * k) // C.W4A8_GROUP_SIZE


def test_nvfp4_storage_shapes_follow_the_swizzle_rules():
    assert nvfp4_storage(64, 512) == {
        C.NVFP4_SUFFIX_WEIGHT: ("U8", (64, 256)),
        C.NVFP4_SUFFIX_SCALE: ("F8_E4M3", (128, 32)),
        C.NVFP4_SUFFIX_SCALE_2: ("F32", ()),
    }
    # Rows round to 16 for the weight and 128 for the scales; scale columns to 4.
    assert nvfp4_storage(300, 1024)[C.NVFP4_SUFFIX_WEIGHT] == ("U8", (304, 512))
    assert nvfp4_storage(300, 1024)[C.NVFP4_SUFFIX_SCALE] == ("F8_E4M3", (384, 64))
