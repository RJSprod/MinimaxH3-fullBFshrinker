"""W4A8 and NVFP4 storage contracts, and the dependency capability probe."""

from __future__ import annotations

import pytest
import torch

from h3converter import constants as C
from h3converter.quant import capability
from h3converter.quant.nvfp4 import NVFP4Error
from h3converter.quant.nvfp4 import dequantize_layer as nvfp4_dequantize
from h3converter.quant.nvfp4 import layer_config as nvfp4_config
from h3converter.quant.nvfp4 import quantize_layer as nvfp4_quantize
from h3converter.quant.w4a8 import W4A8Error
from h3converter.quant.w4a8 import dequantize_layer as w4a8_dequantize
from h3converter.quant.w4a8 import layer_config as w4a8_config
from h3converter.quant.w4a8 import quantize_layer as w4a8_quantize

SHAPE = (128, 512)


@pytest.fixture(scope="module")
def weight():
    generator = torch.Generator().manual_seed(31337)
    return (torch.randn(*SHAPE, generator=generator) * 0.05).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def probe_report():
    return capability.probe()


def test_probe_confirms_the_w4a8_layout_and_contract(probe_report):
    assert probe_report.group_ok("w4a8."), probe_report.blocking_reason(C.FORMAT_W4A8)
    assert probe_report.ok_for(C.FORMAT_W4A8)


def test_probe_confirms_the_nvfp4_layout_and_contract(probe_report):
    assert probe_report.group_ok("nvfp4."), probe_report.blocking_reason(C.FORMAT_NVFP4)
    assert probe_report.ok_for(C.FORMAT_NVFP4)


def test_probe_verifies_group_scale_arithmetic(probe_report):
    check = next(c for c in probe_report.checks if c.name == "w4a8.group_scale_count")
    assert check.ok, check.detail


def test_probe_reports_a_blocking_reason_when_a_check_fails():
    report = capability.CapabilityReport()
    report.add("runtime.tensor_op", True)
    report.add("w4a8.layout_available", False, "AsymW4A8Int8Layout is not present")
    assert not report.ok_for(C.FORMAT_W4A8)
    assert "AsymW4A8Int8Layout is not present" in report.blocking_reason(C.FORMAT_W4A8)


# ---------------------------------------------------------------------------
# W4A8
# ---------------------------------------------------------------------------

def test_w4a8_produces_the_documented_tensor_group(weight):
    result = w4a8_quantize("blocks.0.attn.qkv_proj", weight)
    n, k = SHAPE

    packed = result.tensors[C.W4A8_SUFFIX_WEIGHT]
    assert packed.dtype == torch.int8
    assert tuple(packed.shape) == (n, k // 2), "two 4-bit codes per stored byte"

    s_rel = result.tensors[C.W4A8_SUFFIX_S_REL]
    assert s_rel.dtype == torch.float8_e4m3fn
    assert tuple(s_rel.shape) == (n, k // C.W4A8_GROUP_SIZE)

    s_channel = result.tensors[C.W4A8_SUFFIX_S_CHANNEL]
    assert s_channel.dtype == torch.float32
    assert tuple(s_channel.shape) == (n,)

    codebook = result.tensors[C.W4A8_SUFFIX_CODEBOOK]
    assert codebook.dtype == torch.float32
    assert tuple(codebook.shape) == (C.W4A8_CODEBOOK_ENTRIES,)


def test_w4a8_stores_one_group_scale_per_sixteen_weights(weight):
    result = w4a8_quantize("blocks.0.mlp.fc1", weight)
    logical = SHAPE[0] * SHAPE[1]
    assert result.tensors[C.W4A8_SUFFIX_S_REL].numel() == logical // 16


def test_w4a8_logical_shape_is_recoverable_from_storage(weight):
    result = w4a8_quantize("blocks.0.attn.out_proj", weight)
    packed = result.tensors[C.W4A8_SUFFIX_WEIGHT]
    assert (packed.shape[0], packed.shape[1] * 2) == SHAPE


def test_w4a8_layer_config_declares_the_format_and_geometry():
    config = w4a8_config()
    assert config["format"] == C.QUANT_FORMAT_W4A8 == "asym_w4a8_int8"
    assert config["group_size"] == 16
    assert config["convrot_groupsize"] == 256
    assert config["convrot"] is True


def test_w4a8_round_trip_is_accurate_and_finite(weight):
    result = w4a8_quantize("blocks.0.mlp.fc2", weight, measure=True)
    restored = w4a8_dequantize(result.tensors, SHAPE)

    assert tuple(restored.shape) == SHAPE
    assert torch.isfinite(restored).all()
    relative = (restored.float() - weight.float()).norm() / weight.float().norm()
    # comfy-kitchen documents ~0.073 relative L2 for this layout.
    assert relative < 0.12
    assert result.stats["rel_l2"] == pytest.approx(float(relative), rel=1e-3)


def test_w4a8_scales_contain_no_nan_or_inf(weight):
    result = w4a8_quantize("blocks.0.attn.qkv_proj", weight)
    assert torch.isfinite(result.tensors[C.W4A8_SUFFIX_S_REL].float()).all()
    assert torch.isfinite(result.tensors[C.W4A8_SUFFIX_S_CHANNEL]).all()
    assert torch.isfinite(result.tensors[C.W4A8_SUFFIX_CODEBOOK]).all()


def test_w4a8_rejects_a_width_that_breaks_the_convrot_group():
    bad = torch.randn(16, 300).to(torch.bfloat16)
    with pytest.raises(W4A8Error, match="ConvRot group"):
        w4a8_quantize("blocks.0.attn.qkv_proj", bad)


def test_w4a8_rejects_a_non_2d_weight():
    with pytest.raises(W4A8Error, match="2D"):
        w4a8_quantize("blocks.0.attn.qkv_proj", torch.randn(4, 4, 4).to(torch.bfloat16))


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------

def test_nvfp4_produces_the_documented_tensor_group(weight):
    result = nvfp4_quantize("blocks.0.attn.qkv_proj", weight)
    n, k = SHAPE

    packed = result.tensors[C.NVFP4_SUFFIX_WEIGHT]
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (n, k // 2)

    block_scale = result.tensors[C.NVFP4_SUFFIX_SCALE]
    assert block_scale.dtype == torch.float8_e4m3fn
    assert block_scale.numel() >= (n * k) // C.NVFP4_GROUP_SIZE

    global_scale = result.tensors[C.NVFP4_SUFFIX_SCALE_2]
    assert global_scale.dtype == torch.float32
    assert float(global_scale.reshape(-1)[0]) > 0.0


def test_nvfp4_omits_input_scale_by_default(weight):
    """Absent input_scale means the runtime derives one per activation."""
    result = nvfp4_quantize("blocks.0.mlp.fc1", weight)
    assert C.NVFP4_SUFFIX_INPUT_SCALE not in result.tensors


def test_nvfp4_writes_input_scale_when_calibration_supplies_one(weight):
    result = nvfp4_quantize("blocks.0.mlp.fc1", weight, input_scale=0.0125)
    stored = result.tensors[C.NVFP4_SUFFIX_INPUT_SCALE]
    assert stored.dtype == torch.float32
    assert float(stored) == pytest.approx(0.0125)


def test_nvfp4_round_trip_is_accurate_and_finite(weight):
    result = nvfp4_quantize("blocks.0.mlp.fc2", weight, measure=True)
    restored = nvfp4_dequantize(result.tensors, SHAPE)

    assert tuple(restored.shape) == SHAPE
    assert torch.isfinite(restored).all()
    relative = (restored.float() - weight.float()).norm() / weight.float().norm()
    assert relative < 0.16


def test_nvfp4_layer_config_declares_the_format_and_group():
    config = nvfp4_config()
    assert config["format"] == C.QUANT_FORMAT_NVFP4 == "nvfp4"
    assert config["group_size"] == 16


def test_nvfp4_rejects_a_non_2d_weight():
    with pytest.raises(NVFP4Error, match="2D"):
        nvfp4_quantize("blocks.0.mlp.fc1", torch.randn(2, 2, 2).to(torch.bfloat16))
