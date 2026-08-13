"""The golden Option A target, reproduced analytically.

The observed reference artifact
``10Eros_Max_h3_fl2va_test4_pruned-w4a8_convrot.safetensors`` is 12,540,857,840
bytes and 1,132 tensors, with a specific dtype census. These tests derive that
census from the H3 architecture plus this project's conversion policy, with no
reference to the observed numbers except as the expected result.

If the policy ever stops selecting exactly the right 200 layers, stops
preserving a precision island, or changes a storage shape, the arithmetic stops
landing on the reference and these tests fail.
"""

from __future__ import annotations

import pytest

from h3converter import constants as C
from h3converter.h3_reference import full_source_inventory, reference_geometry
from h3converter.safetensor_io import DTYPE_ITEMSIZE
from h3converter.validate import predict_inventory


@pytest.fixture(scope="module")
def inventory():
    return predict_inventory(reference_geometry(), C.FORMAT_W4A8)


def test_tensor_count_matches_reference(inventory):
    assert inventory["tensor_count"] == C.REFERENCE_INVENTORY["tensor_count"] == 1132


def test_quantized_layer_count_is_exactly_200(inventory):
    assert inventory["quantized_layer_count"] == 200


def test_dtype_counts_match_reference(inventory):
    assert inventory["dtype_counts"] == C.REFERENCE_INVENTORY["dtype_counts"]
    assert inventory["dtype_counts"] == {"I8": 200, "F8_E4M3": 200, "BF16": 322, "F32": 410}


def test_dtype_element_counts_match_reference(inventory):
    assert inventory["dtype_elements"] == C.REFERENCE_INVENTORY["dtype_elements"]


def test_packed_weights_are_two_per_byte(inventory):
    """9,633,792,000 stored bytes <-> 19,267,584,000 logical 4-bit weights."""
    packed = inventory["dtype_elements"]["I8"]
    assert packed == 9_633_792_000
    assert packed * 2 == 19_267_584_000


def test_one_fp8_group_scale_per_sixteen_weights(inventory):
    logical = inventory["dtype_elements"]["I8"] * 2
    scales = inventory["dtype_elements"]["F8_E4M3"]
    assert scales == logical // C.W4A8_GROUP_SIZE
    assert scales == 1_204_224_000


def test_f32_tensors_are_channel_scales_codebooks_and_islands(inventory):
    """410 F32 = 200 s_channel + 200 codebooks + 10 preserved fp32 tensors."""
    assert inventory["dtype_counts"]["F32"] == 200 + 200 + 10


def test_file_size_matches_reference_within_header_size(inventory):
    """Data bytes plus the JSON header must reproduce the observed file size."""
    data_bytes = inventory["data_bytes"]
    assert data_bytes == 12_540_714_592

    header_bytes = C.REFERENCE_INVENTORY["file_bytes"] - data_bytes
    # ~127 bytes of JSON per tensor entry is the expected magnitude.
    assert 0 < header_bytes < 400_000
    assert header_bytes / inventory["tensor_count"] < 400


def test_source_inventory_reproduces_the_66gb_source_size():
    """The full BF16 source the reference was made from is ~66 GB."""
    specs = full_source_inventory(reference_geometry())
    total = sum(spec.numel * DTYPE_ITEMSIZE[spec.dtype] for spec in specs)
    assert 65_000_000_000 < total < 67_000_000_000


def test_adaln_shapes_match_reference():
    geometry = reference_geometry()
    assert (geometry.block_adaln_width, C.ADALN_CURVE_RANK) == C.REFERENCE_INVENTORY["block_adaln_shape"]
    assert (geometry.final_adaln_width, C.ADALN_CURVE_RANK) == C.REFERENCE_INVENTORY["final_adaln_shape"]
    assert geometry.block_adaln_width == 96768
    assert geometry.final_adaln_width == 10752


def test_option_b_lands_in_the_same_size_class():
    """Option B targets 12-13 GB from the same pruned representation."""
    inventory = predict_inventory(reference_geometry(), C.FORMAT_NVFP4)
    assert C.OPTION_B_SIZE_BAND_BYTES[0] < inventory["data_bytes"] < C.OPTION_B_SIZE_BAND_BYTES[1]
