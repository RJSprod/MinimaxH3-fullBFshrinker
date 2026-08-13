"""Option B: NVFP4, via comfy-kitchen's ``TensorCoreNVFP4Layout``.

Storage contract (ComfyUI ``comfy/ops.py``, ``nvfp4`` branch):

    <layer>.weight          uint8, packed E2M1 pairs, [N, K/2]
    <layer>.weight_scale    float8_e4m3, per-group-of-16 block scales, swizzled
    <layer>.weight_scale_2  float32 scalar, per-tensor global scale
    <layer>.input_scale     float32 scalar, OPTIONAL

``input_scale`` is the activation scale. When it is absent the runtime derives
one per activation at inference time
(``amax / (F8_E4M3_MAX * F4_E2M1_MAX)``), which is what makes Option B
require no user calibration at all. Baking a static scale in is a throughput
optimisation, not a correctness requirement, and is only done when a calibration
source is explicitly supplied -- see :mod:`h3converter.calibration`.
"""

from __future__ import annotations

import torch

from h3converter import constants as C
from h3converter.quant.w4a8 import QuantizedLayer


class NVFP4Error(RuntimeError):
    pass


def layer_config() -> dict[str, object]:
    return {
        "format": C.QUANT_FORMAT_NVFP4,
        "group_size": C.NVFP4_GROUP_SIZE,
    }


def quantize_layer(
    layer: str,
    weight: torch.Tensor,
    device: str = "cpu",
    input_scale: float | None = None,
    measure: bool = False,
) -> QuantizedLayer:
    """Quantize one [out, in] weight into the NVFP4 storage group."""
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    if weight.dim() != 2:
        raise NVFP4Error(f"{layer}: NVFP4 requires a 2D weight, got {tuple(weight.shape)}")
    n, k = weight.shape

    source = weight.to(device=device, dtype=torch.bfloat16, copy=False)
    qdata, params = TensorCoreNVFP4Layout.quantize(source)
    produced = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

    tensors: dict[str, torch.Tensor] = {}
    for suffix, tensor in produced.items():
        if tensor is None:
            continue
        tensors[suffix] = tensor.detach().to("cpu").contiguous()

    if input_scale is not None:
        tensors[C.NVFP4_SUFFIX_INPUT_SCALE] = torch.tensor(float(input_scale), dtype=torch.float32)

    _check_contract(layer, tensors, n, k)

    stats: dict[str, float] = {}
    if measure:
        restored = TensorCoreNVFP4Layout.dequantize(qdata, params)
        stats = _measure(source, restored[:n, :k])
    del qdata, params, produced, source
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return QuantizedLayer(layer=layer, tensors=tensors, config=layer_config(), stats=stats)


def _check_contract(layer: str, tensors: dict[str, torch.Tensor], n: int, k: int) -> None:
    weight = tensors.get(C.NVFP4_SUFFIX_WEIGHT)
    if weight is None:
        raise NVFP4Error(f"{layer}: quantization did not produce a packed weight")
    if weight.dtype != torch.uint8:
        raise NVFP4Error(f"{layer}: packed weight has dtype {weight.dtype}, expected uint8")

    block_scale = tensors.get(C.NVFP4_SUFFIX_SCALE)
    if block_scale is None:
        raise NVFP4Error(f"{layer}: quantization did not produce weight_scale block scales")
    if block_scale.dtype != torch.float8_e4m3fn:
        raise NVFP4Error(
            f"{layer}: weight_scale has dtype {block_scale.dtype}, expected float8_e4m3fn"
        )

    global_scale = tensors.get(C.NVFP4_SUFFIX_SCALE_2)
    if global_scale is None:
        raise NVFP4Error(f"{layer}: quantization did not produce weight_scale_2")
    if global_scale.dtype != torch.float32:
        raise NVFP4Error(f"{layer}: weight_scale_2 has dtype {global_scale.dtype}, expected float32")
    if not torch.isfinite(global_scale).all() or float(global_scale.reshape(-1)[0]) <= 0.0:
        raise NVFP4Error(f"{layer}: weight_scale_2 is not a positive finite scale")

    # There must be one block scale per group of 16 logical weights, before the
    # swizzle padding. Padding may add rows/columns but never remove coverage.
    minimum_groups = (n * k) // C.NVFP4_GROUP_SIZE
    if block_scale.numel() < minimum_groups:
        raise NVFP4Error(
            f"{layer}: {block_scale.numel()} block scales cannot cover {n * k} weights "
            f"at group size {C.NVFP4_GROUP_SIZE} (need at least {minimum_groups})"
        )
    if not torch.isfinite(block_scale.to(torch.float32)).all():
        raise NVFP4Error(f"{layer}: weight_scale contains non-finite values")


def _measure(reference: torch.Tensor, restored: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    denom = float(ref.norm())
    diff = restored.float() - ref
    return {
        "rel_l2": float(diff.norm() / denom) if denom else 0.0,
        "max_abs_error": float(diff.abs().max()),
        "weight_amax": float(ref.abs().max()),
    }


def dequantize_layer(tensors: dict[str, torch.Tensor], logical_shape: tuple[int, int],
                     output_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode a stored NVFP4 group back to a dense weight (validation only)."""
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    params = TensorCoreNVFP4Layout.Params(
        scale=tensors[C.NVFP4_SUFFIX_SCALE_2],
        block_scale=tensors[C.NVFP4_SUFFIX_SCALE],
        orig_dtype=output_dtype,
        orig_shape=tuple(logical_shape),
    )
    restored = TensorCoreNVFP4Layout.dequantize(tensors[C.NVFP4_SUFFIX_WEIGHT], params)
    n, k = logical_shape
    return restored[:n, :k]
