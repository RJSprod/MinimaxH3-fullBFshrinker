"""Option A: ``asym_w4a8_int8`` with ConvRot, via comfy-kitchen.

The quantization itself is delegated to ``AsymW4A8Int8Layout`` rather than
reimplemented. That matters: the layout owns the ConvRot rotation, the
Lloyd-Max codebook decision and the ALS group-scale refinement, and the runtime
decodes with the mirror image of that exact code. A hand-rolled packer that
merely produced tensors of the right shape would be silently wrong.

What this module owns is the *checkpoint* side: mapping the layout's key
suffixes onto canonical checkpoint names, emitting the per-layer configuration
the ComfyUI loader reads, and measuring the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from h3converter import constants as C


@dataclass
class QuantizedLayer:
    """One quantized layer, ready for the writer."""

    layer: str
    tensors: dict[str, torch.Tensor]     # suffix -> tensor (suffix "" is the weight)
    config: dict[str, object]            # per-layer entry for _quantization_metadata
    stats: dict[str, float] = field(default_factory=dict)


class W4A8Error(RuntimeError):
    pass


def layer_config() -> dict[str, object]:
    """The per-layer descriptor written into ``_quantization_metadata``.

    ``format``, ``group_size`` and ``convrot_groupsize`` are what ComfyUI's
    loader reads back. ``convrot`` is redundant for this format (the layout is
    always rotated) but is present in the reference artifact's metadata, so it
    is emitted for conformance.
    """
    return {
        "format": C.QUANT_FORMAT_W4A8,
        "group_size": C.W4A8_GROUP_SIZE,
        "convrot_groupsize": C.W4A8_CONVROT_GROUPSIZE,
        "convrot": True,
    }


def quantize_layer(
    layer: str,
    weight: torch.Tensor,
    device: str = "cpu",
    measure: bool = False,
) -> QuantizedLayer:
    """Quantize one [out, in] weight into the W4A8 storage group."""
    from comfy_kitchen.tensor import AsymW4A8Int8Layout

    if weight.dim() != 2:
        raise W4A8Error(f"{layer}: W4A8 requires a 2D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % C.W4A8_CONVROT_GROUPSIZE or k % C.W4A8_GROUP_SIZE:
        raise W4A8Error(
            f"{layer}: input width {k} must divide by the ConvRot group "
            f"({C.W4A8_CONVROT_GROUPSIZE}) and the quant group ({C.W4A8_GROUP_SIZE})"
        )

    source = weight.to(device=device, dtype=torch.bfloat16, copy=False)
    qdata, params = AsymW4A8Int8Layout.quantize(
        source,
        group_size=C.W4A8_GROUP_SIZE,
        convrot_groupsize=C.W4A8_CONVROT_GROUPSIZE,
    )
    produced = AsymW4A8Int8Layout.state_dict_tensors(qdata, params)

    tensors: dict[str, torch.Tensor] = {}
    for suffix, tensor in produced.items():
        if tensor is None:
            continue
        tensors[suffix] = tensor.detach().to("cpu").contiguous()

    _check_contract(layer, tensors, n, k)

    stats: dict[str, float] = {}
    if measure:
        restored = AsymW4A8Int8Layout.dequantize(qdata, params)
        stats = _measure(source, restored)
    del qdata, params, produced, source
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return QuantizedLayer(layer=layer, tensors=tensors, config=layer_config(), stats=stats)


def _check_contract(layer: str, tensors: dict[str, torch.Tensor], n: int, k: int) -> None:
    """Fail loudly if the library did not produce the expected storage group.

    Per §28 of the design: never silently emit something other than W4A8.
    """
    required = {
        C.W4A8_SUFFIX_WEIGHT: (torch.int8, (n, k // 2)),
        C.W4A8_SUFFIX_S_REL: (torch.float8_e4m3fn, (n, k // C.W4A8_GROUP_SIZE)),
        C.W4A8_SUFFIX_S_CHANNEL: (torch.float32, (n,)),
        C.W4A8_SUFFIX_CODEBOOK: (torch.float32, (C.W4A8_CODEBOOK_ENTRIES,)),
    }
    for suffix, (dtype, shape) in required.items():
        name = f"weight{suffix}"
        tensor = tensors.get(suffix)
        if tensor is None:
            raise W4A8Error(f"{layer}: quantization did not produce {name}")
        if tensor.dtype != dtype:
            raise W4A8Error(f"{layer}: {name} has dtype {tensor.dtype}, expected {dtype}")
        if tuple(tensor.shape) != shape:
            raise W4A8Error(f"{layer}: {name} has shape {tuple(tensor.shape)}, expected {shape}")

    s_rel = tensors[C.W4A8_SUFFIX_S_REL]
    expected_groups = (n * k) // C.W4A8_GROUP_SIZE
    if s_rel.numel() != expected_groups:
        raise W4A8Error(
            f"{layer}: {s_rel.numel()} group scales for {n * k} logical weights; "
            f"group size {C.W4A8_GROUP_SIZE} requires {expected_groups}"
        )

    scales = s_rel.to(torch.float32)
    if not torch.isfinite(scales).all():
        raise W4A8Error(f"{layer}: weight_s_rel contains non-finite values")
    if not torch.isfinite(tensors[C.W4A8_SUFFIX_S_CHANNEL]).all():
        raise W4A8Error(f"{layer}: weight_s_channel contains non-finite values")
    if not torch.isfinite(tensors[C.W4A8_SUFFIX_CODEBOOK]).all():
        raise W4A8Error(f"{layer}: weight_codebook contains non-finite values")


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
    """Decode a stored W4A8 group back to a dense weight (validation only)."""
    from comfy_kitchen.tensor import AsymW4A8Int8Layout

    params = AsymW4A8Int8Layout.Params(
        scale=tensors[C.W4A8_SUFFIX_S_REL],
        s_channel=tensors[C.W4A8_SUFFIX_S_CHANNEL],
        codebook=tensors.get(C.W4A8_SUFFIX_CODEBOOK),
        correction=tensors.get(C.W4A8_SUFFIX_CORRECTION),
        group_size=C.W4A8_GROUP_SIZE,
        convrot_groupsize=C.W4A8_CONVROT_GROUPSIZE,
        orig_dtype=output_dtype,
        orig_shape=tuple(logical_shape),
    )
    return AsymW4A8Int8Layout.dequantize(tensors[C.W4A8_SUFFIX_WEIGHT], params)
