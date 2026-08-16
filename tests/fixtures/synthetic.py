"""Synthetic H3 checkpoints for testing.

Structurally a real H3 checkpoint -- every tensor name, dtype and shape
relationship the detector, policy and validators care about -- but small enough
to convert in seconds. Written with the official ``safetensors`` serializer, so
the project's own reader is always tested against the reference writer (and,
via validation, vice versa).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from h3converter.h3_detect import H3Geometry
from h3converter.h3_reference import full_source_inventory, prepruned_source_inventory
from h3converter.safetensor_io import ST_TO_TORCH

# A miniature H3: narrow, but with a real block count.
#
# The block count is deliberately not shrunk below the detector's supported
# range -- an end-to-end test that had to disable the architecture gate would
# not be testing the shipping code path. Every input width is a multiple of the
# 256-element ConvRot group, which is the real constraint the layout imposes.
SMALL_GEOMETRY = H3Geometry(
    hidden_size=256,
    num_layers=40,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=128,
    ffn_hidden_size=256,
    latents_dim=4,
    audio_latents_dim=8,
    text_dim=64,
    time_embed_dim=64,
    rope_inv_freq_len=8,
    timestep_input_dim=32,
    time_embed_hidden_size=128,
)


def _fill(name: str, dtype: torch.dtype, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    if not shape:
        return torch.zeros((), dtype=dtype)
    numel = 1
    for d in shape:
        numel *= d

    if name.endswith("norm.weight") or ".norm1." in name or ".norm2." in name or "_norm." in name:
        # RMSNorm gains sit around 1.0.
        values = 1.0 + 0.05 * torch.randn(numel, generator=generator)
    elif name == "rope.inv_freq":
        values = torch.exp(-torch.linspace(0.0, 9.0, numel))
    elif name.endswith(".bias"):
        values = 0.02 * torch.randn(numel, generator=generator)
    else:
        # Kaiming-ish scaling keeps quantization error in a realistic range.
        fan_in = shape[-1] if len(shape) > 1 else numel
        values = torch.randn(numel, generator=generator) * (fan_in ** -0.5)

    return values.reshape(shape).to(dtype)


def build_synthetic_h3(
    path: Path,
    geometry: H3Geometry = SMALL_GEOMETRY,
    seed: int = 20260813,
    float_dtype: str = "BF16",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a full, unpruned synthetic H3 checkpoint to ``path``."""
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for spec in full_source_inventory(geometry, float_dtype=float_dtype):
        tensors[spec.name] = _fill(spec.name, ST_TO_TORCH[spec.dtype], spec.shape, generator)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata or {"model": "synthetic-minimax-h3"})
    return path


def build_prepruned_h3(
    path: Path,
    geometry: H3Geometry = SMALL_GEOMETRY,
    seed: int = 20260816,
    float_dtype: str = "BF16",
    adaln_dtype: str | None = None,
    table_dtype: str = "BF16",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic H3 that is *already* in the AdaLN curve form.

    This is the shape a TenStrip/10Eros-Max checkpoint arrives in: no time
    embedder, an ``adaln_t_table`` at the full 1025x8, and every AdaLN
    projection already reduced to rank 8.

    ``adaln_dtype`` and ``table_dtype`` are separable from ``float_dtype`` so
    the suite can cover both the BF16 curve real checkpoints ship and an F32
    one, either of which the converter must copy through and then accept in its
    own output validation.
    """
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for spec in prepruned_source_inventory(
        geometry, float_dtype=float_dtype, adaln_dtype=adaln_dtype,
        table_dtype=table_dtype,
    ):
        tensors[spec.name] = _fill(spec.name, ST_TO_TORCH[spec.dtype], spec.shape, generator)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata or {"model": "synthetic-minimax-h3-pruned"})
    return path
