"""The canonical tensor inventory of a full, unpruned BF16 H3 checkpoint.

Enumerated from ComfyUI's ``MiniMaxH3Model`` module tree, including the dtypes
each submodule is constructed with -- the patch projections, the output heads
and the time embedder are explicitly float32 there, everything else follows the
model dtype.

This serves two purposes:

* it lets :mod:`h3converter.validate` predict the exact output inventory for a
  given geometry, which is how the golden reference file's 1,132-tensor,
  12,540,857,840-byte census is reproduced analytically; and
* it builds the synthetic fixtures the test suite converts end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

from h3converter import constants as C
from h3converter.h3_detect import H3Geometry


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def reference_geometry(**overrides) -> H3Geometry:
    """H3Geometry populated from the reference defaults, with optional overrides."""
    base = dict(C.H3_REFERENCE_GEOMETRY)
    base.update(overrides)
    return H3Geometry(
        hidden_size=base["hidden_size"],
        num_layers=base["num_layers"],
        token_refiner_num_layers=base["token_refiner_num_layers"],
        num_attention_heads=base["num_attention_heads"],
        attention_head_dim=base["attention_head_dim"],
        ffn_hidden_size=base["ffn_hidden_size"],
        latents_dim=base["latents_dim"],
        audio_latents_dim=base["audio_latents_dim"],
        text_dim=base["text_dim"],
        time_embed_dim=base["time_embed_dim"],
        rope_inv_freq_len=base["rope_inv_freq_len"],
        timestep_input_dim=base["timestep_input_dim"],
        time_embed_hidden_size=base["time_embed_hidden_size"],
    )


def video_patch_dim(geometry: H3Geometry) -> int:
    patch = C.H3_REFERENCE_GEOMETRY["patch_size"]
    return geometry.latents_dim * patch[0] * patch[1] * patch[2]


def _attention_and_mlp(prefix: str, geometry: H3Geometry, dtype: str) -> list[TensorSpec]:
    hidden = geometry.hidden_size
    inner = geometry.attention_inner_dim
    ffn = geometry.ffn_hidden_size
    return [
        # Attention: qkv_proj and out_proj are bias-free in H3.
        TensorSpec(f"{prefix}.attn.qkv_proj.weight", dtype, (inner * 3, hidden)),
        TensorSpec(f"{prefix}.attn.q_norm.weight", dtype, (geometry.attention_head_dim,)),
        TensorSpec(f"{prefix}.attn.k_norm.weight", dtype, (geometry.attention_head_dim,)),
        TensorSpec(f"{prefix}.attn.out_proj.weight", dtype, (hidden, inner)),
        # SwiGLU MLP: fc1 emits gate and value halves, fc2 consumes one half.
        TensorSpec(f"{prefix}.mlp.fc1.weight", dtype, (ffn * 2, hidden)),
        TensorSpec(f"{prefix}.mlp.fc2.weight", dtype, (hidden, ffn)),
    ]


def full_source_inventory(geometry: H3Geometry, float_dtype: str = "BF16") -> list[TensorSpec]:
    """Every tensor a full, unpruned H3 checkpoint contains, in load order."""
    hidden = geometry.hidden_size
    specs: list[TensorSpec] = []

    # fp32 input projections
    specs.append(TensorSpec("video_patch_proj.weight", "F32", (hidden, video_patch_dim(geometry))))
    specs.append(TensorSpec("video_patch_proj.bias", "F32", (hidden,)))
    specs.append(TensorSpec("audio_patch_proj.weight", "F32", (hidden, geometry.audio_latents_dim)))
    specs.append(TensorSpec("audio_patch_proj.bias", "F32", (hidden,)))

    # text conditioning
    specs.append(TensorSpec("condition_proj.weight", float_dtype, (hidden, geometry.text_dim)))
    specs.append(TensorSpec("condition_proj.bias", float_dtype, (hidden,)))

    # full timestep path (fp32; removed entirely by the curve conversion)
    specs.append(TensorSpec("time_embedder.proj_in.weight", "F32",
                            (geometry.time_embed_hidden_size, geometry.timestep_input_dim)))
    specs.append(TensorSpec("time_embedder.proj_in.bias", "F32", (geometry.time_embed_hidden_size,)))
    specs.append(TensorSpec("time_embedder.proj_out.weight", "F32",
                            (geometry.time_embed_dim, geometry.time_embed_hidden_size)))
    specs.append(TensorSpec("time_embedder.proj_out.bias", "F32", (geometry.time_embed_dim,)))

    specs.append(TensorSpec("rope.inv_freq", "F32", (geometry.rope_inv_freq_len,)))

    # token refiner
    for index in range(geometry.token_refiner_num_layers):
        prefix = f"token_refiner.blocks.{index}"
        specs.append(TensorSpec(f"{prefix}.norm1.weight", float_dtype, (hidden,)))
        specs.append(TensorSpec(f"{prefix}.norm2.weight", float_dtype, (hidden,)))
        specs.extend(_attention_and_mlp(prefix, geometry, float_dtype))
    specs.append(TensorSpec("token_refiner.final_norm.weight", float_dtype, (hidden,)))

    # main transformer blocks
    for index in range(geometry.num_layers):
        prefix = f"blocks.{index}"
        specs.append(TensorSpec(f"{prefix}.norm1.weight", float_dtype, (hidden,)))
        specs.append(TensorSpec(f"{prefix}.norm2.weight", float_dtype, (hidden,)))
        specs.extend(_attention_and_mlp(prefix, geometry, float_dtype))
        specs.append(TensorSpec(f"{prefix}.adaln_proj.linear.weight", float_dtype,
                                (geometry.block_adaln_width, geometry.time_embed_dim)))
        specs.append(TensorSpec(f"{prefix}.adaln_proj.linear.bias", float_dtype,
                                (geometry.block_adaln_width,)))

    # final layer
    specs.append(TensorSpec("final_layer.norm.weight", float_dtype, (hidden,)))
    specs.append(TensorSpec("final_layer.adaln_proj.linear.weight", float_dtype,
                            (geometry.final_adaln_width, geometry.time_embed_dim)))
    specs.append(TensorSpec("final_layer.adaln_proj.linear.bias", float_dtype,
                            (geometry.final_adaln_width,)))
    specs.append(TensorSpec("final_layer.video_out.weight", "F32", (video_patch_dim(geometry), hidden)))
    specs.append(TensorSpec("final_layer.video_out.bias", "F32", (video_patch_dim(geometry),)))
    specs.append(TensorSpec("final_layer.audio_out.weight", "F32", (geometry.audio_latents_dim, hidden)))
    specs.append(TensorSpec("final_layer.audio_out.bias", "F32", (geometry.audio_latents_dim,)))

    return specs
