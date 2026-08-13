"""Identify a MiniMax H3 checkpoint from its header alone.

The filename is never trusted. Everything below is read from tensor names,
shapes and dtypes, mirroring ComfyUI's own H3 detector
(``comfy/model_detection.py``) so that a source this module accepts is a source
the target runtime would also recognise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from h3converter import constants as C
from h3converter.safetensor_io import Header

_BLOCK_INDEX = re.compile(r"^blocks\.(\d+)\.")
_REFINER_BLOCK_INDEX = re.compile(r"^token_refiner\.blocks\.(\d+)\.")

# Key suffixes that mean "this checkpoint has already been quantized".
_QUANT_MARKERS = (
    ".comfy_quant",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".weight_scale",
    ".weight_scale_2",
    ".scale_weight",
)


@dataclass
class H3Geometry:
    """Shapes that define an H3 variant."""

    hidden_size: int
    num_layers: int
    token_refiner_num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    ffn_hidden_size: int
    latents_dim: int
    audio_latents_dim: int
    text_dim: int
    time_embed_dim: int
    rope_inv_freq_len: int
    timestep_input_dim: int | None = None
    time_embed_hidden_size: int | None = None
    adaln_curve_grid: int | None = None

    @property
    def attention_inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def block_adaln_width(self) -> int:
        return C.ADALN_BLOCK_EXPAND * C.ADALN_BLOCK_MODALITIES * self.hidden_size

    @property
    def final_adaln_width(self) -> int:
        return C.ADALN_FINAL_EXPAND * C.ADALN_FINAL_MODALITIES * self.hidden_size

    def as_dict(self) -> dict[str, int | None]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "token_refiner_num_layers": self.token_refiner_num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "attention_inner_dim": self.attention_inner_dim,
            "ffn_hidden_size": self.ffn_hidden_size,
            "latents_dim": self.latents_dim,
            "audio_latents_dim": self.audio_latents_dim,
            "text_dim": self.text_dim,
            "time_embed_dim": self.time_embed_dim,
            "timestep_input_dim": self.timestep_input_dim,
            "time_embed_hidden_size": self.time_embed_hidden_size,
            "rope_inv_freq_len": self.rope_inv_freq_len,
            "adaln_curve_grid": self.adaln_curve_grid,
            "block_adaln_width": self.block_adaln_width,
            "final_adaln_width": self.final_adaln_width,
        }


@dataclass
class Detection:
    """Outcome of inspecting one candidate source file."""

    is_h3: bool = False
    convertible: bool = False
    geometry: H3Geometry | None = None
    already_curve_pruned: bool = False
    already_quantized: bool = False
    float_dtype: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    header: Header | None = None

    @property
    def summary(self) -> str:
        if not self.is_h3:
            return "Not a MiniMax H3 checkpoint"
        if self.convertible:
            g = self.geometry
            return (
                f"MiniMax H3, {g.num_layers} blocks, hidden {g.hidden_size}, "
                f"time embed {g.time_embed_dim}, {self.float_dtype}"
            )
        return "MiniMax H3, but not convertible: " + "; ".join(self.errors)


def _count_indexed(keys: Iterable[str], pattern: re.Pattern[str]) -> int:
    highest = -1
    for key in keys:
        m = pattern.match(key)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def detect(header: Header) -> Detection:
    """Classify a parsed header. Never raises for ordinary bad input."""
    result = Detection(header=header)
    keys = header.tensors

    # --- Is this H3 at all? The two patch projections are the signature
    # ComfyUI itself keys on, and no other supported architecture has both.
    if f"{C.KEY_VIDEO_PATCH_PROJ}.weight" not in keys or f"{C.KEY_AUDIO_PATCH_PROJ}.weight" not in keys:
        result.errors.append(
            "missing video_patch_proj/audio_patch_proj - this is not a MiniMax H3 diffusion model"
        )
        return result
    result.is_h3 = True

    # --- Prior conversion state --------------------------------------------
    result.already_curve_pruned = C.ADALN_TABLE_KEY in keys
    result.already_quantized = (
        C.QUANT_METADATA_KEY in header.metadata
        or any(k.endswith(_QUANT_MARKERS) for k in keys)
    )

    # --- Geometry -----------------------------------------------------------
    try:
        geometry = _read_geometry(header)
    except _GeometryError as exc:
        result.errors.append(str(exc))
        return result
    result.geometry = geometry

    # --- Float precision ----------------------------------------------------
    counts = header.dtype_counts()
    float_counts = {d: n for d, n in counts.items() if d in ("BF16", "F16", "F32", "F64")}
    result.float_dtype = max(float_counts, key=float_counts.get) if float_counts else None

    # --- Convertibility -----------------------------------------------------
    result.errors.extend(_convertibility_errors(header, geometry, result))
    result.warnings.extend(_convertibility_warnings(header, geometry, result))
    result.convertible = not result.errors
    return result


class _GeometryError(RuntimeError):
    pass


def _require(header: Header, name: str) -> tuple[int, ...]:
    shape = header.shape_of(name)
    if shape is None:
        raise _GeometryError(f"required tensor {name!r} is missing")
    return shape


def _read_geometry(header: Header) -> H3Geometry:
    keys = header.tensors

    hidden_size = _require(header, f"{C.KEY_VIDEO_PATCH_PROJ}.weight")[0]
    num_layers = _count_indexed(keys, _BLOCK_INDEX)
    refiner_layers = _count_indexed(keys, _REFINER_BLOCK_INDEX)
    if num_layers == 0:
        raise _GeometryError("no 'blocks.N.' transformer blocks found")

    video_out = _require(header, f"{C.KEY_FINAL_VIDEO_OUT}.weight")
    audio_out = _require(header, f"{C.KEY_FINAL_AUDIO_OUT}.weight")
    head_dim = _require(header, "blocks.0.attn.q_norm.weight")[0]
    qkv = _require(header, "blocks.0.attn.qkv_proj.weight")
    fc1 = _require(header, "blocks.0.mlp.fc1.weight")
    cond = _require(header, f"{C.KEY_CONDITION_PROJ}.weight")
    rope = _require(header, C.KEY_ROPE_INV_FREQ)

    if head_dim == 0 or qkv[0] % (3 * head_dim) != 0:
        raise _GeometryError(
            f"blocks.0.attn.qkv_proj has {qkv[0]} outputs, which is not 3 x heads x {head_dim}"
        )

    # patch_size (1, 2, 2) => video_patch_dim = latents_dim * 4
    if video_out[0] % 4 != 0:
        raise _GeometryError(f"final_layer.video_out has {video_out[0]} outputs, not a multiple of the 1x2x2 patch")

    geometry = H3Geometry(
        hidden_size=hidden_size,
        num_layers=num_layers,
        token_refiner_num_layers=refiner_layers,
        num_attention_heads=qkv[0] // (3 * head_dim),
        attention_head_dim=head_dim,
        ffn_hidden_size=fc1[0] // 2,
        latents_dim=video_out[0] // 4,
        audio_latents_dim=audio_out[0],
        text_dim=cond[1],
        time_embed_dim=0,  # filled in below
        rope_inv_freq_len=rope[0],
    )

    if C.ADALN_TABLE_KEY in keys:
        table = _require(header, C.ADALN_TABLE_KEY)
        if len(table) != 2:
            raise _GeometryError(f"{C.ADALN_TABLE_KEY} must be 2D, got shape {table}")
        geometry.adaln_curve_grid = table[0]
        geometry.time_embed_dim = table[1]
    else:
        proj_in = _require(header, f"{C.KEY_TIME_PROJ_IN}.weight")
        proj_out = _require(header, f"{C.KEY_TIME_PROJ_OUT}.weight")
        geometry.timestep_input_dim = proj_in[1]
        geometry.time_embed_hidden_size = proj_in[0]
        geometry.time_embed_dim = proj_out[0]

    return geometry


def _convertibility_errors(header: Header, g: H3Geometry, result: Detection) -> list[str]:
    errors: list[str] = []
    keys = header.tensors

    if result.already_curve_pruned:
        errors.append(
            "source already uses the compact AdaLN curve form (adaln_t_table present) - "
            "this tool converts full, unpruned checkpoints"
        )
    if result.already_quantized:
        errors.append(
            "source is already quantized (quantization metadata or scale tensors present) - "
            "convert from the original BF16 checkpoint instead"
        )

    if not (C.H3_MIN_BLOCKS <= g.num_layers <= C.H3_MAX_BLOCKS):
        errors.append(
            f"{g.num_layers} transformer blocks is outside the supported H3 range "
            f"{C.H3_MIN_BLOCKS}-{C.H3_MAX_BLOCKS}"
        )

    # The full source must carry a complete time embedder; the curve form is
    # synthesised from it and there is no way to recover it otherwise.
    if not result.already_curve_pruned:
        for suffix in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"):
            key = f"{C.KEY_TIME_EMBEDDER}.{suffix}"
            if key not in keys:
                errors.append(f"missing {key} - the full time embedder is required to build the AdaLN curve")

    # AdaLN projections must be full width and consume the time embedding.
    expected_block = (g.block_adaln_width, g.time_embed_dim)
    expected_final = (g.final_adaln_width, g.time_embed_dim)
    for index in range(g.num_layers):
        name = f"blocks.{index}.adaln_proj.linear.weight"
        shape = header.shape_of(name)
        if shape is None:
            errors.append(f"missing {name}")
            break
        if shape != expected_block:
            errors.append(f"{name} has shape {shape}, expected {expected_block}")
            break

    final_name = f"{C.KEY_FINAL_ADALN}.weight"
    final_shape = header.shape_of(final_name)
    if final_shape is None:
        errors.append(f"missing {final_name}")
    elif final_shape != expected_final:
        errors.append(f"{final_name} has shape {final_shape}, expected {expected_final}")

    # Every layer Option A quantizes must exist, be 2D, and have a K that the
    # W4A8 layout accepts. Checking now avoids failing 40 blocks deep.
    for index in range(g.num_layers):
        for family in C.W4A8_BLOCK_LINEARS:
            name = f"blocks.{index}.{family}.weight"
            shape = header.shape_of(name)
            if shape is None:
                errors.append(f"missing {name}")
                return errors
            if len(shape) != 2:
                errors.append(f"{name} must be 2D, got {shape}")
                return errors
            k = shape[1]
            if k % C.W4A8_CONVROT_GROUPSIZE or k % C.W4A8_GROUP_SIZE:
                errors.append(
                    f"{name} input width {k} is not divisible by the ConvRot group "
                    f"({C.W4A8_CONVROT_GROUPSIZE}) and quant group ({C.W4A8_GROUP_SIZE})"
                )
                return errors

    return errors


def _convertibility_warnings(header: Header, g: H3Geometry, result: Detection) -> list[str]:
    warnings: list[str] = []
    reference = C.H3_REFERENCE_GEOMETRY

    for field_name in ("hidden_size", "num_layers", "attention_head_dim",
                       "num_attention_heads", "ffn_hidden_size", "time_embed_dim"):
        actual = getattr(g, field_name)
        expected = reference.get(field_name)
        if expected is not None and actual != expected:
            warnings.append(
                f"{field_name} is {actual}, reference H3 uses {expected} - "
                "conversion will proceed against the detected geometry"
            )

    if result.float_dtype not in (None, "BF16"):
        warnings.append(
            f"source stores most tensors as {result.float_dtype}; the reference source is BF16"
        )

    if g.token_refiner_num_layers == 0:
        warnings.append("no token_refiner blocks found - unusual for H3")

    return warnings
