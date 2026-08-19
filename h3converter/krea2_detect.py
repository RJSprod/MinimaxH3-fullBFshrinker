"""Header-only structural detection for native/mapped Krea 2 checkpoints."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from h3converter.constants import QUANT_METADATA_KEY
from h3converter.safetensor_io import Header

SIGNATURE = "txtfusion.projector.weight"
PATCH_SIZE = 2
HEAD_DIM = 128
_QUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".scale_weight",
    ".input_scale",
    ".comfy_quant",
)


@dataclass(frozen=True)
class Krea2Geometry:
    blocks: int
    features: int
    channels: int
    patch_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    text_layers: int
    layerwise_text_blocks: int
    refiner_text_blocks: int
    text_hidden_dim: int
    mlp_hidden_dim: int
    text_mlp_hidden_dim: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Detection:
    is_krea2: bool = False
    convertible: bool = False
    prefix: str = ""
    geometry: Krea2Geometry | None = None
    already_quantized: bool = False
    float_dtype: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    header: Header | None = None

    @property
    def summary(self) -> str:
        if not self.is_krea2:
            return "Not a Krea 2 checkpoint"
        if self.geometry is None:
            return "Krea 2, but its geometry is unsupported"
        return (
            f"Krea 2, {self.geometry.blocks} blocks, features "
            f"{self.geometry.features}, {self.float_dtype or 'unknown'}"
        )


def _prefix(keys: set[str]) -> str | None:
    matches = [key.removesuffix(SIGNATURE) for key in keys if key.endswith(SIGNATURE)]
    return matches[0] if len(matches) == 1 else None


def _indices(keys: set[str], prefix: str, family: str) -> set[int]:
    pattern = re.compile(re.escape(prefix + family) + r"(\d+)\.")
    return {int(match.group(1)) for key in keys if (match := pattern.match(key))}


def _require_shape(
    header: Header,
    prefix: str,
    suffix: str,
    expected: tuple[int, ...],
    errors: list[str],
) -> None:
    name = prefix + suffix
    actual = header.shape_of(name)
    if actual is None:
        errors.append(f"required tensor {name!r} is missing")
    elif actual != expected:
        errors.append(f"{name} has shape {actual}, expected {expected}")


def _require_family_shape(
    header: Header,
    prefix: str,
    family: str,
    expected: tuple[int, ...],
    label: str,
    errors: list[str],
) -> None:
    matches = [
        name for name, info in header.tensors.items()
        if name.startswith(prefix + family) and info.shape == expected
    ]
    if not matches:
        errors.append(f"{label} tensor with shape {expected} is missing under {prefix + family}")


def detect(header: Header) -> Detection:
    """Detect and fully validate the Comfy/Forge Krea 2 state-dict geometry."""
    out = Detection(header=header)
    keys = set(header.tensors)
    prefix = _prefix(keys)
    if prefix is None:
        out.errors.append("missing or ambiguous txtfusion.projector.weight signature")
        return out
    out.is_krea2, out.prefix = True, prefix

    projector = header.shape_of(prefix + SIGNATURE)
    first = header.shape_of(prefix + "first.weight")
    wq = header.shape_of(prefix + "blocks.0.attn.wq.weight")
    wk = header.shape_of(prefix + "blocks.0.attn.wk.weight")
    if projector != (1, 12):
        out.errors.append(f"{prefix + SIGNATURE} has shape {projector}, expected (1, 12)")
    if first != (6144, 64):
        out.errors.append(f"{prefix}first.weight has shape {first}, expected (6144, 64)")

    features = first[0] if first and len(first) == 2 else 0
    flattened_width = first[1] if first and len(first) == 2 else 0
    channels = flattened_width // (PATCH_SIZE * PATCH_SIZE)
    heads = wq[0] // HEAD_DIM if wq and len(wq) == 2 else 0
    kv_heads = wk[0] // HEAD_DIM if wk and len(wk) == 2 else 0
    text_layers = projector[1] if projector and len(projector) == 2 else 0

    _require_shape(header, prefix, "last.linear.weight", (64, 6144), out.errors)
    _require_shape(header, prefix, "blocks.0.attn.wq.weight", (6144, 6144), out.errors)
    _require_shape(header, prefix, "blocks.0.attn.wk.weight", (1536, 6144), out.errors)
    _require_shape(
        header, prefix, "txtfusion.layerwise_blocks.0.prenorm.scale", (2560,), out.errors
    )
    _require_family_shape(
        header, prefix, "blocks.0.", (16384, 6144), "main MLP hidden projection", out.errors
    )
    _require_family_shape(
        header,
        prefix,
        "txtfusion.layerwise_blocks.0.",
        (6912, 2560),
        "text-fusion MLP hidden projection",
        out.errors,
    )

    block_indices = _indices(keys, prefix, "blocks.")
    layerwise_indices = _indices(keys, prefix, "txtfusion.layerwise_blocks.")
    refiner_indices = _indices(keys, prefix, "txtfusion.refiner_blocks.")
    expected_counts = (
        ("transformer blocks", block_indices, set(range(28))),
        ("layerwise text blocks", layerwise_indices, {0, 1}),
        ("refiner text blocks", refiner_indices, {0, 1}),
    )
    for label, actual, expected in expected_counts:
        if actual != expected:
            out.errors.append(
                f"expected {len(expected)} contiguous {label}, found {len(actual)} "
                f"(indices {sorted(actual)})"
            )

    if channels != 16 or flattened_width != channels * PATCH_SIZE * PATCH_SIZE:
        out.errors.append(
            f"first.weight width {flattened_width} does not describe 16 channels at patch size 2"
        )
    if heads != 48 or kv_heads != 12:
        out.errors.append(f"expected 48 attention heads and 12 KV heads, found {heads} and {kv_heads}")

    out.already_quantized = (
        QUANT_METADATA_KEY in header.metadata
        or any(info.dtype in {"F8_E4M3", "F8_E5M2", "I8", "U8"} for info in header.tensors.values())
        or any(key.endswith(_QUANT_SUFFIXES) for key in keys)
    )
    if out.already_quantized:
        out.errors.append("source is already quantized; use a full BF16/F32 Krea 2 checkpoint")
    payload_dtypes = {info.dtype for info in header.tensors.values()}
    invalid = payload_dtypes - {"BF16", "F32"}
    if invalid and not out.already_quantized:
        out.errors.append(f"Krea v1 accepts only BF16/F32 payloads, found {sorted(invalid)}")

    elements = header.dtype_elements()
    floats = {dtype: elements.get(dtype, 0) for dtype in ("BF16", "F32") if elements.get(dtype)}
    out.float_dtype = max(floats, key=floats.get) if floats else None
    if not out.errors:
        out.geometry = Krea2Geometry(
            blocks=len(block_indices),
            features=features,
            channels=channels,
            patch_size=PATCH_SIZE,
            attention_heads=heads,
            kv_heads=kv_heads,
            head_dim=HEAD_DIM,
            text_layers=text_layers,
            layerwise_text_blocks=len(layerwise_indices),
            refiner_text_blocks=len(refiner_indices),
            text_hidden_dim=2560,
            mlp_hidden_dim=16384,
            text_mlp_hidden_dim=6912,
        )
    out.convertible = out.geometry is not None and not out.errors
    return out
