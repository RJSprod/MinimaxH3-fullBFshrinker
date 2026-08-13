"""Header-only structural detection for Krea 2 diffusion checkpoints."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from h3converter.constants import QUANT_METADATA_KEY
from h3converter.safetensor_io import Header

SIGNATURE = "txtfusion.projector.weight"
REFERENCE = {"blocks": 28, "features": 6144, "channels": 64, "heads": 48,
             "kv_heads": 12, "head_dim": 128, "text_layers": 12, "text_hidden_dim": 2560}
_QUANT_SUFFIXES = (".weight_scale", ".weight_scale_2", ".scale_weight", ".input_scale", ".comfy_quant")

@dataclass(frozen=True)
class Krea2Geometry:
    blocks: int = 28
    features: int = 6144
    channels: int = 64
    attention_heads: int = 48
    kv_heads: int = 12
    head_dim: int = 128
    text_layers: int = 12
    text_hidden_dim: int = 2560
    def as_dict(self):
        return dict(REFERENCE)

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
    def summary(self):
        return "Krea 2, 28 blocks, features 6144, " + (self.float_dtype or "unknown") if self.is_krea2 else "Not a Krea 2 checkpoint"

def _prefix(keys: set[str]) -> str | None:
    matches = [k[:-len(SIGNATURE)] for k in keys if k.endswith(SIGNATURE)]
    return matches[0] if len(matches) == 1 else None

def detect(header: Header) -> Detection:
    out = Detection(header=header)
    keys = set(header.tensors)
    prefix = _prefix(keys)
    if prefix is None:
        out.errors.append("missing or ambiguous txtfusion.projector.weight signature")
        return out
    out.is_krea2, out.prefix = True, prefix
    required = {
        "first.weight": (REFERENCE["features"], REFERENCE["channels"]),
        "blocks.0.attn.wq.weight": (REFERENCE["features"], REFERENCE["features"]),
        "blocks.0.attn.wk.weight": (REFERENCE["kv_heads"] * REFERENCE["head_dim"], REFERENCE["features"]),
        "txtfusion.layerwise_blocks.0.prenorm.scale": (REFERENCE["text_hidden_dim"],),
    }
    for suffix, expected in required.items():
        actual = header.shape_of(prefix + suffix)
        if actual is None:
            out.errors.append(f"required tensor {prefix + suffix!r} is missing")
        elif actual != expected:
            out.errors.append(f"{prefix + suffix} has shape {actual}, expected {expected}")
    indices = {int(m.group(1)) for key in keys if (m := re.match(re.escape(prefix) + r"blocks\.(\d+)\.", key))}
    if indices != set(range(REFERENCE["blocks"])):
        out.errors.append(f"expected 28 contiguous transformer blocks, found {len(indices)}")
    text_indices = {int(m.group(1)) for key in keys if (m := re.match(re.escape(prefix) + r"txtfusion\.layerwise_blocks\.(\d+)\.", key))}
    if text_indices != set(range(REFERENCE["text_layers"])):
        out.errors.append(f"expected 12 contiguous text layers, found {len(text_indices)}")
    out.already_quantized = (QUANT_METADATA_KEY in header.metadata or
        any(i.dtype in {"F8_E4M3", "F8_E5M2", "I8", "U8"} for i in header.tensors.values()) or
        any(k.endswith(_QUANT_SUFFIXES) for k in keys))
    if out.already_quantized:
        out.errors.append("source is already quantized; use a full BF16/F32 Krea 2 checkpoint")
    payload_dtypes = {i.dtype for i in header.tensors.values()}
    invalid = payload_dtypes - {"BF16", "F32"}
    if invalid and not out.already_quantized:
        out.errors.append(f"Krea v1 accepts only BF16/F32 payloads, found {sorted(invalid)}")
    elems = header.dtype_elements()
    floats = {d: elems.get(d, 0) for d in ("BF16", "F32") if elems.get(d, 0)}
    out.float_dtype = max(floats, key=floats.get) if floats else None
    out.geometry = Krea2Geometry() if not any("shape" in e or "expected" in e or "missing" in e for e in out.errors) else None
    out.convertible = out.geometry is not None and not out.errors
    return out
