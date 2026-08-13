"""Post-write validation for mixed-precision Krea 2 outputs."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from h3converter import constants as C
from h3converter.krea2_policy import OutputPlan
from h3converter.safetensor_io import Header, read_header

@dataclass
class Validation:
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    def as_dict(self): return {"ok": self.ok, "errors": self.errors}
    def summary(self): return "valid" if self.ok else "; ".join(self.errors)

def validate_output(path: Path, source: Header, plan: OutputPlan) -> Validation:
    result = Validation()
    try: out = read_header(path)
    except Exception as exc:
        result.errors.append(str(exc)); return result
    if set(out.tensors) != {t.name for t in plan.tensors}:
        result.errors.append("output tensor inventory differs from plan")
    try: metadata = json.loads(out.metadata[C.QUANT_METADATA_KEY])
    except Exception as exc: result.errors.append(f"invalid _quantization_metadata: {exc}"); metadata = {}
    layers = metadata.get("layers", {})
    for target in plan.quant_targets:
        weight, scale = out.get(target.source_key), out.get(target.scale_key)
        if weight is None or weight.dtype != "F8_E4M3": result.errors.append(f"{target.source_key} is not F8_E4M3")
        if scale is None or scale.dtype != "F32" or scale.shape != (): result.errors.append(f"{target.scale_key} is not scalar F32")
        config = layers.get(target.layer)
        if config is None:
            result.errors.append(f"metadata missing {target.layer}")
        elif config.get("format") != C.QUANT_FORMAT_KREA2_FP8:
            result.errors.append(
                f"metadata for {target.layer} declares unsupported format "
                f"{config.get('format')!r}, expected {C.QUANT_FORMAT_KREA2_FP8!r}"
            )
    for key in plan.passthrough_keys:
        a, b = source.tensors[key], out.get(key)
        if b is None or (a.dtype, a.shape) != (b.dtype, b.shape): result.errors.append(f"preserved tensor changed: {key}")
    result.ok = not result.errors
    return result
