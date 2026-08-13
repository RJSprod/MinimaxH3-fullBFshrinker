"""Output metadata and the ``<output>.report.json`` conversion record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from h3converter import constants as C
from h3converter.h3_policy import OutputPlan

# Source metadata keys that would be actively misleading if copied into a
# converted checkpoint. Everything else the author put there is preserved.
_DROPPED_SOURCE_METADATA = {
    C.QUANT_METADATA_KEY,
    "format",
    "quantization",
    "converted_by",
    "converter_version",
}

_SOURCE_METADATA_PREFIX = "source."


def build_quantization_metadata(plan: OutputPlan, layer_configs: Mapping[str, dict]) -> str:
    """The ``_quantization_metadata`` JSON string ComfyUI reads at load time.

    Schema (``comfy/utils.py::convert_old_quants``): a top-level object with a
    ``layers`` map from layer name to that layer's config, which is what the
    loader turns into each layer's ``comfy_quant`` blob.
    """
    layers = {}
    for target in plan.quant_targets:
        config = layer_configs.get(target.layer)
        if config is None:
            raise ValueError(f"no quantization config recorded for {target.layer}")
        layers[target.layer] = config
    payload = {
        "format_version": C.QUANT_METADATA_FORMAT_VERSION,
        "layers": layers,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def build_output_metadata(
    source_metadata: Mapping[str, str],
    source_path: Path,
    output_format: str,
    plan: OutputPlan,
    layer_configs: Mapping[str, dict],
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """safetensors ``__metadata__`` for the converted checkpoint.

    Source metadata is preserved under a ``source.`` prefix rather than
    inline, so a stale ``format`` or quantization descriptor from the original
    file can never be mistaken for a description of this one.
    """
    metadata: dict[str, str] = {}

    for key, value in source_metadata.items():
        if key in _DROPPED_SOURCE_METADATA:
            continue
        metadata[f"{_SOURCE_METADATA_PREFIX}{key}"] = str(value)

    metadata[C.QUANT_METADATA_KEY] = build_quantization_metadata(plan, layer_configs)
    metadata.update(
        {
            "converted_by": f"{C.APP_NAME} {C.APP_VERSION}",
            "converter_id": C.CONVERTER_ID,
            "converter_version": C.APP_VERSION,
            "format_version": C.QUANT_METADATA_FORMAT_VERSION,
            "output_format": output_format,
            "output_format_label": C.FORMAT_LABELS[output_format],
            "source_filename": source_path.name,
            "source_architecture": "minimax_h3",
            "adaln_prune_version": C.ADALN_PRUNE_VERSION,
            "adaln_curve_grid": str(C.ADALN_CURVE_GRID),
            "adaln_curve_rank": str(C.ADALN_CURVE_RANK),
            "quantization_policy": (
                C.W4A8_POLICY_VERSION if output_format == C.FORMAT_W4A8 else C.NVFP4_POLICY_VERSION
            ),
            "quantized_layer_count": str(plan.quantized_layer_count),
            "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    if extra:
        metadata.update({str(k): str(v) for k, v in extra.items()})
    return metadata


# ---------------------------------------------------------------------------
# Source identification
# ---------------------------------------------------------------------------

def fingerprint(path: Path, edge_bytes: int = 16 * 1024 * 1024) -> str:
    """Fast content fingerprint: size plus the head and tail of the file.

    Identifies a specific checkpoint without re-reading 66 GB. A full SHA-256
    is available separately when a caller wants one.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with open(path, "rb") as fh:
        digest.update(fh.read(min(edge_bytes, size)))
        if size > edge_bytes:
            fh.seek(max(size - edge_bytes, edge_bytes))
            digest.update(fh.read(edge_bytes))
    return f"h3fp1:{digest.hexdigest()}"


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024, progress=None) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, size)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Conversion report
# ---------------------------------------------------------------------------

@dataclass
class ConversionReport:
    """Everything needed to reproduce, audit or debug one conversion."""

    application: dict[str, object] = field(default_factory=dict)
    source: dict[str, object] = field(default_factory=dict)
    output: dict[str, object] = field(default_factory=dict)
    architecture: dict[str, object] = field(default_factory=dict)
    pruning: dict[str, object] = field(default_factory=dict)
    quantization: dict[str, object] = field(default_factory=dict)
    calibration: dict[str, object] = field(default_factory=dict)
    environment: dict[str, object] = field(default_factory=dict)
    resources: dict[str, object] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "application": self.application,
            "source": self.source,
            "output": self.output,
            "architecture": self.architecture,
            "pruning": self.pruning,
            "quantization": self.quantization,
            "calibration": self.calibration,
            "environment": self.environment,
            "resources": self.resources,
            "validation": self.validation,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")
        return path


def summarise_layer_stats(stats: Sequence[dict[str, float]]) -> dict[str, object]:
    """Aggregate per-layer quantization error measurements."""
    measured = [s for s in stats if "rel_l2" in s]
    if not measured:
        return {"measured_layers": 0}
    rel = sorted(s["rel_l2"] for s in measured)
    return {
        "measured_layers": len(measured),
        "rel_l2_min": rel[0],
        "rel_l2_median": rel[len(rel) // 2],
        "rel_l2_max": rel[-1],
        "rel_l2_mean": sum(rel) / len(rel),
        "max_abs_error": max(s.get("max_abs_error", 0.0) for s in measured),
    }
