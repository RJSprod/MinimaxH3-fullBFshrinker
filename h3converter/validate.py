"""Post-conversion validation and golden-reference conformance.

Nothing is renamed to its final filename until every check here passes. The
checks are deliberately structural rather than statistical: they read the
written file back and confirm it satisfies the contract the target runtime
loads against, layer by layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from h3converter import constants as C
from h3converter.h3_detect import H3Geometry
from h3converter.h3_policy import OutputPlan, build_output_plan
from h3converter.h3_reference import TensorSpec, full_source_inventory
from h3converter.safetensor_io import (
    DTYPE_ITEMSIZE,
    Header,
    TensorInfo,
    read_header,
)


# ---------------------------------------------------------------------------
# Analytic inventory prediction
# ---------------------------------------------------------------------------

def header_from_specs(specs: Sequence[TensorSpec], metadata: dict[str, str] | None = None) -> Header:
    """A synthetic ``Header`` over a tensor list, for planning without a file."""
    tensors: dict[str, TensorInfo] = {}
    cursor = 0
    for spec in specs:
        nbytes = spec.numel * DTYPE_ITEMSIZE[spec.dtype]
        tensors[spec.name] = TensorInfo(spec.name, spec.dtype, spec.shape, cursor, cursor + nbytes)
        cursor += nbytes
    return Header(
        path=Path("<synthetic>"),
        tensors=tensors,
        metadata=dict(metadata or {}),
        data_start=0,
        file_size=cursor,
    )


def predict_inventory(geometry: H3Geometry, output_format: str) -> dict[str, object]:
    """Exact output census for a geometry, without converting anything.

    Runs the real policy over a synthesised full-H3 header, so this is a
    prediction *by the production code path*, not a parallel model of it.
    """
    header = header_from_specs(full_source_inventory(geometry))
    plan = build_output_plan(header, geometry, output_format)
    return inventory_of_plan(plan)


def inventory_of_plan(plan: OutputPlan) -> dict[str, object]:
    counts = plan.dtype_counts()
    elements = plan.dtype_elements()
    return {
        "tensor_count": len(plan.tensors),
        "quantized_layer_count": plan.quantized_layer_count,
        "dtype_counts": counts,
        "dtype_elements": elements,
        "dtype_bytes": {d: elements[d] * DTYPE_ITEMSIZE[d] for d in elements},
        "data_bytes": plan.total_bytes,
    }


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    check: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"check": self.check, "ok": self.ok, "detail": self.detail}


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    inventory: dict[str, object] = field(default_factory=dict)
    conformance: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, check: str, ok: bool, detail: str = "") -> None:
        self.findings.append(Finding(check, ok, detail))

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.ok]

    def summary(self) -> str:
        failed = self.failures()
        if not failed:
            return f"passed {len(self.findings)} checks"
        return f"{len(failed)} of {len(self.findings)} checks failed: " + "; ".join(
            f"{f.check} ({f.detail})" for f in failed[:4]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "findings": [f.as_dict() for f in self.findings],
            "inventory": self.inventory,
            "conformance": self.conformance,
            "warnings": list(self.warnings),
        }


class ValidationError(RuntimeError):
    pass


def validate_output(
    path: Path,
    plan: OutputPlan,
    geometry: H3Geometry,
    output_format: str,
    sample_layers: int = 4,
    dequantize_samples: bool = True,
) -> ValidationReport:
    """Read the written file back and verify it against the format contract."""
    report = ValidationReport()

    try:
        header = read_header(path)
    except Exception as exc:  # noqa: BLE001
        report.add("file.readable", False, f"{type(exc).__name__}: {exc}")
        return report
    report.add("file.readable", True, f"{len(header.tensors)} tensors, {header.file_size:,} bytes")

    _check_structure(report, header, plan, geometry)
    _check_metadata(report, header, plan, output_format)
    _check_quantized_layers(report, header, plan, output_format)

    if plan.quant_targets:
        _check_scale_health(report, path, plan, output_format)
    if dequantize_samples and plan.quant_targets:
        _check_dequantization(report, path, plan, output_format, sample_layers)

    report.inventory = {
        "tensor_count": len(header.tensors),
        "dtype_counts": header.dtype_counts(),
        "dtype_elements": header.dtype_elements(),
        "dtype_bytes": header.dtype_bytes(),
        "file_bytes": header.file_size,
    }
    report.conformance = _conformance(header, plan, output_format)
    return report


def _check_structure(report: ValidationReport, header: Header, plan: OutputPlan,
                     geometry: H3Geometry) -> None:
    planned = {t.name: t for t in plan.tensors}
    missing = sorted(set(planned) - set(header.tensors))
    extra = sorted(set(header.tensors) - set(planned))
    report.add(
        "structure.inventory_matches_plan",
        not missing and not extra,
        (f"missing {missing[:3]}" if missing else "")
        + (f" unexpected {extra[:3]}" if extra else "")
        or f"{len(planned)} tensors as planned",
    )

    mismatched = [
        name for name, spec in planned.items()
        if name in header.tensors
        and (header.tensors[name].dtype != spec.dtype or header.tensors[name].shape != spec.shape)
    ]
    report.add(
        "structure.dtypes_and_shapes",
        not mismatched,
        f"{len(mismatched)} mismatches, first: {mismatched[0]}" if mismatched else "all match plan",
    )

    # The time embedder must be gone: its behaviour now lives in the table.
    leftover = [k for k in header.tensors if k.startswith(f"{C.KEY_TIME_EMBEDDER}.")]
    report.add(
        "pruning.time_embedder_removed",
        not leftover,
        f"{len(leftover)} time_embedder tensors remain" if leftover else "removed",
    )

    table = header.get(C.ADALN_TABLE_KEY)
    expected_table = (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK)
    report.add(
        "pruning.adaln_table",
        table is not None and table.dtype == "F32" and table.shape == expected_table,
        (f"{table.dtype} {table.shape}" if table else "missing")
        + f" (expected F32 {expected_table})",
    )

    # The reduced projections' dtype comes from the plan, not from a constant.
    # The full path writes BF16 by policy; a pre-pruned source keeps whatever
    # dtype it already used, and F32 there is as legitimate as BF16. Hard-coding
    # BF16 would fail a checkpoint the converter had copied through correctly.
    planned_dtype = {t.name: t.dtype for t in plan.tensors}
    float_dtypes = ("BF16", "F16", "F32", "F64")

    expected_block = (geometry.block_adaln_width, C.ADALN_CURVE_RANK)
    bad_blocks = []
    block_dtypes: set[str] = set()
    for index in range(geometry.num_layers):
        name = f"blocks.{index}.adaln_proj.linear.weight"
        info = header.get(name)
        expected = planned_dtype.get(name)
        if (info is None or info.shape != expected_block
                or info.dtype != expected or info.dtype not in float_dtypes):
            bad_blocks.append(index)
        else:
            block_dtypes.add(info.dtype)
    report.add(
        "pruning.block_adaln_reduced",
        not bad_blocks,
        f"{len(bad_blocks)} projections are not {expected_block} at the planned dtype"
        if bad_blocks
        else f"{geometry.num_layers} x {'/'.join(sorted(block_dtypes))} {expected_block}",
    )

    final_name = f"{C.KEY_FINAL_ADALN}.weight"
    final = header.get(final_name)
    expected_final_dtype = planned_dtype.get(final_name)
    expected_final = (geometry.final_adaln_width, C.ADALN_CURVE_RANK)
    report.add(
        "pruning.final_adaln_reduced",
        final is not None and final.dtype == expected_final_dtype
        and final.dtype in float_dtypes and final.shape == expected_final,
        (f"{final.dtype} {final.shape}" if final else "missing")
        + f" (expected {expected_final_dtype} {expected_final})",
    )

    # Precision islands.
    island_problems = []
    for prefix in C.FP32_PRESERVED_PREFIXES:
        for key in (prefix, f"{prefix}.weight", f"{prefix}.bias"):
            info = header.get(key)
            if info is not None and info.dtype != "F32":
                island_problems.append(f"{key} is {info.dtype}")
    report.add(
        "policy.fp32_islands_preserved",
        not island_problems,
        "; ".join(island_problems) if island_problems else "patch projections and output heads are F32",
    )


def _check_metadata(report: ValidationReport, header: Header, plan: OutputPlan,
                    output_format: str) -> None:
    raw = header.metadata.get(C.QUANT_METADATA_KEY)
    if raw is None:
        report.add("metadata.present", False, f"{C.QUANT_METADATA_KEY} is missing")
        return
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        report.add("metadata.present", False, f"{C.QUANT_METADATA_KEY} is not valid JSON: {exc}")
        return
    report.add("metadata.present", True, f"{len(raw):,} bytes of quantization metadata")

    layers = parsed.get("layers") if isinstance(parsed, dict) else None
    if not isinstance(layers, dict):
        report.add("metadata.layers", False, "no 'layers' object")
        return

    expected = {t.layer for t in plan.quant_targets}
    actual = set(layers)
    report.add(
        "metadata.layers",
        actual == expected,
        f"{len(actual)} entries, expected {len(expected)}"
        + (f"; first missing {sorted(expected - actual)[0]}" if expected - actual else "")
        + (f"; unexpected {sorted(actual - expected)[0]}" if actual - expected else ""),
    )

    report.add(
        "metadata.format_version",
        str(parsed.get("format_version")) == C.QUANT_METADATA_FORMAT_VERSION,
        f"format_version={parsed.get('format_version')!r}, "
        f"expected {C.QUANT_METADATA_FORMAT_VERSION!r}",
    )

    wanted_format = C.QUANT_FORMAT_W4A8 if output_format == C.FORMAT_W4A8 else C.QUANT_FORMAT_NVFP4
    bad_format = [name for name, conf in layers.items()
                  if not isinstance(conf, dict) or conf.get("format") != wanted_format]
    report.add(
        "metadata.layer_format",
        not bad_format,
        f"{len(bad_format)} layers are not {wanted_format}, first: {bad_format[0]}" if bad_format
        else f"all {len(layers)} layers are {wanted_format}",
    )

    if output_format == C.FORMAT_W4A8:
        bad_group = [name for name, conf in layers.items()
                     if int(conf.get("group_size", -1)) != C.W4A8_GROUP_SIZE]
        report.add(
            "metadata.w4a8_group_size",
            not bad_group,
            f"{len(bad_group)} layers do not declare group_size={C.W4A8_GROUP_SIZE}" if bad_group
            else f"group_size={C.W4A8_GROUP_SIZE} on all layers",
        )
        bad_rot = [name for name, conf in layers.items()
                   if int(conf.get("convrot_groupsize", -1)) != C.W4A8_CONVROT_GROUPSIZE]
        report.add(
            "metadata.w4a8_convrot_groupsize",
            not bad_rot,
            f"{len(bad_rot)} layers do not declare convrot_groupsize={C.W4A8_CONVROT_GROUPSIZE}"
            if bad_rot else f"convrot_groupsize={C.W4A8_CONVROT_GROUPSIZE} on all layers",
        )
        not_rotated = [name for name, conf in layers.items() if conf.get("convrot") is not True]
        report.add(
            "metadata.w4a8_convrot_true",
            not not_rotated,
            f"{len(not_rotated)} layers do not declare convrot=true" if not_rotated
            else f"convrot=true on all {len(layers)} layers",
        )
    else:
        bad_group = [name for name, conf in layers.items()
                     if int(conf.get("group_size", -1)) != C.NVFP4_GROUP_SIZE]
        report.add(
            "metadata.nvfp4_group_size",
            not bad_group,
            f"{len(bad_group)} layers do not declare group_size={C.NVFP4_GROUP_SIZE}" if bad_group
            else f"group_size={C.NVFP4_GROUP_SIZE} on all layers",
        )


def _check_quantized_layers(report: ValidationReport, header: Header, plan: OutputPlan,
                            output_format: str) -> None:
    problems: list[str] = []
    scale_count_problems: list[str] = []

    for target in plan.quant_targets:
        n, k = target.logical_shape
        for suffix, (dtype, shape) in target.outputs.items():
            key = target.output_key(suffix)
            info = header.get(key)
            if info is None:
                problems.append(f"{key} missing")
                continue
            if info.dtype != dtype:
                problems.append(f"{key} is {info.dtype}, expected {dtype}")
            if info.shape != shape:
                problems.append(f"{key} is {info.shape}, expected {shape}")

        if output_format == C.FORMAT_W4A8:
            s_rel = header.get(target.output_key(C.W4A8_SUFFIX_S_REL))
            if s_rel is not None:
                expected_groups = (n * k) // C.W4A8_GROUP_SIZE
                if s_rel.numel != expected_groups:
                    scale_count_problems.append(
                        f"{target.layer}: {s_rel.numel} scales for {n * k} weights, "
                        f"expected {expected_groups}"
                    )
            packed = header.get(target.output_key(C.W4A8_SUFFIX_WEIGHT))
            if packed is not None and packed.shape[1] * 2 != k:
                problems.append(
                    f"{target.layer}: packed width {packed.shape[1]} does not recover K={k}"
                )
        else:
            block_scale = header.get(target.output_key(C.NVFP4_SUFFIX_SCALE))
            if block_scale is not None and block_scale.numel < (n * k) // C.NVFP4_GROUP_SIZE:
                scale_count_problems.append(
                    f"{target.layer}: {block_scale.numel} block scales cannot cover {n * k} weights"
                )

    report.add(
        "layers.tensor_group_complete",
        not problems,
        f"{len(problems)} problems, first: {problems[0]}" if problems
        else f"{len(plan.quant_targets)} layers have a complete tensor group",
    )
    report.add(
        "layers.scale_count_matches_group_size",
        not scale_count_problems,
        f"{len(scale_count_problems)} problems, first: {scale_count_problems[0]}"
        if scale_count_problems else "one scale per group on every layer",
    )
    report.add(
        "layers.count",
        len(plan.quant_targets) == plan.geometry.num_layers * len(C.W4A8_BLOCK_LINEARS),
        f"{len(plan.quant_targets)} quantized layers for {plan.geometry.num_layers} blocks",
    )


def _check_scale_health(report: ValidationReport, path: Path, plan: OutputPlan,
                        output_format: str) -> None:
    """Every quantized layer's scale data must be finite and positive.

    This covers all layers, not a sample: a single NaN scale silently destroys
    one layer's output, and reading back only the scale tensors costs a small
    fraction of the file (about 1.2 GB of a 12.5 GB reference checkpoint) while
    a full dequantization pass would re-materialise the whole model.
    """
    import torch

    from safetensors import safe_open

    if output_format == C.FORMAT_W4A8:
        suffixes = (C.W4A8_SUFFIX_S_REL, C.W4A8_SUFFIX_S_CHANNEL, C.W4A8_SUFFIX_CODEBOOK)
        positive = (C.W4A8_SUFFIX_S_CHANNEL,)
    else:
        suffixes = (C.NVFP4_SUFFIX_SCALE, C.NVFP4_SUFFIX_SCALE_2)
        positive = (C.NVFP4_SUFFIX_SCALE_2,)

    problems: list[str] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for target in plan.quant_targets:
                for suffix in suffixes:
                    key = target.output_key(suffix)
                    try:
                        values = handle.get_tensor(key).to(torch.float32)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{key}: unreadable ({type(exc).__name__})")
                        continue
                    if not bool(torch.isfinite(values).all()):
                        problems.append(f"{key}: contains NaN or Inf")
                    elif suffix in positive and not bool((values > 0).all()):
                        problems.append(f"{key}: contains a non-positive scale")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not open output to read scales: {type(exc).__name__}: {exc}")

    report.add(
        "layers.scales_finite",
        not problems,
        f"{len(problems)} problems, first: {problems[0]}" if problems
        else f"scale data on all {len(plan.quant_targets)} layers is finite",
    )


def _check_dequantization(report: ValidationReport, path: Path, plan: OutputPlan,
                          output_format: str, sample_layers: int) -> None:
    """Decode a sample of stored layers to prove the payload is usable."""
    import torch

    from safetensors import safe_open

    if output_format == C.FORMAT_W4A8:
        from h3converter.quant.w4a8 import dequantize_layer
        suffixes = (C.W4A8_SUFFIX_WEIGHT, C.W4A8_SUFFIX_S_REL,
                    C.W4A8_SUFFIX_S_CHANNEL, C.W4A8_SUFFIX_CODEBOOK)
    else:
        from h3converter.quant.nvfp4 import dequantize_layer
        suffixes = (C.NVFP4_SUFFIX_WEIGHT, C.NVFP4_SUFFIX_SCALE, C.NVFP4_SUFFIX_SCALE_2)

    count = min(sample_layers, len(plan.quant_targets))
    if count == 0:
        return
    step = max(1, len(plan.quant_targets) // count)
    sampled = plan.quant_targets[::step][:count]

    problems: list[str] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for target in sampled:
                tensors = {}
                for suffix in suffixes:
                    key = target.output_key(suffix)
                    try:
                        tensors[suffix] = handle.get_tensor(key)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{key}: {type(exc).__name__}")
                        break
                else:
                    try:
                        restored = dequantize_layer(tensors, target.logical_shape)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{target.layer}: dequantize failed ({type(exc).__name__}: {exc})")
                        continue
                    if tuple(restored.shape) != target.logical_shape:
                        problems.append(
                            f"{target.layer}: decoded shape {tuple(restored.shape)} != "
                            f"{target.logical_shape}"
                        )
                    elif not bool(torch.isfinite(restored).all()):
                        problems.append(f"{target.layer}: decoded weight contains non-finite values")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not open output for sampling: {type(exc).__name__}: {exc}")

    report.add(
        "layers.sample_dequantizes",
        not problems,
        "; ".join(problems[:3]) if problems
        else f"{len(sampled)} sampled layers decode to finite weights of the original shape",
    )


def _conformance(header: Header, plan: OutputPlan, output_format: str) -> dict[str, object]:
    """Compare the produced file against the golden reference class."""
    counts = header.dtype_counts()
    reference = C.REFERENCE_INVENTORY
    band = C.OPTION_A_SIZE_BAND_BYTES if output_format == C.FORMAT_W4A8 else C.OPTION_B_SIZE_BAND_BYTES

    matches_reference_geometry = (
        plan.geometry.num_layers == C.H3_REFERENCE_GEOMETRY["num_layers"]
        and plan.geometry.hidden_size == C.H3_REFERENCE_GEOMETRY["hidden_size"]
        and plan.geometry.token_refiner_num_layers
        == C.H3_REFERENCE_GEOMETRY["token_refiner_num_layers"]
    )

    result: dict[str, object] = {
        "output_format": output_format,
        "file_bytes": header.file_size,
        "size_band_bytes": list(band),
        "within_size_band": band[0] <= header.file_size <= band[1],
        "tensor_count": len(header.tensors),
        "dtype_counts": counts,
        "quantized_layer_count": len(plan.quant_targets),
        "compared_to_reference": matches_reference_geometry and output_format == C.FORMAT_W4A8,
    }

    if result["compared_to_reference"]:
        result["reference"] = {
            "file_bytes": reference["file_bytes"],
            "tensor_count": reference["tensor_count"],
            "dtype_counts": reference["dtype_counts"],
        }
        result["tensor_count_matches_reference"] = len(header.tensors) == reference["tensor_count"]
        result["dtype_counts_match_reference"] = counts == reference["dtype_counts"]
        result["size_delta_bytes"] = header.file_size - reference["file_bytes"]
    return result
