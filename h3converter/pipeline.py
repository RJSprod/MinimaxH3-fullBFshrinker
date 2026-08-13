"""End-to-end conversion orchestration.

Ordering here is a data-safety property, not a style choice:

1. inspect and validate the source (read-only),
2. plan the entire output inventory,
3. build and *numerically validate* the AdaLN curve form,
4. only then quantize and stream into ``<final>.partial``,
5. re-read and validate the written file,
6. atomically rename to the final name and write the report.

Any failure before step 6 leaves the source untouched and no final filename in
existence. The source file is opened read-only and never written to.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from h3converter import constants as C
from h3converter import calibration as calib
from h3converter.adaln_curve import CurveBasis, TimeEmbedder, build_curve_basis, table_for_storage
from h3converter.adaln_prune import (
    PruneReport,
    check_report,
    collapse_projection,
    curve_reconstruction_error,
    probe_timesteps,
    validate_projection,
)
from h3converter.h3_detect import Detection, detect
from h3converter.h3_policy import OutputPlan, build_output_plan
from h3converter.hardware import DiskCheck, PeakUsage, check_disk, system_info
from h3converter.logging_setup import active_log_path, get_logger
from h3converter.paths import derive_output_path, partial_path, report_path, unique_output_path
from h3converter.progress import ProgressCallback, ProgressTracker
from h3converter.quant import capability
from h3converter.reports import (
    ConversionReport,
    build_output_metadata,
    fingerprint,
    sha256_file,
    summarise_layer_stats,
)
from h3converter.safetensor_io import (
    PlannedWriter,
    SourceReader,
    finalize,
    read_header,
)
from h3converter.validate import validate_output

log = get_logger("h3converter.pipeline")

CancelCheck = Callable[[], bool]


class ConversionError(RuntimeError):
    """A conversion could not be completed. The source is untouched."""


class Cancelled(RuntimeError):
    """The user stopped the conversion."""


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------

@dataclass
class ConversionRequest:
    source: Path
    output_format: str
    output_path: Path | None = None
    device: str | None = None
    overwrite: bool = False
    # Measure real quantization error on every Nth layer. Measuring every layer
    # would double the quantization work for little extra information.
    measure_every: int = 25
    full_sha256: bool = False
    validate_samples: int = 4
    # Developer-mode: fold the curve mean into the AdaLN biases. See
    # h3converter.adaln_curve for why this is off by default.
    center_basis: bool = False


@dataclass
class SourceAnalysis:
    """Everything screen 2 needs, gathered without reading tensor data."""

    path: Path
    size_bytes: int
    detection: Detection
    plan_preview: dict[str, OutputPlan] = field(default_factory=dict)
    disk: dict[str, DiskCheck] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.detection and self.detection.convertible) and not self.errors

    def estimated_output_bytes(self, output_format: str) -> int | None:
        plan = self.plan_preview.get(output_format)
        return plan.total_bytes if plan else None


@dataclass
class ConversionOutcome:
    ok: bool = False
    cancelled: bool = False
    output_path: Path | None = None
    report_path: Path | None = None
    report: ConversionReport = field(default_factory=ConversionReport)
    error: str | None = None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def analyze_source(source: Path, formats: tuple[str, ...] = (C.FORMAT_W4A8, C.FORMAT_NVFP4)) -> SourceAnalysis:
    """Header-only inspection: identify, validate and size up the source."""
    source = Path(source)
    analysis = SourceAnalysis(path=source, size_bytes=0, detection=Detection())

    try:
        header = read_header(source)
    except Exception as exc:  # noqa: BLE001
        analysis.errors.append(str(exc))
        return analysis

    analysis.size_bytes = header.file_size
    analysis.detection = detect(header)
    if not analysis.detection.convertible or analysis.detection.geometry is None:
        return analysis

    for output_format in formats:
        try:
            plan = build_output_plan(header, analysis.detection.geometry, output_format)
        except Exception as exc:  # noqa: BLE001
            analysis.errors.append(f"{C.FORMAT_LABELS[output_format]}: {exc}")
            continue
        analysis.plan_preview[output_format] = plan
        target = derive_output_path(source, output_format)
        analysis.disk[output_format] = check_disk(target, output_format, plan.total_bytes)

    return analysis


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    request: ConversionRequest,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> ConversionOutcome:
    """Run one full conversion. Never raises for expected failures."""
    outcome = ConversionOutcome()
    report = outcome.report
    tracker = ProgressTracker.for_format(request.output_format, progress_callback)
    usage = PeakUsage()
    usage.reset_cuda_peak()

    writer: PlannedWriter | None = None
    reader: SourceReader | None = None
    partial: Path | None = None

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise Cancelled()

    try:
        report.application = {
            "name": C.APP_NAME,
            "version": C.APP_VERSION,
            "converter_id": C.CONVERTER_ID,
            "log_path": str(active_log_path()) if active_log_path() else None,
        }

        # ---- 1. inspect -------------------------------------------------
        tracker.start_phase("inspect", 4, "Reading checkpoint header")
        source = Path(request.source).resolve()
        if not source.is_file():
            raise ConversionError(f"source file not found: {source}")

        header = read_header(source)
        tracker.advance(1, "Identifying architecture")

        detection = detect(header)
        if not detection.is_h3:
            raise ConversionError(
                "This is not a MiniMax H3 diffusion checkpoint. "
                + "; ".join(detection.errors)
            )
        if not detection.convertible or detection.geometry is None:
            raise ConversionError(
                "This H3 checkpoint cannot be converted: " + "; ".join(detection.errors)
            )
        geometry = detection.geometry
        log.info("Detected %s", detection.summary)
        for warning in detection.warnings:
            log.warning("source: %s", warning)
            report.warnings.append(warning)

        report.architecture = {
            "model": "minimax_h3",
            "detected": geometry.as_dict(),
            "reference": {k: v for k, v in C.H3_REFERENCE_GEOMETRY.items() if k != "patch_size"},
            "source_float_dtype": detection.float_dtype,
            "source_dtype_counts": header.dtype_counts(),
        }
        tracker.advance(1, "Planning output")

        # ---- 2. plan ----------------------------------------------------
        plan = build_output_plan(header, geometry, request.output_format)
        log.info(
            "Output plan: %d tensors, %d quantized layers, %.2f GB of tensor data",
            len(plan.tensors), plan.quantized_layer_count, plan.total_bytes / 1000**3,
        )

        final_path = Path(request.output_path) if request.output_path else derive_output_path(
            source, request.output_format
        )
        if final_path.resolve() == source:
            raise ConversionError("output path would overwrite the source file")
        if final_path.exists() and not request.overwrite:
            final_path = unique_output_path(final_path)
            log.info("Output already exists; writing %s instead", final_path.name)
        partial = partial_path(final_path)

        disk = check_disk(final_path, request.output_format, plan.total_bytes)
        if not disk.ok:
            raise ConversionError("Not enough free disk space. " + disk.message)
        if disk.message:
            report.warnings.append(disk.message)
        tracker.advance(1, "Checking dependencies")

        # ---- 3. capability ----------------------------------------------
        device = request.device or capability.select_device()
        caps = capability.probe(device=device)
        if not caps.ok_for(request.output_format):
            raise ConversionError(
                f"{C.FORMAT_LABELS[request.output_format]} self-test failed: "
                f"{caps.blocking_reason(request.output_format)}"
            )
        log.info("Quantizing on %s with comfy-kitchen %s", caps.device, caps.kitchen_version)

        environment = system_info()
        report.environment = {
            "system": environment.as_dict(),
            "capability_checks": caps.as_dict(),
            "device": caps.device,
        }

        report.source = {
            "path": str(source),
            "filename": source.name,
            "size_bytes": header.file_size,
            "tensor_count": len(header.tensors),
            "fingerprint": fingerprint(source),
            "sha256": None,
            "metadata_keys": sorted(header.metadata.keys()),
        }
        tracker.advance(1, "Source validated")
        check_cancel()

        # ---- 4. calibration (Option B only) -----------------------------
        calibration_plan = None
        if request.output_format == C.FORMAT_NVFP4:
            tracker.start_phase("calibrate", 1, "Resolving activation scales")
            calibration_plan = calib.resolve([t.layer for t in plan.quant_targets])
            log.info("Calibration: %s", calibration_plan.describe())
            report.calibration = calibration_plan.as_dict()
            for note in calibration_plan.notes:
                log.info("calibration: %s", note)
            tracker.advance(1, calibration_plan.describe())
        else:
            report.calibration = {"strategy": "not_applicable", "description": "W4A8 does not quantize activations"}

        # ---- 5. AdaLN basis ---------------------------------------------
        reader = SourceReader(source)
        tracker.start_phase("adaln_basis", 2, "Sampling the timestep curve")
        embedder = TimeEmbedder.from_tensors(
            {key: reader.get(key) for key in plan.dropped_keys}
        )
        tracker.advance(1, f"Fitting rank-{C.ADALN_CURVE_RANK} basis over {C.ADALN_CURVE_GRID} points")

        basis = build_curve_basis(
            embedder,
            grid=C.ADALN_CURVE_GRID,
            rank=C.ADALN_CURVE_RANK,
            center=request.center_basis,
        )
        prune_report = PruneReport(basis_stats=basis.stats())
        prune_report.curve_rel_l2 = curve_reconstruction_error(basis, embedder)
        log.info(
            "Curve basis: %.4e relative error on the interpolated curve, %.6f energy retained",
            prune_report.curve_rel_l2, basis.stats()["energy_retained"],
        )
        tracker.advance(1, f"Curve error {prune_report.curve_rel_l2:.2e}")
        usage.sample()
        check_cancel()

        # ---- 6. open the output -----------------------------------------
        layer_configs = _layer_configs(plan, request.output_format)
        metadata = build_output_metadata(
            source_metadata=header.metadata,
            source_path=source,
            output_format=request.output_format,
            plan=plan,
            layer_configs=layer_configs,
            extra={
                "adaln_basis_centered": str(basis.centered).lower(),
                "calibration": calibration_plan.strategy if calibration_plan else "not_applicable",
            },
        )
        writer = PlannedWriter(partial, plan.tensors, metadata)
        writer.open()
        log.info("Writing %s (%.2f GB planned)", partial.name, writer.data_bytes / 1000**3)

        writer.write(C.ADALN_TABLE_KEY, table_for_storage(basis))

        # ---- 7. collapse AdaLN projections ------------------------------
        probes = probe_timesteps(basis.grid)
        tracker.start_phase("adaln_collapse", len(plan.adaln_targets), "Collapsing AdaLN projections")
        for index, target in enumerate(plan.adaln_targets):
            check_cancel()
            weight = reader.get(target.weight_key)
            bias = reader.get(target.bias_key) if target.bias_key else None

            reduced = collapse_projection(basis, weight, bias, weight_dtype=torch.bfloat16)
            prune_report.projections.append(
                validate_projection(
                    target.prefix, basis, embedder, weight, bias, reduced, t=probes
                )
            )

            writer.write(target.weight_key, reduced.weight)
            if target.bias_key is not None and reduced.bias is not None:
                writer.write(target.bias_key, reduced.bias)

            del weight, bias, reduced
            usage.sample()
            tracker.advance(1, f"{target.prefix} ({index + 1}/{len(plan.adaln_targets)})")

        # Numerical gate: refuse to go further if the curve form is not faithful.
        worst = prune_report.worst
        if worst is not None:
            log.info("Worst AdaLN reconstruction: %s at %.4e relative L2", worst.layer, worst.rel_l2)
            if worst.rel_l2 > C.ADALN_REL_ERROR_WARN:
                prune_report.warnings.append(
                    f"AdaLN reconstruction error {worst.rel_l2:.3%} on {worst.layer} is above the "
                    f"{C.ADALN_REL_ERROR_WARN:.2%} advisory threshold"
                )
        check_report(prune_report)
        report.pruning = prune_report.as_dict()
        report.warnings.extend(prune_report.warnings)
        gc.collect()
        check_cancel()

        # ---- 8. quantize and copy through -------------------------------
        quant_bytes = sum(header.tensors[t.source_key].nbytes for t in plan.quant_targets)
        passthrough_bytes = sum(header.tensors[k].nbytes for k in plan.passthrough_keys)
        tracker.start_phase("quantize", quant_bytes + passthrough_bytes, "Quantizing transformer blocks")

        for key in plan.passthrough_keys:
            check_cancel()
            writer.write(key, reader.get(key))
            tracker.advance(header.tensors[key].nbytes, f"copying {key}")

        layer_stats: list[dict[str, float]] = []
        quantize_layer = _quantizer(request.output_format)
        for index, target in enumerate(plan.quant_targets):
            check_cancel()
            weight = reader.get(target.source_key)
            measure = request.measure_every > 0 and (
                index % request.measure_every == 0 or index == len(plan.quant_targets) - 1
            )
            kwargs = {}
            if request.output_format == C.FORMAT_NVFP4 and calibration_plan is not None:
                scale = calibration_plan.input_scale_for(target.layer)
                if scale is not None:
                    kwargs["input_scale"] = scale

            quantized = quantize_layer(
                target.layer, weight, device=caps.device, measure=measure, **kwargs
            )
            for suffix, tensor in quantized.tensors.items():
                writer.write(target.output_key(suffix), tensor)
            if quantized.stats:
                layer_stats.append({"layer": target.layer, **quantized.stats})

            processed = header.tensors[target.source_key].nbytes
            del weight, quantized
            usage.sample()
            tracker.bytes_written = writer.bytes_written
            tracker.peak_ram_bytes = usage.peak_ram_bytes
            tracker.peak_vram_bytes = usage.peak_vram_bytes
            tracker.advance(processed, f"{target.layer} ({index + 1}/{len(plan.quant_targets)})")

        writer.close()
        writer = None
        reader.close()
        reader = None
        gc.collect()
        if caps.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # ---- 9. validate and publish ------------------------------------
        tracker.start_phase("finalize", 3, "Validating output")
        validation = validate_output(
            partial, plan, geometry, request.output_format,
            sample_layers=request.validate_samples,
        )
        report.validation = validation.as_dict()
        if not validation.ok:
            raise ConversionError("Output validation failed: " + validation.summary())
        log.info("Validation %s", validation.summary())
        tracker.advance(1, "Validated")

        if request.full_sha256:
            report.source["sha256"] = sha256_file(source)
            tracker.advance(1, "Source hashed")
        else:
            tracker.advance(1, "Publishing")

        finalize(partial, final_path)
        partial = None  # published; the cleanup path must not remove it
        outcome.output_path = final_path

        output_size = final_path.stat().st_size
        report.output = {
            "path": str(final_path),
            "filename": final_path.name,
            "size_bytes": output_size,
            "tensor_count": len(plan.tensors),
            "compression_ratio": header.file_size / output_size if output_size else 0.0,
            "format": request.output_format,
            "format_label": C.FORMAT_LABELS[request.output_format],
        }
        report.quantization = {
            "policy": (
                C.W4A8_POLICY_VERSION if request.output_format == C.FORMAT_W4A8
                else C.NVFP4_POLICY_VERSION
            ),
            "format": (
                C.QUANT_FORMAT_W4A8 if request.output_format == C.FORMAT_W4A8
                else C.QUANT_FORMAT_NVFP4
            ),
            "quantized_layers": plan.quantized_layer_count,
            "preserved_tensors": len(plan.passthrough_keys) + 2 * len(plan.adaln_targets),
            "dropped_tensors": len(plan.dropped_keys),
            "group_size": (
                C.W4A8_GROUP_SIZE if request.output_format == C.FORMAT_W4A8 else C.NVFP4_GROUP_SIZE
            ),
            "convrot_groupsize": (
                C.W4A8_CONVROT_GROUPSIZE if request.output_format == C.FORMAT_W4A8 else None
            ),
            "layer_error": summarise_layer_stats(layer_stats),
            "per_layer_samples": layer_stats,
            "dtype_counts": plan.dtype_counts(),
            "dtype_elements": plan.dtype_elements(),
        }
        usage.sample()
        report.resources = {
            "peak_ram_bytes": usage.peak_ram_bytes,
            "peak_vram_bytes": usage.peak_vram_bytes,
            "peak_ram_target_bytes": C.TARGET_PEAK_RAM_BYTES,
            "peak_vram_target_bytes": C.TARGET_PEAK_VRAM_BYTES,
            "elapsed_seconds": tracker.elapsed,
        }

        outcome.report_path = report.write(report_path(final_path))
        tracker.finish(f"Wrote {final_path.name}")
        outcome.ok = True
        log.info(
            "Conversion complete: %s (%.2f GB, %.2fx smaller) in %.1f min",
            final_path.name, output_size / 1000**3,
            header.file_size / output_size if output_size else 0.0,
            tracker.elapsed / 60,
        )
        return outcome

    except Cancelled:
        outcome.cancelled = True
        outcome.error = "Conversion cancelled."
        log.warning("Conversion cancelled by the user")
        return outcome
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        outcome.error = str(exc) or f"{type(exc).__name__}"
        report.errors.append(outcome.error)
        log.exception("Conversion failed: %s", outcome.error)
        return outcome
    finally:
        if writer is not None:
            writer.abort()
        if reader is not None:
            reader.close()
        if partial is not None and partial.exists() and not outcome.ok:
            try:
                partial.unlink()
            except OSError:
                log.warning("Could not remove partial file %s", partial)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _quantizer(output_format: str):
    if output_format == C.FORMAT_W4A8:
        from h3converter.quant.w4a8 import quantize_layer
        return quantize_layer
    from h3converter.quant.nvfp4 import quantize_layer
    return quantize_layer


def _layer_configs(plan: OutputPlan, output_format: str) -> dict[str, dict]:
    if output_format == C.FORMAT_W4A8:
        from h3converter.quant.w4a8 import layer_config
    else:
        from h3converter.quant.nvfp4 import layer_config
    config = layer_config()
    return {target.layer: dict(config) for target in plan.quant_targets}
