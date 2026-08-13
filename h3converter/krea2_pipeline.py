"""Streaming, fail-safe Krea 2 scaled-FP8 conversion orchestration."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from h3converter import constants as C
from h3converter.hardware import check_disk
from h3converter.krea2_detect import detect
from h3converter.krea2_policy import KREA2_FP8_POLICY_VERSION, build_output_plan
from h3converter.krea2_validate import validate_output
from h3converter.paths import derive_output_path, partial_path, report_path, unique_output_path
from h3converter.quant import capability
from h3converter.quant.fp8 import layer_config, quantize
from h3converter.reports import ConversionReport
from h3converter.safetensor_io import PlannedWriter, SourceReader, finalize, read_header

def metadata_for(header, source, plan):
    layers = {t.layer: layer_config() for t in plan.quant_targets}
    return {
        C.QUANT_METADATA_KEY: json.dumps({"format_version": C.QUANT_METADATA_FORMAT_VERSION, "layers": layers}, separators=(",", ":"), sort_keys=True),
        "converted_by": f"{C.APP_NAME} {C.APP_VERSION}", "source_architecture": "krea2",
        "output_format": C.FORMAT_KREA2_FP8, "quantization_policy": KREA2_FP8_POLICY_VERSION,
        "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

def convert(request, progress_callback=None, should_cancel=None):
    from h3converter.pipeline import Cancelled, ConversionError, ConversionOutcome
    from h3converter.progress import ProgressTracker
    outcome, writer, reader, partial = ConversionOutcome(), None, None, None
    tracker = ProgressTracker.for_format(C.FORMAT_KREA2_FP8, progress_callback)
    try:
        tracker.start_phase("inspect", 3, "Reading Krea 2 header")
        source = Path(request.source).resolve(); header = read_header(source); detection = detect(header)
        if not detection.convertible: raise ConversionError("Krea 2 source cannot be converted: " + "; ".join(detection.errors))
        tracker.advance(1, "Building exact FP8 inventory")
        plan = build_output_plan(header, detection.prefix)
        final = Path(request.output_path) if request.output_path else derive_output_path(source, C.FORMAT_KREA2_FP8)
        if final.resolve() == source: raise ConversionError("output path would overwrite the source file")
        if final.exists(): final = unique_output_path(final) if not request.overwrite else (_ for _ in ()).throw(ConversionError("refusing to overwrite an existing output"))
        partial = partial_path(final)
        disk = check_disk(final, C.FORMAT_KREA2_FP8, plan.total_bytes)
        if not disk.ok: raise ConversionError("Not enough free disk space. " + disk.message)
        caps = capability.probe(device=request.device or capability.select_device())
        if not caps.ok_for(C.FORMAT_KREA2_FP8): raise ConversionError("FP8 self-test failed: " + str(caps.blocking_reason(C.FORMAT_KREA2_FP8)))
        tracker.advance(2, "Source and capability validated")
        writer = PlannedWriter(partial, plan.tensors, metadata_for(header, source, plan)); writer.open()
        reader = SourceReader(source, mmap=request.mmap_source, header=header)
        total = sum(i.nbytes for i in header.tensors.values()); tracker.start_phase("quantize", total, "Streaming tensors")
        targets = {t.source_key: t for t in plan.quant_targets}
        for key, info in header.tensors.items():
            if should_cancel and should_cancel(): raise Cancelled()
            if key in targets:
                target = targets[key]; q = quantize(reader.get(key), caps.device)
                writer.write(key, q.weight); writer.write(target.scale_key, q.weight_scale)
            else: writer.write(key, reader.get(key))
            tracker.bytes_written = writer.bytes_written; tracker.advance(info.nbytes, key)
        writer.close(); writer = None; reader.close(); reader = None
        tracker.start_phase("finalize", 2, "Re-opening and validating partial output")
        validation = validate_output(partial, header, plan)
        if not validation.ok: raise ConversionError("Output validation failed: " + validation.summary())
        tracker.advance(1, "Validated")
        finalize(partial, final); partial = None
        outcome.ok = True; outcome.output_path = final
        report = outcome.report
        report.architecture = {"model": "krea2", "detected": detection.geometry.as_dict()}
        report.source = {"path": str(source), "size_bytes": header.file_size, "dtype_counts": header.dtype_counts()}
        report.output = {"path": str(final), "size_bytes": final.stat().st_size, "format": C.FORMAT_KREA2_FP8}
        report.quantization = {
            "policy": KREA2_FP8_POLICY_VERSION,
            "quantized_layers": plan.quantized_layer_count,
            "source_bytes_selected_for_fp8": plan.source_quantized_bytes,
            "source_fraction_selected_for_fp8": plan.source_quantized_bytes / header.file_size,
            "dtype_counts": plan.dtype_counts(),
        }
        report.validation = validation.as_dict(); outcome.report_path = report.write(report_path(final)); tracker.finish(final.name)
        return outcome
    except Cancelled: outcome.cancelled = True; outcome.error = "Conversion cancelled."; return outcome
    except Exception as exc: outcome.error = str(exc); outcome.report.errors.append(str(exc)); return outcome
    finally:
        if writer: writer.abort()
        if reader: reader.close()
        if partial and partial.exists(): partial.unlink(missing_ok=True)
