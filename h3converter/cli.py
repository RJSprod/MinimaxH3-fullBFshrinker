"""Headless command line interface.

The GUI is the supported user path; this exists for automation, remote
machines, and reproducing a conversion from a script. It drives exactly the
same pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from h3converter import constants as C
from h3converter.logging_setup import start_run_log
from h3converter.pipeline import ConversionRequest, analyze_source, convert
from h3converter.progress import ProgressEvent

_FORMAT_ALIASES = {
    "a": C.FORMAT_W4A8,
    "w4a8": C.FORMAT_W4A8,
    C.FORMAT_W4A8: C.FORMAT_W4A8,
    "b": C.FORMAT_NVFP4,
    "nvfp4": C.FORMAT_NVFP4,
}


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:,.2f} {unit}" if unit != "B" else f"{size:,.0f} B"
        size /= 1000
    return f"{size:.2f} TB"


def _print_analysis(analysis) -> None:
    detection = analysis.detection
    print(f"Source:   {analysis.path}")
    print(f"Size:     {_human(analysis.size_bytes)}")
    print(f"Detected: {detection.summary}")
    if detection.geometry:
        geometry = detection.geometry
        print(f"          {geometry.num_layers} blocks, hidden {geometry.hidden_size}, "
              f"{geometry.num_attention_heads}x{geometry.attention_head_dim} heads, "
              f"ffn {geometry.ffn_hidden_size}, time embed {geometry.time_embed_dim}")
        print(f"          token refiner {geometry.token_refiner_num_layers} blocks, "
              f"text dim {geometry.text_dim}")
    print(f"Pruned:    {'yes' if detection.already_curve_pruned else 'no'}")
    print(f"Quantized: {'yes' if detection.already_quantized else 'no'}")

    for warning in detection.warnings:
        print(f"  warning: {warning}")
    for error in detection.errors:
        print(f"  error:   {error}")
    for error in analysis.errors:
        print(f"  error:   {error}")

    for output_format, plan in analysis.plan_preview.items():
        disk = analysis.disk.get(output_format)
        print()
        print(f"  {C.FORMAT_LABELS[output_format]}:")
        print(f"    {len(plan.tensors):,} tensors, {plan.quantized_layer_count} quantized layers")
        print(f"    estimated output {_human(plan.total_bytes)} "
              f"({analysis.size_bytes / plan.total_bytes:.2f}x smaller)")
        print(f"    dtypes {plan.dtype_counts()}")
        if disk:
            state = "ok" if disk.ok else "INSUFFICIENT"
            print(f"    disk {state}: {_human(disk.free_bytes)} free, "
                  f"{_human(disk.required_bytes)} required")


def _progress_printer():
    state = {"last": -1.0, "phase": ""}

    def report(event: ProgressEvent) -> None:
        percent = event.overall * 100
        changed_phase = event.phase != state["phase"]
        if changed_phase:
            state["phase"] = event.phase
            print()
        if changed_phase or percent - state["last"] >= 0.5:
            state["last"] = percent
            detail = event.detail[:64]
            sys.stdout.write(f"\r[{percent:5.1f}%] {event.phase_label:<34} {detail:<64}")
            sys.stdout.flush()

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="h3convert",
        description=f"{C.APP_NAME} {C.APP_VERSION} - convert a full BF16 MiniMax H3 checkpoint",
    )
    parser.add_argument("source", nargs="?", type=Path, help="source .safetensors checkpoint")
    parser.add_argument(
        "-f", "--format", dest="output_format", choices=sorted(_FORMAT_ALIASES),
        help="a/w4a8 for AdaLN-pruned W4A8 ConvRot, b/nvfp4 for AdaLN-pruned NVFP4",
    )
    parser.add_argument("-o", "--output", type=Path, help="output path (default: beside the source)")
    parser.add_argument("--inspect", action="store_true", help="analyse the source and exit")
    parser.add_argument("--check", action="store_true", help="run environment self-tests and exit")
    parser.add_argument("--device", help="torch device for quantization (default: cuda if present)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--sha256", action="store_true", help="also compute the source SHA-256")
    parser.add_argument(
        "--measure-every", type=int, default=25, metavar="N",
        help="measure quantization error on every Nth layer (0 disables)",
    )
    parser.add_argument("--json", action="store_true", help="print the conversion report as JSON")
    parser.add_argument(
        "--center-basis", action="store_true",
        help="developer mode: fold the AdaLN curve mean into the layer biases",
    )
    args = parser.parse_args(argv)

    if args.check:
        from h3converter.bootstrap_checks import run

        return run(verbose=True)

    if args.source is None:
        parser.error("a source checkpoint is required (or use --check)")

    start_run_log("conversion" if not args.inspect else "inspect")

    if args.inspect:
        _print_analysis(analyze_source(args.source))
        return 0

    if not args.output_format:
        parser.error("--format is required (a = W4A8 ConvRot, b = NVFP4)")

    analysis = analyze_source(args.source)
    _print_analysis(analysis)
    if not analysis.ok:
        print("\nSource cannot be converted.", file=sys.stderr)
        return 2

    output_format = _FORMAT_ALIASES[args.output_format]
    print(f"\nConverting to {C.FORMAT_LABELS[output_format]}...")

    outcome = convert(
        ConversionRequest(
            source=args.source,
            output_format=output_format,
            output_path=args.output,
            device=args.device,
            overwrite=args.overwrite,
            measure_every=max(args.measure_every, 0),
            full_sha256=args.sha256,
            center_basis=args.center_basis,
        ),
        progress_callback=_progress_printer(),
    )
    print()

    if args.json:
        json.dump(outcome.report.as_dict(), sys.stdout, indent=2)
        print()

    if outcome.cancelled:
        print("Cancelled.", file=sys.stderr)
        return 130
    if not outcome.ok:
        print(f"\nConversion failed: {outcome.error}", file=sys.stderr)
        return 1

    output = outcome.report.output
    pruning = outcome.report.pruning
    quantization = outcome.report.quantization
    print()
    print(f"  Output      {output['path']}")
    print(f"  Size        {_human(output['size_bytes'])} "
          f"({output['compression_ratio']:.2f}x smaller)")
    print(f"  Quantized   {quantization['quantized_layers']} layers as {quantization['format']}")
    worst = pruning.get("worst_projection") or {}
    if worst:
        print(f"  AdaLN error {worst.get('rel_l2', 0):.3e} relative (worst projection)")
    error = quantization.get("layer_error", {})
    if error.get("measured_layers"):
        print(f"  Weight error {error['rel_l2_median']:.4f} median relative L2 "
              f"over {error['measured_layers']} sampled layers")
    print(f"  Validation  {outcome.report.validation.get('summary')}")
    print(f"  Report      {outcome.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
