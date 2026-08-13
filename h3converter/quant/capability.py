"""Dependency capability probing.

Option A is only acceptable if the installed stack can actually produce the
``asym_w4a8_int8`` *serialized contract* -- not merely if some function named
"W4A8" imports. So the probe quantizes a real tensor, checks that every
required side tensor comes back with the right dtype and shape, dequantizes it,
and confirms the round trip is numerically sane. The same is done for NVFP4.

The probe also cross-checks the storage shape rules this project uses for
output planning (:mod:`h3converter.h3_policy`) against the live library. A
future library revision that changed, say, the NVFP4 block-scale swizzle would
otherwise silently produce a plan with wrong offsets; here it fails loudly at
startup instead.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass, field

import torch

from h3converter import constants as C
from h3converter.h3_policy import nvfp4_storage, w4a8_storage
from h3converter.safetensor_io import TORCH_TO_ST

# Shapes used for the self-tests. Small, but with a K that exercises the real
# ConvRot group (256) and a K/16 that is not a multiple of 4, so the NVFP4
# block-scale padding rule is genuinely tested rather than accidentally exact.
_W4A8_PROBE_SHAPE = (32, 512)
_NVFP4_PROBE_SHAPES = ((32, 512), (300, 1024))

# Round-trip quality gate. comfy-kitchen documents ~0.073 relative L2 for W4A8
# on real DiT weights; on Gaussian noise it lands in the same place. A result
# far above this means the layout is not behaving as documented.
_W4A8_MAX_REL_L2 = 0.15
_NVFP4_MAX_REL_L2 = 0.25


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class CapabilityReport:
    device: str = "cpu"
    torch_version: str = ""
    torch_cuda: str | None = None
    kitchen_version: str | None = None
    backends: dict[str, object] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> Check:
        check = Check(name, ok, detail)
        self.checks.append(check)
        return check

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def group_ok(self, prefix: str) -> bool:
        group = [c for c in self.checks if c.name.startswith(prefix)]
        return bool(group) and all(c.ok for c in group)

    def ok_for(self, output_format: str) -> bool:
        if not self.group_ok("runtime."):
            return False
        prefix = ("w4a8." if output_format == C.FORMAT_W4A8 else
                  "fp8." if output_format == C.FORMAT_KREA2_FP8 else "nvfp4.")
        return self.group_ok(prefix)

    def blocking_reason(self, output_format: str) -> str | None:
        if self.ok_for(output_format):
            return None
        prefix = ("w4a8." if output_format == C.FORMAT_W4A8 else
                  "fp8." if output_format == C.FORMAT_KREA2_FP8 else "nvfp4.")
        failed = [c for c in self.failures() if c.name.startswith(("runtime.", prefix))]
        if not failed:
            return f"no self-test results for {C.FORMAT_LABELS[output_format]}"
        return "; ".join(f"{c.name}: {c.detail}" for c in failed[:3])

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "torch_version": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "comfy_kitchen_version": self.kitchen_version,
            "backends": self.backends,
            "checks": [c.as_dict() for c in self.checks],
        }


def import_kitchen():
    """Import comfy_kitchen with the same backend gating ComfyUI applies.

    ComfyUI disables the accelerated CUDA backend when torch was not built
    against CUDA 13+. Mirroring that here means the converter exercises the
    same code path the target runtime will, rather than a faster one that
    might round differently.
    """
    ck = importlib.import_module("comfy_kitchen")
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version is None:
        ck.registry.disable("cuda")
    else:
        try:
            major = int(str(cuda_version).split(".")[0])
        except ValueError:
            major = 0
        if major < C.REQUIRED_TORCH_CUDA_MAJOR:
            ck.registry.disable("cuda")
    ck.registry.disable("triton")
    return ck


def select_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _rel_l2(reference: torch.Tensor, approx: torch.Tensor) -> float:
    ref = reference.float()
    denom = float(ref.norm())
    if denom == 0.0:
        return 0.0
    return float((approx.float() - ref).norm() / denom)


def probe(device: str | None = None, prefer_cuda: bool = True) -> CapabilityReport:
    """Run every dependency self-test and return a structured report."""
    report = CapabilityReport()
    report.torch_version = torch.__version__
    report.torch_cuda = getattr(torch.version, "cuda", None)
    report.device = device or select_device(prefer_cuda)

    # --- runtime -----------------------------------------------------------
    try:
        probe_tensor = torch.ones(64, 64, device=report.device, dtype=torch.bfloat16)
        value = float((probe_tensor @ probe_tensor)[0, 0])
        report.add(
            "runtime.tensor_op",
            value == 64.0,
            f"{report.device} bf16 matmul returned {value}, expected 64.0",
        )
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        report.add("runtime.tensor_op", False, f"{type(exc).__name__}: {exc}")
        return report

    try:
        ck = import_kitchen()
    except Exception as exc:  # noqa: BLE001
        report.add("runtime.comfy_kitchen", False, f"import failed: {type(exc).__name__}: {exc}")
        return report

    try:
        from importlib.metadata import version

        report.kitchen_version = version("comfy-kitchen")
    except Exception:  # noqa: BLE001
        report.kitchen_version = "unknown"

    try:
        report.backends = {
            name: {"available": info.get("available"), "disabled": info.get("disabled")}
            for name, info in ck.list_backends().items()
        }
    except Exception:  # noqa: BLE001
        report.backends = {}

    report.add("runtime.comfy_kitchen", True, f"comfy-kitchen {report.kitchen_version}")

    _probe_w4a8(report)
    _probe_nvfp4(report)
    _probe_fp8(report)
    return report

def _probe_fp8(report: CapabilityReport) -> None:
    """Exercise the exact scalar-scaled E4M3 serialization adapter."""
    try:
        from h3converter.quant.fp8 import dequantize, layer_config, quantize
        report.add("fp8.layout_available", hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn")
        weight = torch.randn(32, 64, generator=torch.Generator().manual_seed(19), dtype=torch.float32)
        result = quantize(weight.to(torch.bfloat16), device=report.device)
        report.add("fp8.quantize", result.weight.dtype == torch.float8_e4m3fn,
                   f"weight {result.weight.dtype}, scale {result.weight_scale.dtype}")
        contract = (result.weight.shape == weight.shape and result.weight_scale.dtype == torch.float32
                    and result.weight_scale.ndim == 0
                    and layer_config()["format"] == C.QUANT_FORMAT_KREA2_FP8)
        report.add("fp8.serialization_contract", contract, "F8_E4M3 weight + scalar F32 weight_scale")
        restored = dequantize(result.weight, result.weight_scale)
        rel = _rel_l2(weight, restored)
        report.add("fp8.dequantize", bool(torch.isfinite(restored).all()) and rel < 0.03,
                   f"round-trip relative L2 {rel:.4f}")
    except Exception as exc:  # noqa: BLE001
        for name in ("layout_available", "quantize", "serialization_contract", "dequantize"):
            if not any(c.name == f"fp8.{name}" for c in report.checks):
                report.add(f"fp8.{name}", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# W4A8
# ---------------------------------------------------------------------------

def _probe_w4a8(report: CapabilityReport) -> None:
    try:
        from comfy_kitchen.tensor import AsymW4A8Int8Layout
    except Exception as exc:  # noqa: BLE001
        report.add(
            "w4a8.layout_available",
            False,
            "AsymW4A8Int8Layout is not present in the installed comfy-kitchen "
            f"({type(exc).__name__}: {exc}). Option A cannot be produced by this build.",
        )
        return
    report.add("w4a8.layout_available", True, "AsymW4A8Int8Layout imported")

    n, k = _W4A8_PROBE_SHAPE
    try:
        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(n, k, generator=generator).to(report.device).to(torch.bfloat16)
        qdata, params = AsymW4A8Int8Layout.quantize(
            weight,
            group_size=C.W4A8_GROUP_SIZE,
            convrot_groupsize=C.W4A8_CONVROT_GROUPSIZE,
        )
        tensors = AsymW4A8Int8Layout.state_dict_tensors(qdata, params)
    except Exception as exc:  # noqa: BLE001
        report.add("w4a8.quantize", False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
        return
    report.add("w4a8.quantize", True, f"quantized {n}x{k} at group {C.W4A8_GROUP_SIZE}, ConvRot {C.W4A8_CONVROT_GROUPSIZE}")

    # The serialized contract, checked suffix by suffix against the plan.
    expected = w4a8_storage(n, k)
    problems: list[str] = []
    for suffix, (dtype, shape) in expected.items():
        if suffix not in tensors or tensors[suffix] is None:
            problems.append(f"missing '{suffix or 'weight'}'")
            continue
        actual = tensors[suffix]
        actual_dtype = TORCH_TO_ST.get(actual.dtype, str(actual.dtype))
        if actual_dtype != dtype:
            problems.append(f"'{suffix or 'weight'}' dtype {actual_dtype}, expected {dtype}")
        if tuple(actual.shape) != shape:
            problems.append(f"'{suffix or 'weight'}' shape {tuple(actual.shape)}, expected {shape}")
    unexpected = set(tensors) - set(expected)
    if unexpected:
        problems.append(f"unexpected extra tensors {sorted(unexpected)}")

    report.add(
        "w4a8.serialization_contract",
        not problems,
        "; ".join(problems) if problems else
        "weight I8[N,K/2] + weight_s_rel F8_E4M3[N,K/16] + weight_s_channel F32[N] + weight_codebook F32[16]",
    )

    # Scale count must equal logical weights / group size - the arithmetic that
    # defines the group-size-16 contract.
    s_rel = tensors.get(C.W4A8_SUFFIX_S_REL)
    if s_rel is not None:
        expected_groups = (n * k) // C.W4A8_GROUP_SIZE
        report.add(
            "w4a8.group_scale_count",
            s_rel.numel() == expected_groups,
            f"{s_rel.numel()} group scales for {n * k} logical weights "
            f"(expected {expected_groups} at group size {C.W4A8_GROUP_SIZE})",
        )

    try:
        restored = AsymW4A8Int8Layout.dequantize(qdata, params)
        rel = _rel_l2(weight, restored)
        finite = bool(torch.isfinite(restored).all())
        report.add(
            "w4a8.dequantize",
            finite and rel <= _W4A8_MAX_REL_L2,
            f"round-trip relative L2 {rel:.4f}" + ("" if finite else " (non-finite output)"),
        )
    except Exception as exc:  # noqa: BLE001
        report.add("w4a8.dequantize", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------

def _probe_nvfp4(report: CapabilityReport) -> None:
    try:
        from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    except Exception as exc:  # noqa: BLE001
        report.add("nvfp4.layout_available", False, f"{type(exc).__name__}: {exc}")
        return
    report.add("nvfp4.layout_available", True, "TensorCoreNVFP4Layout imported")

    problems: list[str] = []
    last = None
    for n, k in _NVFP4_PROBE_SHAPES:
        try:
            generator = torch.Generator().manual_seed(11)
            weight = torch.randn(n, k, generator=generator).to(report.device).to(torch.bfloat16)
            qdata, params = TensorCoreNVFP4Layout.quantize(weight)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
            last = (weight, qdata, params)
        except Exception as exc:  # noqa: BLE001
            report.add("nvfp4.quantize", False, f"{n}x{k}: {type(exc).__name__}: {exc}")
            return

        expected = nvfp4_storage(n, k)
        for suffix, (dtype, shape) in expected.items():
            if suffix not in tensors or tensors[suffix] is None:
                problems.append(f"{n}x{k} missing '{suffix or 'weight'}'")
                continue
            actual = tensors[suffix]
            actual_dtype = TORCH_TO_ST.get(actual.dtype, str(actual.dtype))
            if actual_dtype != dtype:
                problems.append(f"{n}x{k} '{suffix or 'weight'}' dtype {actual_dtype} != {dtype}")
            if tuple(actual.shape) != shape:
                problems.append(f"{n}x{k} '{suffix or 'weight'}' shape {tuple(actual.shape)} != {shape}")

    report.add("nvfp4.quantize", True, f"quantized {len(_NVFP4_PROBE_SHAPES)} shapes at group {C.NVFP4_GROUP_SIZE}")
    report.add(
        "nvfp4.serialization_contract",
        not problems,
        "; ".join(problems) if problems else
        "weight U8[N,K/2] + weight_scale F8_E4M3 (swizzled) + weight_scale_2 F32",
    )

    if last is not None:
        weight, qdata, params = last
        try:
            restored = TensorCoreNVFP4Layout.dequantize(qdata, params)
            rel = _rel_l2(weight, restored[: weight.shape[0], : weight.shape[1]])
            finite = bool(torch.isfinite(restored).all())
            report.add(
                "nvfp4.dequantize",
                finite and rel <= _NVFP4_MAX_REL_L2,
                f"round-trip relative L2 {rel:.4f}" + ("" if finite else " (non-finite output)"),
            )
        except Exception as exc:  # noqa: BLE001
            report.add("nvfp4.dequantize", False, f"{type(exc).__name__}: {exc}")
