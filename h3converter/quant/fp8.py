"""Scaled E4M3 adapter used by the Krea 2 pipeline."""
from __future__ import annotations
from dataclasses import dataclass
import torch

FP8_MAX = 448.0
@dataclass
class Quantized:
    weight: torch.Tensor
    weight_scale: torch.Tensor

def quantize(weight: torch.Tensor, device: str = "cpu") -> Quantized:
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"FP8 source must be BF16/F32, got {weight.dtype}")
    work = weight.to(device=device, dtype=torch.float32)
    amax = work.abs().max()
    scale = torch.clamp(amax / FP8_MAX, min=torch.finfo(torch.float32).tiny)
    q = torch.clamp(work / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).cpu()
    return Quantized(q, scale.float().cpu())

def dequantize(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * weight_scale.float()

def layer_config() -> None:
    """Return no QUANT_ALGOS descriptor for legacy scaled-FP8 storage.

    Forge Neo recognizes this checkpoint representation from an E4M3
    ``weight`` and its sibling FP32 ``weight_scale``.  Adding either
    ``scaled_fp8`` or ``fp8_scaled`` as a per-module quant format makes Forge
    index ``QUANT_ALGOS`` with a key that does not exist and abort loading.
    """
    return None
