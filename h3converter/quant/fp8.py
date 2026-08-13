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

def layer_config() -> dict[str, object]:
    return {"format": "scaled_fp8", "weight_scale": "weight_scale", "dtype": "float8_e4m3fn"}
