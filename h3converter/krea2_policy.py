"""Versioned Krea 2 FP8 inventory policy.

The allow-list is intentionally expressed as complete layer families rather
than an ``ndim == 2`` heuristic.  It is frozen in
``krea2_fp8_reference_manifest.json`` so policy changes are reviewable.
"""
from __future__ import annotations
from dataclasses import dataclass
from h3converter import constants as C
from h3converter.safetensor_io import DTYPE_ITEMSIZE, Header, PlannedTensor

KREA2_FP8_POLICY_VERSION = C.KREA2_FP8_POLICY_VERSION
TARGET_SUFFIXES = (
    ".attn.wq.weight", ".attn.wk.weight", ".attn.wv.weight", ".attn.wo.weight",
    ".mlp.w1.weight", ".mlp.w2.weight", ".mlp.w3.weight",
)

@dataclass(frozen=True)
class Target:
    source_key: str
    layer: str
    scale_key: str

@dataclass
class OutputPlan:
    tensors: list[PlannedTensor]
    quant_targets: list[Target]
    passthrough_keys: list[str]
    total_bytes: int
    @property
    def quantized_layer_count(self): return len(self.quant_targets)
    def dtype_counts(self):
        out = {}
        for t in self.tensors: out[t.dtype] = out.get(t.dtype, 0) + 1
        return out
    def dtype_elements(self):
        out = {}
        for t in self.tensors:
            n = 1
            for d in t.shape: n *= d
            out[t.dtype] = out.get(t.dtype, 0) + n
        return out

def is_target(key: str, prefix: str = "") -> bool:
    local = key[len(prefix):] if key.startswith(prefix) else key
    return local.startswith("blocks.") and local.endswith(TARGET_SUFFIXES)

def build_output_plan(header: Header, prefix: str = "") -> OutputPlan:
    targets, passthrough, tensors = [], [], []
    for key, info in header.tensors.items():
        if is_target(key, prefix):
            layer = key[:-len(".weight")]
            targets.append(Target(key, layer, layer + ".weight_scale"))
            tensors.append(PlannedTensor(key, "F8_E4M3", info.shape))
            tensors.append(PlannedTensor(layer + ".weight_scale", "F32", ()))
        else:
            passthrough.append(key)
            tensors.append(PlannedTensor(key, info.dtype, info.shape))
    if not targets:
        raise ValueError("Krea FP8 policy selected no transformer weights")
    total = sum(t.nbytes for t in tensors)
    return OutputPlan(tensors, targets, passthrough, total)
