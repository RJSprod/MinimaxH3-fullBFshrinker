"""Versioned Krea 2 FP8 inventory policy.

Krea native/mapped checkpoints do not consistently retain the same symbolic
names for their three MLP projections.  Policy v2 therefore identifies those
projections by *both* their block namespace and exact architectural shapes.
This remains a closed structural allow-list; it is deliberately not an
``all 2D weights`` rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from h3converter import constants as C
from h3converter.safetensor_io import Header, PlannedTensor

KREA2_FP8_POLICY_VERSION = C.KREA2_FP8_POLICY_VERSION
_BLOCK_WEIGHT = re.compile(r"^blocks\.(\d+)\..+\.weight$")
_ATTENTION_SUFFIXES = (
    ".attn.wq.weight",
    ".attn.wk.weight",
    ".attn.wv.weight",
    ".attn.wo.weight",
)
_MLP_SHAPES = {(16384, 6144), (6144, 16384)}
_EXPECTED_BLOCKS = 28
_EXPECTED_ATTENTION_PER_BLOCK = 4
_EXPECTED_MLP_PER_BLOCK = 3


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
    source_quantized_bytes: int

    @property
    def quantized_layer_count(self) -> int:
        return len(self.quant_targets)

    def dtype_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for tensor in self.tensors:
            out[tensor.dtype] = out.get(tensor.dtype, 0) + 1
        return out

    def dtype_elements(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for tensor in self.tensors:
            out[tensor.dtype] = out.get(tensor.dtype, 0) + tensor.numel
        return out


def _local_key(key: str, prefix: str) -> str | None:
    if prefix and not key.startswith(prefix):
        return None
    return key[len(prefix):]


def target_kind(key: str, shape: tuple[int, ...], prefix: str = "") -> str | None:
    """Return the closed policy role for an eligible Krea weight."""
    local = _local_key(key, prefix)
    if local is None or _BLOCK_WEIGHT.match(local) is None:
        return None
    if local.endswith(_ATTENTION_SUFFIXES):
        return "attention"
    if shape in _MLP_SHAPES:
        return "mlp"
    return None


def is_target(key: str, prefix: str = "", shape: tuple[int, ...] | None = None) -> bool:
    """Compatibility helper; shape is required for mapped MLP aliases."""
    return target_kind(key, shape or (), prefix) is not None


def build_output_plan(header: Header, prefix: str = "") -> OutputPlan:
    targets: list[Target] = []
    passthrough: list[str] = []
    tensors: list[PlannedTensor] = []
    roles = {index: {"attention": 0, "mlp": 0} for index in range(_EXPECTED_BLOCKS)}
    source_quantized_bytes = 0

    for key, info in header.tensors.items():
        kind = target_kind(key, info.shape, prefix)
        if kind is not None:
            local = _local_key(key, prefix)
            match = _BLOCK_WEIGHT.match(local or "")
            assert match is not None
            block = int(match.group(1))
            if block in roles:
                roles[block][kind] += 1
            layer = key.removesuffix(".weight")
            targets.append(Target(key, layer, layer + ".weight_scale"))
            tensors.append(PlannedTensor(key, "F8_E4M3", info.shape))
            tensors.append(PlannedTensor(layer + ".weight_scale", "F32", ()))
            source_quantized_bytes += info.nbytes
        else:
            passthrough.append(key)
            tensors.append(PlannedTensor(key, info.dtype, info.shape))

    bad = {
        block: counts for block, counts in roles.items()
        if counts != {"attention": _EXPECTED_ATTENTION_PER_BLOCK, "mlp": _EXPECTED_MLP_PER_BLOCK}
    }
    if bad:
        preview = ", ".join(f"{block}:{counts}" for block, counts in list(bad.items())[:4])
        raise ValueError(
            "Krea FP8 policy inventory is incomplete; expected 4 attention and 3 MLP "
            f"weights in every block ({preview}). Refusing to create a mostly-BF16 output."
        )

    total = sum(tensor.nbytes for tensor in tensors)
    return OutputPlan(tensors, targets, passthrough, total, source_quantized_bytes)
