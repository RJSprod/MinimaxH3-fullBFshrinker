"""The explicit H3 conversion policy.

This module decides, for every tensor in the source, exactly what appears in
the output. It is deliberately written as an enumerated H3 policy rather than a
"quantize every large 2D weight" heuristic: the golden Option A target has
*exactly* 200 quantized layers, and the layers it leaves alone -- AdaLN
projections, the token refiner, the conditioning projection and the fp32 patch
and output heads -- are precision islands that a generic rule would eat.

The output tensor inventory is fully determined here, before any data is read.
That is what lets the writer emit its header first and stream payloads into
place with no large intermediate file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from h3converter import constants as C
from h3converter.h3_detect import H3Geometry
from h3converter.safetensor_io import Header, PlannedTensor


def roundup(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


# ---------------------------------------------------------------------------
# Storage shapes for the two quantized layouts
# ---------------------------------------------------------------------------

def w4a8_storage(n: int, k: int, group_size: int = C.W4A8_GROUP_SIZE) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Serialized tensors for one ``asym_w4a8_int8`` layer of logical shape [n, k].

    Matches comfy_kitchen ``AsymW4A8Int8Layout.state_dict_tensors`` with the
    default symmetric + codebook configuration, which is what the reference
    artifact contains (one I8 and one F8_E4M3 tensor per layer, plus the two
    F32 side tensors).
    """
    return {
        C.W4A8_SUFFIX_WEIGHT: ("I8", (n, k // 2)),
        C.W4A8_SUFFIX_S_REL: ("F8_E4M3", (n, k // group_size)),
        C.W4A8_SUFFIX_S_CHANNEL: ("F32", (n,)),
        C.W4A8_SUFFIX_CODEBOOK: ("F32", (C.W4A8_CODEBOOK_ENTRIES,)),
    }


def nvfp4_storage(n: int, k: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Serialized tensors for one ``nvfp4`` layer of logical shape [n, k].

    NVFP4 pads the weight to a 16x16 tile and stores block scales in the
    swizzled layout, which rounds rows to 128 and scale columns to 4. Both
    rules are asserted against the live library in the capability self-test.
    """
    padded_n = roundup(n, 16)
    padded_k = roundup(k, 16)
    return {
        C.NVFP4_SUFFIX_WEIGHT: ("U8", (padded_n, padded_k // 2)),
        C.NVFP4_SUFFIX_SCALE: ("F8_E4M3", (roundup(n, 128), roundup(padded_k // C.NVFP4_GROUP_SIZE, 4))),
        C.NVFP4_SUFFIX_SCALE_2: ("F32", ()),
    }


# ---------------------------------------------------------------------------
# Plan objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantTarget:
    """One linear whose weight is replaced by a quantized tensor group."""

    layer: str                       # e.g. "blocks.0.attn.qkv_proj"
    source_key: str                  # "<layer>.weight"
    logical_shape: tuple[int, int]   # [out_features, in_features]
    outputs: dict[str, tuple[str, tuple[int, ...]]]  # suffix -> (dtype, shape)

    def output_key(self, suffix: str) -> str:
        return f"{self.source_key}{suffix}"


@dataclass(frozen=True)
class AdalnTarget:
    """One AdaLN projection collapsed onto the shared rank-8 curve basis."""

    prefix: str                      # e.g. "blocks.0.adaln_proj.linear"
    weight_key: str
    bias_key: str | None
    source_shape: tuple[int, int]    # [width, time_embed_dim]
    reduced_shape: tuple[int, int]   # [width, rank]
    weight_dtype: str
    bias_dtype: str | None
    is_final: bool


@dataclass
class OutputPlan:
    """Complete, byte-exact description of the output file."""

    output_format: str
    geometry: H3Geometry
    source_form: str = C.SOURCE_FORM_FULL
    tensors: list[PlannedTensor] = field(default_factory=list)
    quant_targets: list[QuantTarget] = field(default_factory=list)
    adaln_targets: list[AdalnTarget] = field(default_factory=list)
    passthrough_keys: list[str] = field(default_factory=list)
    dropped_keys: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def quantized_layer_count(self) -> int:
        return len(self.quant_targets)

    def dtype_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tensors:
            counts[t.dtype] = counts.get(t.dtype, 0) + 1
        return counts

    def dtype_elements(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for t in self.tensors:
            totals[t.dtype] = totals.get(t.dtype, 0) + t.numel
        return totals


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def quantized_layer_names(geometry: H3Geometry) -> list[str]:
    """The exact set of layers Option A/B quantize.

    50 blocks x {qkv_proj, out_proj, fc1, fc2} = 200 layers for standard H3.
    Nothing outside ``blocks.<i>.`` is ever included: the token refiner shares
    the same submodule names and would otherwise be swept in.
    """
    return [
        f"{C.KEY_BLOCK_PREFIX}{index}.{family}"
        for index in range(geometry.num_layers)
        for family in C.W4A8_BLOCK_LINEARS
    ]


def is_quantizable(layer: str) -> bool:
    """Guard against a layer name that must never reach the quantizer."""
    if not layer.startswith(C.KEY_BLOCK_PREFIX):
        return False
    if any(marker in layer for marker in C.NEVER_QUANTIZE_SUBSTRINGS):
        return False
    return layer.split(".", 2)[-1] in C.W4A8_BLOCK_LINEARS


def adaln_prefixes(geometry: H3Geometry) -> list[str]:
    """The 50 block AdaLN projections plus the final one, in output order."""
    return [
        f"{C.KEY_BLOCK_PREFIX}{index}.adaln_proj.linear" for index in range(geometry.num_layers)
    ] + [C.KEY_FINAL_ADALN]


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

class PolicyError(RuntimeError):
    """The source cannot be mapped onto the H3 policy."""


def build_output_plan(
    header: Header,
    geometry: H3Geometry,
    output_format: str,
    source_form: str = C.SOURCE_FORM_FULL,
) -> OutputPlan:
    """Derive the full output inventory from the source header.

    Raises ``PolicyError`` rather than silently skipping anything: a layer the
    policy names but the source lacks is a mismatch that must stop the run.

    The two source forms differ only in the timestep path. A full source has
    its time embedder dropped and its AdaLN projections collapsed onto the
    curve basis; a pre-pruned source already carries both the table and the
    reduced projections, and they are passed straight through -- the curve is
    never rebuilt, re-fitted or re-collapsed.
    """
    if output_format not in (C.FORMAT_W4A8, C.FORMAT_NVFP4):
        raise PolicyError(f"unknown output format {output_format!r}")
    if source_form not in (C.SOURCE_FORM_FULL, C.SOURCE_FORM_PREPRUNED):
        raise PolicyError(f"unknown source form {source_form!r}")

    plan = OutputPlan(output_format=output_format, geometry=geometry, source_form=source_form)
    prepruned = source_form == C.SOURCE_FORM_PREPRUNED

    quant_layers = set(quantized_layer_names(geometry))
    # Nothing is collapsed on the pre-pruned path, so there are no AdaLN
    # targets and every adaln tensor falls through to the passthrough branch.
    adaln_map = {} if prepruned else {p: idx for idx, p in enumerate(adaln_prefixes(geometry))}
    storage_fn = w4a8_storage if output_format == C.FORMAT_W4A8 else nvfp4_storage

    emitted: set[str] = set()

    def emit(name: str, dtype: str, shape: tuple[int, ...]) -> None:
        if name in emitted:
            raise PolicyError(f"tensor {name!r} would be written twice")
        emitted.add(name)
        plan.tensors.append(PlannedTensor(name, dtype, shape))

    # The curve table replaces the time embedder; emit it first so the table is
    # at a stable, easily inspected position in the file. A pre-pruned source
    # already has one, which the loop below copies through where it stands.
    if not prepruned:
        emit(C.ADALN_TABLE_KEY, "F32", (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK))

    for key, info in header.tensors.items():
        # 1. The full time embedder disappears entirely - its behaviour is
        #    baked into adaln_t_table.
        if key.startswith(f"{C.KEY_TIME_EMBEDDER}."):
            plan.dropped_keys.append(key)
            continue

        stem, _, param = key.rpartition(".")

        # 2. AdaLN projections collapse onto the rank-8 basis.
        if stem in adaln_map:
            if param == "weight":
                if len(info.shape) != 2:
                    raise PolicyError(f"{key} must be 2D, got {info.shape}")
                bias_key = f"{stem}.bias"
                bias_info = header.get(bias_key)
                reduced = (info.shape[0], C.ADALN_CURVE_RANK)
                plan.adaln_targets.append(
                    AdalnTarget(
                        prefix=stem,
                        weight_key=key,
                        bias_key=bias_key if bias_info else None,
                        source_shape=(info.shape[0], info.shape[1]),
                        reduced_shape=reduced,
                        # The golden reference stores the reduced projections as
                        # BF16; the runtime casts them to its adaln dtype anyway.
                        weight_dtype="BF16",
                        bias_dtype=bias_info.dtype if bias_info else None,
                        is_final=(stem == C.KEY_FINAL_ADALN),
                    )
                )
                emit(key, "BF16", reduced)
            elif param == "bias":
                emit(key, info.dtype, info.shape)
            else:
                plan.passthrough_keys.append(key)
                emit(key, info.dtype, info.shape)
            continue

        # 3. The 200 quantized GEMMs.
        if param == "weight" and stem in quant_layers:
            if not is_quantizable(stem):
                raise PolicyError(f"{stem} matched the quantized set but failed the guard")
            if len(info.shape) != 2:
                raise PolicyError(f"{key} must be 2D to quantize, got {info.shape}")
            n, k = info.shape
            outputs = storage_fn(n, k)
            plan.quant_targets.append(
                QuantTarget(layer=stem, source_key=key, logical_shape=(n, k), outputs=outputs)
            )
            for suffix, (dtype, shape) in outputs.items():
                emit(f"{key}{suffix}", dtype, shape)
            continue

        # 4. Everything else is copied through byte-for-byte.
        plan.passthrough_keys.append(key)
        emit(key, info.dtype, info.shape)

    _verify_plan(plan, header, geometry, source_form)
    return plan


def _verify_plan(plan: OutputPlan, header: Header, geometry: H3Geometry,
                 source_form: str) -> None:
    prepruned = source_form == C.SOURCE_FORM_PREPRUNED
    expected_quant = len(quantized_layer_names(geometry))
    if len(plan.quant_targets) != expected_quant:
        found = {t.layer for t in plan.quant_targets}
        missing = sorted(set(quantized_layer_names(geometry)) - found)
        raise PolicyError(
            f"policy selected {len(plan.quant_targets)} layers to quantize but H3 with "
            f"{geometry.num_layers} blocks requires exactly {expected_quant}"
            + (f"; first missing: {missing[0]}" if missing else "")
        )

    expected_adaln = 0 if prepruned else geometry.num_layers + 1
    if len(plan.adaln_targets) != expected_adaln:
        raise PolicyError(
            f"found {len(plan.adaln_targets)} AdaLN projections to collapse, expected "
            f"{expected_adaln} for a {source_form} source"
        )

    if prepruned:
        # Every curve tensor must be present in the output, unchanged. The
        # table is checked here because it is the one tensor the full path
        # synthesises -- if it ever ended up regenerated on this path, the
        # promise that the source curve is preserved would be silently broken.
        table = header.get(C.ADALN_TABLE_KEY)
        planned_table = next((t for t in plan.tensors if t.name == C.ADALN_TABLE_KEY), None)
        if table is None or planned_table is None:
            raise PolicyError(f"{C.ADALN_TABLE_KEY} is missing from a pre-pruned conversion")
        if planned_table.dtype != table.dtype or planned_table.shape != table.shape:
            raise PolicyError(
                f"{C.ADALN_TABLE_KEY} must be copied unchanged: source is {table.dtype} "
                f"{table.shape}, plan has {planned_table.dtype} {planned_table.shape}"
            )
        if C.ADALN_TABLE_KEY not in plan.passthrough_keys:
            raise PolicyError(f"{C.ADALN_TABLE_KEY} must be copied, not rebuilt")

        by_name = {t.name: t for t in plan.tensors}
        for prefix in adaln_prefixes(geometry):
            for key in (f"{prefix}.weight", f"{prefix}.bias"):
                source = header.get(key)
                if source is None:
                    continue
                planned = by_name.get(key)
                if planned is None:
                    raise PolicyError(f"{key} is missing from the output plan")
                if planned.dtype != source.dtype or planned.shape != source.shape:
                    raise PolicyError(
                        f"{key} must be copied unchanged: source is {source.dtype} "
                        f"{source.shape}, plan has {planned.dtype} {planned.shape}"
                    )

    # Nothing in a preserved family may have been routed to the quantizer.
    for target in plan.quant_targets:
        for marker in C.NEVER_QUANTIZE_SUBSTRINGS:
            if marker in target.layer:
                raise PolicyError(f"{target.layer} is in a preserved family but was selected for quantization")

    if prepruned:
        if plan.dropped_keys:
            raise PolicyError(
                f"a pre-pruned source has no time embedder to drop, but the plan drops "
                f"{len(plan.dropped_keys)} tensors"
            )
    elif not plan.dropped_keys:
        raise PolicyError("no time_embedder tensors were dropped - source is not in the expected full form")

    # Precision islands must survive at their source dtype.
    by_name = {t.name: t for t in plan.tensors}
    for prefix in C.FP32_PRESERVED_PREFIXES:
        for key in (prefix, f"{prefix}.weight", f"{prefix}.bias"):
            source = header.get(key)
            if source is None:
                continue
            planned = by_name.get(key)
            if planned is None:
                raise PolicyError(f"{key} is a preserved tensor but is missing from the output plan")
            if planned.dtype != source.dtype:
                raise PolicyError(
                    f"{key} must keep its source dtype {source.dtype}, plan has {planned.dtype}"
                )
