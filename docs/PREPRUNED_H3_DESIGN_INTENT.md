# 10Eros-Max Pre-Pruned H3 Conversion — High-Level Design Intent

**Version:** 1.1
**Date:** 2026-08-16
**Supersedes:** v1.0 (`10Eros_Max_Prepruned_High_Level_Design_Intent.txt`)
**Status:** design intent, not yet implemented

---

## 0. What changed in v1.1, and why

v1.0 was reviewed against the actual converter source. The design intent is sound and
survives intact; the changes below correct places where v1.0 described the codebase, the
target runtime, or the arithmetic inaccurately. Every correction is traceable to a file in
this repository or to the ComfyUI source it targets.

| # | v1.0 said | Reality | Change in v1.1 |
|---|---|---|---|
| 1 | Profiles 2/3 use "the converter's tested H3-compatible INT8 storage path" | **No such path exists.** `h3converter/quant/` contains only `w4a8.py` and `nvfp4.py`. `docs/COMPATIBILITY.md` mentions INT8 only as something the self-test exists to *prevent* being emitted by accident. | §4 now states plainly that Profiles 2 and 3 require a **new** quantizer, capability probe, validation branch and metadata path. Sized as new work, not configuration. |
| 2 | "INT8 / INT8 ConvRot" | ComfyUI's `QUANT_ALGOS` has no INT8 ConvRot format. The ConvRot entries are `convrot_w4a4` and `asym_w4a8_int8`. The weight-only INT8 entry is **`int8_tensorwise`**. | §4 names the exact format string and drops "INT8 ConvRot" everywhere. |
| 3 | Detection requires "50 main transformer blocks" | `constants.H3_MIN_BLOCKS=40`, `H3_MAX_BLOCKS=64`. Hard-coding 50 would reject valid H3 fine-tunes the existing path already accepts. | §3 keys on the supported range; 50 is the reference value, not the gate. |
| 4 | Detection criteria listed, no ambiguity rule | A checkpoint carrying *both* `adaln_t_table` and `time_embedder.*` is undecidable — either path could be what the runtime uses. | §3 adds explicit accept/reject rules for all four presence combinations. |
| 5 | "Copy the existing AdaLN curve-form tensors bit-for-bit" | `validate._check_structure` hard-codes `BF16` for the reduced AdaLN projections (`pruning.block_adaln_reduced`, `pruning.final_adaln_reduced`). A source storing them as F32 is bit-for-bit copied and then **fails its own output validation**. | §3 and §8 require the BF16 expectation to become "matches the source dtype". |
| 6 | Profile sizes given as round numbers | The arithmetic was never shown, so it could not be checked. | §4 derives all three from the reference geometry. All three v1.0 figures hold: 12.54 / 14.23 / 16.76 GB. |
| 7 | Silent on NVFP4 | The converter has two output formats. v1.0 defines three profiles that are all W4A8-family and never says what happens to Option B on a pre-pruned source. | §4.4 makes NVFP4 an explicit, supported fourth choice on this path. |
| 8 | "Extend the current application" (correct, but unspecific) | Eight modules hard-fail or mis-report on a pre-pruned source, not just the detector. | §8 enumerates every one with file and symbol. |
| 9 | Did not mention the golden reference | `constants.REFERENCE_INVENTORY` is derived from `10Eros_Max_h3_fl2va_test4_pruned-w4a8_convrot.safetensors` — a **10Eros-Max** W4A8 artifact. Profile 1's target is already this repo's conformance reference. | §2.3 states this. It materially de-risks Profile 1. |
| 10 | "Report mean and worst relative L2 for K and fc2" | Default `measure_every=25` measures 8 of 200 layers, and `_measure` is whole-tensor only — it cannot produce a K-slice figure. | §6 specifies the sampling and slicing changes needed to make this criterion achievable. |

Nothing in §1, §2, §5 or the overall shape of §7–§9 of v1.0 was found to be wrong.

---

## 1. Purpose

Extend the existing MiniMax H3 converter so it accepts TenStrip/10Eros-Max checkpoints that
are **already** in the compact AdaLN-pruned H3 form (~40.2 GB) and reduces them to
12.5–16.8 GB depending on the selected quality profile.

This is a second *input* path, not a replacement for the current full-BF16 flow:

```
Existing:                          New:
~66 GB full H3 BF16                ~40.2 GB already-pruned H3
  -> build AdaLN curve form          -> validate existing curve form
  -> quantize                        -> copy curve tensors unchanged
  -> ~12.5 GB                        -> quantize
                                     -> 12.5-16.8 GB
```

If the source already contains a valid AdaLN curve representation, the converter must never
rebuild or re-prune it.

The two source forms differ only in their front half. They converge on one shared
quantization, validation and output stage.

---

## 2. Source model characteristics

**Target repository:** <https://huggingface.co/TenStrip/10Eros-Max/tree/main>

The repository contains multiple ~40.2 GB MiniMax H3 checkpoints, including beta1/beta2 and
explicitly named BF16-pruned variants.

### 2.1 What the model is

The author describes 10Eros-Max as a weight-space graft/merge rather than a conventional
training fine-tune. Learned patterns from LTX 2.3, Wan 2.2 and, in later versions, Krea 2 are
added into H3's existing attention/MLP weights while the underlying H3 architecture remains
intact.

### 2.2 Model-specific information from the author

- H3's fused Q/K/V weights were temporarily unfused for selective grafting, then re-fused into
  normal `qkv_proj` tensors for ComfyUI inference.
- The **K projection** was deliberately isolated/preserved — the author found it sensitive to
  H3 amplification/calibration and to audio quality.
- **MLP `fc2`** was also deliberately protected in the graft workflow.
- H3 uses a unified audio/video transformer, so video-oriented weight changes can affect audio
  behaviour.

The model can therefore be quantized normally, but the converter should offer optional
precision profiles that protect these audio-sensitive paths.

### 2.3 This model is already the converter's conformance reference

`h3converter/constants.py::REFERENCE_INVENTORY` is derived from, and checked against, the
observed artifact `10Eros_Max_h3_fl2va_test4_pruned-w4a8_convrot.safetensors` —
12,540,857,840 bytes, 1,132 tensors, 200 quantized layers. That is a 10Eros-Max checkpoint in
exactly the form Profile 1 produces.

Consequence: **Profile 1 is not a new target.** It is the target the existing test suite
already reproduces analytically (`tests/unit/test_reference_conformance.py`). The new work for
Profile 1 is confined to the input path. Profiles 2 and 3 are the genuinely new output
territory.

---

## 3. New source detection mode

Add a source classification alongside the implicit `FULL_H3_BF16`:

```
PREPRUNED_H3_FLOAT
```

### 3.1 Qualifying criteria

Header inspection only — no tensor data is read. A checkpoint qualifies if and only if:

| # | Criterion | Checked against |
|---|---|---|
| 1 | MiniMax H3 architecture (`video_patch_proj` + `audio_patch_proj` both present) | existing `h3_detect.detect` signature test |
| 2 | Transformer block count within the supported range | `H3_MIN_BLOCKS` (40) – `H3_MAX_BLOCKS` (64); 50 for reference H3 |
| 3 | `adaln_t_table` present, dtype `F32`, shape exactly `[ADALN_CURVE_GRID, ADALN_CURVE_RANK]` = `[1025, 8]` | `constants.ADALN_TABLE_KEY` |
| 4 | **No** `time_embedder.*` tensors | `constants.KEY_TIME_EMBEDDER` |
| 5 | Every `blocks.<i>.adaln_proj.linear.weight` has shape `[block_adaln_width, 8]` | `H3Geometry.block_adaln_width` (96,768 at hidden 5376) |
| 6 | `final_layer.adaln_proj.linear.weight` has shape `[final_adaln_width, 8]` | `H3Geometry.final_adaln_width` (10,752) |
| 7 | No quantization metadata key | `constants.QUANT_METADATA_KEY` |
| 8 | No quantization side tensors | `h3_detect._QUANT_MARKERS` (`.comfy_quant`, `.weight_scale`, `.weight_s_rel`, `.weight_s_channel`, `.weight_codebook`, `.weight_scale_2`, `.scale_weight`) |
| 9 | All four target matrices per block are 2D, floating point, K divisible by 256 and 16 | existing `_convertibility_errors` loop |

Criterion 3 is exact, not approximate. The table's second dimension **is** the AdaLN
projections' input width at runtime — `comfy/ldm/minimax/model.py` registers the buffer as
`[adaln_curve_grid, time_embed_dim]` and lerps between adjacent rows. A table of a different
rank is a different model, not a variant to adapt to.

**Filenames are never trusted.** `pruned`, `bf16`, `beta1`, `beta2` carry no weight in the
decision. This matches the existing detector's stated contract.

### 3.2 Ambiguity rules (new in v1.1)

| `adaln_t_table` | `time_embedder.*` | Classification |
|---|---|---|
| absent | present | `FULL_H3_BF16` — existing path, unchanged |
| present | absent | `PREPRUNED_H3_FLOAT` — new path |
| present | present | **Reject.** Undecidable: both modulation paths are materially present and the converter cannot know which one the source is meant to use. |
| absent | absent | **Reject.** No timestep path at all; not a loadable H3 checkpoint. |

The two rejections must produce distinct, specific error messages. "Ambiguous" and "broken"
are different problems for the user.

### 3.3 Behaviour on this path

For `PREPRUNED_H3_FLOAT` inputs, **do not**:

- rebuild the timestep curve,
- run SVD,
- recreate `adaln_t_table`,
- collapse AdaLN projections again,
- modify the existing reduced AdaLN tensors in any way.

Copy `adaln_t_table` and every reduced AdaLN projection **bit-for-bit, at their source dtype**.

The dtype clause is load-bearing. `validate.py` currently asserts the reduced projections are
`BF16`, because the full path writes them as BF16 by policy. On this path the source decides.
A source that stores them as F32 is legitimate, is bit-for-bit copied, and must not then be
failed by our own output validator. The check becomes "matches the source dtype and is a float
type", not "is BF16".

### 3.4 What is no longer an error

`h3_detect._convertibility_errors` currently emits a hard error when `already_curve_pruned` is
true:

> *"source already uses the compact AdaLN curve form (adaln_t_table present) — this tool
> converts full, unpruned checkpoints"*

That error becomes conditional on the source failing §3.1/§3.2 rather than on the curve form
being present. Similarly, the "missing time embedder" error must apply only to
`FULL_H3_BF16`. Both already sit behind `if not result.already_curve_pruned:` guards in part;
the guard must be extended to the classification, not the single boolean.

---

## 4. Quantization profiles

### 4.0 The format actually available for the 8-bit tiers

ComfyUI's `QUANT_ALGOS` registry (`comfy/quant_ops.py`) contains one weight-only INT8 entry:

```
int8_tensorwise    storage torch.int8    param: weight_scale    quantize_input: False
```

There is **no** INT8 ConvRot format. `convrot_w4a4` and `asym_w4a8_int8` are the ConvRot
entries and both are 4-bit weight formats. v1.0's "INT8 / INT8 ConvRot" wording described
something that does not exist in the target runtime, and all references to it are removed.

`quantize_input: False` is a point in this design's favour: activations stay in BF16, so the
protected paths keep full activation precision as well as 8-bit weights.

**Two things must be resolved before implementation, by probe rather than by reading:**

1. **Scale granularity.** The registry entry declares a single `weight_scale`; the loader in
   `comfy/ops.py` reads it as a 1-D per-output-channel tensor. Per-channel is both more likely
   and far better for quality. Resolve it by round-tripping a real tensor through the installed
   build in the capability probe — the same method `quant/capability.py` already uses for W4A8
   and NVFP4 — and fail loudly on mismatch. Do not infer it from the format's name. The size
   impact either way is negligible (N floats vs 1); the *quality* impact is large.
2. **Mixed-format checkpoints.** `_quantization_metadata.layers` is a per-layer map with a
   per-layer `format` field, so a checkpoint with different formats on different linears is
   representable. That it *loads* must be confirmed on the target runtime before Profiles 2/3
   are considered deliverable (see §9).

**There is no existing INT8 code in this converter.** Profiles 2 and 3 require, as new work:

- `h3converter/quant/int8.py` — quantize / dequantize / `layer_config()`, mirroring `w4a8.py`'s
  structure and its `_check_contract` strictness;
- an `int8.*` group in `quant/capability.py::probe`, with the same
  quantize → contract-check → dequantize → round-trip-error shape as `_probe_w4a8`;
- an `int8_storage()` function in `h3_policy.py` and per-family storage selection in
  `build_output_plan`;
- per-layer expected-format validation in `validate._check_metadata`, which today asserts one
  format across all layers;
- `CapabilityReport.ok_for` / `blocking_reason` gating for the new profile ids.

### 4.1 Reference arithmetic

All sizes below are derived from `H3_REFERENCE_GEOMETRY` (hidden 5376, 50 blocks, 56×128
heads, ffn 14336). Per-block parameter counts:

| Family | Logical shape | Params/block | Params × 50 |
|---|---|---|---|
| `attn.qkv_proj` | [21504, 5376] | 115,605,504 | 5,780,275,200 |
| `attn.out_proj` | [5376, 7168] | 38,535,168 | 1,926,758,400 |
| `mlp.fc1` | [28672, 5376] | 154,140,672 | 7,707,033,600 |
| `mlp.fc2` | [5376, 14336] | 77,070,336 | 3,853,516,800 |
| **Total** | | **385,351,680** | **19,267,584,000** |

Storage cost per parameter:

- **W4A8 ConvRot** — 0.5 B packed weight + 1 B per group of 16 (F8_E4M3 `weight_s_rel`) =
  **0.5625 B/param**, plus per-channel F32 and a 16-entry codebook per layer.
- **`int8_tensorwise`** — **1.0 B/param**, plus a scale of at most N floats per layer.

Non-quantized remainder (norms, biases, AdaLN, token refiner, `condition_proj`, F32 islands):
842,459,392 BF16 elements + 4,444,952 F32 elements ≈ **1.703 GB**.

### 4.2 The three profiles

| | Profile 1 — Compact / Reference | Profile 2 — Audio-Safer | Profile 3 — Audio-Conservative |
|---|---|---|---|
| id | `compact_w4a8` | `audio_safer` | `audio_conservative` |
| `attn.qkv_proj` | W4A8 ConvRot | W4A8 ConvRot | **`int8_tensorwise`** |
| `attn.out_proj` | W4A8 ConvRot | W4A8 ConvRot | W4A8 ConvRot |
| `mlp.fc1` | W4A8 ConvRot | W4A8 ConvRot | W4A8 ConvRot |
| `mlp.fc2` | W4A8 ConvRot | **`int8_tensorwise`** | **`int8_tensorwise`** |
| Data bytes | 12,540,714,592 | ≈ 14,226,628,192 | ≈ 16,755,498,592 |
| **Output** | **≈ 12.54 GB** | **≈ 14.23 GB** | **≈ 16.76 GB** |
| vs 40.2 GB source | 3.2× smaller | 2.8× smaller | 2.4× smaller |
| New code needed | input path only | + INT8 quantizer | + INT8 quantizer |

W4A8 parameters are fixed by `constants.py` and are not user-adjustable: format
`asym_w4a8_int8`, W4 group size 16, ConvRot group size 256.

All three v1.0 size estimates are confirmed by this arithmetic. Profile 1 lands on the golden
reference exactly, because it *is* the golden reference policy.

#### Profile 1 — Compact / Reference

Maximum compression using the existing H3 W4A8 reference policy. Calibration-free, and
designed as a low-error 4-bit DiT format. **Test this first when minimum size is the priority.**

*Risk:* the fused `qkv_proj` includes the K slice the author deliberately preserved, and `fc2`
is quantized too. The graft is not removed, but these audio-sensitive values are represented at
lower precision.

#### Profile 2 — Audio-Safer

Protects the author's specifically identified MLP path while retaining most of the compression
benefit. `fc2` was intentionally protected during grafting; keeping it at 8 bits materially
lowers quantization error for that path for +1.69 GB.

#### Profile 3 — Audio-Conservative

Protects both areas the author identifies as important to audio/calibration.

The final H3 checkpoint stores Q/K/V as one fused `qkv_proj` tensor. No standard ComfyUI
quantized-linear format can keep only the K third at higher precision inside a fused matrix
without a custom segmented layout that the runtime would not load. Promoting the whole
`qkv_proj` to INT8 is the clean, runtime-compatible way to give K more precision.

**Do not unfuse Q/K/V in the final checkpoint**, under any profile.

### 4.3 The quality premise must be measured, not assumed

The profiles rest on "8-bit is safer for audio than 4-bit here". That is very likely true but
is not free:

- W4A8 ConvRot is documented at ~0.073 relative L2 on real DiT weights (`capability.py`), and
  it is a *sophisticated* 4-bit format — rotation, Lloyd-Max codebook, ALS-refined group scales
  at group 16, plus a per-channel scale.
- INT8 with a **per-channel** scale should land near ~0.014 relative L2 on
  Gaussian-distributed weights — roughly a 5× improvement, and the profiles are clearly worth it.
- INT8 with a single **tensor-wide** scale is a different story. One scale for 115M values is
  set by the global outlier, and on an outlier-heavy grafted tensor the effective precision can
  fall far below the nominal 8 bits — conceivably below W4A8's grouped, rotated 4 bits.

This is exactly why §4.0 requires the granularity to be probed. If the installed build turns
out to be genuinely tensor-wide, Profiles 2 and 3 must be re-evaluated against measured
per-slice error (§6) before they are offered in the GUI, not shipped on the assumption that
more bits is always better.

### 4.4 NVFP4 on this path (gap closed from v1.0)

The converter has a second existing output format, `nvfp4` (Option B), which v1.0 did not
mention. A pre-pruned source is a perfectly valid input for it, and `analyze_source` previews
both formats today.

Decision: **NVFP4 remains available on the pre-pruned path as a fourth choice, unprofiled.**
It keeps its existing policy, calibration step and Blackwell (sm_100+) runtime requirement. It
is not folded into the profile axis, because the profile axis exists to trade W4A8 4-bit
precision for INT8 8-bit precision on two specific families, and that trade has no NVFP4
analogue worth the combinatorial surface.

The output-format registry therefore becomes four entries — three W4A8-family profiles plus
`nvfp4` — rather than a `format × profile` matrix.

---

## 5. Preserved tensors

For all profiles, preserve the pre-pruned structure:

- `adaln_t_table`
- `blocks.*.adaln_proj.linear.*`
- `final_layer.adaln_proj.linear.*`
- `token_refiner.*`
- `condition_proj`
- video/audio patch projections
- final video/audio output heads
- norms and biases
- existing precision islands (`constants.FP32_PRESERVED_PREFIXES`)

Preserve source dtype unless a known-good H3 runtime contract requires otherwise. The existing
`NEVER_QUANTIZE_SUBSTRINGS` guard already enforces most of this and needs no change; note that
`token_refiner` blocks carry the same `attn.qkv_proj` / `mlp.fc1` submodule names as the main
blocks and are excluded only because selection is anchored to the `blocks.<i>.` prefix.

---

## 6. Quality and validation

Because 10Eros-Max is a weight-space graft, validation includes model-specific measurements in
addition to normal checkpoint validation.

### 6.1 Per-slice error reporting

For sampled blocks, dequantize and report relative error separately for:

- Q slice of fused `qkv_proj`
- K slice of fused `qkv_proj`
- V slice of fused `qkv_proj`
- `mlp.fc1`
- `mlp.fc2`

Slice boundaries come from geometry, not convention: `qkv_proj` is `[3 × heads × head_dim,
hidden]`, so the three equal output regions are rows `[0, 7168)`, `[7168, 14336)`,
`[14336, 21504)` at reference geometry.

**The qkv tensor is split into three logical regions for validation only. Serialization
remains fused.**

Report **mean and worst relative L2 for the K slice and for `fc2`** across all sampled layers.

Two existing behaviours have to change for this to be achievable:

1. `quant/w4a8.py::_measure` computes one whole-tensor `rel_l2`. It needs an optional row-range
   argument so the same dequantized tensor yields Q/K/V figures without a second decode.
2. The default `measure_every=25` samples 8 of 200 layers. A "worst K error" over 8 layers is
   a weak claim. Use a denser default (every 5th layer, i.e. 40 of 200) on this path, or
   measure every layer for the `qkv_proj`/`fc2` families specifically. Measurement reuses the
   already-in-memory weight and adds one dequantize per sampled layer.

`reports.summarise_layer_stats` and the `quantization.per_layer_samples` block extend to carry
the per-slice figures.

### 6.2 Generation tests

Compare the ~40.2 GB source against each profile using identical prompts, seeds and inference
settings, focusing on:

- speech/audio intelligibility
- audio continuity
- voice stability
- audio/video synchronization
- temporal motion
- visual character of the graft
- prompt adherence
- complete loss or weakening of audio

If Profile 1 preserves audio adequately, it remains the preferred ~12.5 GB output. If audio
degrades, test Profile 2, then Profile 3.

### 6.3 Structural validation on the pre-pruned path

The full path's numerical gate (`ADALN_REL_ERROR_WARN` / `ADALN_REL_ERROR_ABORT`, enforced by
`adaln_prune.check_report`) has nothing to measure here — no basis is fitted, so there is no
reconstruction error to bound. It is replaced by structural assertions:

- `adaln_t_table` in the output is byte-identical to the source's;
- every reduced AdaLN projection is byte-identical to the source's;
- the output contains no `time_embedder.*` tensors (trivially true — the source had none).

`report.pruning` keeps its key in the JSON schema and gains an explicit
`{"mode": "preserved_from_source", ...}` shape, so downstream readers of the report do not have
to special-case a missing section.

---

## 7. User experience

When a compatible source is selected, the GUI shows:

```
MiniMax H3 detected
Source form: Already AdaLN-pruned floating-point H3
AdaLN table: F32 [1025, 8]
Additional pruning required: No
```

Then offers:

```
1. Compact W4A8          ~12.5 GB   Maximum compression
2. Audio-Safer           ~14.2 GB   Keeps MLP fc2 at INT8
3. Audio-Conservative    ~16.8 GB   Keeps QKV and MLP fc2 at INT8
4. NVFP4                 ~12.5 GB   Blackwell GPUs only
```

Sizes shown are the real planned totals from `OutputPlan.total_bytes`, not the estimates in
this document — the plan is fully determined before any data is read, and screen 2 already
displays it that way today.

Each option is disabled with a reason when its capability probe fails, as the two existing
format buttons already are. Profile 1 and NVFP4 land at a similar size for different reasons;
the subtitles must make the Blackwell requirement unmissable so NVFP4 is not chosen by accident
on unsupported hardware.

All other decisions remain automatic. The existing full ~66 GB H3 conversion path must
continue to work unchanged, and continues to present its own two choices.

---

## 8. Implementation principle

**Do not build a separate converter.** Extend the current application so both source states
converge into the same quantization/output pipeline:

```
FULL_H3_BF16                    PREPRUNED_H3_FLOAT
  -> validate                     -> validate existing AdaLN curve form
  -> build AdaLN curve form       -> copy curve tensors unchanged
  -> selected profile             -> selected profile
  -> output                       -> output
```

Continue using the existing project-local `.venv`, pinned dependencies, comfy-kitchen layouts,
streaming safetensors I/O, `.partial` output, post-write validation, atomic finalization and
JSON conversion reports.

### 8.1 Every site that must change

Eight modules hard-fail or mis-report on a pre-pruned source today. This is the full list; it
is what makes the estimate honest.

| File | Symbol | Today | Required |
|---|---|---|---|
| `h3_detect.py` | `Detection` | `already_curve_pruned: bool` | add `source_form` (`FULL_H3_BF16` / `PREPRUNED_H3_FLOAT`) and surface it in `summary` |
| `h3_detect.py` | `_convertibility_errors` | curve form → hard error; missing time embedder → hard error | gate both on `source_form`; add the §3.1/§3.2 checks |
| `h3_policy.py` | `build_output_plan` | always drops `time_embedder.*`, always collapses AdaLN | on the pre-pruned path, pass `adaln_t_table` and the reduced projections through untouched, with **zero** `adaln_targets` |
| `h3_policy.py` | `_verify_plan` | **raises** `"no time_embedder tensors were dropped"` when `dropped_keys` is empty; requires `len(adaln_targets) == num_layers + 1` | both assertions become source-form-specific |
| `h3_policy.py` | `w4a8_storage` / storage selection | one storage function per format | per-family storage selection driven by the profile |
| `pipeline.py` | `convert` steps 5 and 7 | reads the time embedder from `plan.dropped_keys`, fits the basis, collapses 51 projections | skip entirely on the pre-pruned path; the curve tensors ride the passthrough loop |
| `pipeline.py` | `report.quantization` | single `policy`/`format`/`group_size`; `preserved_tensors` counts `2 × len(adaln_targets)` | per-family format map; corrected preserved-tensor count when there are no AdaLN targets |
| `constants.py` | `PHASE_WEIGHTS` | must sum to 1.0 per format; pre-pruned skips `adaln_basis` (0.10) + `adaln_collapse` (0.20) | new weight sets per source form — progress would otherwise stall at 70% |
| `constants.py` | `OPTION_A_SIZE_BAND_BYTES` (11–14 GB) | Profile 2 at 14.23 GB is outside it; Profile 3 far outside | per-profile advisory bands |
| `constants.py` | `DISK_*_MULTIPLIER`, `FORMAT_LABELS`, `FORMAT_FILENAME_INFIX` | keyed by the two formats | entries for every new profile id |
| `constants.py` | `W4A8_BLOCK_LINEARS` | name now misleading — the four families are quantized under every profile, at differing formats | rename to `BLOCK_LINEARS` |
| `paths.py` | `derive_output_path` | one infix per format | distinct infix per profile, or the three profiles collide on one filename and silently take the `_2` / `_3` uniquifier |
| `validate.py` | `pruning.block_adaln_reduced`, `pruning.final_adaln_reduced` | hard-code `BF16` | expect the source dtype (see §3.3) |
| `validate.py` | `_check_metadata` | asserts one `wanted_format` across all layers | per-layer expected format from the plan |
| `validate.py` | `_check_scale_health`, `_check_dequantization` | branch on `output_format` for suffixes and decoder | branch per layer, since one file now mixes formats |
| `validate.py` | `_conformance` | `compared_to_reference` is true for any W4A8 output at reference geometry | restrict to Profile 1 — Profiles 2/3 have deliberately different dtype censuses |
| `h3_reference.py` | `full_source_inventory` | builds the full unpruned inventory | add a pre-pruned inventory so `predict_inventory` and the fixtures cover this path |
| `quant/capability.py` | `probe`, `ok_for`, `blocking_reason` | `w4a8.` / `nvfp4.` groups | add the `int8.` group and profile-aware gating |
| `gui.py` | screen 2 | two format buttons, *"exactly one decision: A or B"* | four options plus the source-form banner from §7 |
| `cli.py` | `_FORMAT_ALIASES` | `a`/`w4a8`, `b`/`nvfp4` | profile aliases (`1`/`2`/`3` or the ids), kept backward compatible |

`quantized_layer_names`, `is_quantizable`, `NEVER_QUANTIZE_SUBSTRINGS` and the whole
`safetensor_io` streaming writer need no change — the layer *selection* is identical across all
three profiles; only the per-family storage format differs.

### 8.2 Suggested sequencing

1. **Detection + policy + pipeline for `PREPRUNED_H3_FLOAT`, Profile 1 only.** Delivers a
   ~12.5 GB output against a target this repo already reproduces analytically. Lowest risk,
   highest confidence, and it exercises every structural change without any new numerics.
2. **Per-slice error measurement (§6.1).** Needed to judge whether Profiles 2/3 are worth
   shipping, and useful on Profile 1 immediately.
3. **`int8_tensorwise` quantizer + capability probe**, granularity resolved by probe (§4.0).
4. **Profiles 2 and 3**, gated on the mixed-format runtime load test (§9).

Steps 1 and 2 are worth shipping on their own if step 3 stalls on runtime compatibility.

---

## 9. Acceptance criteria

The feature is complete when:

**Detection and input handling**
1. TenStrip beta/pruned checkpoints are classified `PREPRUNED_H3_FLOAT` from tensor structure
   alone, with filenames ignored.
2. A checkpoint with both `adaln_t_table` and `time_embedder.*` is rejected with a distinct
   "ambiguous" message; one with neither is rejected with a distinct "no timestep path" message.
3. `adaln_t_table` and all 51 reduced AdaLN projections are byte-identical between source and
   output, at the source dtype, verified by a test that compares raw bytes.
4. A source whose reduced AdaLN projections are F32 converts and passes output validation.

**Outputs**

5. Profile 1 produces the ~12.5 GB W4A8 class and matches `REFERENCE_INVENTORY`'s tensor count
   and dtype census at reference geometry.
6. Profile 2 produces a ~14.2 GB mixed W4A8/INT8 output.
7. Profile 3 produces a ~16.8 GB mixed W4A8/INT8 output.
8. All outputs retain the standard **fused** `qkv_proj` architecture.
9. The three profiles write three distinct filenames beside the same source.
10. All outputs load in the intended ComfyUI H3 runtime. **For Profiles 2 and 3 this includes
    an explicit mixed-format load test** — one checkpoint whose blocks contain both
    `asym_w4a8_int8` and `int8_tensorwise` linears — recorded with the ComfyUI commit used.

**Measurement**

11. Per-Q/K/V and `fc2` quantization error is recorded in the JSON report, with mean and worst
    relative L2 for the K slice and `fc2` over a sample of at least 40 of the 200 layers.
12. The `int8_tensorwise` scale granularity is confirmed by capability probe against the
    installed build, and recorded in `environment.capability_checks`.

**Safety and regression**

13. Source checkpoints are never modified; no surviving `.partial`; cancellation leaves nothing
    behind. (Already guaranteed by the existing pipeline; re-asserted on the new path.)
14. The existing full ~66 GB H3 workflow remains regression-tested and byte-identical in
    output for Profile 1 / Option A.
15. Every output of every profile is refused as an input, via the existing `_QUANT_MARKERS`
    detection — including `.weight_scale`, which the INT8 profiles introduce.

---

## 10. Open questions

| # | Question | Blocks | How to resolve |
|---|---|---|---|
| 1 | Is `int8_tensorwise`'s `weight_scale` per-channel or tensor-wide in the pinned build? | Profiles 2/3 quality claim | Capability probe (§4.0); resolve before writing the quantizer |
| 2 | Does ComfyUI load a checkpoint mixing `asym_w4a8_int8` and `int8_tensorwise` across linears within one block? | Profiles 2/3 entirely | Runtime load test (§9.10) on the target machine |
| 3 | Does comfy-kitchen expose an INT8 layout to delegate to, or must the converter own the numerics? | Implementation shape of `quant/int8.py` | Inspect the pinned `comfy-kitchen==0.2.31`. If not, this is the first place the converter implements quantization numerics itself rather than delegating — that departure from the existing design principle should be a conscious, documented decision |
| 4 | ~~Is the 40.2 GB figure BF16 or F32 for the non-quantized remainder?~~ | — | **Resolved 2026-08-16** — see §11. BF16, 40.22 GB. |

None of these block the Profile 1 pre-pruned path in §8.2 step 1.

---

## 11. Confirmed against a real checkpoint

`10Eros_Max_h3_fl2va_beta2_pruned.safetensors` (40.22 GB) was run through the current GUI on
the target machine. It was refused, as expected — the feature is unimplemented — but the
inspection screen confirms most of §3 already holds against a real file:

| Reported | Confirms |
|---|---|
| Detected model: MiniMax H3 | §3.1 criterion 1 |
| 50 transformer blocks (+2 token refiner) | §3.1 criterion 2; reference geometry |
| hidden 5376, 56×128 heads, ffn 14336 | `H3_REFERENCE_GEOMETRY` exactly |
| **time embed 8** | §3.1 criterion 3 — this value is read from `adaln_t_table`'s second dimension, so the curve table is present and is rank 8 |
| Pruning state: curve-pruned already | `already_curve_pruned` is set correctly |
| Quantization state: none | §3.1 criteria 7–8 |
| Source precision: BF16 | resolves open question 4 |

Only one compatibility error was produced — the §3.4 refusal at `h3_detect.py:234-238`. The
AdaLN shape checks at `_convertibility_errors` (every block against `[96768, 8]`, final against
`[10752, 8]`) passed silently, and the time-embedder requirement was correctly skipped by its
existing `if not result.already_curve_pruned:` guard.

**This narrows §8.2 step 1 considerably.** Detection already parses and validates the curve
form correctly; it refuses on policy, not on failing to understand the file. The work is the
refusal gate plus the plan/pipeline/validation branches in §8.1 — not new detection logic.

Target machine for reference: RTX 5090 (sm_120), 34.19 GB VRAM, 102.56 GB RAM. Available RAM
at inspection time was 16.42 GB against a `TARGET_PEAK_RAM_BYTES` of 80 GB; that target is a
ceiling rather than a requirement, since the pipeline streams one tensor at a time and the
largest source tensor is `mlp.fc1` at ~308 MB in BF16. Worth measuring on the first real run
rather than assuming.

---

## Reference sources

- TenStrip/10Eros-Max — <https://huggingface.co/TenStrip/10Eros-Max>
- Comfy Kitchen W4A8 design — <https://github.com/Comfy-Org/comfy-kitchen/pull/90>
- ComfyUI MiniMax H3 implementation — <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py>
- ComfyUI quantization registry (`QUANT_ALGOS`, `int8_tensorwise`) — `comfy/quant_ops.py`
- ComfyUI quantized-layer loader — `comfy/ops.py`
- This repository — `h3converter/constants.py`, `docs/COMPATIBILITY.md`
