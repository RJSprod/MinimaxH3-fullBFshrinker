# MiniMax H3 Checkpoint Converter

Converts a full, unpruned **MiniMax H3 BF16** diffusion checkpoint (~66 GB) into one of two
ComfyUI-compatible ~12.5 GB checkpoints:

| | Output | Format |
|---|---|---|
| **A** | AdaLN-pruned W4A8 ConvRot | `asym_w4a8_int8`, W4 group 16, ConvRot group 256 |
| **B** | AdaLN-pruned NVFP4 | `nvfp4`, group 16 |

The user picks a file and picks A or B. Everything else — environment bootstrap, architecture
detection, AdaLN curve construction, layer policy, quantization, validation — is automatic.

```
start_windows.bat  →  Browse…  →  pick .safetensors  →  [A] or [B]  →  wait  →  done
```

---

## Install and run (Windows)

1. Download or clone this repository.
2. Double-click **`start_windows.bat`**.

The launcher downloads a pinned [uv](https://docs.astral.sh/uv/), installs a pinned CPython 3.12,
creates `.venv\` inside the project, syncs the locked dependency set, runs environment self-tests,
and opens the GUI. No preinstalled Python, no CUDA Toolkit, no manual venv activation.

`update_windows.bat` pulls the latest source, re-syncs the lock and re-runs the self-tests.

### Command line

The GUI is the supported path, but the same pipeline is available headless:

```
uv run h3convert --check                              # environment self-test
uv run h3convert model_bf16.safetensors --inspect     # identify and size up a source
uv run h3convert model_bf16.safetensors --format a    # W4A8 ConvRot
uv run h3convert model_bf16.safetensors --format b    # NVFP4
```

---

## What the conversion actually does

### 1. Identify the source (header only)

The filename is never trusted. A 66 GB file is identified from its safetensors header alone —
tensor names, shapes and dtypes — using the same signature ComfyUI's own detector keys on
(`video_patch_proj` + `audio_patch_proj`). The converter refuses a source that is already
curve-pruned, already quantized, or structurally not H3, and it reports the detected geometry
against the reference:

```
hidden 5376 · 50 blocks · 56×128 heads · ffn 14336 · time embed 2688 · 2 token-refiner blocks
```

### 2. Replace the timestep path with the curve form

A full H3 checkpoint modulates every block through a large timestep path:

```
e(t) = time_embedder(t)                       # [2688]
m(t) = adaln_proj.linear(silu(e(t))) + bias   # [96768] per block
```

The 51 full-width AdaLN projections are `96768 × 2688` (and `10752 × 2688` for the final layer) —
about **26 GB of the 66 GB source**. The curve form replaces them, and the time embedder, with a
shared rank-8 basis of the *same function*:

```
pos  = clamp(t,0,1) · (grid-1)
i0   = min(floor(pos), grid-2)
c(t) = lerp(table[i0], table[i0+1], pos-i0)   # [8], from adaln_t_table [1025, 8]
m(t) = adaln_proj.linear(c(t)) + bias         # no silu in curve mode
```

Finding that basis is a truncated SVD of the sampled curve. With `G = g(t_grid)` factored as
`U S Vᵀ` and `V₈` its leading 8 right singular vectors:

```
adaln_t_table = G V₈        [1025, 8]  F32
W_reduced     = W V₈        [96768, 8] BF16     bias unchanged
```

so `W_reduced @ c(t) == W @ g(t)` up to the rank-8 truncation. This works because the H3 timestep
curve is extremely smooth: the frequency embedding uses periods from 1 down to 1e-4 over
`t ∈ [0,1]`, so every component is a low-order function of `t`.

Both error sources — the SVD truncation and the runtime's linear interpolation between table rows —
are measured directly, at **grid-cell midpoints** (the interpolation's worst case) rather than at
grid nodes, against the exact full-precision path. If the error exceeds the gate the conversion
**aborts before anything destructive happens**.

### 3. Quantize exactly 200 layers

Per DiT block, four GEMM families: `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`.
50 blocks × 4 = **200 layers**. This is an enumerated H3 policy, not an "all large 2D weights"
heuristic — the token refiner uses the *same submodule names* and would be swept in by any
generic rule.

Preserved at reference precision:

| Tensor | Precision |
|---|---|
| `blocks.*.adaln_proj.linear`, `final_layer.adaln_proj.linear` | BF16 (reduced to `[·, 8]`) |
| `token_refiner.*` linears | BF16 |
| `condition_proj` | BF16 |
| `video_patch_proj`, `audio_patch_proj` | F32 |
| `final_layer.video_out`, `final_layer.audio_out` | F32 |
| `adaln_t_table` | F32 |
| norms, biases | source dtype, copied bit-for-bit |

### 4. Stream, validate, then publish

The output inventory is fully determined **before any data is read**, so the header is written
first and every payload streams into its final offset. There is no 40 GB intermediate file and no
in-memory state dict; the largest single working set is one AdaLN projection.

Writes go to `<final>.safetensors.partial`. The file is then re-opened and validated — inventory,
dtypes, shapes, per-layer tensor groups, group-scale arithmetic, metadata schema, finiteness of
every layer's scale data, and sampled dequantization — and only then atomically renamed. **The
source is opened read-only and never modified.** A `<output>.report.json` records everything
needed to audit or reproduce the run.

---

## The storage contract

Both formats are produced by [comfy-kitchen](https://pypi.org/project/comfy-kitchen/)'s layouts
rather than a hand-rolled packer, because the runtime decodes with the mirror image of that exact
code — the ConvRot rotation, the Lloyd-Max codebook decision and the ALS group-scale refinement
all have to match bit for bit.

**Option A** — `asym_w4a8_int8`, per layer of logical shape `[N, K]`:

| Key | dtype | Shape |
|---|---|---|
| `<layer>.weight` | I8 | `[N, K/2]` — two 4-bit codes per byte |
| `<layer>.weight_s_rel` | F8_E4M3 | `[N, K/16]` — one scale per 16 weights |
| `<layer>.weight_s_channel` | F32 | `[N]` |
| `<layer>.weight_codebook` | F32 | `[16]` |

**Option B** — `nvfp4`:

| Key | dtype | Shape |
|---|---|---|
| `<layer>.weight` | U8 | `[N, K/2]` |
| `<layer>.weight_scale` | F8_E4M3 | swizzled `[roundup(N,128), roundup(K/16,4)]` |
| `<layer>.weight_scale_2` | F32 | scalar |
| `<layer>.input_scale` | F32 | scalar, **optional** — see calibration below |

Quantization metadata goes in the safetensors `__metadata__` under `_quantization_metadata`, the
key ComfyUI's `convert_old_quants` reads:

```json
{"format_version": "1.0",
 "layers": {"blocks.0.attn.qkv_proj": {"format": "asym_w4a8_int8",
                                        "group_size": 16,
                                        "convrot_groupsize": 256,
                                        "convrot": true}, ...}}
```

### Option B needs no calibration

ComfyUI's NVFP4 linear reads `input_scale` with `getattr(self, 'input_scale', None)`, and when it
is absent derives a scale from the activation itself (`amax / (448 · 6)`). Omitting the tensor is a
supported first-class mode, not a degraded fallback — and it adapts to whatever resolution,
duration and conditioning the user actually runs. That is how "the user must not perform manual
calibration" is satisfied.

A versioned static-calibration path exists for the throughput case: drop a checksummed pack of
per-layer activation maxima into `calibration_assets/` and the converter bakes `input_scale`
tensors instead. The chosen strategy, pack version and layer coverage are always recorded in the
report. No pack ships with the repository.

---

## Conformance to the golden reference

The reference artifact `10Eros_Max_h3_fl2va_test4_pruned-w4a8_convrot.safetensors` is
**12,540,857,840 bytes / 1,132 tensors**. Its full census is *derived* by this project from the H3
architecture plus the conversion policy, and checked in
`tests/unit/test_reference_conformance.py`:

| dtype | Tensors | Elements | Source |
|---|---|---|---|
| I8 | 200 | 9,633,792,000 | packed 4-bit weights |
| F8_E4M3 | 200 | 1,204,224,000 | group scales, `19,267,584,000 / 16` |
| F32 | 410 | 4,444,952 | 200 `s_channel` + 200 codebooks + 10 fp32 islands |
| BF16 | 322 | 842,459,392 | norms, biases, AdaLN, token refiner, `condition_proj` |
| **total** | **1,132** | | **12,540,714,592 data bytes + ~143 KB header** |

Every figure lands on the observed file. That agreement is what makes the Option A target
verifiable rather than guessed — and if the policy ever selects the wrong layer set or drops a
precision island, the arithmetic stops matching and the tests fail.

---

## Development

```
uv sync --extra dev
uv run pytest                       # 166 tests
uv run pytest tests/unit -q         # fast: no conversions
```

Tests convert a miniature but structurally faithful synthetic H3 checkpoint (40 blocks, hidden 256)
end to end for both formats. The block count is deliberately not shrunk below the detector's
supported range — a test that had to disable the architecture gate would not be testing the
shipping path.

### Layout

```
h3converter/
  constants.py        contracts: geometry, storage keys, thresholds, reference inventory
  safetensor_io.py    header reader, planned streaming writer, atomic finalize
  h3_detect.py        architecture identification from the header
  h3_policy.py        the explicit 200-layer policy and the full output plan
  h3_reference.py     canonical H3 tensor inventory
  adaln_curve.py      timestep curve sampling and rank-8 basis
  adaln_prune.py      projection collapse and the numerical quality gate
  quant/              capability probe, W4A8 and NVFP4 engines
  calibration.py      activation-scale strategy
  pipeline.py         orchestration
  validate.py         post-conversion validation and conformance
  reports.py          checkpoint metadata and report.json
  gui.py / cli.py     interfaces
```

See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for the pinned dependency matrix and the
verification status of each acceptance criterion.
