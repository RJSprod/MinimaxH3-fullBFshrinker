# Pinned compatibility matrix and verification status

## Pinned versions

Everything is pinned exactly in `pyproject.toml`. A release must be reproducible months later, so
nothing floats.

| Component | Pin | Why this one |
|---|---|---|
| Python | `3.12.*` (`.python-version`) | uv-managed; the tested CPython line |
| uv | `0.12.3` (`start_windows.bat`) | pinned standalone release, downloaded to `installer_files\uv\` |
| torch | `2.13.0` from `https://download.pytorch.org/whl/cu130` | CUDA 13 is required — see below |
| comfy-kitchen | `0.2.31` | first PyPI wheel confirmed to ship `AsymW4A8Int8Layout` |
| safetensors | `0.8.0` | |
| PySide6 | `6.11.1` | native GUI |
| numpy | `2.5.2` | |
| psutil | `7.2.2` | RAM and peak-usage instrumentation |
| packaging | `26.3` | |
| pytest | `9.1.1` (dev extra) | |

### Why CUDA 13

ComfyUI's `comfy/quant_ops.py` disables comfy-kitchen's accelerated CUDA backend when
`torch.version.cuda < 13`:

> *"You need pytorch with cu130 or higher to use optimized CUDA operations."*

The converter mirrors that gate (`h3converter/quant/capability.py::import_kitchen`) so it exercises
the same code path the target runtime will, rather than a faster one that might round differently.
The cu130 wheels bundle the CUDA runtime, so no separate CUDA Toolkit install is needed — only a
driver new enough for CUDA 13. **The application never installs or updates a driver.**

### W4A8 availability — resolved

The design document flagged a risk that the public comfy-kitchen wheel might not contain
`AsymW4A8Int8Layout`, and listed a fork or a vendored implementation as fallbacks. **None of that is
needed.** The stock PyPI wheel `comfy-kitchen==0.2.31` ships:

- `comfy_kitchen/tensor/w4a8_int8.py` — `AsymW4A8Int8Layout`
- `comfy_kitchen/backends/eager/w4a8_int8.py` — a pure-PyTorch implementation
- `comfy_kitchen/backends/triton/w4a8_int8.py` — the accelerated path

The eager backend means quantization runs correctly with or without CUDA. `h3convert --check`
verifies this on the actual installed build and refuses Option A if it is ever absent, rather than
silently emitting W4A4 or INT8.

Upstream provenance: comfy-kitchen PR #90 (`AsymW4A8Int8Layout`) and ComfyUI PR #15308
(`asym_w4a8_int8` in the quantization registry).

### Target runtime

| Format | Runtime requirement |
|---|---|
| A — `asym_w4a8_int8` | ComfyUI with `asym_w4a8_int8` in `QUANT_ALGOS` (ComfyUI ≥ v0.31.0) plus a comfy-kitchen exposing `AsymW4A8Int8Layout`. Weight decode needs sm_80+. |
| B — `nvfp4` | Stock ComfyUI NVFP4 path. The accelerated matmul needs sm_100+ (Blackwell). |

Record the exact ComfyUI commit used for release validation here when the runtime test is run.

---

## Contract sources

Every on-disk detail is taken from the target runtime's own source rather than inferred:

| What | Source |
|---|---|
| H3 module tree, AdaLN fan-out, curve interpolation, SiLU placement | `comfy/ldm/minimax/model.py` |
| H3 detection signature and geometry derivation | `comfy/model_detection.py` |
| Per-layer key suffixes and load-time expectations | `comfy/ops.py` (`asym_w4a8_int8` / `nvfp4` branches) |
| `QUANT_ALGOS` registry, storage dtypes | `comfy/quant_ops.py` |
| `_quantization_metadata` schema | `comfy/utils.py::convert_old_quants` |
| Quantize/dequantize numerics, ConvRot, codebook, packing | `comfy_kitchen/tensor/w4a8_int8.py`, `comfy_kitchen/backends/eager/` |

---

## Verification status

### Verified in this repository (`uv run pytest` — 166 tests)

- The golden reference census — 1,132 tensors, 200/200/322/410 by dtype, 12,540,714,592 data bytes
  — is reproduced analytically from the architecture plus the policy, by the production code path.
- The 66 GB source size is reproduced from the same model of the architecture.
- Exactly 200 layers are selected for a 50-block H3; AdaLN projections, the token refiner,
  `condition_proj` and the fp32 heads are never selected.
- Group-scale arithmetic (`N·K / 16`) holds for every reference layer, and packed storage recovers
  the logical width.
- The timestep embedding and the table interpolation match independent transcriptions of
  `comfy/ldm/minimax/model.py`.
- The rank-8 / 1025-point curve reproduces the modulation output to < 1e-4 relative on the fixture,
  measured at grid-cell midpoints; the abort gate fires on a bad basis.
- W4A8 round-trip relative L2 lands at ~0.073, matching comfy-kitchen's documented figure.
- Both formats convert end to end; the output loads with the reference safetensors library.
- 13 deliberate corruptions of a real output — missing scale, missing codebook, wrong scale count,
  wrong/absent metadata, `convrot` disabled, leftover time embedder, wrong table shape, downcast
  fp32 island, NaN scales, garbage file — are each caught by validation.
- Data safety: source unmodified, no surviving `.partial`, no clobbered output, cancellation leaves
  nothing behind, converted files are refused as input.
- The GUI constructs and drives all four screens (verified offscreen).

### Requires the target machine — not verifiable in this environment

| Item | Why | How to close it |
|---|---|---|
| `uv.lock` | `download.pytorch.org` is unreachable from the build sandbox. The rest of the dependency set resolves cleanly (44 packages). | Run `uv lock` once on a machine with normal network access, commit the result. `start_windows.bat` already falls back to `uv sync` when the lock is absent. |
| Real 66 GB conversion | No H3 checkpoint or 66 GB of disk here | Run `h3convert <source> --format a --json` and compare `validation.conformance` against the reference row |
| RTX 5090 / CUDA 13 path | No GPU in this environment; all quantization was exercised through comfy-kitchen's eager backend | `h3convert --check` on the target machine |
| Peak RAM / VRAM against the 96 GB / 32 GB targets | Needs the real model | The report records measured peaks in `resources` |
| ComfyUI load + generation smoke test (design §29) | Needs the runtime and the real checkpoint | Load both outputs, confirm 200 W4A8 linears load and ConvRot runs, then a short generation |
| Quality benchmarks (design §30) | Needs generation | BF16 vs pruned-BF16 vs A vs B at identical prompts, seeds, frame counts, steps and flow shifts |

Nothing in the second table is blocked by the code; each is a measurement that needs hardware and a
real checkpoint.

---

## Known assumptions

**The AdaLN basis is not mean-centred by default.** Centring would buy one extra deviation
direction, but it moves the entire mean modulation into a BF16 bias vector where a ~0.4% relative
rounding error lands on the dominant term — and it would change the BF16/F32 tensor census away
from the reference. The uncentred basis leaves source biases untouched. `--center-basis` enables
the alternative for experiments; both truncation errors are always reported.

**Reduced AdaLN projections are written as BF16.** This matches the reference artifact. ComfyUI
constructs those modules as fp32 in curve mode and casts on load, so F32 would also be loadable, at
double the size for that tensor family.

**The output size band is advisory, not a gate.** A fine-tune with a different token-refiner depth
legitimately shifts the total. Size is reported in `validation.conformance`, never used to fail a
conversion.
