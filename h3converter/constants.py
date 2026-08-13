"""Fixed facts about MiniMax H3, the two output formats, and this application.

Everything in this module is a *contract*, not a tuning knob. The values were
derived from, and cross-checked against, three independent sources:

  1. ComfyUI's H3 implementation (``comfy/ldm/minimax/model.py``) and its
     architecture detector (``comfy/model_detection.py``), which define what a
     loadable H3 checkpoint must contain.
  2. ComfyUI's quantized-layer loader (``comfy/ops.py``) and quantization
     registry (``comfy/quant_ops.py``), which define the on-disk key names and
     metadata schema for ``asym_w4a8_int8`` and ``nvfp4``.
  3. The observed golden reference artifact
     ``10Eros_Max_h3_fl2va_test4_pruned-w4a8_convrot.safetensors``
     (12,540,857,840 bytes, 1,132 tensors).

Source (1) + (2) predict source (3) exactly -- see ``REFERENCE_INVENTORY`` and
``tests/unit/test_reference_conformance.py``. That agreement is what makes the
Option A target verifiable rather than guessed.
"""

from __future__ import annotations

APP_NAME = "MiniMax H3 Checkpoint Converter"
APP_VERSION = "1.0.0"
CONVERTER_ID = "minimax-h3-converter"

# Versioned policies. Bump these whenever the layer selection or the numerical
# recipe changes, so a report always identifies exactly what produced a file.
W4A8_POLICY_VERSION = "h3_w4a8_policy_v1"
NVFP4_POLICY_VERSION = "h3_nvfp4_policy_v1"
ADALN_PRUNE_VERSION = "h3_adaln_curve_v1"

# --------------------------------------------------------------------------
# Output format identifiers (user-facing labels and internal keys)
# --------------------------------------------------------------------------
FORMAT_W4A8 = "w4a8_convrot"
FORMAT_NVFP4 = "nvfp4"
FORMAT_KREA2_FP8 = "krea2_fp8_scaled"

FORMAT_LABELS = {
    FORMAT_W4A8: "AdaLN-pruned W4A8 ConvRot",
    FORMAT_NVFP4: "AdaLN-pruned NVFP4",
    FORMAT_KREA2_FP8: "FP8 Scaled (Forge Neo / ComfyUI)",
}

# Filename infixes used when naming the output beside the source.
FORMAT_FILENAME_INFIX = {
    FORMAT_W4A8: "pruned_w4a8_convrot",
    FORMAT_NVFP4: "pruned_nvfp4",
    FORMAT_KREA2_FP8: "fp8_scaled",
}

KREA2_FP8_POLICY_VERSION = "krea2_fp8_policy_v1"

# --------------------------------------------------------------------------
# Quantization contract (ComfyUI comfy/quant_ops.py + comfy/ops.py)
# --------------------------------------------------------------------------

# Value of the per-layer "format" field in the checkpoint metadata.
QUANT_FORMAT_W4A8 = "asym_w4a8_int8"
QUANT_FORMAT_NVFP4 = "nvfp4"

# safetensors __metadata__ key holding the JSON quantization descriptor.
QUANT_METADATA_KEY = "_quantization_metadata"
QUANT_METADATA_FORMAT_VERSION = "1.0"

# W4A8: group and rotation geometry. Both are part of the stored per-layer
# config, so a mismatch here silently produces a checkpoint the runtime
# decodes incorrectly. They are never user-adjustable.
W4A8_GROUP_SIZE = 16
W4A8_CONVROT_GROUPSIZE = 256
W4A8_CODEBOOK_ENTRIES = 16

# Per-layer tensor key suffixes, matching comfy_kitchen
# AsymW4A8Int8Layout.state_dict_tensors() and the ComfyUI loader.
W4A8_SUFFIX_WEIGHT = ""            # int8, [N, K // 2]  (two 4-bit codes per byte)
W4A8_SUFFIX_S_REL = "_s_rel"       # float8_e4m3fn, [N, K // group_size]
W4A8_SUFFIX_S_CHANNEL = "_s_channel"  # float32, [N]
W4A8_SUFFIX_CODEBOOK = "_codebook"    # float32, [16]
W4A8_SUFFIX_CORRECTION = "_correction"  # float32, [groups, N] (asymmetric only)

# NVFP4 group size and key suffixes (TensorCoreNVFP4Layout).
NVFP4_GROUP_SIZE = 16
NVFP4_SUFFIX_WEIGHT = ""           # uint8, [N, K // 2]
NVFP4_SUFFIX_SCALE = "_scale"      # float8_e4m3fn block scales (swizzled)
NVFP4_SUFFIX_SCALE_2 = "_scale_2"  # float32 scalar, per-tensor global scale
NVFP4_SUFFIX_INPUT_SCALE = "_input_scale"  # float32 scalar, optional (calibrated)

# --------------------------------------------------------------------------
# H3 AdaLN curve form (comfy/ldm/minimax/model.py)
# --------------------------------------------------------------------------
# The runtime reads the reduced representation as:
#
#     pos   = clamp(t, 0, 1) * (grid - 1)
#     i0    = min(floor(pos), grid - 2)
#     c(t)  = lerp(table[i0], table[i0 + 1], pos - i0)      # [rank]
#     out   = adaln_proj.linear(c(t))                        # no SiLU
#
# whereas the full form computes:
#
#     out   = adaln_proj.linear(silu(time_embedder(t)))      # SiLU applied
#
# Grid and rank are fixed by the golden target geometry and are not options.
ADALN_CURVE_GRID = 1025
ADALN_CURVE_RANK = 8
ADALN_TABLE_KEY = "adaln_t_table"

# AdaLN projection fan-out, from AdalnProj(t_dim, hidden, expand, modalities):
#   DiT block  : expand=6 (shift/scale/gate x2), modalities=3 -> 18 * hidden
#   Final layer: expand=2 (shift/scale),         modalities=1 ->  2 * hidden
ADALN_BLOCK_EXPAND = 6
ADALN_BLOCK_MODALITIES = 3
ADALN_FINAL_EXPAND = 2
ADALN_FINAL_MODALITIES = 1

# The timestep frequency embedding inside TimeEmbedder.forward:
#   half  = freq_dim // 2
#   freqs = exp(-log(10000) * arange(half) / half)
#   emb   = cat([cos(t * freqs), sin(t * freqs)])     # cos BEFORE sin
#   e(t)  = proj_out(silu(proj_in(emb)))
TIME_EMBED_MAX_PERIOD = 10000.0
TIME_DOMAIN_MIN = 0.0
TIME_DOMAIN_MAX = 1.0

# --------------------------------------------------------------------------
# H3 tensor key names
# --------------------------------------------------------------------------
KEY_VIDEO_PATCH_PROJ = "video_patch_proj"
KEY_AUDIO_PATCH_PROJ = "audio_patch_proj"
KEY_CONDITION_PROJ = "condition_proj"
KEY_TIME_EMBEDDER = "time_embedder"
KEY_TIME_PROJ_IN = "time_embedder.proj_in"
KEY_TIME_PROJ_OUT = "time_embedder.proj_out"
KEY_ROPE_INV_FREQ = "rope.inv_freq"
KEY_FINAL_VIDEO_OUT = "final_layer.video_out"
KEY_FINAL_AUDIO_OUT = "final_layer.audio_out"
KEY_FINAL_ADALN = "final_layer.adaln_proj.linear"
KEY_BLOCK_PREFIX = "blocks."
KEY_TOKEN_REFINER_PREFIX = "token_refiner."

# The four GEMM families quantized per DiT block for Option A.
# 50 blocks x 4 = exactly 200 quantized layers.
W4A8_BLOCK_LINEARS = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)

# Layers that must stay in their reference precision island under both options.
FP32_PRESERVED_PREFIXES = (
    KEY_VIDEO_PATCH_PROJ,
    KEY_AUDIO_PATCH_PROJ,
    KEY_FINAL_VIDEO_OUT,
    KEY_FINAL_AUDIO_OUT,
    "rope.inv_freq",
)

# Layers that are 2D and large enough to look quantizable but must not be
# touched, under either option.
NEVER_QUANTIZE_SUBSTRINGS = (
    "adaln_proj",
    KEY_TOKEN_REFINER_PREFIX,
    KEY_CONDITION_PROJ,
    KEY_TIME_EMBEDDER,
    KEY_VIDEO_PATCH_PROJ,
    KEY_AUDIO_PATCH_PROJ,
    KEY_FINAL_VIDEO_OUT,
    KEY_FINAL_AUDIO_OUT,
)

# --------------------------------------------------------------------------
# Expected source geometry
# --------------------------------------------------------------------------
# Defaults from MiniMaxH3Model.__init__. Detection reads the real values from
# the checkpoint; these are the reference point a source is reported against,
# and the basis of the "materially different architecture" refusal.
H3_REFERENCE_GEOMETRY = {
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "rope_inv_freq_len": 16,
}

# A source is accepted only within these bounds. H3 fine-tunes vary in weights,
# not in shape; anything outside this is a different architecture.
H3_MIN_BLOCKS = 40
H3_MAX_BLOCKS = 64

# --------------------------------------------------------------------------
# Golden Option A reference inventory
# --------------------------------------------------------------------------
# Derived analytically from the H3 reference geometry plus the storage
# contract, then checked against the observed reference file. Every figure
# below is reproduced by ``h3converter.validate.predict_inventory``.
#
#   I8      200 tensors   9,633,792,000 elements   (packed 4-bit weights)
#   F8_E4M3 200 tensors   1,204,224,000 elements   (per-group scales, K/16)
#   F32     410 tensors       4,444,952 elements   (200 s_channel + 200
#                                                   codebooks + 10 islands)
#   BF16    322 tensors     842,459,392 elements   (norms, biases, AdaLN,
#                                                   token_refiner, cond_proj)
#   total  1132 tensors  12,540,714,592 data bytes + ~143 KB header
#                        = 12,540,857,840 observed file bytes
REFERENCE_INVENTORY = {
    "file_bytes": 12_540_857_840,
    "tensor_count": 1132,
    "quantized_layer_count": 200,
    "dtype_counts": {"I8": 200, "F8_E4M3": 200, "BF16": 322, "F32": 410},
    "dtype_elements": {
        "I8": 9_633_792_000,
        "F8_E4M3": 1_204_224_000,
        "BF16": 842_459_392,
        "F32": 4_444_952,
    },
    "adaln_table_shape": (ADALN_CURVE_GRID, ADALN_CURVE_RANK),
    "block_adaln_shape": (96768, ADALN_CURVE_RANK),
    "final_adaln_shape": (10752, ADALN_CURVE_RANK),
}

# Acceptable output size band for the conformance report (not a hard failure;
# fine-tunes with a different token_refiner depth legitimately shift this).
OPTION_A_SIZE_BAND_BYTES = (11_000_000_000, 14_000_000_000)
OPTION_B_SIZE_BAND_BYTES = (11_000_000_000, 15_000_000_000)

# --------------------------------------------------------------------------
# Pruning quality gates
# --------------------------------------------------------------------------
# Relative L2 error of the reconstructed AdaLN modulation output, measured at
# off-grid timesteps against the exact full-precision path. Exceeding the abort
# threshold stops the conversion before anything destructive happens.
ADALN_REL_ERROR_WARN = 5e-3
ADALN_REL_ERROR_ABORT = 2.5e-2

# --------------------------------------------------------------------------
# Disk space policy
# --------------------------------------------------------------------------
# The source is read in place and never copied, and there is no large
# intermediate file, so the requirement is driven by the planned output size
# plus working headroom -- not by the 66 GB input.
#
# The multipliers are set so a reference-sized 12.54 GB output reproduces the
# figures in the design spec: 25 GB required / 40 GB preferred for Option A,
# 35 GB / 50 GB for Option B. They scale down honestly for smaller models
# instead of demanding 25 GB to write a 15 MB file.
DISK_REQUIRED_MULTIPLIER = {
    FORMAT_W4A8: 2.0,
    FORMAT_NVFP4: 2.8,
    FORMAT_KREA2_FP8: 1.2,
}
DISK_PREFERRED_MULTIPLIER = {
    FORMAT_W4A8: 3.2,
    FORMAT_NVFP4: 4.0,
    FORMAT_KREA2_FP8: 1.5,
}
# Absolute working headroom on top of the output itself, for the log, the
# report and filesystem overhead.
DISK_WORKING_RESERVE_BYTES = 1 * 1000**3

# Reference-sized expectations, quoted in documentation and the GUI.
DISK_REFERENCE_OUTPUT_BYTES = 12_540_857_840

# --------------------------------------------------------------------------
# Hardware policy
# --------------------------------------------------------------------------
# Blackwell (sm_100/sm_120) is required for the accelerated NVFP4 matmul path;
# W4A8 runs the INT8 GEMM and needs only sm_80+. The converter itself can quantize
# on CPU, but a checkpoint is only *validated* on hardware that can run it.
MIN_COMPUTE_CAPABILITY_W4A8 = (8, 0)
MIN_COMPUTE_CAPABILITY_NVFP4 = (10, 0)
REQUIRED_TORCH_CUDA_MAJOR = 13

# Peak resource targets on the reference machine (96 GB RAM / 32 GB VRAM).
TARGET_PEAK_RAM_BYTES = 80 * 1000**3
TARGET_PEAK_VRAM_BYTES = 28 * 1000**3

# Progress weighting per phase. Values must sum to 1.0 per format.
PHASE_WEIGHTS = {
    FORMAT_W4A8: {
        "inspect": 0.05,
        "adaln_basis": 0.10,
        "adaln_collapse": 0.20,
        "quantize": 0.60,
        "finalize": 0.05,
    },
    FORMAT_NVFP4: {
        "inspect": 0.05,
        "adaln_basis": 0.10,
        "adaln_collapse": 0.20,
        "calibrate": 0.10,
        "quantize": 0.50,
        "finalize": 0.05,
    },
    FORMAT_KREA2_FP8: {"inspect": 0.05, "quantize": 0.90, "finalize": 0.05},
}
