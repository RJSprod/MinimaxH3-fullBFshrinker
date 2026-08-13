from pathlib import Path

from h3converter.krea2_detect import detect
from h3converter.krea2_policy import build_output_plan, is_target
from h3converter.safetensor_io import Header, TensorInfo

def _header(prefix=""):
    specs = {
        "first.weight": (6144, 64),
        "last.linear.weight": (64, 6144),
        "blocks.0.attn.wq.weight": (6144, 6144),
        "blocks.0.attn.wk.weight": (1536, 6144),
        "blocks.0.mlp.w1.weight": (16384, 6144),
        "txtfusion.projector.weight": (1, 12),
        "txtfusion.layerwise_blocks.0.mlp.w1.weight": (6912, 2560),
    }
    for i in range(28): specs.setdefault(f"blocks.{i}.attn.wq.weight", (6144, 6144))
    for i in range(2):
        specs[f"txtfusion.layerwise_blocks.{i}.prenorm.scale"] = (2560,)
        specs[f"txtfusion.refiner_blocks.{i}.prenorm.scale"] = (2560,)
    tensors = {}
    cursor = 0
    for key, shape in specs.items():
        size = 2
        for d in shape: size *= d
        tensors[prefix + key] = TensorInfo(prefix + key, "BF16", shape, cursor, cursor + size)
        cursor += size
    return Header(Path("synthetic.safetensors"), tensors, {}, 0, cursor)

def test_detects_cronos_mapped_geometry_with_common_prefix():
    result = detect(_header("model.diffusion_model."))
    assert result.is_krea2 and result.convertible
    assert result.prefix == "model.diffusion_model."
    assert result.geometry.blocks == 28
    assert result.geometry.text_layers == 12
    assert result.geometry.layerwise_text_blocks == 2
    assert result.geometry.refiner_text_blocks == 2
    assert result.geometry.channels == 16
    assert result.geometry.patch_size == 2
    assert result.geometry.attention_heads == 48
    assert result.geometry.kv_heads == 12
    assert result.geometry.mlp_hidden_dim == 16384
    assert result.geometry.text_mlp_hidden_dim == 6912
    assert result.float_dtype == "BF16"

def test_rejects_quantized_krea_source():
    header = _header()
    header.metadata["_quantization_metadata"] = "{}"
    result = detect(header)
    assert result.is_krea2 and result.already_quantized and not result.convertible

def test_policy_is_allow_list_not_all_matrices():
    header = _header()
    plan = build_output_plan(header)
    assert is_target("blocks.1.attn.wq.weight")
    assert not is_target("txtfusion.projector.weight")
    assert "txtfusion.projector.weight" in plan.passthrough_keys
    assert all(t.layer.startswith("blocks.") for t in plan.quant_targets)
