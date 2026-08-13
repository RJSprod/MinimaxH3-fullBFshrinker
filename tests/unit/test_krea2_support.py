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
        "txtfusion.projector.weight": (1, 12),
        "txtfusion.layerwise_blocks.0.mlp.w1.weight": (6912, 2560),
    }
    for i in range(28):
        for name, shape in {
            "attn.wq": (6144, 6144), "attn.wk": (1536, 6144),
            "attn.wv": (1536, 6144), "attn.wo": (6144, 6144),
            # Mapped checkpoints may rename these. Policy uses namespace+shape.
            "mlp.in_a": (16384, 6144), "mlp.in_b": (16384, 6144),
            "mlp.out": (6144, 16384),
        }.items():
            specs.setdefault(f"blocks.{i}.{name}.weight", shape)
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
    assert is_target("blocks.1.mapped.feed_forward.weight", shape=(16384, 6144))
    assert not is_target("blocks.1.some_other_matrix.weight", shape=(6144, 6144))
    assert not is_target("txtfusion.projector.weight")
    assert "txtfusion.projector.weight" in plan.passthrough_keys
    assert all(t.layer.startswith("blocks.") for t in plan.quant_targets)
    assert plan.quantized_layer_count == 196
    # This fixture is intentionally dominated by the seven block matrices.
    # A regression to attention-only selection (112 layers) cannot pass this.
    source_bytes = sum(info.nbytes for info in header.tensors.values())
    assert plan.total_bytes < source_bytes * 0.6


def test_policy_refuses_attention_only_mostly_bf16_output():
    header = _header()
    tensors = {
        key: info for key, info in header.tensors.items()
        if ".mlp." not in key
    }
    incomplete = Header(header.path, tensors, header.metadata, header.data_start, header.file_size)
    try:
        build_output_plan(incomplete)
    except ValueError as exc:
        assert "Refusing to create a mostly-BF16 output" in str(exc)
    else:
        raise AssertionError("incomplete Krea FP8 inventory was accepted")
