"""Header parsing, planned streaming writes, and atomic finalisation."""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from h3converter.safetensor_io import (
    PlannedTensor,
    PlannedWriter,
    SafetensorsError,
    finalize,
    read_header,
    tensor_bytes,
)


def _write_simple(path):
    tensors = {
        "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "b": torch.ones(8, dtype=torch.bfloat16),
        "c": torch.zeros(2, 2, dtype=torch.int8),
    }
    save_file(tensors, str(path), metadata={"note": "hello"})
    return tensors


# ---------------------------------------------------------------------------
# Header reading
# ---------------------------------------------------------------------------

def test_reads_dtypes_shapes_and_metadata(tmp_path):
    path = tmp_path / "simple.safetensors"
    _write_simple(path)

    header = read_header(path)
    assert set(header.tensors) == {"a", "b", "c"}
    assert header.tensors["a"].dtype == "F32"
    assert header.tensors["a"].shape == (3, 4)
    assert header.tensors["b"].dtype == "BF16"
    assert header.metadata["note"] == "hello"
    assert header.dtype_counts() == {"F32": 1, "BF16": 1, "I8": 1}
    assert header.dtype_elements() == {"F32": 12, "BF16": 8, "I8": 4}


def test_header_read_does_not_depend_on_tensor_data(tmp_path):
    """A truncated data region is still parseable up to the point it is wrong."""
    path = tmp_path / "simple.safetensors"
    _write_simple(path)
    original = path.read_bytes()

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(original[:-4])
    with pytest.raises(SafetensorsError, match="outside the"):
        read_header(truncated)


def test_rejects_implausible_header_length(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40) + b"{}")
    with pytest.raises(SafetensorsError, match="implausible header length"):
        read_header(path)


def test_rejects_non_json_header(tmp_path):
    body = b"not json at all"
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(SafetensorsError, match="not valid JSON"):
        read_header(path)


def test_rejects_inconsistent_offsets(tmp_path):
    header = {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 12]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 12)
    with pytest.raises(SafetensorsError, match="needs 16"):
        read_header(path)


def test_rejects_unknown_dtype(tmp_path):
    header = {"a": {"dtype": "F4_MYSTERY", "shape": [4], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
    with pytest.raises(SafetensorsError, match="unsupported dtype"):
        read_header(path)


# ---------------------------------------------------------------------------
# Byte-exact serialisation of exotic dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.int8,
                                   torch.uint8, torch.float32])
def test_tensor_bytes_round_trips_every_stored_dtype(dtype):
    source = (torch.randn(6, 4) * 3).to(dtype)
    payload = bytes(tensor_bytes(source))
    assert len(payload) == source.numel() * source.element_size()

    restored = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(source.shape)
    assert torch.equal(restored.view(torch.uint8), source.view(torch.uint8))


# ---------------------------------------------------------------------------
# Planned writing
# ---------------------------------------------------------------------------

def test_planned_writer_output_loads_with_the_reference_library(tmp_path):
    values = {
        "x": torch.randn(4, 6).to(torch.bfloat16),
        "y": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "s": torch.tensor(2.5, dtype=torch.float32),
    }
    plan = [PlannedTensor.of(name, t.dtype, t.shape) for name, t in values.items()]

    path = tmp_path / "planned.safetensors"
    with PlannedWriter(path, plan, {"converted_by": "test"}) as writer:
        # Deliberately out of plan order: payloads seek to their planned offset.
        for name in ("s", "y", "x"):
            writer.write(name, values[name])

    restored = load_file(str(path))
    assert set(restored) == set(values)
    for name, tensor in values.items():
        # Compare raw bytes so bfloat16 and the 0-dim scalar are covered too.
        assert bytes(tensor_bytes(restored[name])) == bytes(tensor_bytes(tensor))
        assert restored[name].shape == tensor.shape
        assert restored[name].dtype == tensor.dtype
    assert read_header(path).metadata["converted_by"] == "test"


def test_planned_writer_refuses_a_dtype_or_shape_that_is_not_planned(tmp_path):
    plan = [PlannedTensor.of("x", torch.bfloat16, (4, 4))]
    writer = PlannedWriter(tmp_path / "p.safetensors", plan)
    writer.open()
    with pytest.raises(SafetensorsError, match="planned dtype"):
        writer.write("x", torch.zeros(4, 4, dtype=torch.float32))
    with pytest.raises(SafetensorsError, match="planned shape"):
        writer.write("x", torch.zeros(2, 4, dtype=torch.bfloat16))
    with pytest.raises(SafetensorsError, match="not in the output plan"):
        writer.write("z", torch.zeros(4, 4, dtype=torch.bfloat16))
    writer.abort()


def test_incomplete_output_is_rejected_and_deleted(tmp_path):
    """A conversion that stops early must not leave a loadable-looking file."""
    plan = [
        PlannedTensor.of("x", torch.bfloat16, (4, 4)),
        PlannedTensor.of("y", torch.bfloat16, (4, 4)),
    ]
    path = tmp_path / "incomplete.safetensors"
    writer = PlannedWriter(path, plan)
    writer.open()
    writer.write("x", torch.zeros(4, 4, dtype=torch.bfloat16))
    assert writer.missing == ["y"]

    with pytest.raises(SafetensorsError, match="incomplete"):
        writer.close()
    assert not path.exists()


def test_duplicate_plan_entries_are_rejected(tmp_path):
    plan = [PlannedTensor.of("x", torch.bfloat16, (2,)), PlannedTensor.of("x", torch.bfloat16, (2,))]
    with pytest.raises(SafetensorsError, match="duplicate tensor"):
        PlannedWriter(tmp_path / "d.safetensors", plan)


def test_abort_removes_the_partial_file(tmp_path):
    path = tmp_path / "aborted.safetensors"
    writer = PlannedWriter(path, [PlannedTensor.of("x", torch.bfloat16, (4,))])
    writer.open()
    assert path.exists()
    writer.abort()
    assert not path.exists()


def test_exception_inside_the_context_manager_aborts(tmp_path):
    path = tmp_path / "boom.safetensors"
    plan = [PlannedTensor.of("x", torch.bfloat16, (4,)), PlannedTensor.of("y", torch.bfloat16, (4,))]
    with pytest.raises(RuntimeError, match="simulated"):
        with PlannedWriter(path, plan) as writer:
            writer.write("x", torch.zeros(4, dtype=torch.bfloat16))
            raise RuntimeError("simulated failure")
    assert not path.exists()


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

def test_finalize_is_an_atomic_rename(tmp_path):
    partial = tmp_path / "out.safetensors.partial"
    final = tmp_path / "out.safetensors"
    partial.write_bytes(b"payload")

    finalize(partial, final)
    assert final.read_bytes() == b"payload"
    assert not partial.exists()


def test_finalize_never_overwrites_an_existing_output(tmp_path):
    partial = tmp_path / "out.safetensors.partial"
    final = tmp_path / "out.safetensors"
    partial.write_bytes(b"new")
    final.write_bytes(b"existing")

    with pytest.raises(SafetensorsError, match="refusing to overwrite"):
        finalize(partial, final)
    assert final.read_bytes() == b"existing"
