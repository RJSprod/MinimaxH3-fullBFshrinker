"""Direct-read and memory-mapped source access must agree exactly."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from h3converter.safetensor_io import (
    SafetensorsError,
    SourceReader,
    read_header,
    tensor_bytes,
)

TENSORS = {
    "f32": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    "bf16": (torch.randn(8, 6) * 4).to(torch.bfloat16),
    "i8": torch.arange(-8, 8, dtype=torch.int8).reshape(4, 4),
    "u8": torch.arange(0, 16, dtype=torch.uint8).reshape(2, 8),
    "fp8": (torch.randn(4, 8) * 2).to(torch.float8_e4m3fn),
    "scalar": torch.tensor(2.5, dtype=torch.float32),
    "vector": torch.linspace(0, 1, 17, dtype=torch.float32),
    "empty": torch.zeros(0, 4, dtype=torch.float32),
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("reader") / "sample.safetensors"
    save_file(TENSORS, str(path), metadata={"note": "reader test"})
    return path


def test_direct_reads_match_the_written_tensors(checkpoint):
    with SourceReader(checkpoint) as reader:
        assert not reader.mmap
        for name, expected in TENSORS.items():
            actual = reader.get(name)
            assert actual.dtype == expected.dtype, name
            assert actual.shape == expected.shape, name
            assert bytes(tensor_bytes(actual)) == bytes(tensor_bytes(expected)), name


def test_direct_reads_are_byte_identical_to_the_memory_mapped_path(checkpoint):
    with SourceReader(checkpoint, mmap=False) as direct, \
         SourceReader(checkpoint, mmap=True) as mapped:
        for name in TENSORS:
            a, b = direct.get(name), mapped.get(name)
            assert a.dtype == b.dtype, name
            assert a.shape == b.shape, name
            assert bytes(tensor_bytes(a)) == bytes(tensor_bytes(b)), name


def test_reader_exposes_keys_and_metadata(checkpoint):
    with SourceReader(checkpoint) as reader:
        assert set(reader.keys()) == set(TENSORS)
        assert reader.metadata()["note"] == "reader test"


def test_reader_reuses_a_header_it_is_given(checkpoint):
    header = read_header(checkpoint)
    with SourceReader(checkpoint, header=header) as reader:
        assert reader.get("f32").shape == (3, 4)


def test_unknown_tensor_is_a_clear_error(checkpoint):
    with SourceReader(checkpoint) as reader:
        with pytest.raises(SafetensorsError, match="no tensor named"):
            reader.get("nope")


def test_truncated_source_is_reported_not_silently_short(checkpoint, tmp_path):
    """A short read must raise, never hand back a half-filled tensor."""
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(checkpoint.read_bytes()[:-32])

    header = read_header(checkpoint)  # the intact header, so offsets look valid
    with SourceReader(truncated, header=header) as reader:
        with pytest.raises(SafetensorsError, match="unexpected end of file"):
            for name in TENSORS:
                reader.get(name)


def test_close_is_idempotent(checkpoint):
    reader = SourceReader(checkpoint)
    reader.close()
    reader.close()


def test_large_reads_are_assembled_from_short_reads(tmp_path):
    """readinto can return a short count; the loop must keep going."""
    path = tmp_path / "big.safetensors"
    payload = torch.randn(400_000, dtype=torch.float32)
    save_file({"big": payload}, str(path))

    with SourceReader(path) as reader:
        restored = reader.get("big")
    assert torch.equal(restored, payload)
