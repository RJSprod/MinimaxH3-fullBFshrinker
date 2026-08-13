"""safetensors reading and streaming writing.

Two things matter here:

* **Header-only inspection.** A 66 GB source must be identified, validated and
  sized up without reading tensor data. ``read_header`` parses just the JSON
  header at the front of the file.

* **Planned streaming writes.** The output is written to ``<final>.partial``
  with the complete tensor inventory declared up front, so the header can be
  emitted first and every payload streamed into its final offset. That avoids
  both a full in-memory state dict and any large intermediate file. The final
  name only appears after validation, via an atomic rename.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

# safetensors header size prefix: little-endian u64
_HEADER_LEN_BYTES = 8
# Refuse absurd headers before allocating. Real H3 headers are ~150 KB.
_MAX_HEADER_BYTES = 256 * 1024 * 1024

ST_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

TORCH_TO_ST: dict[torch.dtype, str] = {v: k for k, v in ST_TO_TORCH.items()}

DTYPE_ITEMSIZE: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


class SafetensorsError(RuntimeError):
    """Raised for malformed or unreadable safetensors input."""


# ---------------------------------------------------------------------------
# Header inspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class Header:
    path: Path
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]
    data_start: int
    file_size: int

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str) -> TensorInfo | None:
        return self.tensors.get(name)

    def shape_of(self, name: str) -> tuple[int, ...] | None:
        info = self.tensors.get(name)
        return info.shape if info else None

    def dtype_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for info in self.tensors.values():
            counts[info.dtype] = counts.get(info.dtype, 0) + 1
        return counts

    def dtype_elements(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for info in self.tensors.values():
            totals[info.dtype] = totals.get(info.dtype, 0) + info.numel
        return totals

    def dtype_bytes(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for info in self.tensors.values():
            totals[info.dtype] = totals.get(info.dtype, 0) + info.nbytes
        return totals


def read_header(path: str | os.PathLike[str]) -> Header:
    """Parse a safetensors header without touching tensor data.

    Also verifies that the declared offsets are internally consistent and fit
    inside the file, which catches truncated or corrupted downloads before a
    multi-hour conversion starts.
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size < _HEADER_LEN_BYTES:
        raise SafetensorsError(f"{path.name}: file is too small to be a safetensors file")

    with open(path, "rb") as fh:
        raw_len = fh.read(_HEADER_LEN_BYTES)
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len == 0 or header_len > _MAX_HEADER_BYTES:
            raise SafetensorsError(f"{path.name}: implausible header length {header_len}")
        if _HEADER_LEN_BYTES + header_len > file_size:
            raise SafetensorsError(f"{path.name}: header extends past end of file (truncated?)")
        raw_header = fh.read(header_len)

    try:
        parsed = json.loads(raw_header)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetensorsError(f"{path.name}: header is not valid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise SafetensorsError(f"{path.name}: header is not a JSON object")

    raw_metadata = parsed.pop("__metadata__", {}) or {}
    if not isinstance(raw_metadata, dict):
        raise SafetensorsError(f"{path.name}: __metadata__ is not an object")
    # Checkpoint metadata is untrusted input: keep it as opaque strings, never
    # evaluate it, never let it name a path or a module.
    metadata = {str(k): str(v) for k, v in raw_metadata.items()}

    data_start = _HEADER_LEN_BYTES + header_len
    available = file_size - data_start

    tensors: dict[str, TensorInfo] = {}
    for name, spec in parsed.items():
        if not isinstance(spec, dict):
            raise SafetensorsError(f"{path.name}: entry {name!r} is not an object")
        try:
            dtype = str(spec["dtype"])
            shape = tuple(int(d) for d in spec["shape"])
            begin, end = (int(x) for x in spec["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(f"{path.name}: malformed entry for {name!r} ({exc})") from exc

        if dtype not in DTYPE_ITEMSIZE:
            raise SafetensorsError(f"{path.name}: unsupported dtype {dtype!r} for {name!r}")
        if begin < 0 or end < begin or end > available:
            raise SafetensorsError(
                f"{path.name}: tensor {name!r} offsets [{begin}, {end}] fall outside the "
                f"{available}-byte data region"
            )

        numel = 1
        for d in shape:
            if d < 0:
                raise SafetensorsError(f"{path.name}: negative dimension in {name!r}")
            numel *= d
        expected = numel * DTYPE_ITEMSIZE[dtype]
        if end - begin != expected:
            raise SafetensorsError(
                f"{path.name}: tensor {name!r} declares {end - begin} bytes but "
                f"{shape} of {dtype} needs {expected}"
            )
        tensors[str(name)] = TensorInfo(str(name), dtype, shape, begin, end)

    return Header(
        path=path,
        tensors=tensors,
        metadata=metadata,
        data_start=data_start,
        file_size=file_size,
    )


# ---------------------------------------------------------------------------
# Streaming reads
# ---------------------------------------------------------------------------

class SourceReader:
    """Reads one tensor at a time from a source checkpoint.

    Two strategies, because the source is ~66 GB:

    ``mmap=False`` (default) reads each tensor with an ordinary seek and read,
    using the byte offsets already parsed from the header. Every tensor is
    visited exactly once by the conversion, so there is nothing for a mapping
    to amortise, and direct reads keep the resident set bounded by one tensor
    instead of letting the OS cache pull the whole file into RAM. It also keeps
    I/O failures as ordinary Python exceptions: on Windows a failed page fault
    against a mapped file raises EXCEPTION_IN_PAGE_ERROR, a structured
    exception that no ``except`` clause can catch and that terminates the
    process with no traceback at all.

    ``mmap=True`` uses ``safetensors.safe_open`` instead, which memory-maps the
    file. Kept for comparison and for callers that want lazy paging.
    """

    def __init__(self, path: str | os.PathLike[str], mmap: bool = False,
                 header: Header | None = None):
        self.path = Path(path)
        self.mmap = mmap
        self._handle = None
        self._fh = None
        self._header = header if header is not None else read_header(self.path)

        if mmap:
            from safetensors import safe_open  # lazy: the header path is torch-free

            self._handle = safe_open(str(self.path), framework="pt", device="cpu")
        else:
            self._fh = open(self.path, "rb", buffering=0)

    def __enter__(self) -> "SourceReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None and hasattr(handle, "__exit__"):
            handle.__exit__(None, None, None)
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def keys(self) -> list[str]:
        return list(self._header.tensors)

    def metadata(self) -> dict[str, str]:
        return dict(self._header.metadata)

    def get(self, name: str) -> torch.Tensor:
        if self._handle is not None:
            return self._handle.get_tensor(name)
        return self._read_tensor(name)

    def _read_tensor(self, name: str) -> torch.Tensor:
        info = self._header.tensors.get(name)
        if info is None:
            raise SafetensorsError(f"{self.path.name}: no tensor named {name!r}")

        dtype = ST_TO_TORCH.get(info.dtype)
        if dtype is None:
            raise SafetensorsError(f"{name}: unsupported dtype {info.dtype}")
        if info.numel == 0:
            return torch.empty(info.shape, dtype=dtype)

        nbytes = info.nbytes
        buffer = torch.empty(nbytes, dtype=torch.uint8)
        view = memoryview(buffer.numpy())

        self._fh.seek(self._header.data_start + info.begin)
        filled = 0
        while filled < nbytes:
            # buffering=0 means read() can return a short count on large reads.
            read = self._fh.readinto(view[filled:])
            if not read:
                raise SafetensorsError(
                    f"{self.path.name}: unexpected end of file reading {name!r} "
                    f"({filled} of {nbytes} bytes); the source may be truncated"
                )
            filled += read

        return buffer.view(dtype).reshape(info.shape)


# ---------------------------------------------------------------------------
# Planned streaming writes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedTensor:
    """One entry of the output inventory, known before any data is produced."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @staticmethod
    def of(name: str, dtype: torch.dtype | str, shape: Sequence[int]) -> "PlannedTensor":
        st = dtype if isinstance(dtype, str) else TORCH_TO_ST.get(dtype)
        if st is None:
            raise SafetensorsError(f"{name}: dtype {dtype} has no safetensors encoding")
        return PlannedTensor(name, st, tuple(int(d) for d in shape))

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_ITEMSIZE[self.dtype]


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    """Raw little-endian payload of ``tensor``, dtype preserved exactly.

    Goes through a flat uint8 view so bfloat16 and the fp8 types -- which numpy
    cannot represent -- serialise byte-for-byte like any other dtype.
    """
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return memoryview(b"")
    return memoryview(flat.view(torch.uint8).numpy())


class PlannedWriter:
    """Writes a safetensors file whose inventory is declared before the data.

    Payloads may be supplied in any order; each is seeked to its planned
    offset. ``close`` refuses to finish while any planned tensor is missing, so
    a partial run can never be renamed into a plausible-looking output.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        plan: Sequence[PlannedTensor],
        metadata: Mapping[str, str] | None = None,
    ):
        self.path = Path(path)
        self._plan = list(plan)
        if not self._plan:
            raise SafetensorsError("output plan is empty")

        seen: set[str] = set()
        for entry in self._plan:
            if entry.name in seen:
                raise SafetensorsError(f"duplicate tensor {entry.name!r} in output plan")
            seen.add(entry.name)

        self._by_name = {e.name: e for e in self._plan}
        self._offsets: dict[str, tuple[int, int]] = {}
        self._written: set[str] = set()
        self._metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        self._fh = None
        self._data_start = 0
        self.bytes_written = 0

        cursor = 0
        for entry in self._plan:
            self._offsets[entry.name] = (cursor, cursor + entry.nbytes)
            cursor += entry.nbytes
        self.data_bytes = cursor

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "PlannedWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def _build_header(self) -> bytes:
        header: dict[str, object] = {}
        if self._metadata:
            header["__metadata__"] = self._metadata
        for entry in self._plan:
            begin, end = self._offsets[entry.name]
            header[entry.name] = {
                "dtype": entry.dtype,
                "shape": list(entry.shape),
                "data_offsets": [begin, end],
            }
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Pad so the data region starts 8-byte aligned, matching the reference
        # serializer. Trailing spaces are valid JSON whitespace.
        pad = (-(_HEADER_LEN_BYTES + len(raw))) % 8
        return raw + b" " * pad

    def open(self) -> None:
        if self._fh is not None:
            raise SafetensorsError("writer is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw_header = self._build_header()
        self._data_start = _HEADER_LEN_BYTES + len(raw_header)

        self._fh = open(self.path, "wb+")
        self._fh.write(struct.pack("<Q", len(raw_header)))
        self._fh.write(raw_header)
        # Size the file up front so a full disk fails here rather than 90% in.
        self._fh.truncate(self._data_start + self.data_bytes)

    def write(self, name: str, tensor: torch.Tensor) -> None:
        """Store one planned tensor. Dtype and shape must match the plan."""
        if self._fh is None:
            raise SafetensorsError("writer is not open")
        entry = self._by_name.get(name)
        if entry is None:
            raise SafetensorsError(f"{name!r} is not in the output plan")
        if name in self._written:
            raise SafetensorsError(f"{name!r} was already written")

        actual_dtype = TORCH_TO_ST.get(tensor.dtype)
        if actual_dtype != entry.dtype:
            raise SafetensorsError(
                f"{name!r}: planned dtype {entry.dtype}, got {actual_dtype or tensor.dtype}"
            )
        if tuple(tensor.shape) != entry.shape:
            raise SafetensorsError(
                f"{name!r}: planned shape {entry.shape}, got {tuple(tensor.shape)}"
            )

        payload = tensor_bytes(tensor)
        begin, end = self._offsets[name]
        if len(payload) != end - begin:
            raise SafetensorsError(
                f"{name!r}: payload is {len(payload)} bytes, plan reserved {end - begin}"
            )
        self._fh.seek(self._data_start + begin)
        self._fh.write(payload)
        self._written.add(name)
        self.bytes_written += len(payload)

    @property
    def missing(self) -> list[str]:
        return [e.name for e in self._plan if e.name not in self._written]

    def close(self) -> None:
        if self._fh is None:
            return
        missing = self.missing
        if missing:
            self.abort()
            preview = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise SafetensorsError(f"output is incomplete: {len(missing)} tensors never written: {preview}{more}")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None

    def abort(self) -> None:
        """Close and discard the partial file. Never raises."""
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


def finalize(partial: Path, final: Path) -> None:
    """Atomically publish a validated ``.partial`` file under its final name."""
    if not partial.exists():
        raise SafetensorsError(f"{partial.name} does not exist; nothing to finalize")
    if final.exists():
        raise SafetensorsError(f"refusing to overwrite existing output {final.name}")
    os.replace(partial, final)


def iter_plan_bytes(plan: Sequence[PlannedTensor]) -> Iterator[tuple[str, int]]:
    for entry in plan:
        yield entry.name, entry.nbytes


def plan_total_bytes(plan: Sequence[PlannedTensor]) -> int:
    return sum(e.nbytes for e in plan)
