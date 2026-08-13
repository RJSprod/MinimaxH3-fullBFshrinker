"""Real, measurable conversion progress.

Progress is driven by work actually completed -- tensors collapsed, layers
quantized, bytes written -- weighted per phase. There is no timer-based
animation anywhere: a stalled conversion must look stalled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from h3converter import constants as C

PHASE_LABELS = {
    "inspect": "Inspecting source",
    "adaln_basis": "Building AdaLN curve basis",
    "adaln_collapse": "Collapsing AdaLN projections",
    "calibrate": "Resolving activation scales",
    "quantize": "Quantizing transformer blocks",
    "finalize": "Writing and validating output",
}


@dataclass
class ProgressEvent:
    phase: str
    phase_label: str
    overall: float           # 0.0 - 1.0
    phase_fraction: float    # 0.0 - 1.0
    detail: str = ""
    elapsed_seconds: float = 0.0
    bytes_written: int = 0
    peak_ram_bytes: int = 0
    peak_vram_bytes: int = 0


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class ProgressTracker:
    """Weighted multi-phase progress with monotonic overall fraction."""

    weights: dict[str, float]
    callback: ProgressCallback | None = None
    started_at: float = field(default_factory=time.monotonic)
    bytes_written: int = 0
    peak_ram_bytes: int = 0
    peak_vram_bytes: int = 0

    _phase: str = ""
    _phase_total: float = 1.0
    _phase_done: float = 0.0
    _completed_weight: float = 0.0
    _last_overall: float = 0.0

    @classmethod
    def for_format(cls, output_format: str, callback: ProgressCallback | None = None) -> "ProgressTracker":
        return cls(weights=dict(C.PHASE_WEIGHTS[output_format]), callback=callback)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def start_phase(self, phase: str, total_units: float = 1.0, detail: str = "") -> None:
        if phase not in self.weights:
            raise KeyError(f"unknown phase {phase!r} for this format")
        if self._phase and self._phase != phase:
            self._completed_weight += self.weights[self._phase]
        self._phase = phase
        self._phase_total = max(float(total_units), 1e-9)
        self._phase_done = 0.0
        self.emit(detail)

    def advance(self, units: float = 1.0, detail: str = "") -> None:
        self._phase_done = min(self._phase_done + float(units), self._phase_total)
        self.emit(detail)

    def set_phase_progress(self, done: float, detail: str = "") -> None:
        self._phase_done = min(max(float(done), 0.0), self._phase_total)
        self.emit(detail)

    def finish(self, detail: str = "") -> None:
        if self._phase:
            self._completed_weight += self.weights[self._phase]
            self._phase_done = self._phase_total
        self._last_overall = 1.0
        self.emit(detail)

    @property
    def overall(self) -> float:
        if not self._phase:
            return self._last_overall
        fraction = self._phase_done / self._phase_total
        value = self._completed_weight + self.weights[self._phase] * fraction
        # Never move backwards, even if a phase reports a smaller total later.
        self._last_overall = max(self._last_overall, min(value, 1.0))
        return self._last_overall

    def emit(self, detail: str = "") -> None:
        if self.callback is None:
            return
        self.callback(
            ProgressEvent(
                phase=self._phase,
                phase_label=PHASE_LABELS.get(self._phase, self._phase),
                overall=self.overall,
                phase_fraction=self._phase_done / self._phase_total if self._phase else 0.0,
                detail=detail,
                elapsed_seconds=self.elapsed,
                bytes_written=self.bytes_written,
                peak_ram_bytes=self.peak_ram_bytes,
                peak_vram_bytes=self.peak_vram_bytes,
            )
        )
