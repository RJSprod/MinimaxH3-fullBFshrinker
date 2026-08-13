"""Activation scaling for Option B.

The user never calibrates anything. There are two ways that promise is kept,
and the converter picks between them automatically.

**Dynamic (default).** ComfyUI's NVFP4 linear reads ``input_scale`` with
``getattr(self, 'input_scale', None)``, and when it is absent
``QuantizedTensor.from_float`` derives a scale from the activation itself:
``amax / (F8_E4M3_MAX * F4_E2M1_MAX)``. Omitting the tensor is therefore a
supported, first-class mode -- not a degraded fallback -- and it adapts to
whatever resolution, duration and conditioning the user actually runs. This is
what a conversion produces unless a calibration pack is installed.

**Static (optional).** A versioned calibration pack may ship per-layer
activation maxima collected offline from real H3 inference. When one is present
and matches the source's layer set, those maxima are baked into ``input_scale``
tensors, which lets the runtime skip the per-activation reduction. The pack is
content-addressed and verified by SHA-256 before use.

Either way the choice, the pack version and the layer coverage are recorded in
the conversion report, so a checkpoint always states how it was scaled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from h3converter.paths import calibration_assets_dir

# NVFP4 activation scale denominator: F8_E4M3_MAX * F4_E2M1_MAX = 448 * 6.
# Matches comfy_kitchen.float_utils and TensorCoreNVFP4Layout.quantize.
NVFP4_ACT_SCALE_DENOMINATOR = 448.0 * 6.0

STRATEGY_DYNAMIC = "dynamic_activation_scale"
STRATEGY_STATIC = "static_calibrated_input_scale"

MANIFEST_NAME = "manifest.json"


class CalibrationError(RuntimeError):
    pass


@dataclass
class CalibrationPack:
    """A verified, versioned set of per-layer activation maxima."""

    name: str
    version: str
    case_count: int
    coverage: dict[str, object]
    activation_amax: dict[str, float]
    source_file: str

    def input_scale_for(self, layer: str) -> float | None:
        amax = self.activation_amax.get(layer)
        if amax is None or amax <= 0.0:
            return None
        return float(amax) / NVFP4_ACT_SCALE_DENOMINATOR


@dataclass
class CalibrationPlan:
    """What the pipeline will do about activation scales, and why."""

    strategy: str
    pack: CalibrationPack | None = None
    covered_layers: int = 0
    total_layers: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        return self.strategy == STRATEGY_STATIC

    def input_scale_for(self, layer: str) -> float | None:
        if not self.is_static or self.pack is None:
            return None
        return self.pack.input_scale_for(layer)

    def describe(self) -> str:
        if self.strategy == STRATEGY_DYNAMIC:
            return (
                "dynamic activation scaling (no input_scale tensors; the runtime derives "
                "a scale per activation)"
            )
        return (
            f"static input_scale from calibration pack {self.pack.name} v{self.pack.version} "
            f"({self.covered_layers}/{self.total_layers} layers covered)"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "description": self.describe(),
            "pack_name": self.pack.name if self.pack else None,
            "pack_version": self.pack.version if self.pack else None,
            "case_count": self.pack.case_count if self.pack else 0,
            "coverage": self.pack.coverage if self.pack else {},
            "layers_covered": self.covered_layers,
            "layers_total": self.total_layers,
            "notes": list(self.notes),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(assets_dir: Path | None = None) -> dict[str, object] | None:
    """Read and sanity-check the calibration manifest, if one is installed."""
    assets_dir = assets_dir or calibration_assets_dir()
    manifest_path = assets_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"calibration manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CalibrationError("calibration manifest is not a JSON object")
    return manifest


def load_pack(assets_dir: Path | None = None) -> CalibrationPack | None:
    """Load the installed calibration pack, verifying its checksums.

    Returns ``None`` when no pack is installed, which is the normal case: the
    repository ships the manifest describing the format but no activation data,
    because activation maxima can only be collected from real H3 inference.
    """
    assets_dir = assets_dir or calibration_assets_dir()
    manifest = load_manifest(assets_dir)
    if manifest is None:
        return None

    files = manifest.get("files") or []
    if not isinstance(files, list) or not files:
        return None

    entry = files[0]
    if not isinstance(entry, dict) or "name" not in entry:
        raise CalibrationError("calibration manifest 'files' entries need a 'name'")

    # Reject anything that is not a plain filename inside the assets directory:
    # the manifest is data, and must never be able to name an arbitrary path.
    name = str(entry["name"])
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise CalibrationError(f"calibration file name {name!r} is not a plain filename")

    data_path = assets_dir / name
    if not data_path.is_file():
        return None

    expected_sha = entry.get("sha256")
    if expected_sha:
        actual = _sha256(data_path)
        if actual != str(expected_sha).lower():
            raise CalibrationError(
                f"calibration file {name} failed its checksum "
                f"(expected {expected_sha}, got {actual})"
            )

    try:
        with open(data_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"calibration data {name} is unreadable: {exc}") from exc

    raw = payload.get("activation_amax") if isinstance(payload, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise CalibrationError(f"calibration data {name} has no 'activation_amax' mapping")

    activation_amax: dict[str, float] = {}
    for layer, value in raw.items():
        try:
            amax = float(value)
        except (TypeError, ValueError):
            raise CalibrationError(f"activation_amax[{layer!r}] is not a number") from None
        if amax <= 0.0 or amax != amax or amax == float("inf"):
            raise CalibrationError(f"activation_amax[{layer!r}] = {amax} is not a usable maximum")
        activation_amax[str(layer)] = amax

    return CalibrationPack(
        name=str(manifest.get("calibration_set", "unnamed")),
        version=str(manifest.get("version", "0")),
        case_count=int(manifest.get("cases", 0) or 0),
        coverage=dict(manifest.get("coverage") or {}),
        activation_amax=activation_amax,
        source_file=name,
    )


def resolve(layers: list[str], assets_dir: Path | None = None) -> CalibrationPlan:
    """Choose the activation-scaling strategy for this conversion."""
    plan = CalibrationPlan(strategy=STRATEGY_DYNAMIC, total_layers=len(layers))

    try:
        pack = load_pack(assets_dir)
    except CalibrationError as exc:
        # A broken pack must not silently downgrade the output without saying so.
        plan.notes.append(f"installed calibration pack rejected: {exc}")
        pack = None

    if pack is None:
        plan.notes.append(
            "no calibration pack installed; the runtime derives an activation scale per "
            "activation, which needs no user input and adapts to the actual workload"
        )
        return plan

    covered = sum(1 for layer in layers if pack.input_scale_for(layer) is not None)
    if covered == 0:
        plan.notes.append(
            f"calibration pack {pack.name} v{pack.version} covers none of this source's "
            "layers; falling back to dynamic activation scaling"
        )
        return plan

    plan.strategy = STRATEGY_STATIC
    plan.pack = pack
    plan.covered_layers = covered
    if covered < len(layers):
        plan.notes.append(
            f"{len(layers) - covered} layers are not in the pack and keep dynamic scaling"
        )
    return plan
