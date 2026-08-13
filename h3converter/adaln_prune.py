"""Collapse H3's full-width AdaLN projections onto the shared curve basis.

Each source projection is a [width, time_embed_dim] matrix -- 96768 x 2688 for a
DiT block, about half a gigabyte in BF16. There are 51 of them, roughly 26 GB of
the 66 GB source, and they are what the curve form eliminates. They are
therefore processed strictly one at a time and in row chunks, so peak memory
stays bounded regardless of model size.

Nothing here is destructive: the reduced projections are validated against the
exact full-precision path *before* the pipeline is allowed to proceed to
quantization or to write a final filename.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from h3converter import constants as C
from h3converter.adaln_curve import CurveBasis, TimeEmbedder

# Elements per projection chunk. ~64M float32 = 256 MB of working set, which
# keeps a 96768 x 2688 projection bounded without making the matmul inefficient.
_PROJECT_ELEM_BUDGET = 1 << 26

_COMPUTE_DTYPE = torch.float32


@dataclass
class ReducedProjection:
    """One collapsed AdaLN projection, ready to serialise."""

    weight: torch.Tensor          # [width, rank], stored dtype (BF16)
    bias: torch.Tensor | None     # source dtype, unchanged unless the basis is centred


@dataclass
class ProjectionError:
    """Reconstruction quality for one projection, measured off-grid."""

    layer: str
    rel_l2: float
    max_abs: float
    reference_scale: float
    finite: bool

    @property
    def ok(self) -> bool:
        return self.finite and self.rel_l2 <= C.ADALN_REL_ERROR_ABORT

    @property
    def warn(self) -> bool:
        return self.finite and C.ADALN_REL_ERROR_WARN < self.rel_l2 <= C.ADALN_REL_ERROR_ABORT


@dataclass
class PruneReport:
    """Aggregate numerical evidence that the curve form reproduces the source."""

    basis_stats: dict[str, object] = field(default_factory=dict)
    curve_rel_l2: float = 0.0
    projections: list[ProjectionError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def worst(self) -> ProjectionError | None:
        finite = [p for p in self.projections if p.finite]
        if not finite:
            return self.projections[0] if self.projections else None
        return max(finite, key=lambda p: p.rel_l2)

    @property
    def ok(self) -> bool:
        return bool(self.projections) and all(p.ok for p in self.projections)

    def as_dict(self) -> dict[str, object]:
        worst = self.worst
        return {
            "basis": self.basis_stats,
            "curve_rel_l2": self.curve_rel_l2,
            "projection_count": len(self.projections),
            "worst_projection": (
                {
                    "layer": worst.layer,
                    "rel_l2": worst.rel_l2,
                    "max_abs": worst.max_abs,
                    "reference_scale": worst.reference_scale,
                }
                if worst
                else None
            ),
            "mean_rel_l2": (
                sum(p.rel_l2 for p in self.projections) / len(self.projections)
                if self.projections
                else 0.0
            ),
            "abort_threshold": C.ADALN_REL_ERROR_ABORT,
            "warn_threshold": C.ADALN_REL_ERROR_WARN,
            "warnings": list(self.warnings),
        }


class PruneError(RuntimeError):
    """Raised when the reduced AdaLN representation is not faithful enough."""


def collapse_projection(
    basis: CurveBasis,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> ReducedProjection:
    """Re-express one projection in the curve basis: [width, D] -> [width, rank]."""
    if weight.dim() != 2:
        raise PruneError(f"AdaLN weight must be 2D, got {tuple(weight.shape)}")
    width, in_dim = weight.shape
    if in_dim != basis.basis.shape[0]:
        raise PruneError(
            f"AdaLN weight consumes {in_dim} inputs but the curve basis covers "
            f"{basis.basis.shape[0]}"
        )

    projector = basis.basis.to(_COMPUTE_DTYPE)
    reduced = torch.empty((width, basis.rank), dtype=_COMPUTE_DTYPE)

    chunk = max(1, _PROJECT_ELEM_BUDGET // max(in_dim, 1))
    for start in range(0, width, chunk):
        stop = min(start + chunk, width)
        reduced[start:stop] = weight[start:stop].to(_COMPUTE_DTYPE) @ projector

    if not torch.isfinite(reduced).all():
        raise PruneError("AdaLN projection produced non-finite values")

    new_bias = bias
    if basis.mean is not None:
        folded = basis.fold_bias(weight, bias)
        new_bias = folded.to(bias.dtype if bias is not None else torch.float32)

    return ReducedProjection(weight=reduced.to(weight_dtype).contiguous(), bias=new_bias)


def probe_timesteps(grid: int = C.ADALN_CURVE_GRID, count: int = 129, seed: int = 20260813) -> torch.Tensor:
    """Timesteps to measure reconstruction at.

    Deliberately biased towards the worst case: grid-cell midpoints, where the
    runtime's linear interpolation is furthest from the true curve. Grid points
    themselves would only measure the SVD truncation and would flatter the
    result.
    """
    generator = torch.Generator().manual_seed(seed)
    cells = max(grid - 1, 1)
    midpoints = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
    stride = max(1, cells // max(count // 2, 1))
    sampled = midpoints[::stride][: max(count // 2, 1)]
    random = torch.rand(max(count - sampled.numel() - 2, 1), generator=generator, dtype=torch.float64)
    edges = torch.tensor([0.0, 1.0], dtype=torch.float64)
    return torch.cat([edges, sampled, random]).clamp(0.0, 1.0)


def curve_reconstruction_error(basis: CurveBasis, embedder: TimeEmbedder,
                               t: torch.Tensor | None = None) -> float:
    """Relative L2 error of the reconstructed AdaLN input curve itself.

    This is the model-independent half of the check: how well ``basis`` plus the
    runtime's interpolation reproduce ``g(t) = silu(e(t))``.
    """
    if t is None:
        t = probe_timesteps(basis.grid)
    truth = embedder.adaln_input(t)
    coords = basis.interpolate(t)
    approx = coords @ basis.basis.transpose(0, 1)
    if basis.mean is not None:
        approx = approx + basis.mean
    denom = float(truth.norm())
    if denom == 0.0:
        return 0.0
    return float((truth - approx).norm() / denom)


def validate_projection(
    layer: str,
    basis: CurveBasis,
    embedder: TimeEmbedder,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    reduced: ReducedProjection,
    t: torch.Tensor | None = None,
    row_sample: int = 4096,
) -> ProjectionError:
    """Compare reduced against full modulation output at off-grid timesteps.

    Uses an evenly strided row sample so the cost does not scale with the
    96768-row projections while still spanning every modulation slot (shift,
    scale and gate for all three modalities).
    """
    if t is None:
        t = probe_timesteps(basis.grid)

    width = weight.shape[0]
    if width > row_sample:
        rows = torch.linspace(0, width - 1, row_sample).long()
    else:
        rows = torch.arange(width)

    g = embedder.adaln_input(t).to(_COMPUTE_DTYPE)                    # [M, D]
    coords = basis.interpolate(t).to(_COMPUTE_DTYPE)                  # [M, rank]

    w_rows = weight[rows].to(_COMPUTE_DTYPE)                          # [R, D]
    r_rows = reduced.weight[rows].to(_COMPUTE_DTYPE)                  # [R, rank]

    truth = w_rows @ g.transpose(0, 1)                                # [R, M]
    approx = r_rows @ coords.transpose(0, 1)

    if bias is not None:
        truth = truth + bias[rows].to(_COMPUTE_DTYPE).unsqueeze(1)
    if reduced.bias is not None:
        approx = approx + reduced.bias[rows].to(_COMPUTE_DTYPE).unsqueeze(1)

    finite = bool(torch.isfinite(approx).all() and torch.isfinite(truth).all())
    if not finite:
        return ProjectionError(layer, float("inf"), float("inf"), 0.0, False)

    diff = truth - approx
    scale = float(truth.norm())
    rel = float(diff.norm() / scale) if scale > 0 else 0.0
    return ProjectionError(
        layer=layer,
        rel_l2=rel,
        max_abs=float(diff.abs().max()),
        reference_scale=scale,
        finite=True,
    )


def check_report(report: PruneReport) -> None:
    """Raise if the pruned representation is not good enough to continue.

    Called before any quantization or output finalisation, so a bad basis costs
    the user minutes rather than a corrupted 12 GB checkpoint.
    """
    non_finite = [p.layer for p in report.projections if not p.finite]
    if non_finite:
        raise PruneError(
            "AdaLN reconstruction produced non-finite values for: " + ", ".join(non_finite[:5])
        )

    failed = [p for p in report.projections if p.rel_l2 > C.ADALN_REL_ERROR_ABORT]
    if failed:
        worst = max(failed, key=lambda p: p.rel_l2)
        raise PruneError(
            f"AdaLN curve reconstruction error {worst.rel_l2:.4%} on {worst.layer} exceeds the "
            f"{C.ADALN_REL_ERROR_ABORT:.2%} limit ({len(failed)} of {len(report.projections)} "
            "projections failed). The rank-8 / 1025-point geometry is fixed by the target "
            "format, so this source's timestep path cannot be represented in the curve form."
        )
