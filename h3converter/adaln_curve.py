"""Construction of H3's compact AdaLN curve representation.

Background
----------
A full H3 checkpoint modulates every block through a large timestep path::

    e(t) = time_embedder(t)                      # [time_embed_dim], e.g. 2688
    m(t) = adaln_proj.linear(silu(e(t))) + bias  # [18 * hidden] per block

The curve form replaces the time embedder and the full-width projections with a
small shared basis of the *same function*::

    pos  = clamp(t, 0, 1) * (grid - 1)
    i0   = min(floor(pos), grid - 2)
    c(t) = lerp(table[i0], table[i0 + 1], pos - i0)   # [rank], from adaln_t_table
    m(t) = adaln_proj.linear(c(t)) + bias             # no silu in curve mode

So the conversion has to find one shared ``rank``-dimensional subspace that the
whole curve ``g(t) = silu(e(t))`` lives in, tabulate the curve's coordinates in
that subspace on the ``grid``, and re-express every projection matrix in it.

That is exactly a truncated SVD of the sampled curve. With ``G = g(t_grid)``
factored as ``U S Vᵀ`` and ``V₈`` its leading 8 right singular vectors:

    table  = G V₈          (the curve's coordinates, [grid, 8])
    W_red  = W V₈          (each projection re-expressed, [width, 8])

and ``W_red @ c(t) == W @ g(t)`` up to the rank-8 truncation, with the bias
untouched. The runtime's linear interpolation between grid rows is the only
other approximation, and 1025 rows over t ∈ [0, 1] make it negligible; both
error sources are measured directly in :mod:`h3converter.adaln_prune`.

Why the basis is not mean-centred by default
--------------------------------------------
Folding the curve's mean into the bias would buy one extra deviation direction,
but it moves the entire mean modulation into a BF16 bias vector, where a ~0.4%
relative rounding error lands directly on the dominant term. Leaving the mean
inside the basis keeps it spread across the F32 table and the reduced matrix,
where it is averaged over 8 accumulated products instead. It also leaves the
source biases untouched, and matches the golden reference's BF16 bias
inventory. ``center=True`` remains available for developer experiments and both
truncation errors are always reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from h3converter import constants as C

# All curve mathematics runs in float64. The tensors involved are tiny
# (1025 x 2688 at most) and the whole point of this stage is accuracy.
_DTYPE = torch.float64


@dataclass
class TimeEmbedder:
    """The source checkpoint's timestep path, in float64.

    Mirrors ``comfy/ldm/minimax/model.py::TimeEmbedder.forward`` exactly,
    including the cos-before-sin concatenation order, which is what makes the
    sampled curve the same function the full checkpoint evaluates.
    """

    proj_in_weight: torch.Tensor
    proj_in_bias: torch.Tensor
    proj_out_weight: torch.Tensor
    proj_out_bias: torch.Tensor

    @property
    def freq_dim(self) -> int:
        return self.proj_in_weight.shape[1]

    @property
    def out_dim(self) -> int:
        return self.proj_out_weight.shape[0]

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor]) -> "TimeEmbedder":
        def take(name: str) -> torch.Tensor:
            key = f"{C.KEY_TIME_EMBEDDER}.{name}"
            if key not in tensors:
                raise KeyError(f"time embedder tensor {key!r} is missing from the source")
            return tensors[key].to(_DTYPE)

        return cls(
            proj_in_weight=take("proj_in.weight"),
            proj_in_bias=take("proj_in.bias"),
            proj_out_weight=take("proj_out.weight"),
            proj_out_bias=take("proj_out.bias"),
        )

    def frequency_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(C.TIME_EMBED_MAX_PERIOD)
            * torch.arange(half, dtype=_DTYPE, device=t.device)
            / half
        )
        args = t.to(_DTYPE)[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """e(t) for a batch of timesteps: [M] -> [M, time_embed_dim]."""
        hidden = torch.nn.functional.linear(
            self.frequency_embedding(t), self.proj_in_weight, self.proj_in_bias
        )
        hidden = torch.nn.functional.silu(hidden)
        return torch.nn.functional.linear(hidden, self.proj_out_weight, self.proj_out_bias)

    def adaln_input(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = silu(e(t)) - what the full-form AdaLN linear actually consumes."""
        return torch.nn.functional.silu(self.embed(t))


def grid_timesteps(grid: int = C.ADALN_CURVE_GRID) -> torch.Tensor:
    """The ``grid`` evenly spaced timesteps the table is defined on."""
    return torch.linspace(C.TIME_DOMAIN_MIN, C.TIME_DOMAIN_MAX, grid, dtype=_DTYPE)


@dataclass
class CurveBasis:
    """A rank-``k`` basis of the AdaLN input curve, plus its tabulated coordinates."""

    table: torch.Tensor              # [grid, rank] float64 (stored as F32)
    basis: torch.Tensor              # [time_embed_dim, rank] float64, orthonormal
    mean: torch.Tensor | None        # [time_embed_dim] float64 when centred
    singular_values: torch.Tensor    # full spectrum, float64
    rel_error_uncentred: float
    rel_error_centred: float
    centered: bool

    @property
    def grid(self) -> int:
        return int(self.table.shape[0])

    @property
    def rank(self) -> int:
        return int(self.table.shape[1])

    def interpolate(self, t: torch.Tensor) -> torch.Tensor:
        """Reproduce the runtime's table lookup exactly, for validation.

        Mirrors ``MiniMaxH3Model._forward``: clamp to [0, 1], scale to a
        fractional grid index, clamp the lower row so t = 1.0 reads the last
        interval instead of running past the table, then lerp.
        """
        pos = t.to(_DTYPE).clamp(0.0, 1.0) * (self.grid - 1)
        i0 = pos.floor().long().clamp(max=self.grid - 2)
        frac = (pos - i0).unsqueeze(1)
        return torch.lerp(self.table[i0], self.table[i0 + 1], frac)

    def project_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """W [width, time_embed_dim] -> W_red [width, rank]."""
        if weight.shape[1] != self.basis.shape[0]:
            raise ValueError(
                f"weight has input width {weight.shape[1]}, basis expects {self.basis.shape[0]}"
            )
        return weight.to(_DTYPE) @ self.basis

    def fold_bias(self, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
        """Bias for the reduced layer. Unchanged unless the basis is centred."""
        if self.mean is None:
            return bias
        shift = weight.to(_DTYPE) @ self.mean
        return shift if bias is None else bias.to(_DTYPE) + shift

    def stats(self) -> dict[str, object]:
        spectrum = self.singular_values
        total = float((spectrum**2).sum())
        kept = float((spectrum[: self.rank] ** 2).sum())
        return {
            "grid": self.grid,
            "rank": self.rank,
            "centered": self.centered,
            "energy_retained": kept / total if total > 0 else 1.0,
            "rel_error_uncentered": self.rel_error_uncentred,
            "rel_error_centered": self.rel_error_centred,
            "singular_values_head": [float(v) for v in spectrum[: self.rank + 2]],
        }


def build_curve_basis(
    embedder: TimeEmbedder,
    grid: int = C.ADALN_CURVE_GRID,
    rank: int = C.ADALN_CURVE_RANK,
    center: bool = False,
) -> CurveBasis:
    """Sample g(t) on the grid and fit the shared rank-``k`` subspace."""
    if rank < 1:
        raise ValueError("rank must be positive")
    if grid < 2:
        raise ValueError("grid must have at least two points")

    t = grid_timesteps(grid)
    curve = embedder.adaln_input(t)  # [grid, time_embed_dim]
    if not torch.isfinite(curve).all():
        raise ValueError("time embedding produced non-finite values; source weights are corrupt")
    if rank > min(curve.shape):
        raise ValueError(f"rank {rank} exceeds the sampled curve's dimensions {tuple(curve.shape)}")

    mean = curve.mean(dim=0)
    centred = curve - mean

    spectrum_raw = torch.linalg.svdvals(curve)
    spectrum_centred = torch.linalg.svdvals(centred)

    def tail_error(spectrum: torch.Tensor, reference: torch.Tensor) -> float:
        total = float((reference**2).sum())
        if total <= 0:
            return 0.0
        tail = float((spectrum[rank:] ** 2).sum())
        return math.sqrt(max(tail, 0.0) / total)

    energy = curve
    rel_uncentred = tail_error(spectrum_raw, energy)
    rel_centred = tail_error(spectrum_centred, energy)

    fitted = centred if center else curve
    _, spectrum, vh = torch.linalg.svd(fitted, full_matrices=False)
    basis = vh[:rank].transpose(0, 1).contiguous()  # [D, rank], orthonormal columns
    table = (fitted @ basis).contiguous()           # [grid, rank]

    return CurveBasis(
        table=table,
        basis=basis,
        mean=mean if center else None,
        singular_values=spectrum,
        rel_error_uncentred=rel_uncentred,
        rel_error_centred=rel_centred,
        centered=center,
    )


def table_for_storage(basis: CurveBasis) -> torch.Tensor:
    """adaln_t_table as it is written to disk: F32 [grid, rank]."""
    return basis.table.to(torch.float32).contiguous()
