"""The AdaLN curve basis, projection collapse, and the pruning quality gate.

The reference behaviour these tests check against is re-implemented
independently below, straight from ComfyUI's ``comfy/ldm/minimax/model.py``, so
a change to either the sampled function or the interpolation rule shows up as a
disagreement rather than as two copies of the same mistake.
"""

from __future__ import annotations

import math

import pytest
import torch

from h3converter import constants as C
from h3converter.adaln_curve import (
    CurveBasis,
    TimeEmbedder,
    build_curve_basis,
    grid_timesteps,
    table_for_storage,
)
from h3converter.adaln_prune import (
    PruneError,
    PruneReport,
    ProjectionError,
    check_report,
    collapse_projection,
    curve_reconstruction_error,
    probe_timesteps,
    validate_projection,
)


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def reference_time_embedding(t, w_in, b_in, w_out, b_out, freq_dim):
    """comfy/ldm/minimax/model.py::TimeEmbedder.forward, transcribed."""
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = torch.nn.functional.linear(emb, w_in, b_in)
    return torch.nn.functional.linear(torch.nn.functional.silu(hidden), w_out, b_out)


def reference_table_lookup(table, t):
    """MiniMaxH3Model._forward's curve interpolation, transcribed."""
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


@pytest.fixture(scope="module")
def embedder():
    generator = torch.Generator().manual_seed(4242)
    freq_dim, hidden, out = 256, 512, 192
    return TimeEmbedder(
        proj_in_weight=(torch.randn(hidden, freq_dim, generator=generator) * 0.05).double(),
        proj_in_bias=(torch.randn(hidden, generator=generator) * 0.02).double(),
        proj_out_weight=(torch.randn(out, hidden, generator=generator) * 0.05).double(),
        proj_out_bias=(torch.randn(out, generator=generator) * 0.02).double(),
    )


@pytest.fixture(scope="module")
def basis(embedder):
    return build_curve_basis(embedder)


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def test_time_embedding_matches_the_runtime_definition(embedder):
    t = torch.tensor([0.0, 0.137, 0.5, 0.913, 1.0])
    ours = embedder.embed(t).float()
    theirs = reference_time_embedding(
        t,
        embedder.proj_in_weight.float(), embedder.proj_in_bias.float(),
        embedder.proj_out_weight.float(), embedder.proj_out_bias.float(),
        embedder.freq_dim,
    )
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_frequency_embedding_is_cos_then_sin(embedder):
    """The concatenation order is cos-before-sin, unlike most DiT codebases."""
    t = torch.tensor([0.25])
    emb = embedder.frequency_embedding(t)
    half = embedder.freq_dim // 2
    assert emb.shape == (1, embedder.freq_dim)
    assert emb[0, 0] == pytest.approx(math.cos(0.25), abs=1e-12)
    assert emb[0, half] == pytest.approx(math.sin(0.25), abs=1e-12)


def test_adaln_input_applies_silu(embedder):
    """The full form feeds silu(e(t)) to the projection; the curve form does not."""
    t = torch.tensor([0.3, 0.7])
    raw = embedder.embed(t)
    assert torch.allclose(embedder.adaln_input(t), torch.nn.functional.silu(raw))


def test_time_embedder_reports_a_missing_tensor():
    with pytest.raises(KeyError, match="proj_in.weight"):
        TimeEmbedder.from_tensors({})


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def test_table_has_the_fixed_golden_geometry(basis):
    assert basis.table.shape == (C.ADALN_CURVE_GRID, C.ADALN_CURVE_RANK) == (1025, 8)
    stored = table_for_storage(basis)
    assert stored.dtype == torch.float32
    assert stored.shape == (1025, 8)
    assert torch.isfinite(stored).all()


def test_basis_is_orthonormal(basis):
    gram = basis.basis.transpose(0, 1) @ basis.basis
    assert torch.allclose(gram, torch.eye(basis.rank, dtype=gram.dtype), atol=1e-10)


def test_grid_spans_the_normalised_timestep_domain():
    t = grid_timesteps()
    assert t.numel() == C.ADALN_CURVE_GRID
    assert float(t[0]) == C.TIME_DOMAIN_MIN
    assert float(t[-1]) == C.TIME_DOMAIN_MAX


def test_interpolation_matches_the_runtime_lookup(basis):
    t = torch.tensor([0.0, 1e-6, 0.123456, 0.5, 0.999999, 1.0], dtype=torch.float64)
    ours = basis.interpolate(t)
    theirs = reference_table_lookup(basis.table, t)
    assert torch.allclose(ours, theirs, atol=1e-12)


def test_interpolation_clamps_out_of_range_timesteps(basis):
    inside = basis.interpolate(torch.tensor([0.0, 1.0], dtype=torch.float64))
    outside = basis.interpolate(torch.tensor([-4.0, 7.0], dtype=torch.float64))
    assert torch.allclose(inside, outside)


def test_smooth_h3_style_curve_is_captured_at_rank_8(basis, embedder):
    """The timestep path is a smooth low-order curve, which is why rank 8 works."""
    assert curve_reconstruction_error(basis, embedder) < 1e-5
    assert basis.stats()["energy_retained"] > 1.0 - 1e-9


def test_rank_beyond_the_curve_dimension_is_refused(embedder):
    with pytest.raises(ValueError, match="rank"):
        build_curve_basis(embedder, grid=4, rank=8)


# ---------------------------------------------------------------------------
# Projection collapse
# ---------------------------------------------------------------------------

def test_collapse_produces_the_reduced_shape_and_dtype(basis, embedder):
    generator = torch.Generator().manual_seed(9)
    width = 18 * 64
    weight = (torch.randn(width, embedder.out_dim, generator=generator) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(width, generator=generator) * 0.01).to(torch.bfloat16)

    reduced = collapse_projection(basis, weight, bias)
    assert reduced.weight.shape == (width, C.ADALN_CURVE_RANK)
    assert reduced.weight.dtype == torch.bfloat16
    # With an uncentred basis the source bias passes through untouched.
    assert reduced.bias is bias


def test_collapsed_projection_reproduces_the_full_modulation(basis, embedder):
    generator = torch.Generator().manual_seed(11)
    width = 512
    weight = (torch.randn(width, embedder.out_dim, generator=generator) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(width, generator=generator) * 0.01).to(torch.bfloat16)

    reduced = collapse_projection(basis, weight, bias)
    error = validate_projection("test.adaln", basis, embedder, weight, bias, reduced)
    assert error.finite
    assert error.ok
    # BF16 storage of the reduced matrix dominates; the basis itself is exact.
    assert error.rel_l2 < C.ADALN_REL_ERROR_WARN


def test_centred_basis_folds_the_mean_into_the_bias(embedder):
    centred = build_curve_basis(embedder, center=True)
    assert centred.mean is not None

    generator = torch.Generator().manual_seed(13)
    weight = (torch.randn(64, embedder.out_dim, generator=generator) * 0.02).to(torch.bfloat16)
    bias = torch.zeros(64, dtype=torch.bfloat16)

    reduced = collapse_projection(centred, weight, bias)
    assert reduced.bias is not bias
    assert not torch.allclose(reduced.bias.float(), torch.zeros(64))

    error = validate_projection("test.adaln", centred, embedder, weight, bias, reduced)
    assert error.ok


def test_collapse_rejects_a_mismatched_input_width(basis):
    with pytest.raises(PruneError, match="curve basis covers"):
        collapse_projection(basis, torch.zeros(8, 7, dtype=torch.bfloat16), None)


def test_collapse_rejects_a_non_2d_weight(basis):
    with pytest.raises(PruneError, match="must be 2D"):
        collapse_projection(basis, torch.zeros(8, dtype=torch.bfloat16), None)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def test_probe_timesteps_target_the_worst_case_between_grid_points():
    t = probe_timesteps(C.ADALN_CURVE_GRID)
    grid = grid_timesteps(C.ADALN_CURVE_GRID)
    assert t.numel() > 32
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    # Probes must not sit on grid nodes, or only the SVD truncation is measured
    # and the runtime's interpolation error is never exercised.
    on_grid = sum(1 for value in t if float((grid - value).abs().min()) < 1e-9)
    assert on_grid == 2  # exactly the deliberate 0.0 and 1.0 endpoints

    # Half the probes are grid-cell midpoints, the worst case for the lerp.
    spacing = 1.0 / (C.ADALN_CURVE_GRID - 1)
    midpoint_probes = sum(
        1 for value in t
        if abs(float((grid - value).abs().min()) - spacing / 2) < 1e-9
    )
    assert midpoint_probes >= t.numel() // 4


def test_report_aborts_when_reconstruction_is_too_poor():
    report = PruneReport()
    report.projections = [
        ProjectionError("blocks.0.adaln_proj.linear", 1e-4, 1e-5, 1.0, True),
        ProjectionError("blocks.1.adaln_proj.linear", 0.4, 1.0, 1.0, True),
    ]
    with pytest.raises(PruneError, match="exceeds the"):
        check_report(report)


def test_report_aborts_on_non_finite_reconstruction():
    report = PruneReport()
    report.projections = [ProjectionError("blocks.0.adaln_proj.linear", float("inf"),
                                          float("inf"), 0.0, False)]
    with pytest.raises(PruneError, match="non-finite"):
        check_report(report)


def test_report_passes_when_every_projection_is_accurate():
    report = PruneReport()
    report.projections = [ProjectionError(f"blocks.{i}.adaln_proj.linear", 1e-4, 1e-5, 1.0, True)
                          for i in range(4)]
    check_report(report)
    assert report.ok
    assert report.as_dict()["worst_projection"]["rel_l2"] == pytest.approx(1e-4)
