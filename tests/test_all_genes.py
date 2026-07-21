"""Smooth every gene once, derive any signature/gene from the result -- no re-smoothing.

``smooth(..., all_genes=True)`` (and :func:`spatial_smooth.smooth_all`) run the pipeline over the
full ``(n_obs, n_vars)`` matrix instead of the requested subset. Two loads:

* the per-step cache key stops depending on *which* genes were asked for (the input is always the
  whole matrix), so a second signature -- or a single gene, or ``"blend"`` -- through the **same**
  pipeline re-smooths **nothing**; the diffusion-map GP in particular runs exactly once; and
* the derived score is *exactly* the per-signature result for the linear neighbour smoothers
  (:class:`KnnGaussian`, :class:`Kde` are column-independent) and matches it to floating-point
  precision for the GP (mellon's Nystrom solve rounds differently at a different matrix width).

The reuse claim is asserted by counting ``Step.apply`` calls; the fidelity claim by comparing the
``all_genes`` score against a from-scratch per-subset smooth.
"""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from spatial_smooth.core import CACHE_LAYER_PREFIX
from conftest import _make_adata, needs_kompot


def _count_apply(monkeypatch, step_cls) -> dict:
    calls = {"n": 0}
    orig = step_cls.apply

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(step_cls, "apply", wrapper)
    return calls


def _cache_layers(adata):
    return [name for name in adata.layers if str(name).startswith(CACHE_LAYER_PREFIX)]


# --------------------------------------------------------------------------------------- #
# fidelity -- the linear neighbour smoothers derive *exactly* the per-subset score          #
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("steps", ["spatial", "spatial-kde"])
def test_all_genes_linear_is_exact(adata, signature, steps):
    """KnnGaussian/Kde are column-independent, so gather-from-all == smooth-the-subset, exactly."""
    whole = _make_adata()
    subset = _make_adata()
    ss.smooth(whole, signature, "sig", steps=steps, all_genes=True)
    ss.smooth(subset, signature, "sig", steps=steps)  # per-subset, the old path

    np.testing.assert_array_equal(
        whole.obs["sig"].to_numpy(), subset.obs["sig"].to_numpy()
    )
    # The raw score is the signature's subset either way -- all_genes changes only the smoothing.
    np.testing.assert_array_equal(
        whole.obs["sig_raw"].to_numpy(), subset.obs["sig_raw"].to_numpy()
    )


def test_all_genes_single_gene_is_exact(adata):
    """A single gene derived from the full spatial layer equals smoothing that one gene alone."""
    whole = _make_adata()
    one = _make_adata()
    gene = [list(whole.var_names)[-1]]  # a background gene, not in the planted signature
    ss.smooth(whole, gene, "g", steps="spatial", all_genes=True)
    ss.smooth(one, gene, "g", steps="spatial")
    np.testing.assert_array_equal(whole.obs["g"].to_numpy(), one.obs["g"].to_numpy())


# --------------------------------------------------------------------------------------- #
# reuse -- one smooth serves every signature, single gene and blend through a pipeline      #
# --------------------------------------------------------------------------------------- #
def test_all_genes_reuses_one_spatial_smooth_across_signatures(adata, signature, monkeypatch):
    """After the full matrix is smoothed once, every derivation is a pure cache hit."""
    calls = _count_apply(monkeypatch, ss.KnnGaussian)

    ss.smooth(adata, signature, "sigA", steps="spatial", all_genes=True)  # 1 compute (the miss)
    assert calls["n"] == 1
    assert len(_cache_layers(adata)) == 1

    # A different signature and a single gene, same pipeline -> both served from the one layer.
    ss.smooth(adata, signature[:1], "one", steps="spatial", all_genes=True)
    ss.smooth(adata, [list(adata.var_names)[-1]], "bg", steps="spatial", all_genes=True)
    assert calls["n"] == 1, "no re-smoothing: the full spatial layer serves every gene"
    assert len(_cache_layers(adata)) == 1


def test_smooth_all_warms_the_cache_without_scoring(adata, signature, monkeypatch):
    """smooth_all writes no signature but leaves the pipeline cached for later derivations."""
    ss.smooth_all(adata, steps="spatial")
    assert _cache_layers(adata)                 # cache warmed ...
    assert ss.list_results(adata) == []         # ... but nothing scored
    assert not any(str(c) == "sig" for c in adata.obs.columns)

    calls = _count_apply(monkeypatch, ss.KnnGaussian)
    ss.smooth(adata, signature, "sig", steps="spatial", all_genes=True)
    assert calls["n"] == 0, "the signature is derived from the warmed layer -- no smoothing"


def test_smooth_all_without_cache_warns(adata):
    with pytest.warns(UserWarning, match="smooth_all\\(cache=False\\)"):
        ss.smooth_all(adata, steps="spatial", cache=False)
    assert _cache_layers(adata) == []


def test_all_genes_records_the_flag_in_provenance(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial", all_genes=True)
    assert ss.provenance(adata, "sig")["all_genes"] is True
    ss.smooth(adata, signature, "sub", steps="spatial")
    assert ss.provenance(adata, "sub")["all_genes"] is False


# --------------------------------------------------------------------------------------- #
# the payoff -- the diffusion-map GP runs exactly once across dm + single gene + blend      #
# --------------------------------------------------------------------------------------- #
@needs_kompot
def test_all_genes_gp_runs_once_across_dm_gene_and_blend(adata, signature, monkeypatch):
    fast_dm = [ss.KompotGP(n_landmarks=100)]
    gp_calls = _count_apply(monkeypatch, ss.KompotGP)

    ss.smooth_all(adata, steps="spatial")                 # warm spatial (for the blend's left)
    ss.smooth_all(adata, steps=fast_dm, auto_embed=False)  # the ONE GP solve, over every gene
    assert gp_calls["n"] == 1

    ss.smooth(adata, signature, "dm", steps=fast_dm, auto_embed=False, all_genes=True)
    ss.smooth(adata, signature[:1], "one", steps=fast_dm, auto_embed=False, all_genes=True)
    ss.smooth(
        adata, signature, "bl",
        steps=ss.Blend(left="spatial", right=fast_dm), auto_embed=False, all_genes=True,
    )
    assert gp_calls["n"] == 1, "the GP solve was reused by dm, the single gene, and blend"


@needs_kompot
def test_all_genes_gp_matches_subset_to_precision(adata, signature, monkeypatch):
    """The GP derived from the full layer equals the per-subset GP to floating-point precision.

    Not asserted bit-identical: mellon's Nystrom solve batches the columns, so a 248-wide solve
    rounds a shade differently from a 3-wide one (~1e-13 in float64). The score is stored float32,
    where that difference vanishes, but the tolerance is kept explicit and honest.
    """
    fast_dm = [ss.KompotGP(n_landmarks=100)]
    whole = _make_adata()
    subset = _make_adata()
    ss.smooth(whole, signature, "dm", steps=fast_dm, auto_embed=False, all_genes=True)
    ss.smooth(subset, signature, "dm", steps=fast_dm, auto_embed=False)  # per-subset

    np.testing.assert_allclose(
        whole.obs["dm"].to_numpy(), subset.obs["dm"].to_numpy(), atol=1e-6
    )


@needs_kompot
def test_all_genes_blend_matches_per_subset_blend(adata, signature):
    """A blend built from full layers matches the per-subset blend (GP precision aside)."""
    fast_dm = [ss.KompotGP(n_landmarks=100)]
    spec = ss.Blend(left="spatial", right=fast_dm, calibrate="std")

    whole = _make_adata()
    subset = _make_adata()
    ss.smooth(whole, signature, "bl", steps=spec, auto_embed=False, all_genes=True)
    ss.smooth(subset, signature, "bl", steps=spec, auto_embed=False)

    np.testing.assert_allclose(
        whole.obs["bl"].to_numpy(), subset.obs["bl"].to_numpy(), atol=1e-5
    )
