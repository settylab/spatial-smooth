"""Kernel-level properties, including the row-stochastic commutation the scoring rests on."""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from spatial_smooth.core import _combine, _gene_matrix, _raw_stats


def test_operator_is_row_stochastic(adata):
    W, sigma = ss.knn_gaussian_operator(adata.obsm["spatial"], k=20)
    assert sigma > 0
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=1e-12)
    assert (W.data >= 0).all()
    assert W.shape == (adata.n_obs, adata.n_obs)


def test_operator_preserves_constants(adata):
    """A row-stochastic smoother leaves a constant field untouched -- the basis of the
    gene-level/score-level equivalence below."""
    W, _ = ss.knn_gaussian_operator(adata.obsm["spatial"], k=15)
    constant = np.full(adata.n_obs, 3.7)
    np.testing.assert_allclose(W @ constant, constant, rtol=1e-12, atol=1e-12)


def test_gene_level_and_score_level_smoothing_agree(adata, signature):
    """Smoothing the genes then z-scoring == z-scoring then smoothing the score.

    This is exactly why `smooth()` may smooth per gene (so a Gaussian-process step, which does
    *not* commute, stays meaningful) without changing the answer for the row-stochastic kernels.
    """
    coords = adata.obsm["spatial"]
    raw = _gene_matrix(adata, signature, None)
    stats = _raw_stats(raw)

    ss.smooth(adata, signature, "sig", steps="spatial")
    gene_level = adata.obs["sig"].to_numpy()

    raw_score = _combine(raw, "mean_z", stats)
    score_level, _ = ss.smooth_field_knn_gaussian(coords, raw_score)

    np.testing.assert_allclose(gene_level, score_level, rtol=1e-5, atol=1e-6)


def test_knn_k_of_one_is_the_identity(adata, signature):
    ss.smooth(adata, signature, "sig", steps=[ss.KnnGaussian(k=1)])
    np.testing.assert_allclose(
        adata.obs["sig"].to_numpy(), adata.obs["sig_raw"].to_numpy(), rtol=1e-5, atol=1e-6
    )


def test_kde_rejects_non_2d_basis(adata, signature):
    pytest.importorskip("KDEpy")
    with pytest.raises(ValueError, match="2-D only"):
        ss.smooth(adata, signature, "sig", steps=[ss.Kde(basis="DM_EigenVectors")])


def test_bad_sigma_rejected(adata):
    with pytest.raises(ValueError, match="sigma must be positive"):
        ss.knn_gaussian_operator(adata.obsm["spatial"], sigma=0.0)


def test_median_nn_distance_needs_two_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        ss.median_nn_distance(np.zeros((1, 2)))


def test_smoothers_are_pure_functions(adata, signature):
    """The kernels do not touch the AnnData; only `smooth()` writes."""
    before = adata.X.copy()
    ss.smooth_matrix_knn_gaussian(adata.obsm["spatial"], adata.X[:, :3])
    np.testing.assert_array_equal(adata.X, before)
    assert "signature" not in adata.obs
