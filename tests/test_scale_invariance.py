"""Bandwidths are scale-invariant: rescaling the coordinates must not change the field."""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from conftest import needs_kompot

SCALE = 1000.0


def test_median_nn_distance_scales_linearly(adata, adata_scaled):
    small = ss.median_nn_distance(adata.obsm["spatial"])
    large = ss.median_nn_distance(adata_scaled.obsm["spatial"])
    assert large == pytest.approx(SCALE * small, rel=1e-9)


def test_knn_gaussian_is_scale_invariant(adata, adata_scaled, signature):
    """`sigma=None` + `sigma_factor` gives the identical field on rescaled coordinates."""
    ss.smooth(adata, signature, "sig", steps="spatial")
    ss.smooth(adata_scaled, signature, "sig", steps="spatial")
    np.testing.assert_allclose(
        adata.obs["sig"].to_numpy(), adata_scaled.obs["sig"].to_numpy(), rtol=1e-5, atol=1e-6
    )
    # ... and the recorded bandwidth tracks the coordinate scale exactly.
    small = ss.provenance(adata, "sig")["steps"][0]["resolved"]["sigma_used"]
    large = ss.provenance(adata_scaled, "sig")["steps"][0]["resolved"]["sigma_used"]
    assert large == pytest.approx(SCALE * small, rel=1e-9)


def test_explicit_sigma_is_not_scale_invariant(adata, adata_scaled, signature):
    """The escape hatch behaves as advertised: an absolute sigma is in coordinate units."""
    step = ss.KnnGaussian(sigma=1.5)
    ss.smooth(adata, signature, "sig", steps=[step])
    ss.smooth(adata_scaled, signature, "sig", steps=[step])
    assert not np.allclose(adata.obs["sig"].to_numpy(), adata_scaled.obs["sig"].to_numpy())


def test_kde_is_scale_invariant(adata, adata_scaled, signature):
    pytest.importorskip("KDEpy")
    ss.smooth(adata, signature, "sig", steps="spatial-kde")
    ss.smooth(adata_scaled, signature, "sig", steps="spatial-kde")
    a = adata.obs["sig"].to_numpy()
    b = adata_scaled.obs["sig"].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.mean() > 0.95
    # The grid is rebuilt in the new units, so agreement is to grid precision, not exact.
    np.testing.assert_allclose(a[finite], b[finite], rtol=1e-3, atol=1e-4)


@needs_kompot
def test_kompot_gp_ls_factor_is_scale_invariant(adata, adata_scaled, signature):
    """mellon infers `ls` from nearest-neighbour distances, so `ls_factor` is unit-free."""
    step = ss.KompotGP(basis="spatial", ls_factor=0.3, n_landmarks=100)
    ss.smooth(adata, signature, "sig", steps=[step])
    ss.smooth(adata_scaled, signature, "sig", steps=[step])
    a = adata.obs["sig"].to_numpy()
    b = adata_scaled.obs["sig"].to_numpy()
    assert float(np.corrcoef(a, b)[0, 1]) > 0.999
