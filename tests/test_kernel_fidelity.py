"""Properties the docstrings assert, checked in code.

Two of these exist because a skeptic found the prose and the behaviour disagreed:

* **D1** -- ``Kde`` claimed to be row-stochastic (constants preserved) and to satisfy the exact
  gene-then-score / score-then-gene commutation. Both were asserted only in prose, and the first
  was *false* at the constant boundary: a zero-variance gene crashed the estimator.
* **D2** -- the ``k``-neighbour truncation makes the kNN-Gaussian kernel implicitly
  density-adaptive, so the nominal ``sigma`` overstates the bandwidth the data actually sees.
  ``provenance()`` must report the effective one.
"""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from spatial_smooth.core import _combine, _gene_matrix, _raw_stats


def _constant_gene_adata(scale: float = 100.0):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 400
    coords = rng.random((n, 2)) * scale
    X = np.column_stack(
        [rng.random(n), rng.random(n), np.zeros(n)]  # 'c' is identically zero
    ).astype(np.float32)
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["a", "b", "c"]))
    adata.obsm["spatial"] = coords
    adata.obs["grp"] = pd.Categorical(["x"] * (n // 2) + ["y"] * (n - n // 2))
    return adata


# ------------------------------------------------------------------ D1: the constant boundary
def test_kde_survives_a_zero_variance_gene():
    """A panel gene absent from the selected cells is routine, not pathological."""
    pytest.importorskip("KDEpy")
    adata = _constant_gene_adata()
    ss.smooth(adata, ["a", "b", "c"], "sig", steps="spatial-kde")
    assert np.isfinite(adata.obs["sig"].to_numpy()).all()


def test_kde_on_a_single_constant_gene():
    pytest.importorskip("KDEpy")
    adata = _constant_gene_adata()
    ss.smooth(adata, ["c"], "sig", steps="spatial-kde")
    assert np.isfinite(adata.obs["sig"].to_numpy()).all()


def test_kde_survives_a_gene_constant_only_on_the_subset():
    """The realistic path: a gene that is constant *after* filtering to one region."""
    pytest.importorskip("KDEpy")
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(2)
    n = 400
    grp = np.array(["x"] * (n // 2) + ["y"] * (n - n // 2))
    a = rng.random(n)
    b = np.where(grp == "y", 0.0, rng.random(n))  # 'b' is constant on subset y
    adata = ad.AnnData(
        X=np.column_stack([a, b]).astype(np.float32), var=pd.DataFrame(index=["a", "b"])
    )
    adata.obsm["spatial"] = rng.random((n, 2)) * 100.0
    adata.obs["grp"] = pd.Categorical(grp)

    out = ss.smooth(adata, ["a", "b"], "sig", steps="spatial-kde", subset_key="grp", include=["y"])
    assert np.isfinite(out.obs["sig"].to_numpy()).all()


@pytest.mark.parametrize("step", ["knn", "kde"])
def test_row_stochastic_kernels_preserve_constants(adata, step):
    """The property both `steps.py` and `core.py` assert for BOTH kernels."""
    if step == "kde":
        pytest.importorskip("KDEpy")
    coords = adata.obsm["spatial"]
    constant = np.full(adata.n_obs, 3.7)
    if step == "knn":
        out, _ = ss.smooth_field_knn_gaussian(coords, constant)
    else:
        out, _ = ss.smooth_field_kde(coords, constant)
    np.testing.assert_allclose(out, constant, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("steps,kernel", [("spatial", "knn"), ("spatial-kde", "kde")])
def test_gene_level_equals_score_level_for_both_kernels(adata, signature, steps, kernel):
    """The exact commutation `core.py` claims -- previously tested for knn only."""
    if kernel == "kde":
        pytest.importorskip("KDEpy")
    coords = adata.obsm["spatial"]
    raw = _gene_matrix(adata, signature, None)
    stats = _raw_stats(raw)

    ss.smooth(adata, signature, "sig", steps=steps)
    gene_level = adata.obs["sig"].to_numpy()

    raw_score = _combine(raw, "mean_z", stats)
    if kernel == "knn":
        score_level, _ = ss.smooth_field_knn_gaussian(coords, raw_score)
    else:
        score_level, _ = ss.smooth_field_kde(coords, raw_score)

    np.testing.assert_allclose(gene_level, score_level, rtol=1e-4, atol=1e-5)


# --------------------------------------------------- D2: truncation, and what provenance says
def test_provenance_reports_the_effective_bandwidth(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial")
    resolved = ss.provenance(adata, "sig")["steps"][0]["resolved"]

    for key in ("sigma_used", "sigma_effective", "kernel_mass_retained",
                "sigma_effective_p1", "sigma_effective_p99"):
        assert key in resolved, f"provenance must record {key!r}"

    assert 0.0 < resolved["kernel_mass_retained"] <= 1.0
    assert 0.0 < resolved["sigma_effective"] <= resolved["sigma_used"] * 1.05
    assert resolved["sigma_effective_p1"] <= resolved["sigma_effective"] <= resolved["sigma_effective_p99"]


def test_truncation_shrinks_the_effective_bandwidth(adata, signature):
    """A small `k` truncates the Gaussian, so the kernel behaves narrower than `sigma`."""
    ss.smooth(adata, signature, "wide", steps=[ss.KnnGaussian(k=adata.n_obs)])
    with pytest.warns(UserWarning, match="truncates the kernel"):
        ss.smooth(adata, signature, "tight", steps=[ss.KnnGaussian(k=8)])

    wide = ss.provenance(adata, "wide")["steps"][0]["resolved"]
    tight = ss.provenance(adata, "tight")["steps"][0]["resolved"]

    # Same nominal sigma (both infer it from the same coordinates) ...
    assert wide["sigma_used"] == pytest.approx(tight["sigma_used"], rel=1e-9)
    # ... but the truncated kernel keeps less mass and acts narrower.
    assert tight["kernel_mass_retained"] < wide["kernel_mass_retained"]
    assert tight["sigma_effective"] < wide["sigma_effective"]


def test_severe_truncation_warns_and_names_both_numbers(adata, signature):
    with pytest.warns(UserWarning) as record:
        ss.smooth(adata, signature, "sig", steps=[ss.KnnGaussian(k=6)])
    message = str(record[0].message)
    assert "effective bandwidth" in message
    assert "nominal sigma" in message
    assert "Raise k" in message


def test_mild_truncation_does_not_warn(adata, signature):
    """The shipped default must not cry wolf."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ss.smooth(adata, signature, "sig", steps="spatial")


def test_effective_bandwidth_is_scale_invariant(adata, adata_scaled, signature):
    ss.smooth(adata, signature, "sig", steps="spatial")
    ss.smooth(adata_scaled, signature, "sig", steps="spatial")
    small = ss.provenance(adata, "sig")["steps"][0]["resolved"]
    large = ss.provenance(adata_scaled, "sig")["steps"][0]["resolved"]
    assert large["sigma_effective"] == pytest.approx(1000.0 * small["sigma_effective"], rel=1e-6)
    assert large["kernel_mass_retained"] == pytest.approx(small["kernel_mass_retained"], rel=1e-6)
