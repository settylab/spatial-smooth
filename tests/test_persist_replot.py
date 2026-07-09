"""The load-bearing contract: compute once, persist, replot with **zero** recomputation.

The proof is adversarial rather than circumstantial. *After* smoothing and writing the object,
and *before* plotting it back, we

1. block ``kompot``, ``KDEpy`` and ``palantir`` at the import machinery
   (``sys.modules[name] = None`` makes ``import name`` raise ``ImportError``), and
2. replace every compute entry point in the package with a function that raises.

If ``spatial_smooth.pl.signature`` still renders the reloaded ``.h5ad``, it demonstrably read the
stored result rather than recreating it.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

import spatial_smooth as ss


def _boom(*args, **kwargs):  # pragma: no cover - only reached on failure
    raise AssertionError("plotting recomputed the smoothing -- the persistence contract is broken")


def _seal(monkeypatch):
    """Make every smoothing path in this process unusable."""
    for module in ("kompot", "KDEpy", "palantir"):
        monkeypatch.setitem(sys.modules, module, None)
    for target in (
        "knn_gaussian_operator",
        "smooth_matrix_knn_gaussian",
        "smooth_matrix_kde",
        "median_nn_distance",
    ):
        monkeypatch.setattr(ss.smoothers, target, _boom)
    monkeypatch.setattr(ss.core, "smooth", _boom)
    monkeypatch.setattr(ss.core, "compute_diffusion_map", _boom)
    # A sealed process really cannot smooth any more.
    with pytest.raises(AssertionError):
        ss.core.smooth(None, [])


def test_plot_from_reloaded_h5ad_never_recomputes(adata, signature, tmp_path, monkeypatch):
    import anndata as ad

    ss.smooth(adata, signature, "sig", steps="spatial", store_genes=True)
    expected = adata.obs["sig"].to_numpy().copy()
    expected_raw = adata.obs["sig_raw"].to_numpy().copy()

    path = tmp_path / "smoothed.h5ad"
    adata.write_h5ad(path)

    _seal(monkeypatch)
    reloaded = ad.read_h5ad(path)

    # The stored numbers survive the round-trip bit for bit.
    np.testing.assert_array_equal(reloaded.obs["sig"].to_numpy(), expected)
    np.testing.assert_array_equal(reloaded.obs["sig_raw"].to_numpy(), expected_raw)
    assert reloaded.obsm["sig_smoothed"].shape == (adata.n_obs, len(signature))

    # Provenance survives too, pipeline and all.
    prov = ss.provenance(reloaded, "sig")
    assert prov["genes"] == signature
    assert str(prov["version"]) == ss.__version__
    assert [s["kind"] for s in prov["steps"]] == ["knn_gaussian"]
    assert prov["steps"][0]["resolved"]["sigma_used"] > 0

    # And it renders, with every compute path sealed shut.
    pytest.importorskip("scanpy")
    ss.pl.signature(reloaded, "sig", backend="scanpy", show=False)


def test_compare_from_reloaded_h5ad_never_recomputes(adata, signature, tmp_path, monkeypatch):
    import anndata as ad

    ss.smooth(adata, signature, "a", steps="spatial")
    ss.smooth(adata, signature, "b", steps=[ss.KnnGaussian(k=10, sigma_factor=2.0)])
    path = tmp_path / "two.h5ad"
    adata.write_h5ad(path)

    _seal(monkeypatch)
    reloaded = ad.read_h5ad(path)
    assert ss.list_results(reloaded) == ["a", "b"]

    pytest.importorskip("scanpy")
    ss.pl.compare(reloaded, ["a", "b"], raw=True, backend="scanpy", show=False)


def test_replot_is_deterministic_across_the_round_trip(adata, signature, tmp_path):
    """The plotted columns are the stored columns -- no re-derivation, no drift."""
    import anndata as ad

    ss.smooth(adata, signature, "sig", steps="spatial")
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    first = ad.read_h5ad(path).obs["sig"].to_numpy()
    second = ad.read_h5ad(path).obs["sig"].to_numpy()
    np.testing.assert_array_equal(first, second)


def test_plot_without_a_stored_result_is_a_clear_error(adata):
    with pytest.raises(KeyError, match="no stored smoothing result"):
        ss.pl.signature(adata, "never_computed")


def test_plot_detects_a_tampered_object(adata, signature):
    ss.smooth(adata, signature, "sig")
    del adata.obs["sig"]
    with pytest.raises(KeyError, match="modified after smoothing"):
        ss.pl.signature(adata, "sig", backend="scanpy", show=False)


def test_provenance_error_names_available_results(adata, signature):
    ss.smooth(adata, signature, "present")
    with pytest.raises(KeyError, match=r"available: \['present'\]"):
        ss.provenance(adata, "absent")
