"""The composition matrix: spatial only, cell-state only, and both."""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from conftest import needs_kompot


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def test_spatial_only_denoises(adata, signature):
    """The point of smoothing: recover the planted field better than the raw score does."""
    ss.smooth(adata, signature, "sig", steps="spatial")

    assert "sig" in adata.obs and "sig_raw" in adata.obs
    smoothed = adata.obs["sig"].to_numpy()
    raw = adata.obs["sig_raw"].to_numpy()
    truth = adata.obs["truth"].to_numpy()
    assert np.isfinite(smoothed).all()

    assert smoothed.std() < raw.std()  # noise is suppressed ...
    assert _corr(smoothed, truth) > _corr(raw, truth) + 0.2  # ... and signal is recovered
    assert _corr(smoothed, truth) > 0.9

    prov = ss.provenance(adata, "sig")
    assert [s["kind"] for s in prov["steps"]] == ["knn_gaussian"]
    assert prov["steps"][0]["resolved"]["sigma_used"] > 0
    assert prov["genes"] == signature


def test_no_steps_is_identity(adata, signature):
    ss.smooth(adata, signature, "sig", steps="none")
    np.testing.assert_allclose(adata.obs["sig"].to_numpy(), adata.obs["sig_raw"].to_numpy())
    assert ss.provenance(adata, "sig")["steps"] == []


def test_kde_step_runs(adata, signature):
    pytest.importorskip("KDEpy")
    ss.smooth(adata, signature, "sig", steps="spatial-kde")
    out = adata.obs["sig"].to_numpy()
    assert np.isfinite(out).all()
    assert out.std() < adata.obs["sig_raw"].to_numpy().std()
    assert ss.provenance(adata, "sig")["steps"][0]["resolved"]["bw_used"] > 0


@needs_kompot
def test_dm_only(adata, signature):
    ss.smooth(adata, signature, "sig", steps=[ss.KompotGP(n_landmarks=100)], auto_embed=False)
    out = adata.obs["sig"].to_numpy()
    assert np.isfinite(out).all()
    assert out.std() < adata.obs["sig_raw"].to_numpy().std()
    prov = ss.provenance(adata, "sig")
    assert prov["steps"][0]["kind"] == "kompot_gp"
    assert prov["steps"][0]["basis"] == "DM_EigenVectors"


@needs_kompot
def test_composition_matrix_all_three_differ(adata, signature):
    """spatial-only, dm-only and dm+spatial are three distinct, valid fields."""
    ss.smooth(adata, signature, "sp", steps="spatial")
    ss.smooth(adata, signature, "dm", steps=[ss.KompotGP(n_landmarks=100)], auto_embed=False)
    ss.smooth(
        adata,
        signature,
        "both",
        steps=[ss.KompotGP(n_landmarks=100), ss.KnnGaussian()],
        auto_embed=False,
    )

    sp = adata.obs["sp"].to_numpy()
    dm = adata.obs["dm"].to_numpy()
    both = adata.obs["both"].to_numpy()
    raw = adata.obs["sp_raw"].to_numpy()

    for field in (sp, dm, both):
        assert np.isfinite(field).all()
        assert field.shape == raw.shape

    # All three are genuinely different fields ...
    assert not np.allclose(sp, dm)
    assert not np.allclose(sp, both)
    assert not np.allclose(dm, both)
    # ... but composing keeps the cell-state structure and adds spatial coherence, so `both`
    # is the smoothest of the three.
    assert both.std() < dm.std()

    assert sorted(ss.list_results(adata)) == ["both", "dm", "sp"]
    assert [s["kind"] for s in ss.provenance(adata, "both")["steps"]] == [
        "kompot_gp",
        "knn_gaussian",
    ]


@needs_kompot
def test_composition_order_matters(adata, signature):
    """The second step consumes the first step's output, so the order is not symmetric."""
    ss.smooth(
        adata, signature, "dm_first",
        steps=[ss.KompotGP(n_landmarks=100), ss.KnnGaussian()], auto_embed=False,
    )
    ss.smooth(
        adata, signature, "sp_first",
        steps=[ss.KnnGaussian(), ss.KompotGP(n_landmarks=100)], auto_embed=False,
    )
    assert not np.allclose(adata.obs["dm_first"].to_numpy(), adata.obs["sp_first"].to_numpy())


def test_shorthands_resolve():
    assert [s.kind for s in ss.resolve_steps("dm+spatial")] == ["kompot_gp", "knn_gaussian"]
    assert [s.kind for s in ss.resolve_steps("spatial+dm")] == ["knn_gaussian", "kompot_gp"]
    assert ss.resolve_steps("spatial-gp")[0].basis == "spatial"
    assert ss.resolve_steps("spatial-gp")[0].ls_factor == 0.3
    assert ss.resolve_steps("dm")[0].ls_factor == 10.0
    assert ss.resolve_steps([]) == []
    with pytest.raises(ValueError, match="unknown steps shorthand"):
        ss.resolve_steps("nonsense")


def test_cell_filter_subsets_and_returns_new_object(adata, signature):
    out = ss.smooth(adata, signature, "sig", subset_key="region", exclude=["left"])
    assert out.n_obs < adata.n_obs
    assert (out.obs["region"] == "right").all()
    assert ss.provenance(out, "sig")["n_obs"] == out.n_obs

    with pytest.raises(ValueError, match="removed all"):
        ss.smooth(adata, signature, "sig", subset_key="region", include=["nowhere"])


def test_store_genes_writes_obsm(adata, signature):
    ss.smooth(adata, signature, "sig", store_genes=True)
    assert adata.obsm["sig_smoothed"].shape == (adata.n_obs, len(signature))
    assert ss.provenance(adata, "sig")["obsm_key_genes"] == "sig_smoothed"


def test_layer_and_score_options(adata, signature):
    ss.smooth(adata, signature, "z", layer="logcounts", score="mean_z")
    ss.smooth(adata, signature, "m", layer="logcounts", score="mean")
    assert not np.allclose(adata.obs["z"], adata.obs["m"])
    with pytest.raises(ValueError, match="unknown score"):
        ss.smooth(adata, signature, "x", score="median")


def test_missing_gene_and_basis_errors(adata, signature):
    with pytest.raises(KeyError, match="not in adata.var_names"):
        ss.smooth(adata, ["Sig1", "Nope"], "sig")
    del adata.obsm["spatial"]
    with pytest.raises(KeyError, match="spatial"):
        ss.smooth(adata, signature, "sig", steps="spatial")


def test_dm_basis_missing_without_auto_embed(adata, signature):
    del adata.obsm["DM_EigenVectors"]
    with pytest.raises(KeyError, match="DM_EigenVectors"):
        ss.smooth(adata, signature, "sig", steps="dm", auto_embed=False)
