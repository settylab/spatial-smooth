"""The ``blend`` mode: a symmetric, range-calibrated mean of two independent views.

``blend`` runs two branches independently on the raw expression, standardises each branch's
score, averages them, and rescales the average back onto the raw score's scale. The load-bearing
property is the calibration: a blended field must share a colour bar with ``raw``, ``spatial`` and
``dm`` -- it must live in raw-score units, not the z-units the un-calibrated average lives in --
while staying a distinct field, equal to neither parent.
"""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss
from conftest import needs_kompot


def _rng(x) -> float:
    x = np.asarray(x, float)
    return float(x.max() - x.min())


# --------------------------------------------------------------------------------------- #
# spec plumbing -- no smoothing backend needed                                             #
# --------------------------------------------------------------------------------------- #
def test_as_blend_recognises_the_spec():
    assert isinstance(ss.as_blend("blend"), ss.Blend)
    assert isinstance(ss.as_blend("  BLEND "), ss.Blend)  # whitespace/case insensitive
    b = ss.Blend(left="spatial", right="dm", calibrate="iqr")
    assert ss.as_blend(b) is b
    # everything else falls through to the linear resolver
    assert ss.as_blend("spatial") is None
    assert ss.as_blend("dm+spatial") is None
    assert ss.as_blend([ss.KnnGaussian()]) is None


def test_blend_defaults_are_spatial_and_dm():
    b = ss.Blend()
    assert [s.kind for s in ss.resolve_steps(b.left)] == ["knn_gaussian"]
    assert [s.kind for s in ss.resolve_steps(b.right)] == ["kompot_gp"]
    assert b.calibrate == "std"


def test_blend_is_not_a_linear_pipeline():
    """`resolve_steps('blend')` must fail loudly and point at the right entry point."""
    with pytest.raises(ValueError, match="not a linear pipeline"):
        ss.resolve_steps("blend")


# --------------------------------------------------------------------------------------- #
# calibration math -- exercised with two spatial branches, so no kompot required           #
# --------------------------------------------------------------------------------------- #
def _two_spatial_branches():
    """Two genuinely different spatial smoothers -> a well-defined blend without kompot."""
    return ss.Blend(
        left=[ss.KnnGaussian()],
        right=[ss.KnnGaussian(k=60, sigma=2.0)],
        calibrate="std",
    )


def test_blend_std_calibration_matches_raw_moments(adata, signature):
    """`calibrate='std'` pins the blended field's mean and std to the raw score's, exactly."""
    ss.smooth(adata, signature, "bl", steps=_two_spatial_branches())
    bl = adata.obs["bl"].to_numpy()
    raw = adata.obs["bl_raw"].to_numpy()

    assert np.isfinite(bl).all()
    # First two moments match to numerical precision -- that is what the affine map guarantees.
    assert bl.mean() == pytest.approx(raw.mean(), abs=1e-5)
    assert bl.std() == pytest.approx(raw.std(), rel=1e-5)
    # And the overall extent is comparable, not an order of magnitude off.
    assert 0.5 < _rng(bl) / _rng(raw) < 2.0


def test_blend_calibration_is_not_z_units(adata, signature):
    """Without calibration the field sits on a different scale; calibration fixes exactly that."""
    left = [ss.KnnGaussian()]
    right = [ss.KnnGaussian(k=60, sigma=2.0)]
    ss.smooth(adata, signature, "cal", steps=ss.Blend(left=left, right=right, calibrate="std"))
    ss.smooth(adata, signature, "raw_z", steps=ss.Blend(left=left, right=right, calibrate="none"))

    raw = adata.obs["cal_raw"].to_numpy()
    calibrated = adata.obs["cal"].to_numpy()
    uncalibrated = adata.obs["raw_z"].to_numpy()

    # The calibrated field matches the raw score's spread; the un-calibrated one does not.
    assert calibrated.std() == pytest.approx(raw.std(), rel=1e-5)
    assert abs(uncalibrated.std() - raw.std()) > 0.1 * raw.std()
    # The two blends are the same field up to an affine map, so they stay perfectly correlated:
    # calibration only rescales, it never reorders cells.
    assert np.corrcoef(calibrated, uncalibrated)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_blend_iqr_calibration_matches_robust_stats(adata, signature):
    ss.smooth(
        adata,
        signature,
        "bl",
        steps=ss.Blend(left=[ss.KnnGaussian()], right=[ss.KnnGaussian(k=60, sigma=2.0)], calibrate="iqr"),
    )
    bl = adata.obs["bl"].to_numpy()
    raw = adata.obs["bl_raw"].to_numpy()
    assert np.median(bl) == pytest.approx(np.median(raw), abs=1e-5)
    q_bl = np.percentile(bl, 75) - np.percentile(bl, 25)
    q_raw = np.percentile(raw, 75) - np.percentile(raw, 25)
    assert q_bl == pytest.approx(q_raw, rel=1e-5)


def test_blend_rejects_unknown_calibration(adata, signature):
    with pytest.raises(ValueError, match="unknown blend calibrate"):
        ss.smooth(
            adata,
            signature,
            "bl",
            steps=ss.Blend(left=[ss.KnnGaussian()], right=[ss.KnnGaussian(k=60)], calibrate="nope"),
        )


def test_blend_provenance_records_both_branches(adata, signature):
    ss.smooth(adata, signature, "bl", steps=_two_spatial_branches())
    prov = ss.provenance(adata, "bl")
    (step,) = prov["steps"]
    assert step["kind"] == "blend"
    assert step["calibrate"] == "std"
    assert [s["kind"] for s in step["left"]] == ["knn_gaussian"]
    assert [s["kind"] for s in step["right"]] == ["knn_gaussian"]
    # The realised affine map is recorded, so a stored result documents its own calibration.
    resolved = step["resolved"]
    assert resolved["blend_std"] == pytest.approx(resolved["raw_std"], rel=1e-5)
    assert set(resolved) >= {"scale", "shift", "blend_std_precalibration", "blend_std", "raw_std"}


def test_blend_store_genes_warns(adata, signature):
    """A blend combines scores, not a gene matrix, so `store_genes` writes nothing and warns."""
    with pytest.warns(UserWarning, match="store_genes has no effect"):
        ss.smooth(adata, signature, "bl", steps=_two_spatial_branches(), store_genes=True)
    assert "bl_smoothed" not in adata.obsm


# --------------------------------------------------------------------------------------- #
# the real thing -- the default spatial+cell-state blend                                   #
# --------------------------------------------------------------------------------------- #
@needs_kompot
def test_blend_default_is_spatial_and_dm_and_distinct(adata, signature):
    """`steps='blend'` combines the spatial and cell-state views, distinct from both parents."""
    fast_dm = [ss.KompotGP(n_landmarks=100)]
    ss.smooth(adata, signature, "sp", steps="spatial")
    ss.smooth(adata, signature, "dm", steps=fast_dm, auto_embed=False)
    ss.smooth(
        adata, signature, "bl",
        steps=ss.Blend(left="spatial", right=fast_dm, calibrate="std"), auto_embed=False,
    )

    sp = adata.obs["sp"].to_numpy()
    dm = adata.obs["dm"].to_numpy()
    bl = adata.obs["bl"].to_numpy()
    raw = adata.obs["bl_raw"].to_numpy()

    assert np.isfinite(bl).all()
    # (a) calibrated to the raw score's scale ...
    assert bl.std() == pytest.approx(raw.std(), rel=1e-5)
    assert bl.mean() == pytest.approx(raw.mean(), abs=1e-5)
    assert 0.5 < _rng(bl) / _rng(raw) < 2.0
    # ... and comparable to the other modes, not orders of magnitude off.
    assert 0.3 < bl.std() / sp.std() < 3.0
    assert 0.3 < bl.std() / dm.std() < 3.0
    # (b) a genuinely distinct field: equal to neither parent ...
    assert not np.allclose(bl, sp)
    assert not np.allclose(bl, dm)
    # ... and symmetric -- it does not collapse onto one parent the way a composition does.
    corr_sp = np.corrcoef(bl, sp)[0, 1]
    corr_dm = np.corrcoef(bl, dm)[0, 1]
    assert corr_sp > 0.5 and corr_dm > 0.5
    assert abs(corr_sp - corr_dm) < 0.25


@needs_kompot
def test_blend_string_shorthand_runs(adata, signature):
    """The bare `steps='blend'` shorthand runs end to end and stores a blend result."""
    ss.smooth(adata, signature, "bl", steps="blend")
    assert "bl" in adata.obs and "bl_raw" in adata.obs
    assert np.isfinite(adata.obs["bl"].to_numpy()).all()
    assert ss.provenance(adata, "bl")["steps"][0]["kind"] == "blend"
    assert adata.obs["bl"].to_numpy().std() == pytest.approx(
        adata.obs["bl_raw"].to_numpy().std(), rel=1e-5
    )
