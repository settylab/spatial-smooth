"""The smoothing cache: reuse a smoother's output instead of recomputing it.

The cache memoizes each pipeline step keyed by a stable hash of (input matrix bytes, canonical
step parameters, basis bytes). Two loads bear the weight:

* a repeated identical computation is served from the cache, so the smoother does **not** run
  again -- most valuably, ``"blend"``'s two branches reproduce a prior ``"spatial"`` and ``"dm"``
  call, so the expensive diffusion-map GP runs *once* across all three modes; and
* the reused result is **byte-identical** to a fresh compute -- the cache short-cuts the smoother,
  never the scoring or provenance, so a cache-on run and a cache-off run agree bit for bit.

Invalidation is by construction: change the input, a parameter, or the embedding and the hash
changes, so a stale result can never be served.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import spatial_smooth as ss
from spatial_smooth.core import CACHE_LAYER_PREFIX, _cache_key
from conftest import _make_adata, needs_kompot


def _count_apply(monkeypatch, step_cls) -> dict:
    """Wrap ``step_cls.apply`` with a call counter; monkeypatch auto-restores it after the test."""
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
# claim 1 -- a repeated identical smooth is a genuine cache hit                             #
# --------------------------------------------------------------------------------------- #
def test_repeated_smooth_is_a_cache_hit(adata, signature, monkeypatch):
    """The identical (matrix, step, basis) recurs -> the smoother is not re-run, result identical."""
    calls = _count_apply(monkeypatch, ss.KnnGaussian)

    ss.smooth(adata, signature, "a", steps="spatial")
    first = adata.obs["a"].to_numpy().copy()
    assert calls["n"] == 1  # computed once on the miss

    # Same genes, same step, same basis -> same key. Only the output name differs.
    ss.smooth(adata, signature, "b", steps="spatial")
    assert calls["n"] == 1  # the smoother did NOT run again -- served from the cache
    np.testing.assert_array_equal(adata.obs["b"].to_numpy(), first)


def test_blend_left_branch_reuses_a_prior_spatial(adata, signature, monkeypatch):
    """A blend branch that repeats a prior standalone step hits the cache (no kompot needed)."""
    calls = _count_apply(monkeypatch, ss.KnnGaussian)

    ss.smooth(adata, signature, "sp", steps="spatial")  # 1 compute
    ss.smooth(
        adata, signature, "bl",
        steps=ss.Blend(left="spatial", right=[ss.KnnGaussian(k=60, sigma=2.0)]),
    )
    # Without the cache this is 3 KnnGaussian runs (sp + blend-left + blend-right). The blend's
    # left branch is byte-for-byte the standalone `sp`, so it hits: 2 runs, not 3.
    assert calls["n"] == 2


# --------------------------------------------------------------------------------------- #
# claim 2 -- blend reuses spatial + dm, the DM GP runs ONCE, output byte-identical          #
# --------------------------------------------------------------------------------------- #
@needs_kompot
def test_blend_reuses_spatial_and_dm_so_gp_runs_once(adata, signature, monkeypatch):
    fast_dm = [ss.KompotGP(n_landmarks=100)]
    gp_calls = _count_apply(monkeypatch, ss.KompotGP)

    ss.smooth(adata, signature, "sp", steps="spatial", auto_embed=False)
    ss.smooth(adata, signature, "dm", steps=fast_dm, auto_embed=False)
    ss.smooth(
        adata, signature, "bl",
        steps=ss.Blend(left="spatial", right=fast_dm), auto_embed=False,
    )
    # The GP ran exactly once across all three modes: blend's right branch hit `dm`'s cache.
    assert gp_calls["n"] == 1

    # ... and blend's output is byte-identical to the same three modes run with the cache off.
    off = _make_adata()
    ss.smooth(off, signature, "sp", steps="spatial", auto_embed=False, cache=False)
    ss.smooth(off, signature, "dm", steps=fast_dm, auto_embed=False, cache=False)
    ss.smooth(
        off, signature, "bl",
        steps=ss.Blend(left="spatial", right=fast_dm), auto_embed=False, cache=False,
    )
    np.testing.assert_array_equal(adata.obs["bl"].to_numpy(), off.obs["bl"].to_numpy())
    # Cache off really did recompute the GP twice (dm + blend's branch); the count proves the
    # single-run above was the cache doing its job, not the object being reused some other way.
    assert gp_calls["n"] == 1 + 2


# --------------------------------------------------------------------------------------- #
# claim 3 -- a changed input, parameter or basis misses and recomputes                     #
# --------------------------------------------------------------------------------------- #
def test_changed_input_param_or_basis_misses(adata, signature, monkeypatch):
    calls = _count_apply(monkeypatch, ss.KnnGaussian)

    ss.smooth(adata, signature, "a", steps="spatial")
    assert calls["n"] == 1
    ss.smooth(adata, signature, "a2", steps="spatial")  # identical -> hit
    assert calls["n"] == 1

    # Changed expression -> new matrix bytes -> miss.
    adata.X = (np.asarray(adata.X) * 1.7).astype(np.float32)
    ss.smooth(adata, signature, "b", steps="spatial")
    assert calls["n"] == 2

    # Changed parameter -> new signature -> miss.
    ss.smooth(adata, signature, "c", steps=[ss.KnnGaussian(k=50)])
    assert calls["n"] == 3

    # Changed embedding -> new basis bytes -> miss.
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"]) * 2.0
    ss.smooth(adata, signature, "d", steps="spatial")
    assert calls["n"] == 4


# --------------------------------------------------------------------------------------- #
# claim 4 -- a condition-aware GP keys on the grouping *contents*, not just its name        #
# --------------------------------------------------------------------------------------- #
# The step's parameter signature records `groupby`'s column *name* and the `condition` label,
# but a GP fitted with groupby/condition trains only on `obs[groupby] == condition`. If the key
# hashed the name alone, mutating that column in place -- same name, different assignment --
# would leave the key unchanged and serve a result fitted on the OLD grouping: a silent stale
# hit. So `_cache_key` folds the column's contents in; change the grouping, change the key.


def _reversed_contents(col):
    """The same categorical column with its per-cell values reversed (contents, not dtype)."""
    import pandas as pd

    return pd.Categorical(np.asarray(col.astype(str))[::-1])


def test_cache_key_folds_groupby_column_contents(adata):
    """A groupby GP's key changes when the grouping column's contents change in place."""
    step = ss.KompotGP(groupby="condition", condition="a")
    matrix = np.asarray(adata.X, dtype=np.float64)

    before = _cache_key(step, matrix, adata)
    adata.obs["condition"] = _reversed_contents(adata.obs["condition"])
    after = _cache_key(step, matrix, adata)
    assert before != after, "a mutated grouping column must change the key (else a stale hit)"


def test_cache_key_ignores_unrelated_obs_for_non_groupby_steps(adata):
    """A step without `groupby` is not keyed on any obs column -- the same change leaves it alone."""
    matrix = np.asarray(adata.X, dtype=np.float64)
    plain = ss.KnnGaussian()                          # no groupby
    gp_plain = ss.KompotGP()                          # groupby is None
    before_plain = _cache_key(plain, matrix, adata)
    before_gp = _cache_key(gp_plain, matrix, adata)

    adata.obs["condition"] = _reversed_contents(adata.obs["condition"])
    assert _cache_key(plain, matrix, adata) == before_plain
    assert _cache_key(gp_plain, matrix, adata) == before_gp


@needs_kompot
def test_mutated_grouping_misses_and_recomputes(adata, signature, monkeypatch):
    """End to end: flip the grouping in place -> the GP re-runs and the score changes.

    Before the key folded the grouping's contents, the second call was a false hit that replayed
    the field fitted on the *old* grouping. Now it misses, refits on the new grouping, and the two
    fields differ.
    """
    calls = _count_apply(monkeypatch, ss.KompotGP)
    step = [ss.KompotGP(n_landmarks=100, groupby="condition", condition="a")]

    ss.smooth(adata, signature, "g1", steps=step, auto_embed=False)
    assert calls["n"] == 1
    first = adata.obs["g1"].to_numpy().copy()

    # A genuinely different assignment of the same two labels to the same cells.
    rng = np.random.default_rng(1)
    import pandas as pd

    adata.obs["condition"] = pd.Categorical(np.where(rng.random(adata.n_obs) < 0.5, "a", "b"))
    ss.smooth(adata, signature, "g2", steps=step, auto_embed=False)
    assert calls["n"] == 2, "the mutated grouping must MISS the cache and recompute the GP"
    assert not np.allclose(first, adata.obs["g2"].to_numpy()), (
        "refitting on a different grouping must yield a different field"
    )


# --------------------------------------------------------------------------------------- #
# storage contract -- the artifact is a layer, the index is in uns, nothing lands in obs   #
# --------------------------------------------------------------------------------------- #
def test_cached_artifact_is_a_layer_not_obs(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial")

    layers = _cache_layers(adata)
    assert len(layers) == 1  # the smoothed matrix is a layer ...
    assert adata.layers[layers[0]].shape == adata.shape  # ... of the full (n_obs, n_vars) shape

    # The uns entry holds only the index (a JSON string), never a matrix.
    assert ss.CACHE_KEY in adata.uns
    assert isinstance(adata.uns[ss.CACHE_KEY], str)
    index = json.loads(adata.uns[ss.CACHE_KEY])
    assert set(index["entries"]) and all("layer" in e for e in index["entries"].values())

    # Nothing cache-related leaked into obs.
    assert not any(str(col).startswith(CACHE_LAYER_PREFIX) for col in adata.obs.columns)


def test_normal_outputs_are_untouched_by_the_cache(adata, signature):
    """A cache hit short-cuts the smoother, not the score/provenance: outputs match cache-off."""
    off = _make_adata()
    ss.smooth(adata, signature, "sig", steps="spatial")                # cache on (default)
    ss.smooth(off, signature, "sig", steps="spatial", cache=False)     # cache off

    np.testing.assert_array_equal(adata.obs["sig"].to_numpy(), off.obs["sig"].to_numpy())
    np.testing.assert_array_equal(adata.obs["sig_raw"].to_numpy(), off.obs["sig_raw"].to_numpy())
    assert json.dumps(ss.provenance(adata, "sig")["steps"], sort_keys=True) == json.dumps(
        ss.provenance(off, "sig")["steps"], sort_keys=True
    )


# --------------------------------------------------------------------------------------- #
# the guards -- opt-out, clear, and the size cap                                            #
# --------------------------------------------------------------------------------------- #
def test_cache_can_be_disabled(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial", cache=False)
    assert _cache_layers(adata) == []
    assert ss.CACHE_KEY not in adata.uns


def test_clear_smooth_cache_drops_artifacts_keeps_results(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial")
    assert _cache_layers(adata)

    removed = ss.clear_smooth_cache(adata)
    assert removed == 1
    assert _cache_layers(adata) == []
    assert ss.CACHE_KEY not in adata.uns

    # The stored *result* is untouched -- only the reuse cache is gone.
    assert "sig" in adata.obs
    assert ss.provenance(adata, "sig")["steps"][0]["kind"] == "knn_gaussian"


def test_cache_max_entries_evicts_lru(adata, signature, monkeypatch):
    ss.smooth(adata, signature, "a", steps="spatial", cache_max_entries=1)
    before = adata.obs["a"].to_numpy().copy()
    ss.smooth(adata, signature, "b", steps=[ss.KnnGaussian(k=50)], cache_max_entries=1)
    assert len(_cache_layers(adata)) == 1  # the older entry was evicted at the cap

    # Re-requesting the evicted computation is a miss -> recompute, still byte-identical.
    calls = _count_apply(monkeypatch, ss.KnnGaussian)
    ss.smooth(adata, signature, "a_again", steps="spatial", cache_max_entries=1)
    assert calls["n"] == 1
    np.testing.assert_array_equal(adata.obs["a_again"].to_numpy(), before)


# --------------------------------------------------------------------------------------- #
# serialization -- the cache rides along in .h5ad and a reloaded object still hits          #
# --------------------------------------------------------------------------------------- #
def test_cache_survives_h5ad_roundtrip_and_still_hits(adata, signature, tmp_path, monkeypatch):
    import anndata as ad

    ss.smooth(adata, signature, "sig", steps="spatial")
    expected = adata.obs["sig"].to_numpy().copy()

    path = tmp_path / "cached.h5ad"
    adata.write_h5ad(path)
    reloaded = ad.read_h5ad(path)
    assert _cache_layers(reloaded)
    assert ss.CACHE_KEY in reloaded.uns

    # A repeat smooth on the reloaded object is served from the round-tripped cache.
    calls = _count_apply(monkeypatch, ss.KnnGaussian)
    ss.smooth(reloaded, signature, "sig2", steps="spatial")
    assert calls["n"] == 0  # pure hit -- the smoother never ran
    np.testing.assert_array_equal(reloaded.obs["sig2"].to_numpy(), expected)
