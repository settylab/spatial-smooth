"""The compute entry point: :func:`smooth`, plus the AnnData storage contract.

Everything :func:`smooth` produces is written into the ``AnnData`` object under documented keys.
Nothing else is needed to render the result later -- :mod:`spatial_smooth.plot` reads those keys
and never recomputes, so a smoothed object can be written to ``.h5ad``, shipped, reloaded, and
plotted on a laptop without ``kompot``, ``KDEpy`` or ``palantir`` installed.

Storage contract
----------------
=====================================  ==============================================================
key                                    contents
=====================================  ==============================================================
``adata.obs[name]``                    smoothed signature score, ``float32``
``adata.obs[f"{name}_raw"]``           unsmoothed score from the same genes and combiner
``adata.obsm[f"{name}_smoothed"]``     ``(n_obs, n_genes)`` smoothed expression (``store_genes=True``)
``adata.uns["spatial_smooth"][name]``  provenance: genes, pipeline, resolved bandwidths, version
``adata.layers["_sscache_<hash>"]``    smoothing-cache artifact: a step's output (``cache=True``)
``adata.uns["spatial_smooth_cache"]``  smoothing-cache index: hash -> layer-key + params (JSON)
=====================================  ==============================================================

The last two rows are the reuse cache (:func:`smooth`'s ``cache`` argument): each smoother's
output, keyed by a hash of its input, parameters and basis, so a repeated computation -- e.g.
``"blend"``'s branches re-running the ``"spatial"`` and ``"dm"`` steps -- is served from the layer
instead of recomputed. Matrices live in ``layers``, the tiny index in ``uns``. Both serialize with
the object; :func:`clear_smooth_cache` drops them, ``smooth(..., cache=False)`` opts out.

Reuse across *signatures* is one flag further: ``smooth(..., all_genes=True)`` (and
:func:`smooth_all`) smooth the **whole** ``(n_obs, n_vars)`` matrix, so the cache key stops
depending on which genes you asked for. Smooth every gene through ``"spatial"`` and ``"dm"`` once
and every later signature, single gene and ``"blend"`` reads those two pre-smoothed layers -- the
diffusion-map GP runs exactly once total, and each signature's score is gathered from the full
layer rather than re-smoothed.

Scoring contract
----------------
The multi-gene score is ``mean_z`` by default: each gene is standardised and the standardised
genes are averaged. **The mean and standard deviation always come from the raw matrix**, for both
the raw and the smoothed score. Two consequences, both intended:

* raw and smoothed scores share one scale, so they can go on a common colour bar; and
* for a row-stochastic smoother (:class:`~spatial_smooth.steps.KnnGaussian`,
  :class:`~spatial_smooth.steps.Kde`) *smoothing the genes and then scoring* is exactly
  *scoring and then smoothing the score* -- the two orders commute, because such a smoother is
  linear and maps constants to themselves. Gene-level is what the pipeline does, which keeps a
  Gaussian-process step (which does not commute) meaningful in the same framework.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._deps import require
from .steps import DM_KEY, Step, StepSpec, as_blend, resolve_steps

__all__ = [
    "smooth",
    "smooth_all",
    "select_cells",
    "provenance",
    "list_results",
    "compute_diffusion_map",
    "clear_smooth_cache",
    "UNS_KEY",
    "CACHE_KEY",
    "SCORE_METHODS",
]

#: Top-level ``adata.uns`` key under which every result's provenance lives.
UNS_KEY = "spatial_smooth"

#: ``adata.uns`` key holding the smoothing cache *index* (hash -> layer-key + param
#: signature + resolved params), JSON-encoded. The smoothed matrices themselves live in
#: ``adata.layers`` under :data:`CACHE_LAYER_PREFIX`-namespaced keys, never here -- the index
#: stays small and the artefacts are ordinary layers you can inspect or clear.
CACHE_KEY = "spatial_smooth_cache"

#: Namespace prefix for cache-artefact layers. A leading underscore and this obvious tag make
#: every cached matrix trivially recognisable and clearable (:func:`clear_smooth_cache`).
CACHE_LAYER_PREFIX = "_sscache_"

#: Default LRU cap on the number of cached smoothings kept on one AnnData. Bounds the ``.h5ad``
#: bloat the operator flagged; override per call with ``smooth(..., cache_max_entries=N)``.
SMOOTH_CACHE_MAX_ENTRIES = 64

#: Supported multi-gene score combiners.
SCORE_METHODS = ("mean_z", "mean")


# --------------------------------------------------------------------------------------- #
# embedding                                                                                #
# --------------------------------------------------------------------------------------- #
def compute_diffusion_map(
    adata,
    *,
    obsm_key: str = DM_KEY,
    n_components: int = 10,
    knn: int = 30,
    n_pca_components: int = 50,
    use_hvg: bool = False,
    random_state: int = 0,
    recompute: bool = False,
):
    """Compute a Palantir diffusion map and store it in ``adata.obsm[obsm_key]``.

    A thin wrapper over ``palantir.utils.run_pca`` + ``palantir.utils.run_diffusion_maps``. The
    diffusion map is the cell-state embedding a :class:`~spatial_smooth.steps.KompotGP` step
    smooths over by default: nearby cells are transcriptionally similar, so smoothing there
    denoises along biological structure rather than physical position.

    Idempotent -- returns immediately if ``obsm_key`` already exists, unless ``recompute``.

    Parameters
    ----------
    adata
        Normalised, log-transformed expression.
    obsm_key
        Destination key. The default matches kompot's expectation.
    n_components, knn, n_pca_components, use_hvg, random_state
        Forwarded to Palantir.
    recompute
        Recompute even when ``obsm_key`` is present.

    Returns
    -------
    AnnData
        The same object, for chaining.
    """
    if obsm_key in adata.obsm and not recompute:
        return adata
    palantir = require("palantir")

    if "X_pca" not in adata.obsm or recompute:
        palantir.utils.run_pca(adata, n_components=n_pca_components, use_hvg=use_hvg, pca_key="X_pca")
    palantir.utils.run_diffusion_maps(
        adata,
        n_components=n_components,
        knn=knn,
        seed=random_state,
        pca_key="X_pca",
        eigvec_key=obsm_key,
    )
    return adata


# --------------------------------------------------------------------------------------- #
# helpers                                                                                  #
# --------------------------------------------------------------------------------------- #
def select_cells(
    adata,
    obs_key: str,
    *,
    include: Optional[Iterable] = None,
    exclude: Optional[Iterable] = None,
):
    """Boolean mask over ``adata.obs`` selecting the cells to keep.

    ``include`` keeps only the listed values; ``exclude`` drops the listed values; together they
    apply in that order. Values are compared as strings, so numeric and categorical columns both
    work.
    """
    np = require("numpy")
    if obs_key not in adata.obs:
        raise KeyError(f"{obs_key!r} not in adata.obs (have {list(adata.obs.columns)[:20]})")
    values = adata.obs[obs_key].astype(str).to_numpy()
    mask = np.ones(adata.n_obs, dtype=bool)
    if include is not None:
        mask &= np.isin(values, [str(v) for v in include])
    if exclude is not None:
        mask &= ~np.isin(values, [str(v) for v in exclude])
    return mask


def _require_finite_genes(matrix, genes: Sequence[str]) -> None:
    """Reject NaN/inf in the expression matrix, naming the gene that carries it.

    Validated here, at the single point every pipeline passes through, rather than inside each
    step: `KnnGaussian` and `Kde` guarded themselves while `KompotGP` did not, so `steps="dm"`
    returned an all-NaN score for every cell with no exception and no warning. A step-local
    invariant is only as good as the steps that implement it.
    """
    np = require("numpy")
    bad = ~np.isfinite(matrix)
    if not bad.any():
        return
    columns = np.flatnonzero(bad.any(axis=0))
    named = ", ".join(f"{genes[j]!r} ({int(bad[:, j].sum())} cells)" for j in columns[:5])
    more = " ..." if columns.size > 5 else ""
    raise ValueError(
        f"expression contains non-finite values in {columns.size} gene(s): {named}{more}. "
        "Drop or impute them before smoothing -- a missing value is neither a constant nor a "
        "measurement, and smoothing would spread it across the tissue."
    )


def _gene_matrix(adata, genes: Sequence[str], layer: Optional[str]):
    """Dense ``(n_obs, n_genes)`` float64 matrix for ``genes`` out of ``layer`` (or ``.X``)."""
    np = require("numpy")
    idx = adata.var_names.get_indexer(genes)
    missing = [g for g, i in zip(genes, idx) if i < 0]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise KeyError(f"genes not in adata.var_names: {shown}")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"layer {layer!r} not in adata.layers (have {list(adata.layers)})")
    source = adata.X if layer is None else adata.layers[layer]
    block = source[:, idx]
    if hasattr(block, "toarray"):
        block = block.toarray()
    return np.asarray(block, dtype=np.float64)


def _combine(matrix, score: str, stats):
    """Collapse an ``(n, g)`` matrix to an ``(n,)`` score using statistics from the raw matrix."""
    np = require("numpy")
    if score == "mean":
        return matrix.mean(axis=1)
    if score == "mean_z":
        mu, sd = stats
        return ((matrix - mu) / sd).mean(axis=1)
    raise ValueError(f"unknown score {score!r}; use one of {SCORE_METHODS}")


def _raw_stats(matrix):
    np = require("numpy")
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)  # constant genes contribute nothing
    return mu, sd


# --------------------------------------------------------------------------------------- #
# smoothing cache                                                                          #
# --------------------------------------------------------------------------------------- #
# A *step* is a pure function of three things: the matrix it consumes, its own parameters, and
# the basis (``obsm``) it smooths over. Hash those three and you can reuse a smoother's output
# whenever the identical computation recurs -- which is exactly what happens when a four-mode
# figure runs ``spatial``, ``dm`` and ``blend``: blend's two branches reproduce the ``spatial``
# and ``dm`` steps verbatim, so with the cache on the expensive diffusion-map GP runs *once*
# total instead of twice. Invalidation is automatic: change the input, a parameter, or the
# embedding and the hash changes, so a stale result can never be served.


def _step_param_signature(step: Step) -> str:
    """Canonical, deterministic parameter signature of a step (class + sorted fields)."""
    return json.dumps(step.to_dict(), sort_keys=True)


def _array_digest(h, tag: bytes, arr) -> None:
    """Fold an array's shape, dtype and exact bytes into a running hash under ``tag``."""
    np = require("numpy")
    arr = np.ascontiguousarray(arr)
    h.update(tag)
    h.update(repr(arr.shape).encode())
    h.update(b"|")
    h.update(str(arr.dtype).encode())
    h.update(b"|")
    h.update(arr.tobytes())


def _group_bytes(column) -> bytes:
    """Deterministic byte encoding of an ``obs`` column's *values* (order-sensitive).

    NUL-joined string values, so two columns with different contents can never collide and no
    concatenation aliases another (``["a", "bc"]`` differs from ``["ab", "c"]``).
    """
    np = require("numpy")
    return "\x00".join(str(v) for v in np.asarray(column)).encode("utf-8")


def _cache_key(step: Step, input_matrix, adata) -> str:
    """``sha256`` over (input matrix bytes, canonical step params, basis bytes, grouping bytes).

    The input matrix is hashed as its exact stored floats -- ``np.ascontiguousarray(...).tobytes()``
    plus shape and dtype -- never a re-derived or rounded copy, so the key matches the values a
    cache hit will replay bit for bit.

    A condition-aware :class:`~spatial_smooth.steps.KompotGP` (``groupby``/``condition``) *fits* on
    the cells with ``adata.obs[groupby] == condition`` and evaluates everywhere. The step's
    parameter signature records the ``groupby`` *column name* and the ``condition`` *label* -- but
    not the column's *contents*. Were only the name hashed, mutating that grouping column in place
    between two otherwise-identical calls would leave the key unchanged and serve a **stale** result
    fitted on the old grouping. So the column's values are folded into the key: change the grouping,
    change the key, miss the cache. (For steps without ``groupby`` nothing extra is hashed, so their
    keys are unaffected.)
    """
    h = hashlib.sha256()
    _array_digest(h, b"matrix|", input_matrix)
    h.update(b"step|")
    h.update(_step_param_signature(step).encode())
    _array_digest(h, b"basis|", adata.obsm[step.basis])
    groupby = getattr(step, "groupby", None)
    if groupby is not None and groupby in adata.obs:
        h.update(b"groupby|")
        h.update(str(groupby).encode())
        h.update(b"|groupvals|")
        h.update(_group_bytes(adata.obs[groupby]))
    return h.hexdigest()


def _empty_cache_index() -> Dict[str, Any]:
    return {"version": 1, "entries": {}, "order": []}


def _load_cache_index(adata) -> Dict[str, Any]:
    """Decode the JSON cache index from ``adata.uns`` (empty scaffold if absent/corrupt)."""
    raw = adata.uns.get(CACHE_KEY)
    if raw is None:
        return _empty_cache_index()
    if not isinstance(raw, str):  # h5ad may hand back a numpy str_
        raw = str(raw)
    try:
        idx = json.loads(raw)
    except (ValueError, TypeError):  # pragma: no cover - defensive against a mangled index
        return _empty_cache_index()
    idx.setdefault("version", 1)
    idx.setdefault("entries", {})
    idx.setdefault("order", [])
    return idx


def _save_cache_index(adata, idx: Dict[str, Any]) -> None:
    adata.uns[CACHE_KEY] = json.dumps(idx)


def _cache_get(adata, key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Return ``(matrix, resolved)`` for a cache hit, or ``None`` on a miss.

    A hit bumps the key to most-recently-used. A dangling entry (index says hit, layer gone --
    e.g. a manual ``del adata.layers[...]``) is pruned and treated as a miss.
    """
    np = require("numpy")
    idx = _load_cache_index(adata)
    entry = idx["entries"].get(key)
    if entry is None:
        return None
    layer_key = entry["layer"]
    if layer_key not in adata.layers:
        idx["entries"].pop(key, None)
        if key in idx["order"]:
            idx["order"].remove(key)
        _save_cache_index(adata, idx)
        return None
    cols = list(entry["cols"])
    full = np.asarray(adata.layers[layer_key])
    matrix = np.ascontiguousarray(full[:, cols], dtype=np.float64)
    if key in idx["order"]:
        idx["order"].remove(key)
    idx["order"].append(key)
    _save_cache_index(adata, idx)
    return matrix, dict(entry.get("resolved", {}))


def _cache_put(adata, key: str, step: Step, out_matrix, genes, resolved, cache_max_entries) -> None:
    """Store a step's output as a namespaced layer and record the index entry (LRU-evicting)."""
    np = require("numpy")
    idx = _load_cache_index(adata)
    entries, order = idx["entries"], idx["order"]

    # A layer must be (n_obs, n_vars); a signature smoothing is (n_obs, n_genes). Scatter the
    # signature columns into a full-width layer (rest NaN) and record which columns they are, so
    # a hit gathers exactly the stored floats back. float64 is kept, not the float32 of
    # `store_genes`, so a replayed matrix is byte-identical to a fresh compute.
    cols = [int(c) for c in adata.var_names.get_indexer(genes)]
    layer_key = f"{CACHE_LAYER_PREFIX}{key}"
    full = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float64)
    full[:, cols] = np.asarray(out_matrix, dtype=np.float64)
    adata.layers[layer_key] = full

    entries[key] = {
        "layer": layer_key,
        "cols": cols,
        "resolved": resolved,
        "params": _step_param_signature(step),
    }
    if key in order:
        order.remove(key)
    order.append(key)

    cap = SMOOTH_CACHE_MAX_ENTRIES if cache_max_entries is None else int(cache_max_entries)
    if cap is not None and cap > 0:
        while len(order) > cap:
            evicted = order.pop(0)
            old = entries.pop(evicted, None)
            if old is not None and old["layer"] in adata.layers:
                del adata.layers[old["layer"]]
    _save_cache_index(adata, idx)


def clear_smooth_cache(adata) -> int:
    """Remove every smoothing-cache artefact from ``adata`` and return how many layers were dropped.

    Deletes the :data:`CACHE_LAYER_PREFIX`-namespaced layers and the :data:`CACHE_KEY` index in
    ``uns``. Stored *results* (``obs``/``obsm``/``uns['spatial_smooth']``) are untouched -- only
    the reuse cache is cleared, so plotting a previously smoothed result still works afterwards.
    """
    removed = 0
    for layer_key in [
        name
        for name in list(adata.layers)
        if isinstance(name, str) and name.startswith(CACHE_LAYER_PREFIX)
    ]:
        del adata.layers[layer_key]
        removed += 1
    adata.uns.pop(CACHE_KEY, None)
    return removed


def _run_pipeline(pipeline, raw_matrix, adata, genes, progress, *, cache=False, cache_max_entries=None):
    """Apply a linear pipeline to ``raw_matrix``, returning the smoothed matrix and step records.

    Factored out of :func:`smooth` so that a :class:`~spatial_smooth.steps.Blend` can run two
    independent branches through the identical machinery. Each step consumes the previous step's
    output; the shape is asserted to be preserved so a misbehaving step fails loudly.

    When ``cache`` is set, each step's output is memoized on ``adata`` keyed by
    :func:`_cache_key`; a hit skips the smoother and replays the stored matrix (and its resolved
    params) verbatim, so the step records -- and every downstream output -- are byte-identical to
    a cache-off run.
    """
    np = require("numpy")
    matrix = raw_matrix
    step_records: List[Dict[str, Any]] = []
    for step in pipeline:
        key = _cache_key(step, matrix, adata) if cache else None
        hit = _cache_get(adata, key) if cache else None
        if hit is not None:
            matrix, resolved = hit
        else:
            matrix, resolved = step.apply(matrix, adata, genes, progress=progress)
            matrix = np.asarray(matrix, dtype=np.float64)
            if cache:
                _cache_put(adata, key, step, matrix, genes, resolved, cache_max_entries)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != raw_matrix.shape:  # pragma: no cover - defensive
            raise RuntimeError(
                f"step {type(step).__name__} changed the matrix shape "
                f"{raw_matrix.shape} -> {matrix.shape}"
            )
        record = step.to_dict()
        record["resolved"] = resolved
        step_records.append(record)
    return matrix, step_records


def _standardize(x):
    """Zero-mean, unit-variance version of a 1-D score, guarding a constant field."""
    np = require("numpy")
    sd = float(x.std())
    sd = 1.0 if sd == 0 else sd
    return (x - x.mean()) / sd


def _affine_to_match(values, target, *, method: str):
    """Return ``(a, b)`` so ``a * values + b`` matches ``target``'s centre and spread.

    ``method="std"`` matches the mean and standard deviation (the first two moments exactly);
    ``method="iqr"`` matches the median and inter-quartile range (robust to outlier cells). The
    map is affine and, because both scales are non-negative, monotone -- it never reorders cells.
    """
    np = require("numpy")
    if method == "std":
        src_center, src_scale = float(values.mean()), float(values.std())
        tgt_center, tgt_scale = float(target.mean()), float(target.std())
    elif method == "iqr":
        src_center = float(np.median(values))
        sq1, sq3 = np.percentile(values, [25, 75])
        src_scale = float(sq3 - sq1)
        tgt_center = float(np.median(target))
        tq1, tq3 = np.percentile(target, [25, 75])
        tgt_scale = float(tq3 - tq1)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown calibrate method {method!r}")
    src_scale = 1.0 if src_scale == 0 else src_scale
    a = tgt_scale / src_scale
    b = tgt_center - a * src_center
    return a, b


def _blend_field(left_score, right_score, raw_score, *, calibrate: str):
    """Symmetric mean of two standardized scores, affinely calibrated to ``raw_score``'s scale.

    Returns ``(blended_score, resolved)`` where ``resolved`` records the calibration method and
    the realised scale/shift plus the pre- and post-calibration spreads, for provenance.
    """
    np = require("numpy")
    from .steps import BLEND_CALIBRATIONS

    if calibrate not in BLEND_CALIBRATIONS:
        raise ValueError(
            f"unknown blend calibrate {calibrate!r}; use one of {BLEND_CALIBRATIONS}"
        )
    z = 0.5 * (_standardize(left_score) + _standardize(right_score))
    if calibrate == "none":
        blended, a, b = z, 1.0, 0.0
    else:
        a, b = _affine_to_match(z, raw_score, method=calibrate)
        blended = a * z + b
    resolved = {
        "calibrate": calibrate,
        "scale": float(a),
        "shift": float(b),
        "blend_std_precalibration": float(z.std()),
        "blend_std": float(np.asarray(blended).std()),
        "raw_std": float(np.asarray(raw_score).std()),
    }
    return blended, resolved


# --------------------------------------------------------------------------------------- #
# the compute entry point                                                                  #
# --------------------------------------------------------------------------------------- #
def smooth(
    adata,
    genes: Sequence[str],
    name: str = "signature",
    *,
    steps: StepSpec = "spatial",
    layer: Optional[str] = None,
    score: str = "mean_z",
    subset_key: Optional[str] = None,
    include: Optional[Iterable] = None,
    exclude: Optional[Iterable] = None,
    store_genes: bool = False,
    auto_embed: bool = True,
    progress: bool = False,
    cache: bool = True,
    cache_max_entries: Optional[int] = None,
    all_genes: bool = False,
    copy: bool = False,
):
    """Smooth a gene signature through a pipeline of steps and score it per cell.

    .. warning::

       The smoothed score is **for visualization only**. It is spatially autocorrelated by
       construction, so any statistic computed on it (differential expression, clustering,
       correlation, a p-value of any kind) will be badly over-confident. Plot ``obs[name]``;
       analyse ``obs[f"{name}_raw"]``.

    The one-liner smooths over physical coordinates with a Gaussian kNN kernel::

        import spatial_smooth as ss
        ss.smooth(adata, ["Prox1", "Neurod6"], "hippocampus")
        ss.pl.signature(adata, "hippocampus")

    Choose *what to smooth over* with ``steps``: ``"spatial"`` (default), ``"dm"`` (the
    expression manifold, via ``kompot.smooth_expression``), or ``"dm+spatial"`` to compose both
    -- the spatial step then consumes the manifold-denoised expression. ``"blend"`` is different:
    it runs the spatial and cell-state views *independently* on the raw expression and returns a
    symmetric, range-calibrated mean of the two, so it stays distinct from both parents (see
    :class:`~spatial_smooth.steps.Blend`). Pass :class:`~spatial_smooth.steps.Step` objects
    instead of a shorthand for full control.

    Parameters
    ----------
    adata
        Normalised, log-transformed expression with the required ``obsm`` bases.
    genes
        Signature genes. Duplicates are dropped, order preserved. One gene is fine.
    name
        Base name for the outputs (see the module docstring's storage contract).
    steps
        A shorthand (``"spatial"``, ``"dm"``, ``"dm+spatial"``, ``"spatial+dm"``,
        ``"spatial-kde"``, ``"spatial-gp"``, ``"none"``, ``"blend"``), a single ``Step``, a
        :class:`~spatial_smooth.steps.Blend`, or a sequence of steps. A list of steps is applied
        left to right, each consuming the previous step's output; ``"blend"`` /
        :class:`~spatial_smooth.steps.Blend` instead combines two independent branches (see
        above).
    layer
        Expression layer to read (``None`` -> ``adata.X``). Should be log-normalised.
    score
        Multi-gene combiner: ``"mean_z"`` (default) or ``"mean"``.
    subset_key, include, exclude
        Optional cell filter applied *before* smoothing (see :func:`select_cells`). Filtered-out
        cells neither train nor receive the field. **When a filter removes cells the returned
        object is a new, smaller AnnData** -- use the return value.
    store_genes
        Also write the smoothed ``(n_obs, n_genes)`` expression matrix to
        ``adata.obsm[f"{name}_smoothed"]``.
    auto_embed
        Compute a Palantir diffusion map when a step needs ``obsm["DM_EigenVectors"]`` and it is
        absent. Set ``False`` to fail loudly instead.
    progress
        Show the GP backend's progress bar.
    cache
        Memoize each smoother's output on ``adata``, keyed by a stable hash of its input matrix,
        parameters and basis, and reuse it on a hit (default ``True``). Its purpose is to avoid
        recomputing the same smoothing twice -- most visibly, ``"blend"``'s two branches reuse the
        results of a prior ``"spatial"`` and ``"dm"`` call, so the diffusion-map GP runs once
        across all three modes. Cached matrices live in ``adata.layers`` under
        ``"_sscache_<hash>"`` and their index in ``adata.uns['spatial_smooth_cache']``; both ride
        along in ``.h5ad``. Set ``False`` to neither read nor write the cache;
        :func:`clear_smooth_cache` removes it wholesale.
    cache_max_entries
        LRU cap on how many cached smoothings ``adata`` retains (``None`` ->
        :data:`SMOOTH_CACHE_MAX_ENTRIES`). Bounds the ``.h5ad`` growth caching introduces.
    all_genes
        Smooth **every** ``var`` through the pipeline (over the full ``(n_obs, n_vars)`` matrix),
        then derive this signature's score by gathering its columns from that full result. The
        point is reuse *across signatures*: because the smoother now sees the whole matrix, the
        cache key (:func:`_cache_key`) no longer depends on *which* genes you asked for, so a
        second call for a different signature -- or a single gene -- through the **same** pipeline
        is a cache hit and re-smooths nothing. Compute ``"spatial"`` and ``"dm"`` once with
        ``all_genes=True`` and every later signature, single gene and ``"blend"`` reads those two
        pre-smoothed layers -- the diffusion-map GP runs exactly once total. Requires ``cache=True``
        (the default) to reuse across calls; see :func:`smooth_all` to warm the layers up front
        without scoring a throwaway signature.

        The derived score is *exactly* the per-signature result for the linear neighbour smoothers
        (:class:`~spatial_smooth.steps.KnnGaussian`, :class:`~spatial_smooth.steps.Kde`), which are
        column-independent, and matches it to floating-point precision (~1e-13) for
        :class:`~spatial_smooth.steps.KompotGP`, whose Nystrom solve rounds differently at a
        different matrix width. All signatures derived from one full layer are mutually exact.
        Every ``var`` must be finite (not just the signature genes), or the call raises naming the
        offending gene.
    copy
        Work on a copy and leave the input untouched.

    Returns
    -------
    AnnData
        The object carrying the result. Identical to the input when ``copy=False`` and no cell
        filter was applied.

    Raises
    ------
    KeyError
        A gene, layer, or ``obsm`` basis is missing (the message names it).
    ImportError
        An optional backend a step needs is not installed (the message gives the pip line).

    See Also
    --------
    spatial_smooth.plot.signature : render a stored result without recomputing it.
    provenance : read back exactly what was run.
    """
    np = require("numpy")

    genes = list(dict.fromkeys(genes))
    if not genes:
        raise ValueError("`genes` is empty")
    if score not in SCORE_METHODS:
        raise ValueError(f"unknown score {score!r}; use one of {SCORE_METHODS}")

    blend_spec = as_blend(steps)
    if blend_spec is not None:
        from .steps import BLEND_CALIBRATIONS

        if blend_spec.calibrate not in BLEND_CALIBRATIONS:  # fail before the branches run
            raise ValueError(
                f"unknown blend calibrate {blend_spec.calibrate!r}; use one of {BLEND_CALIBRATIONS}"
            )
        left_pipeline: List[Step] = resolve_steps(blend_spec.left)
        right_pipeline: List[Step] = resolve_steps(blend_spec.right)
        pipeline: List[Step] = left_pipeline + right_pipeline  # for embed/basis validation only
    else:
        pipeline = resolve_steps(steps)

    if copy:
        adata = adata.copy()

    if subset_key is not None and (include is not None or exclude is not None):
        mask = select_cells(adata, subset_key, include=include, exclude=exclude)
        if mask.sum() == 0:
            raise ValueError(f"the cell filter on {subset_key!r} removed all {adata.n_obs} cells")
        if mask.sum() < adata.n_obs:
            adata = adata[mask].copy()

    if auto_embed:
        for step in pipeline:
            if step.basis == DM_KEY and DM_KEY not in adata.obsm:
                compute_diffusion_map(adata, obsm_key=DM_KEY)
    for step in pipeline:
        if step.basis not in adata.obsm:
            raise KeyError(
                f"step {type(step).__name__} needs adata.obsm[{step.basis!r}]; "
                f"available: {sorted(adata.obsm)}"
            )

    raw_matrix = _gene_matrix(adata, genes, layer)
    _require_finite_genes(raw_matrix, genes)
    stats = _raw_stats(raw_matrix)
    raw_score = _combine(raw_matrix, score, stats)

    # Choose what the *pipeline* smooths. With `all_genes`, every var goes through the smoother
    # (so the cache key stops depending on the signature and the work is shared across calls); the
    # signature's smoothed columns are then gathered out of the full result for scoring. Scoring
    # itself -- `stats`, `raw_score`, `mean_z` -- always comes from the signature's raw subset,
    # unchanged. Without `all_genes` the pipeline smooths exactly the signature subset, as before.
    if all_genes:
        pipe_genes = list(map(str, adata.var_names))
        pipe_input = _gene_matrix(adata, pipe_genes, layer)
        _require_finite_genes(pipe_input, pipe_genes)
        sig_cols = [int(c) for c in adata.var_names.get_indexer(genes)]

        def _sig(mat):
            return np.ascontiguousarray(np.asarray(mat)[:, sig_cols], dtype=np.float64)
    else:
        pipe_genes = genes
        pipe_input = raw_matrix

        def _sig(mat):
            return mat

    genes_key = ""
    if blend_spec is not None:
        # Two branches, run independently on the same raw expression, then symmetrically
        # averaged and range-calibrated. Because neither branch consumes the other, the blend
        # stays distinct from both parents -- unlike a linear "dm+spatial" composition.
        left_matrix, left_records = _run_pipeline(
            left_pipeline, pipe_input, adata, pipe_genes, progress,
            cache=cache, cache_max_entries=cache_max_entries,
        )
        right_matrix, right_records = _run_pipeline(
            right_pipeline, pipe_input, adata, pipe_genes, progress,
            cache=cache, cache_max_entries=cache_max_entries,
        )
        left_score = _combine(_sig(left_matrix), score, stats)
        right_score = _combine(_sig(right_matrix), score, stats)
        score_values, blend_resolved = _blend_field(
            left_score, right_score, raw_score, calibrate=blend_spec.calibrate
        )
        step_records = [
            {
                "kind": "blend",
                "calibrate": str(blend_spec.calibrate),
                "left": left_records,
                "right": right_records,
                "resolved": blend_resolved,
            }
        ]
        if store_genes:
            # A blend combines *scores*, not gene matrices, so there is no single smoothed
            # (n_obs, n_genes) field to store. Be explicit rather than write something wrong.
            warnings.warn(
                "store_genes has no effect for steps='blend': a blend combines the per-branch "
                "scores, not a single smoothed gene matrix, so no obsm layer is written.",
                stacklevel=2,
            )
    else:
        matrix, step_records = _run_pipeline(
            pipeline, pipe_input, adata, pipe_genes, progress,
            cache=cache, cache_max_entries=cache_max_entries,
        )
        smoothed = _sig(matrix)
        score_values = _combine(smoothed, score, stats)
        if store_genes:
            genes_key = f"{name}_smoothed"
            adata.obsm[genes_key] = np.asarray(smoothed, dtype=np.float32)

    raw_key = f"{name}_raw"
    adata.obs[raw_key] = raw_score.astype(np.float32)
    adata.obs[name] = np.asarray(score_values, dtype=np.float32)

    from . import __version__

    record = {
        "version": str(__version__),
        "name": str(name),
        "genes": [str(g) for g in genes],
        "score": str(score),
        "layer": "" if layer is None else str(layer),
        "all_genes": bool(all_genes),
        "obs_key": str(name),
        "obs_key_raw": str(raw_key),
        "obsm_key_genes": genes_key,
        "n_obs": int(adata.n_obs),
        # Serialised as JSON so the whole pipeline survives an .h5ad round-trip verbatim;
        # `provenance()` decodes it. AnnData's uns writer has no schema for a list of dicts.
        "steps_json": json.dumps(step_records, sort_keys=True),
    }
    if UNS_KEY not in adata.uns or not isinstance(adata.uns[UNS_KEY], dict):
        adata.uns[UNS_KEY] = {}
    adata.uns[UNS_KEY][name] = record
    return adata


def smooth_all(
    adata,
    steps: StepSpec = "spatial",
    *,
    layer: Optional[str] = None,
    auto_embed: bool = True,
    progress: bool = False,
    cache: bool = True,
    cache_max_entries: Optional[int] = None,
    copy: bool = False,
):
    """Smooth **every** gene through ``steps`` once, warming the cache -- no signature scored.

    A pre-pass for the ``all_genes`` workflow: run the pipeline (or, for ``"blend"``, each of its
    two branches) over the full ``(n_obs, n_vars)`` matrix and store the smoothed layers in the
    reuse cache. Nothing is written to ``obs``/``uns['spatial_smooth']`` -- this computes the
    expensive part **once, up front**, so every later ``smooth(..., all_genes=True)`` for a
    signature or a single gene through the same pipeline is a pure cache hit.

    Warm ``"spatial"`` and ``"dm"`` and you have covered ``"spatial"``, ``"dm"``, ``"blend"`` and
    any single gene through either -- the diffusion-map GP runs exactly once::

        ss.smooth_all(adata, steps="spatial")
        ss.smooth_all(adata, steps="dm")            # the one GP solve, over every gene
        ss.smooth(adata, signature, "hippocampus", steps="blend", all_genes=True)   # both hits
        ss.smooth(adata, ["Ascl1"], "ascl1", steps="dm", all_genes=True)            # hit again

    Parameters mirror :func:`smooth`'s (``layer``, ``auto_embed``, ``progress``, ``cache``,
    ``cache_max_entries``, ``copy``); there is no ``genes``, ``name`` or ``score`` because nothing
    is scored. With ``cache=False`` this is a no-op with no lasting effect (the whole point is the
    cache), so it warns.

    Returns
    -------
    AnnData
        The object, cache warmed, for chaining.
    """
    np = require("numpy")

    if copy:
        adata = adata.copy()
    if not cache:
        warnings.warn(
            "smooth_all(cache=False) computes and discards -- it stores nothing, so no later "
            "call can reuse it. Leave cache=True (the default) for the warm-up to have any effect.",
            stacklevel=2,
        )

    blend_spec = as_blend(steps)
    if blend_spec is not None:
        pipelines: List[List[Step]] = [
            resolve_steps(blend_spec.left),
            resolve_steps(blend_spec.right),
        ]
    else:
        pipelines = [resolve_steps(steps)]

    flat = [step for pipeline in pipelines for step in pipeline]
    if auto_embed:
        for step in flat:
            if step.basis == DM_KEY and DM_KEY not in adata.obsm:
                compute_diffusion_map(adata, obsm_key=DM_KEY)
    for step in flat:
        if step.basis not in adata.obsm:
            raise KeyError(
                f"step {type(step).__name__} needs adata.obsm[{step.basis!r}]; "
                f"available: {sorted(adata.obsm)}"
            )

    genes = list(map(str, adata.var_names))
    full = _gene_matrix(adata, genes, layer)
    _require_finite_genes(full, genes)
    for pipeline in pipelines:
        _run_pipeline(
            pipeline, full, adata, genes, progress,
            cache=cache, cache_max_entries=cache_max_entries,
        )
    return adata


# --------------------------------------------------------------------------------------- #
# reading results back                                                                     #
# --------------------------------------------------------------------------------------- #
def list_results(adata) -> List[str]:
    """Names of every stored smoothing result in ``adata``."""
    store = adata.uns.get(UNS_KEY, {})
    return sorted(store) if isinstance(store, dict) else []


def provenance(adata, name: str = "signature") -> Dict[str, Any]:
    """Read back what :func:`smooth` did, with the pipeline decoded.

    The returned dict is the stored record with an extra ``"steps"`` entry: the list of step
    specifications, each including a ``"resolved"`` sub-dict of the values actually used (an
    inferred bandwidth, the kompot version, ...).

    Raises
    ------
    KeyError
        If no result called ``name`` is stored -- the message lists what *is* stored.
    """
    store = adata.uns.get(UNS_KEY, {})
    if not isinstance(store, dict) or name not in store:
        available = list_results(adata)
        hint = f"available: {available}" if available else "nothing has been smoothed yet"
        raise KeyError(
            f"no stored smoothing result named {name!r} in adata.uns[{UNS_KEY!r}] ({hint}). "
            f"Run spatial_smooth.smooth(adata, genes, {name!r}, ...) first."
        )
    record = dict(store[name])
    steps_json = record.get("steps_json", "[]")
    if not isinstance(steps_json, str):  # h5ad may hand back a numpy str_
        steps_json = str(steps_json)
    record["steps"] = json.loads(steps_json)
    record["genes"] = [str(g) for g in record.get("genes", [])]
    return record
