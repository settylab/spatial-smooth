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
=====================================  ==============================================================

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

import json
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ._deps import require
from .steps import DM_KEY, Step, StepSpec, resolve_steps

__all__ = [
    "smooth",
    "select_cells",
    "provenance",
    "list_results",
    "compute_diffusion_map",
    "UNS_KEY",
    "SCORE_METHODS",
]

#: Top-level ``adata.uns`` key under which every result's provenance lives.
UNS_KEY = "spatial_smooth"

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
    copy: bool = False,
):
    """Smooth a gene signature through a pipeline of steps and score it per cell.

    The one-liner smooths over physical coordinates with a Gaussian kNN kernel::

        import spatial_smooth as ss
        ss.smooth(adata, ["Prox1", "Neurod6"], "hippocampus")
        ss.pl.signature(adata, "hippocampus")

    Choose *what to smooth over* with ``steps``: ``"spatial"`` (default), ``"dm"`` (the
    expression manifold, via ``kompot.smooth_expression``), or ``"dm+spatial"`` to compose both
    -- the spatial step then consumes the manifold-denoised expression. Pass
    :class:`~spatial_smooth.steps.Step` objects instead of a shorthand for full control.

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
        ``"spatial-kde"``, ``"spatial-gp"``, ``"none"``), a single ``Step``, or a sequence of
        either. Applied left to right; each step consumes the previous step's output.
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

    pipeline: List[Step] = resolve_steps(steps)

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
    stats = _raw_stats(raw_matrix)

    matrix = raw_matrix
    step_records: List[Dict[str, Any]] = []
    for step in pipeline:
        matrix, resolved = step.apply(matrix, adata, genes, progress=progress)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != raw_matrix.shape:  # pragma: no cover - defensive
            raise RuntimeError(
                f"step {type(step).__name__} changed the matrix shape "
                f"{raw_matrix.shape} -> {matrix.shape}"
            )
        record = step.to_dict()
        record["resolved"] = resolved
        step_records.append(record)

    raw_key = f"{name}_raw"
    adata.obs[raw_key] = _combine(raw_matrix, score, stats).astype(np.float32)
    adata.obs[name] = _combine(matrix, score, stats).astype(np.float32)

    genes_key = ""
    if store_genes:
        genes_key = f"{name}_smoothed"
        adata.obsm[genes_key] = matrix.astype(np.float32)

    from . import __version__

    record = {
        "version": str(__version__),
        "name": str(name),
        "genes": [str(g) for g in genes],
        "score": str(score),
        "layer": "" if layer is None else str(layer),
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
