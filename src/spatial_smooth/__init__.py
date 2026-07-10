"""spatial_smooth -- composable smoothing of gene signatures over space and cell state.

.. warning::

   **This package is for visualization only.** It exists to make spatial regions easier to see.
   Its output is a *rendering aid*, not data.

   Do not feed smoothed values into differential expression, differential abundance, clustering,
   correlation, or any other statistic. Smoothing deliberately makes neighbouring cells resemble
   each other, so a smoothed score is spatially autocorrelated by construction: cells stop being
   independent observations, effective sample size collapses, and p-values come out far too
   small. A test run on smoothed values will find "significant" structure in pure noise.

   Look at the smoothed field. Run the statistics on ``adata.obs[f"{name}_raw"]`` -- or on the
   unsmoothed expression -- with a method that models spatial dependence.

Every cell in a single-cell or spatial assay is measured independently, so a per-cell signature
score is dominated by dropout and sampling noise. Smoothing lets neighbouring cells borrow
statistical strength: a speckled score becomes a coherent field you can actually see. *Which*
neighbours count is the scientific choice this package makes explicit.

* **Spatial smoothing** -- neighbours are physical neighbours (``obsm["spatial"]``). Recovers
  tissue architecture: niches, layers, gradients.
* **Cell-state smoothing** -- neighbours are transcriptional neighbours (a diffusion map of the
  expression manifold). Denoises along biology, ignoring where a cell sits.
* **Both, composed** -- denoise along the manifold first, then smooth the result over the tissue.

The three are one argument apart::

    import spatial_smooth as ss

    ss.smooth(adata, genes, "sig")                        # spatial only  (the default)
    ss.smooth(adata, genes, "sig", steps="dm")            # cell state only
    ss.smooth(adata, genes, "sig", steps="dm+spatial")    # both, in that order
    ss.pl.signature(adata, "sig")                         # raw vs smoothed, on tissue

Results live in the ``AnnData`` (``obs``, ``obsm``, ``uns``), so ``adata.write_h5ad(...)`` and a
later ``ss.pl.signature(...)`` re-render them **without recomputing anything**. See
:mod:`spatial_smooth.core` for the storage contract and :mod:`spatial_smooth.plot` for how
``**kwargs`` reach scanpy/squidpy.

Only ``numpy``, ``scipy``, ``pandas`` and ``anndata`` are hard requirements. The Gaussian-process
step needs ``kompot``, the diffusion map ``palantir``, the KDE step ``KDEpy``, plotting
``scanpy``/``squidpy`` -- each imported lazily and, when absent, reported with the exact
``pip install`` line. Run :func:`check_dependencies` to see where you stand.
"""
from __future__ import annotations

__version__ = "0.1.0"

from ._deps import check_dependencies
from .core import (
    UNS_KEY,
    compute_diffusion_map,
    list_results,
    provenance,
    select_cells,
    smooth,
)
from .smoothers import (
    SpatialSmoothWarning,
    TruncationWarning,
    knn_gaussian_operator,
    median_nn_distance,
    smooth_field_kde,
    smooth_field_knn_gaussian,
    smooth_matrix_kde,
    smooth_matrix_knn_gaussian,
)
from .steps import (
    DM_KEY,
    SHORTHANDS,
    SPATIAL_KEY,
    Kde,
    KnnGaussian,
    KompotGP,
    Step,
    resolve_steps,
)
from . import plot
from . import plot as pl  # scanpy-style alias

__all__ = [
    "__version__",
    # compute
    "smooth",
    "compute_diffusion_map",
    "select_cells",
    # steps
    "Step",
    "KnnGaussian",
    "Kde",
    "KompotGP",
    "resolve_steps",
    "SHORTHANDS",
    "DM_KEY",
    "SPATIAL_KEY",
    # results
    "provenance",
    "list_results",
    "UNS_KEY",
    # warnings
    "SpatialSmoothWarning",
    "TruncationWarning",
    # kernels
    "knn_gaussian_operator",
    "median_nn_distance",
    "smooth_matrix_knn_gaussian",
    "smooth_field_knn_gaussian",
    "smooth_matrix_kde",
    "smooth_field_kde",
    # plotting
    "plot",
    "pl",
    # environment
    "check_dependencies",
]
