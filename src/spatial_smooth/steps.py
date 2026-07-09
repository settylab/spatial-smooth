"""Smoothing steps -- the composable units of a smoothing pipeline.

A *step* smooths an ``(n_obs, n_genes)`` expression matrix over one embedding stored in
``adata.obsm``. A pipeline is an ordered list of steps: each consumes the previous step's
output, so ``[KompotGP(), KnnGaussian()]`` first denoises along the expression manifold and then
smooths the *already-denoised* values over physical space.

Three steps ship with the package:

============================  =========================  ==============================================
step                          default basis              engine
============================  =========================  ==============================================
:class:`KnnGaussian`          ``spatial``                Gaussian kernel over ``k`` nearest neighbours
:class:`Kde`                  ``spatial``                FFT Nadaraya-Watson on a fine grid (KDEpy)
:class:`KompotGP`             ``DM_EigenVectors``        Gaussian-process regression (kompot/mellon)
============================  =========================  ==============================================

Every step is a frozen dataclass: it is a *specification*, not a fitted object. It carries no
data, is trivially serialisable into ``adata.uns`` (:meth:`Step.to_dict`), and can be reused
across datasets.

String shorthands
-----------------
:func:`resolve_steps` turns a shorthand into a pipeline, so the common cases need no imports:

==================  ====================================================  ==========================
shorthand           pipeline                                              meaning
==================  ====================================================  ==========================
``"spatial"``       ``[KnnGaussian()]``                                   spatial smoothing only
``"dm"``            ``[KompotGP()]``                                      cell-state smoothing only
``"dm+spatial"``    ``[KompotGP(), KnnGaussian()]``                       both, cell-state first
``"spatial+dm"``    ``[KnnGaussian(), KompotGP()]``                       both, spatial first
``"spatial-kde"``   ``[Kde()]``                                           spatial, KDE engine
``"spatial-gp"``    ``[KompotGP(basis="spatial", ls_factor=0.3)]``        spatial, GP engine
``"none"``          ``[]``                                                no smoothing (raw only)
==================  ====================================================  ==========================
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._deps import require

__all__ = [
    "Step",
    "KnnGaussian",
    "Kde",
    "KompotGP",
    "resolve_steps",
    "DM_KEY",
    "SPATIAL_KEY",
    "SHORTHANDS",
]

#: Default cell-state embedding key (kompot's and Palantir's convention).
DM_KEY = "DM_EigenVectors"
#: Default physical-coordinate key (the AnnData spatial convention).
SPATIAL_KEY = "spatial"


@dataclass(frozen=True)
class Step:
    """Base class for a smoothing step. Not used directly."""

    #: Key into ``adata.obsm`` giving the coordinates this step smooths over.
    basis: str = SPATIAL_KEY

    #: Human-readable name recorded in provenance.
    kind: str = field(default="step", init=False, repr=False)

    def apply(self, matrix, adata, genes: Sequence[str], *, progress: bool = False):
        """Smooth ``matrix`` (``(n_obs, n_genes)``) over ``adata.obsm[self.basis]``.

        Returns ``(smoothed_matrix, resolved_params)`` where ``resolved_params`` records the
        values actually used (e.g. an inferred bandwidth) for provenance.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable specification of this step (for ``adata.uns``)."""
        d = {"kind": self.kind}
        d.update(asdict(self))
        return d

    def _coords(self, adata):
        np = require("numpy")
        if self.basis not in adata.obsm:
            raise KeyError(
                f"{type(self).__name__} needs adata.obsm[{self.basis!r}], which is absent. "
                f"Available: {sorted(adata.obsm)}."
                + (
                    f" Call spatial_smooth.compute_diffusion_map(adata) to create {DM_KEY!r}, "
                    "or let spatial_smooth.smooth(..., auto_embed=True) do it for you."
                    if self.basis == DM_KEY
                    else ""
                )
            )
        return np.asarray(adata.obsm[self.basis], dtype=np.float64)


@dataclass(frozen=True)
class KnnGaussian(Step):
    """Gaussian kernel over the ``k`` nearest neighbours -- the fast default.

    A row-stochastic linear smoother (see :mod:`spatial_smooth.smoothers`): each cell's value
    becomes the Gaussian-weighted mean of its ``k`` nearest neighbours in ``basis``. On a
    full imaging-based slide (~1.6e5 cells) this runs in about a second, against several
    minutes for the Gaussian process at comparable output quality -- so it is the default for
    smoothing over physical tissue coordinates.

    Parameters
    ----------
    basis
        ``adata.obsm`` key to smooth over. Defaults to ``"spatial"``.
    k
        Neighbours per cell, self included.
    sigma
        Bandwidth in coordinate units. ``None`` (default) infers it scale-invariantly as
        ``sigma_factor`` x the median nearest-neighbour distance.
    sigma_factor
        Multiplier on the median nearest-neighbour distance when ``sigma`` is inferred.
    workers
        Threads for the kd-tree query (``-1`` = all cores).
    """

    basis: str = SPATIAL_KEY
    k: int = 100
    sigma: Optional[float] = None
    sigma_factor: float = 6.0
    workers: int = -1

    kind: str = field(default="knn_gaussian", init=False, repr=False)

    def apply(self, matrix, adata, genes, *, progress: bool = False):
        from .smoothers import smooth_matrix_knn_gaussian

        out, sigma_used = smooth_matrix_knn_gaussian(
            self._coords(adata),
            matrix,
            k=self.k,
            sigma=self.sigma,
            sigma_factor=self.sigma_factor,
            workers=self.workers,
        )
        return out, {"sigma_used": float(sigma_used), "k_used": int(min(self.k, adata.n_obs))}


@dataclass(frozen=True)
class Kde(Step):
    """Fine-grid FFT Nadaraya-Watson smoothing (KDEpy). Two-dimensional bases only.

    Comparable in speed to :class:`KnnGaussian` and also row-stochastic, but resolution-bound:
    detail finer than one grid cell is lost. Useful when you want a rendered *field* rather
    than a neighbour average, or when point density varies enough that a fixed ``k`` misbehaves.

    Parameters
    ----------
    basis
        ``adata.obsm`` key -- must be 2-D.
    grid_points
        Grid resolution per axis.
    bw
        Bandwidth in coordinate units. ``None`` -> ``bw_factor`` x median NN distance.
    bw_factor
        Multiplier on the median nearest-neighbour distance when ``bw`` is inferred.
    min_density_pct
        Percentile of the uniform-KDE density below which a grid cell counts as empty
        background and contributes no signal.
    """

    basis: str = SPATIAL_KEY
    grid_points: int = 1024
    bw: Optional[float] = None
    bw_factor: float = 6.0
    min_density_pct: float = 1.0

    kind: str = field(default="kde", init=False, repr=False)

    def apply(self, matrix, adata, genes, *, progress: bool = False):
        from .smoothers import smooth_matrix_kde

        out, bw_used = smooth_matrix_kde(
            self._coords(adata),
            matrix,
            grid_points=self.grid_points,
            bw=self.bw,
            bw_factor=self.bw_factor,
            min_density_pct=self.min_density_pct,
        )
        return out, {"bw_used": float(bw_used)}


@dataclass(frozen=True)
class KompotGP(Step):
    """Gaussian-process regression per gene via ``kompot.smooth_expression``.

    Fits a GP of each gene over ``basis`` and evaluates it on every cell, so neighbouring cells
    borrow statistical strength. Slower than :class:`KnnGaussian` -- the GP solve dominates --
    but it is the only step that models a length scale explicitly, returns a posterior, and
    supports *fit on one condition, evaluate everywhere* via ``groupby``/``condition``.

    Defaults target the **cell-state** use: ``basis="DM_EigenVectors"`` (a diffusion map of the
    expression manifold) with kompot's native ``ls_factor=10``.

    Length scale is scale-invariant
    -------------------------------
    With ``ls=None`` (the default) kompot/mellon infer the length scale empirically from the
    data's nearest-neighbour distances -- ``ls_base = geometric_mean(nn_distances) * e**3``
    (``mellon.parameters.compute_ls``) -- and multiply it by ``ls_factor``. Because ``ls_base``
    is proportional to the point spacing, the same ``ls_factor`` yields the same smoothing
    regardless of the coordinates' absolute scale (microns vs millimetres).

    Over a **diffusion map** the native ``ls_factor=10`` is right. Over **physical coordinates**
    it is roughly 200x the cell spacing and washes the field into a single global gradient; use
    ``ls_factor=0.3`` there (~6 cell spacings, matching :class:`KnnGaussian`'s default
    footprint). The ``"spatial-gp"`` shorthand does exactly that.

    Parameters
    ----------
    basis
        ``adata.obsm`` key to smooth over. Defaults to ``"DM_EigenVectors"``.
    sigma
        GP noise level.
    ls
        Explicit length scale in coordinate units; bypasses the empirical path entirely.
    ls_factor
        Multiplier on the empirically inferred length scale (used when ``ls is None``).
    n_landmarks
        Inducing points for mellon's Nystrom approximation. A small effective length scale
        needs enough landmarks that the landmark spacing stays below it.
    groupby, condition
        When both are given the GP is *fitted* only on cells with
        ``adata.obs[groupby] == condition`` and *evaluated* on all cells.
    random_state
        Seed for landmark selection.
    """

    basis: str = DM_KEY
    sigma: float = 1.0
    ls: Optional[float] = None
    ls_factor: float = 10.0
    n_landmarks: int = 5000
    groupby: Optional[str] = None
    condition: Optional[str] = None
    random_state: int = 0

    kind: str = field(default="kompot_gp", init=False, repr=False)

    def apply(self, matrix, adata, genes, *, progress: bool = False):
        np = require("numpy")
        pd = require("pandas")
        anndata = require("anndata")
        kompot = require("kompot")
        if not hasattr(kompot, "smooth_expression"):
            raise ImportError(
                "spatial_smooth's KompotGP step needs `kompot.smooth_expression`, which was "
                f"added in kompot 0.8.0 (found {getattr(kompot, '__version__', '?')}).\n"
                '    Install it with:  pip install "kompot>=0.8.0"'
            )
        from kompot import GPSettings, OutputSettings, StorageSettings

        genes = list(genes)
        coords = self._coords(adata)

        # Run kompot on a private, signature-sized AnnData rather than mutating the caller's
        # object: memory scales with len(genes), not the full panel, and kompot's bookkeeping
        # layers/uns never touch `adata`. This also makes the step position-independent -- it
        # smooths whatever matrix the previous step handed it, not `adata.X`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            work = anndata.AnnData(
                X=np.ascontiguousarray(matrix, dtype=np.float32),
                obs=adata.obs.copy(),
                var=pd.DataFrame(index=pd.Index(genes, dtype=str)),
            )
            work.obsm[self.basis] = coords

            result_key = "_ss_gp"
            kompot.smooth_expression(
                work,
                groupby=self.groupby,
                condition=self.condition,
                obsm_key=self.basis,
                layer=None,
                genes=genes,
                gp=GPSettings(
                    sigma=self.sigma,
                    ls=self.ls,
                    ls_factor=self.ls_factor,
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state,
                ),
                storage=StorageSettings(result_key=result_key, overwrite=True),
                output=OutputSettings(progress=progress),
            )

        hits = [
            layer
            for layer in work.layers
            if layer.startswith(result_key) and layer.endswith("_smoothed")
        ]
        if not hits:
            raise RuntimeError(
                "kompot.smooth_expression produced no smoothed layer "
                f"(result_key={result_key!r}); layers={list(work.layers)}"
            )
        out = work.layers[hits[0]]
        out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
        return np.asarray(out, dtype=np.float64), {
            "kompot_version": str(getattr(kompot, "__version__", "?")),
            "n_landmarks_used": int(min(self.n_landmarks, adata.n_obs)),
        }


#: Mapping from string shorthand to a pipeline factory. See the module docstring.
SHORTHANDS = {
    "none": lambda: [],
    "raw": lambda: [],
    "spatial": lambda: [KnnGaussian()],
    "spatial-knn": lambda: [KnnGaussian()],
    "spatial-kde": lambda: [Kde()],
    "spatial-gp": lambda: [KompotGP(basis=SPATIAL_KEY, ls_factor=0.3)],
    "dm": lambda: [KompotGP()],
    "dm+spatial": lambda: [KompotGP(), KnnGaussian()],
    "spatial+dm": lambda: [KnnGaussian(), KompotGP()],
}

StepSpec = Union[str, Step, Sequence[Union[str, Step]], None]


def resolve_steps(spec: StepSpec) -> List[Step]:
    """Turn a shorthand, a step, or a sequence of either into a list of :class:`Step`.

    Examples
    --------
    >>> resolve_steps("spatial")                       # doctest: +ELLIPSIS
    [KnnGaussian(basis='spatial', k=100, ...)]
    >>> len(resolve_steps("dm+spatial"))
    2
    >>> resolve_steps(None)
    []
    """
    if spec is None:
        return []
    if isinstance(spec, Step):
        return [spec]
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in SHORTHANDS:
            raise ValueError(
                f"unknown steps shorthand {spec!r}; expected one of "
                f"{sorted(SHORTHANDS)} or a list of Step objects"
            )
        return SHORTHANDS[key]()
    steps: List[Step] = []
    for item in spec:
        steps.extend(resolve_steps(item))
    return steps
