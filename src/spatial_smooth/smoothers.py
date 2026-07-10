"""Low-level smoothing kernels.

Each function here maps ``(coords, values) -> smoothed values`` and knows nothing about
:class:`~anndata.AnnData`. Two of them are *linear and row-stochastic* -- they replace each
point's value by a weighted average of its neighbours' values with weights summing to one:

* :func:`knn_gaussian_operator` -- a sparse Gaussian kernel over the ``k`` nearest neighbours.
* :func:`smooth_matrix_kde` -- a fine-grid FFT Nadaraya-Watson estimator.

Row-stochasticity is the property that makes the package's scoring contract exact: a constant
field is left unchanged, so smoothing the individual genes and then combining them into a
z-scored signature score gives *exactly* the same answer as combining first and smoothing the
score (see :func:`spatial_smooth.core.smooth`). The Gaussian-process smoother
(:class:`~spatial_smooth.steps.KompotGP`) is linear but not row-stochastic, so it is applied
per gene.

All bandwidths default to a **scale-invariant** setting: a multiple of the median
nearest-neighbour distance of ``coords``. The same factor therefore produces the same amount of
smoothing whether coordinates are in microns, millimetres, or arbitrary embedding units.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

from ._deps import require

#: Warn when the ``k``-neighbour truncation discards more than this fraction of the kernel.
#: Lives here, not in `steps`, so every public entry point inherits the warning -- the low-level
#: functions are exported too, and a user calling `smooth_field_knn_gaussian` deserves the same
#: disclosure as one calling `smooth(steps=[KnnGaussian(...)])`.
MIN_KERNEL_MASS = 0.9

__all__ = [
    "median_nn_distance",
    "knn_gaussian_operator",
    "smooth_matrix_knn_gaussian",
    "smooth_field_knn_gaussian",
    "smooth_matrix_kde",
    "smooth_field_kde",
]


def _require_finite(matrix, what: str) -> None:
    """Reject NaN/inf before it can be mistaken for something else.

    A NaN column is not a constant column, and it is not a smoothable one either. Without this
    check ``np.ptp(values)`` is ``nan``, ``not (nan > 0)`` is ``True``, and a gene carrying a
    single missing value would silently take the constant-column branch and come back unsmoothed.
    A loud failure beats a wrong answer that looks like a right one.

    Every step validates through this one function, and :func:`spatial_smooth.core.smooth`
    validates the expression matrix before any step runs. An earlier version guarded only the two
    row-stochastic kernels, and ``steps="dm"`` -- which goes through the Gaussian process -- still
    returned an all-NaN score for every cell in silence.
    """
    np = require("numpy")
    matrix = np.asarray(matrix)
    if not np.all(np.isfinite(matrix)):
        bad = np.argwhere(~np.isfinite(matrix))
        first = tuple(int(i) for i in bad[0])
        raise ValueError(
            f"{what} requires finite values; found {len(bad)} non-finite entr"
            f"{'y' if len(bad) == 1 else 'ies'} (first at index {first}). "
            "Drop or impute them before smoothing -- a missing value is neither a constant nor a "
            "measurement, and silently passing it through would corrupt the field."
        )


def median_nn_distance(coords, *, workers: int = -1) -> float:
    """Median distance from each point to its nearest *other* point.

    The natural length unit of a point cloud, and the basis of every scale-invariant
    bandwidth in this package.
    """
    np = require("numpy")
    cKDTree = require("scipy").spatial.cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 2:
        raise ValueError("need at least 2 points to estimate a nearest-neighbour distance")
    dist, _ = cKDTree(coords).query(coords, k=2, workers=workers)
    nn = dist[:, 1]
    positive = nn[nn > 0]
    if positive.size == 0:  # degenerate: all points coincide
        return 1.0
    return float(np.median(positive))


def knn_gaussian_operator(
    coords,
    *,
    k: int = 400,
    sigma: Optional[float] = None,
    sigma_factor: float = 6.0,
    workers: int = -1,
    return_info: bool = False,
):
    """Build the sparse row-stochastic Gaussian kNN smoothing operator ``W``.

    ``W[i, j] = exp(-d(i, j)**2 / (2 * sigma**2))`` for the ``k`` nearest neighbours ``j`` of
    ``i`` (self included), each row normalised to sum to one. Applying ``W`` to any field
    replaces every value with the Gaussian-weighted mean of its ``k`` nearest neighbours.

    It costs one ``cKDTree`` query (``O(n log n)``) plus a sparse mat-vec (``O(n * k)``) per
    field -- orders of magnitude cheaper than a Gaussian-process fit, and the recommended
    default for smoothing over physical tissue coordinates.

    Parameters
    ----------
    coords
        ``(n, d)`` array of coordinates (physical positions, or any embedding).
    k
        Neighbours per point, including the point itself. Capped at ``n``. See the truncation
        note below -- ``k`` is not a free performance knob, it changes the estimator.
    sigma
        Gaussian bandwidth in coordinate units. ``None`` (default) sets it scale-invariantly
        to ``sigma_factor * median_nn_distance(coords)``.
    sigma_factor
        Multiplier on the median nearest-neighbour distance when ``sigma`` is inferred. The
        default ``6.0`` is ~6 cell spacings.
    workers
        Threads for the kd-tree query (``-1`` uses all cores; cap via the environment on a
        shared machine).
    return_info
        Also return a diagnostics dict (see below).

    Returns
    -------
    W : scipy.sparse.csr_matrix
        ``(n, n)`` row-stochastic operator.
    sigma : float
        The *nominal* bandwidth: the ``sigma`` of the Gaussian before truncation.
    info : dict, only if ``return_info``
        ``kernel_mass_retained`` -- the mean fraction of the untruncated 2-D Gaussian's mass
        that falls inside each point's ``k``-neighbour radius. This assumes locally uniform point
        density, so it is an estimate of the discrete retained-weight fraction rather than an exact
        accounting; it is biased slightly low (conservative). ``sigma_effective`` -- the mean
        bandwidth actually applied, recovered from the weighted second moment
        (``sqrt(E_w[d**2] / 2)``); and its 1st/99th percentiles across points.

    Notes
    -----
    **Truncation makes the kernel density-adaptive.** Neighbours beyond the ``k``-th get zero
    weight even where ``sigma`` would give them a real one, so the *effective* bandwidth is set
    by whichever of ``sigma`` and the ``k``-neighbour radius binds first. Because the radius is
    fixed by a neighbour *count*, it shrinks in dense regions and grows in sparse ones: this is
    a truncated-Gaussian kNN smoother, not a strictly fixed-bandwidth one.

    The default ``k=400`` with ``sigma_factor=6.0`` retains ~96% of the kernel mass on 2-D
    tissue (~210 points lie within ``2 * sigma`` at typical imaging-assay densities), so the
    truncation is mild and ``sigma`` and ``sigma_effective`` nearly agree. Lowering ``k``
    tightens the smoother and *reduces* the effective bandwidth: at ``k=100`` only ~58% of the
    mass survives, ``sigma_effective`` falls to ~0.6 * ``sigma``, and it varies ~2.8x across
    cells with local density. Quote ``sigma_effective``, not ``sigma``, in a methods section --
    :func:`spatial_smooth.provenance` records both.
    """
    np = require("numpy")
    scipy = require("scipy")
    cKDTree = scipy.spatial.cKDTree
    sparse = scipy.sparse

    coords = np.asarray(coords, dtype=np.float64)
    _require_finite(coords, "knn_gaussian_operator coordinates")
    n = coords.shape[0]
    k = int(min(k, n))
    if k < 1:
        raise ValueError("k must be >= 1")

    dist, idx = cKDTree(coords).query(coords, k=k, workers=workers)
    if dist.ndim == 1:  # k == 1
        dist = dist[:, None]
        idx = idx[:, None]

    if sigma is None:
        nn = dist[:, 1] if dist.shape[1] > 1 else dist[:, 0]
        positive = nn[nn > 0]
        med = float(np.median(positive)) if positive.size else 1.0
        sigma = sigma_factor * med
    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    w /= w.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(n), k)
    W = sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    # Mass of an untruncated 2-D Gaussian inside the k-th neighbour radius, and the bandwidth
    # the truncated kernel actually behaves like (E_w[d^2] = 2 * sigma^2 for a 2-D Gaussian).
    r_k = dist[:, -1]
    mass = 1.0 - np.exp(-(r_k ** 2) / (2.0 * sigma ** 2))
    sigma_eff = np.sqrt((w * dist ** 2).sum(axis=1) / 2.0)
    info = {
        "kernel_mass_retained": float(mass.mean()),
        "sigma_effective": float(sigma_eff.mean()),
        "sigma_effective_p1": float(np.percentile(sigma_eff, 1)),
        "sigma_effective_p99": float(np.percentile(sigma_eff, 99)),
    }

    if info["kernel_mass_retained"] < MIN_KERNEL_MASS:
        warnings.warn(
            f"KnnGaussian(k={k}) truncates the kernel: only "
            f"{info['kernel_mass_retained']:.0%} of the Gaussian mass falls within each "
            f"point's {k}-neighbour radius, so the effective bandwidth is "
            f"{info['sigma_effective']:.3g} (nominal sigma {sigma:.3g}) and varies with local "
            "density. Raise k, or quote the effective bandwidth, not the nominal one.",
            UserWarning,
            stacklevel=2,
        )

    if not return_info:
        return W, sigma
    return W, sigma, info


def smooth_matrix_knn_gaussian(
    coords,
    matrix,
    *,
    k: int = 400,
    sigma: Optional[float] = None,
    sigma_factor: float = 6.0,
    workers: int = -1,
):
    """Apply :func:`knn_gaussian_operator` to every column of ``matrix``.

    See that function's Notes on ``k``-truncation: the kernel is truncated-Gaussian and
    implicitly density-adaptive, and the returned ``sigma`` is the *nominal* one.

    Parameters
    ----------
    coords
        ``(n, d)`` coordinates.
    matrix
        ``(n, g)`` field(s) to smooth -- one column per gene, or a single score column.

    Returns
    -------
    (numpy.ndarray, float)
        The ``(n, g)`` smoothed matrix and the bandwidth used.
    """
    np = require("numpy")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    _require_finite(matrix, "smooth_matrix_knn_gaussian")
    W, sigma_used = knn_gaussian_operator(
        coords, k=k, sigma=sigma, sigma_factor=sigma_factor, workers=workers
    )
    return np.asarray(W @ matrix), sigma_used


def smooth_field_knn_gaussian(coords, values, **kwargs):
    """1-D convenience wrapper around :func:`smooth_matrix_knn_gaussian`.

    Returns ``(smoothed_values, sigma_used)`` where ``smoothed_values`` has shape ``(n,)``.
    """
    out, sigma_used = smooth_matrix_knn_gaussian(coords, values, **kwargs)
    return out[:, 0], sigma_used


def smooth_matrix_kde(
    coords,
    matrix,
    *,
    grid_points: int = 1024,
    bw: Optional[float] = None,
    bw_factor: float = 6.0,
    min_density_pct: float = 1.0,
    workers: int = -1,
):
    """Fine-grid Nadaraya-Watson smoothing via FFT-KDE (`KDEpy <https://kdepy.readthedocs.io>`_).

    Estimates the smoothed field on a regular ``grid_points x grid_points`` grid as the ratio of
    two FFT-accelerated kernel density estimates -- one weighted by the field, one uniform -- then
    bilinearly interpolates back to each input point. The FFT is what makes a *fine* grid cheap:
    cost is ``O(grid_points**2 log grid_points)`` regardless of how the points cluster.

    The estimator is affine, so negative values (z-scored signatures) are handled exactly by
    shifting the field non-negative for the KDE and shifting back afterwards.

    Parameters
    ----------
    coords
        ``(n, 2)`` coordinates. Two-dimensional only -- this is a tissue-plane smoother.
    matrix
        ``(n, g)`` field(s) to smooth.
    grid_points
        Grid resolution per axis.
    bw
        Kernel bandwidth in coordinate units. ``None`` -> ``bw_factor * median_nn_distance``.
    bw_factor
        Multiplier on the median nearest-neighbour distance when ``bw`` is inferred.
    min_density_pct
        Grid cells whose uniform-KDE density falls below this percentile (of the positive
        densities) are treated as empty background: they are excluded from the Nadaraya-Watson
        ratio and then **backfilled with the field's median** before interpolation, which stops
        NaNs bleeding inward from the tissue edge. Background therefore contributes the median,
        not nothing -- it cannot invent structure, but it is not absent either.

    Returns
    -------
    (numpy.ndarray, float)
        The ``(n, g)`` smoothed matrix and the bandwidth used, in the caller's coordinate units.

    Notes
    -----
    The estimate is computed in units of the median nearest-neighbour distance. KDEpy solves for
    its kernel's practical support numerically, and that solve is not scale-free -- it raises on
    a bandwidth of a few hundred microns even though such a bandwidth is perfectly ordinary on a
    tissue section. Rescaling makes the smoother genuinely invariant to the coordinate units.

    A constant column is returned unchanged. Smoothing a constant field is the identity for any
    row-stochastic estimator, and the Nadaraya-Watson ratio cannot express it: KDEpy normalises
    its weights by their sum, so an all-zero weight vector (which a constant column becomes after
    the affine shift) divides by zero. A gene absent from the selected cells is routine, so this
    is a real input, not a pathological one.
    """
    np = require("numpy")
    scipy = require("scipy")
    FFTKDE = require("KDEpy").FFTKDE
    from scipy.interpolate import RegularGridInterpolator

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "smooth_matrix_kde is 2-D only (tissue plane); got coords with shape "
            f"{coords.shape}. Use method='knn_gaussian' or 'gp' for higher-dimensional bases."
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    _require_finite(coords, "smooth_matrix_kde coordinates")
    _require_finite(matrix, "smooth_matrix_kde")

    spacing = median_nn_distance(coords, workers=workers)
    if bw is None:
        bw = bw_factor * spacing
    bw = float(bw)
    if not bw > 0:
        raise ValueError(f"bw must be positive, got {bw}")

    # KDEpy solves for its kernel's practical support numerically, and that solve is not
    # scale-free: it fails outright once the bandwidth is large in absolute units (microns on a
    # whole slide). Work in units of the point spacing, where the bandwidth is O(1), and report
    # the bandwidth back in the caller's units. The estimator is invariant to this rescaling.
    coords = coords / spacing
    bw_scaled = bw / spacing

    # Uniform density on the shared grid: the Nadaraya-Watson denominator.
    grid, dens_1 = FFTKDE(bw=bw_scaled).fit(coords).evaluate(grid_points)
    ax_x = np.unique(grid[:, 0])
    ax_y = np.unique(grid[:, 1])
    gx, gy = ax_x.size, ax_y.size
    d1 = dens_1.reshape(gx, gy)
    threshold = np.percentile(d1[d1 > 0], min_density_pct)
    background = d1 < threshold

    n = matrix.shape[0]
    out = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        values = matrix[:, j]
        if np.ptp(values) == 0:
            # Constant column: the identity for a row-stochastic smoother, and the only finite
            # input for which KDEpy's weight normalisation divides by zero. See the Notes above.
            # `== 0` rather than `not > 0`: the latter also captures NaN, which is not constant.
            out[:, j] = values
            continue
        shift = float(values.min())
        v_pos = values - shift  # KDEpy weights must be non-negative
        _, dens_v = FFTKDE(bw=bw_scaled).fit(coords, weights=v_pos).evaluate(grid_points)
        dv = dens_v.reshape(gx, gy)
        with np.errstate(invalid="ignore", divide="ignore"):
            field = (dv * v_pos.sum()) / (d1 * n)
        field = field + shift  # undo the shift (the estimator is affine)
        field[background] = np.nan
        filled = np.where(np.isfinite(field), field, np.nanmedian(field))
        interp = RegularGridInterpolator(
            (ax_x, ax_y), filled, bounds_error=False, fill_value=np.nan
        )
        out[:, j] = interp(coords)
    return out, bw


def smooth_field_kde(coords, values, **kwargs):
    """1-D convenience wrapper around :func:`smooth_matrix_kde`."""
    out, bw = smooth_matrix_kde(coords, values, **kwargs)
    return out[:, 0], bw
