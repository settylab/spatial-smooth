"""Low-level smoothing kernels.

Each function here maps ``(coords, values) -> smoothed values`` and knows nothing about
:class:`~anndata.AnnData`. Two of them are *linear and row-stochastic* -- they replace each
point's value by a weighted average of its neighbours' values with weights summing to one:

* :func:`knn_gaussian_operator` -- a sparse Gaussian kernel over the ``k`` nearest neighbours.
* :func:`kde_operator_apply` -- a fine-grid FFT Nadaraya-Watson estimator.

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

from typing import Optional, Tuple

from ._deps import require

__all__ = [
    "median_nn_distance",
    "knn_gaussian_operator",
    "smooth_matrix_knn_gaussian",
    "smooth_field_knn_gaussian",
    "smooth_matrix_kde",
    "smooth_field_kde",
]


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
    k: int = 100,
    sigma: Optional[float] = None,
    sigma_factor: float = 6.0,
    workers: int = -1,
):
    """Build the sparse row-stochastic Gaussian kNN smoothing operator ``W``.

    ``W[i, j] = exp(-d(i, j)**2 / (2 * sigma**2))`` for the ``k`` nearest neighbours ``j`` of
    ``i`` (self included), each row normalised to sum to one. Applying ``W`` to any field
    replaces every value with the Gaussian-weighted mean of its ``k`` nearest neighbours.

    This is the classic fixed-bandwidth spatial smoother. It costs one ``cKDTree`` query
    (``O(n log n)``) plus a sparse mat-vec (``O(n * k)``) per field -- orders of magnitude
    cheaper than a Gaussian-process fit, and the recommended default for smoothing over
    physical tissue coordinates.

    Parameters
    ----------
    coords
        ``(n, d)`` array of coordinates (physical positions, or any embedding).
    k
        Neighbours per point, including the point itself. Capped at ``n``.
    sigma
        Gaussian bandwidth in coordinate units. ``None`` (default) sets it scale-invariantly
        to ``sigma_factor * median_nn_distance(coords)``.
    sigma_factor
        Multiplier on the median nearest-neighbour distance when ``sigma`` is inferred. The
        default ``6.0`` (~6 cell spacings) reproduces the ~50 um bandwidth conventionally used
        on imaging-based spatial assays with ~8 um cell spacing.
    workers
        Threads for the kd-tree query (``-1`` uses all cores; cap via the environment on a
        shared machine).

    Returns
    -------
    W : scipy.sparse.csr_matrix
        ``(n, n)`` row-stochastic operator.
    sigma : float
        The bandwidth actually used.

    Notes
    -----
    ``k`` truncates the kernel: neighbours beyond the ``k``-th are given zero weight even if
    ``sigma`` would assign them a non-negligible one. Keep ``k`` comfortably larger than the
    number of points within ``~2 * sigma`` (the default ``k=100`` / ``sigma_factor=6`` pairing
    is calibrated for 2-D tissue).
    """
    np = require("numpy")
    scipy = require("scipy")
    cKDTree = scipy.spatial.cKDTree
    sparse = scipy.sparse

    coords = np.asarray(coords, dtype=np.float64)
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
    return W, sigma


def smooth_matrix_knn_gaussian(
    coords,
    matrix,
    *,
    k: int = 100,
    sigma: Optional[float] = None,
    sigma_factor: float = 6.0,
    workers: int = -1,
):
    """Apply :func:`knn_gaussian_operator` to every column of ``matrix``.

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
    W, sigma_used = knn_gaussian_operator(
        coords, k=k, sigma=sigma, sigma_factor=sigma_factor, workers=workers
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
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
        densities) are treated as empty background and left out of the interpolation, so blank
        tissue does not invent signal.

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
