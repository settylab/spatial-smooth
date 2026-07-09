"""Plotting -- a thin, transparent wrapper over scanpy and squidpy.

This module **never computes anything**. It reads the keys :func:`spatial_smooth.core.smooth`
wrote into the ``AnnData`` (``obs[name]``, ``obs[f"{name}_raw"]``,
``uns["spatial_smooth"][name]``), works out sensible defaults from the recorded provenance, and
hands everything to an existing plotting function. Write a smoothed object to ``.h5ad``, reload
it anywhere, and these calls render it -- no ``kompot``, ``KDEpy`` or ``palantir`` needed, and no
smoothing repeated.

Backends and where your ``**kwargs`` go
---------------------------------------
========================  ==================================  ======================================
``backend``               underlying call                     when to use it
========================  ==================================  ======================================
``"squidpy"``             ``squidpy.pl.spatial_scatter``      tissue coordinates, optional image
``"scanpy"``              ``scanpy.pl.embedding``             any ``obsm`` basis, imageless
``"scanpy-spatial"``      ``scanpy.pl.spatial``               Visium-style ``uns["spatial"]`` image
``"auto"`` (default)      squidpy if installed, else scanpy   --
========================  ==================================  ======================================

Every ``**kwargs`` is forwarded **verbatim** to that function. ``color`` is set by this module
(to the raw and smoothed obs columns); passing it raises ``TypeError`` rather than being silently
ignored. Everything else -- ``cmap``, ``size``, ``figsize``, ``vmin``/``vmax``, ``title``,
``save``, ``ax`` -- is the backend's own parameter, documented in the backend's own docstring.
Defaults this module supplies (a perceptually uniform colour map, percentile colour limits where
the backend supports them, a grey ``na_color``) are applied only when you have not passed that
key yourself.

Two conventions this module *does* normalise, because leaving them to the backend produced plots
that differed for the same data:

**Orientation.** Imaging platforms store cell centroids as image coordinates -- origin top-left,
y increasing *downward*. ``squidpy.pl.spatial_scatter`` honours that; ``scanpy.pl.embedding``
treats ``obsm["spatial"]`` as an abstract Cartesian embedding and draws y upward, mirroring the
tissue vertically. Whenever the basis being drawn is ``"spatial"``, this module inverts the
y-axis and sets equal aspect, so every backend renders the section as the microscope saw it.

**``size`` is backend-native and is deliberately not translated.** In ``scanpy`` it is the marker
area in points squared (scanpy's own default, ``~120000 / n_obs``, is usually right); in
``squidpy`` it is a *scale factor* on the inferred spot size (default ``1.0``). A ``size=6`` that
looks correct in one is nearly invisible in the other. Omit it and take the backend's default
unless you have a reason not to.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ._deps import have, require
from .core import provenance
from .steps import SPATIAL_KEY

__all__ = ["signature", "compare", "available_backends", "BACKENDS"]

#: ``backend -> (module, attribute)`` of the function each backend delegates to.
BACKENDS = {
    "squidpy": ("squidpy", "pl.spatial_scatter"),
    "scanpy": ("scanpy", "pl.embedding"),
    "scanpy-spatial": ("scanpy", "pl.spatial"),
}

# Defaults injected per backend, only for keys the caller did not supply. Kept to parameters
# each backend actually declares -- squidpy's `spatial_scatter` forwards unknown keys to
# matplotlib's scatter, where a stray `vmin` would be silently mis-applied.
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "squidpy": {"cmap": "viridis", "na_color": "#d9d9d9"},
    "scanpy": {"cmap": "viridis", "vmin": "p1", "vmax": "p99", "na_color": "#d9d9d9"},
    "scanpy-spatial": {"cmap": "viridis", "vmin": "p1", "vmax": "p99", "na_color": "#d9d9d9"},
}


def available_backends() -> List[str]:
    """Backends whose plotting library is importable right now."""
    out = []
    if have("squidpy"):
        out.append("squidpy")
    if have("scanpy"):
        out += ["scanpy", "scanpy-spatial"]
    return out


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)} or 'auto'")
        return backend
    if have("squidpy"):
        return "squidpy"
    if have("scanpy"):
        return "scanpy"
    raise ImportError(
        "spatial_smooth needs a plotting backend: install `scanpy` "
        '(pip install "scanpy>=1.9") or `squidpy` (pip install "squidpy>=1.4").'
    )


def _default_basis(adata, records: Sequence[Dict[str, Any]]) -> str:
    """Plot over the last step's basis when that is a 2-D physical basis, else fall back."""
    for record in reversed(records):
        for step in reversed(record.get("steps", [])):
            basis = step.get("basis")
            if basis == SPATIAL_KEY and basis in adata.obsm:
                return basis
    for candidate in (SPATIAL_KEY, "X_umap", "umap", "X_pca"):
        if candidate in adata.obsm:
            return candidate
    raise KeyError(
        f"cannot pick a plotting basis: adata.obsm has {sorted(adata.obsm)}. "
        "Pass basis=... explicitly."
    )


def _colors(adata, record: Dict[str, Any], raw: bool) -> List[str]:
    smoothed, unsmoothed = record["obs_key"], record["obs_key_raw"]
    if smoothed not in adata.obs:
        raise KeyError(
            f"adata.uns['spatial_smooth'][{record['name']!r}] exists but adata.obs[{smoothed!r}] "
            "does not -- the object was modified after smoothing. Re-run spatial_smooth.smooth()."
        )
    if raw and unsmoothed in adata.obs:
        return [unsmoothed, smoothed]
    return [smoothed]


def _as_axes(result) -> List[Any]:
    """Normalise the several shapes scanpy/squidpy return into a flat list of Axes."""
    if result is None:
        return []
    if hasattr(result, "flatten"):  # numpy array of Axes
        return list(result.flatten())
    if isinstance(result, (list, tuple)):
        out: List[Any] = []
        for item in result:
            out.extend(_as_axes(item))
        return out
    return [result] if hasattr(result, "yaxis") else []


def _apply_image_convention(axes) -> None:
    """Draw a spatial basis the way an imaging assay stores it: y increasing downward.

    Cell centroids from imaging platforms are *image* coordinates -- the origin is the top-left
    of the slide and y grows downward. ``squidpy.pl.spatial_scatter`` honours that and inverts
    the y-axis; ``scanpy.pl.embedding`` treats any ``obsm`` basis as an abstract Cartesian
    embedding and leaves y increasing upward. Rendering one section through both backends
    therefore produced vertically mirrored tissue -- a plot that looks entirely plausible and is
    upside down.

    We normalise onto squidpy's convention: the tissue as the microscope saw it. Equal aspect
    goes with it, because anisotropic scaling of physical coordinates distorts anatomy.
    """
    for ax in axes:
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_aspect("equal")


def _dispatch(adata, backend: str, color: List[str], basis: str, kwargs: Dict[str, Any]):
    if "color" in kwargs:
        raise TypeError(
            "`color` is set from the stored result and cannot be overridden; it names the raw "
            f"and smoothed obs columns ({color}). To colour cells by something else, call the "
            "backend function directly (see spatial_smooth.plot.BACKENDS)."
        )
    merged = dict(_DEFAULTS[backend])
    merged.update(kwargs)
    merged["color"] = color

    if backend == "squidpy":
        squidpy = require("squidpy")
        merged.setdefault("shape", None)  # point cloud, not a Visium hex grid
        merged.setdefault("spatial_key", basis)
        # squidpy has no `show`; it draws unconditionally and returns axes on `return_ax`.
        # Accept `show` anyway so the backends present one interface, and translate it --
        # forwarding it would land in `matplotlib.scatter` as an unknown keyword.
        show = merged.pop("show", None)
        if show is False:
            merged.setdefault("return_ax", True)
        return squidpy.pl.spatial_scatter(adata, **merged)

    scanpy = require("scanpy")
    on_tissue = backend == "scanpy-spatial" or basis == SPATIAL_KEY

    if not on_tissue:
        merged.setdefault("basis", basis)
        return scanpy.pl.embedding(adata, **merged)

    # A spatial basis through a scanpy backend: draw it, then impose the image convention
    # before anything is shown. `sc.pl.spatial` already inverts (it overlays a tissue image),
    # so this is a no-op there beyond enforcing equal aspect.
    plt = require("matplotlib").pyplot
    show = merged.pop("show", None)
    if backend == "scanpy-spatial":
        result = scanpy.pl.spatial(adata, show=False, **merged)
    else:
        merged.setdefault("basis", basis)
        result = scanpy.pl.embedding(adata, show=False, **merged)

    _apply_image_convention(_as_axes(result))
    if show is False:
        return result
    plt.show()
    return None


def signature(
    adata,
    name: str = "signature",
    *,
    raw: bool = True,
    backend: str = "auto",
    basis: Optional[str] = None,
    **kwargs,
):
    """Plot a stored smoothing result -- raw next to smoothed, by default.

    Reads only what :func:`spatial_smooth.smooth` stored. Nothing is recomputed, so this works on
    a reloaded ``.h5ad`` in an environment without the smoothing backends installed.

    Parameters
    ----------
    adata
        Object carrying a result named ``name``.
    name
        The result to plot.
    raw
        Show the unsmoothed score alongside the smoothed one (two panels).
    backend
        ``"auto"``, ``"squidpy"``, ``"scanpy"``, or ``"scanpy-spatial"``. See the module
        docstring for what each delegates to.
    basis
        ``adata.obsm`` key to lay the cells out on. Defaults to the last smoothing step's basis
        when that is ``"spatial"``, else the first of ``spatial``/``X_umap``/``X_pca`` present.
    **kwargs
        Forwarded verbatim to the backend's plotting function.

    Returns
    -------
    Whatever the backend returns (usually ``None`` when it shows, or axes when ``show=False``).

    Raises
    ------
    KeyError
        If no result named ``name`` is stored (the message says how to create it).
    """
    record = provenance(adata, name)
    color = _colors(adata, record, raw)
    resolved = _resolve_backend(backend)
    basis = basis or _default_basis(adata, [record])
    return _dispatch(adata, resolved, color, basis, kwargs)


def compare(
    adata,
    names: Sequence[str],
    *,
    raw: bool = False,
    backend: str = "auto",
    basis: Optional[str] = None,
    **kwargs,
):
    """Plot several stored results side by side -- e.g. one panel per smoothing pipeline.

    ``compare(adata, ["sig_spatial", "sig_dm", "sig_both"])`` puts the three composition modes on
    a common canvas. With ``raw=True`` the first panel is the (shared) unsmoothed score.
    """
    if isinstance(names, str):
        names = [names]
    records = [provenance(adata, n) for n in names]
    color: List[str] = []
    if raw and records:
        color.append(records[0]["obs_key_raw"])
    for record in records:
        color.extend(c for c in _colors(adata, record, raw=False) if c not in color)
    resolved = _resolve_backend(backend)
    basis = basis or _default_basis(adata, records)
    return _dispatch(adata, resolved, color, basis, kwargs)
