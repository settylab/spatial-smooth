"""Every backend must draw a tissue section the same way up.

Imaging platforms store cell centroids as *image* coordinates: origin top-left, y increasing
downward. ``squidpy.pl.spatial_scatter`` honours that; ``scanpy.pl.embedding`` does not, and
drew ``obsm["spatial"]`` as a Cartesian embedding with y increasing upward -- mirroring the
tissue vertically. The mirrored plot is entirely plausible to look at, which is exactly what
makes it dangerous.

These tests do not check "it rendered". They plant an asymmetric feature at a known place and
assert it lands in the same *screen* quadrant under every backend.
"""
from __future__ import annotations

import numpy as np
import pytest

import spatial_smooth as ss

pytest.importorskip("scanpy")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The blob sits at small y. In image convention (y down) that is the TOP of the picture.
BLOB_XY = (50.0, 10.0)


def _as_axes(result):
    """Flatten whatever a backend returned into a list of Axes.

    Deliberately defined here rather than imported from the package, so this module collects
    and runs against *any* version of `spatial_smooth` -- including one that predates the fix.
    """
    if result is None:
        return []
    if hasattr(result, "flatten"):
        return list(result.flatten())
    if isinstance(result, (list, tuple)):
        out = []
        for item in result:
            out.extend(_as_axes(item))
        return out
    return [result] if hasattr(result, "yaxis") else []


@pytest.fixture
def tissue():
    """Cells over a square, with a signature blob at low y (top of the image)."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 600
    xy = rng.random((n, 2)) * 100.0
    field = np.exp(-((xy - np.array(BLOB_XY)) ** 2).sum(1) / (2 * 9.0 ** 2))
    X = np.column_stack([field + rng.normal(scale=0.05, size=n)] * 2).astype(np.float32)
    adata = ad.AnnData(X=np.clip(X, 0, None), var=pd.DataFrame(index=["g1", "g2"]))
    adata.obsm["spatial"] = xy
    adata.obsm["X_umap"] = rng.random((n, 2))
    ss.smooth(adata, ["g1", "g2"], "sig")
    return adata


def _blob_is_at_top(ax) -> bool:
    """Does the planted blob render in the upper half of the drawn canvas?

    Answered in *display* space, so it is independent of how the backend chose to orient the
    data axes -- which is the only way to catch a flip.
    """
    x, y = BLOB_XY
    (_, blob_py) = ax.transData.transform((x, y))
    (_, mid_py) = ax.transData.transform((x, 50.0))  # centre of the tissue
    return blob_py > mid_py  # larger display y == higher on screen


@pytest.mark.parametrize("backend", ["scanpy", "squidpy"])
def test_spatial_basis_uses_the_image_convention(tissue, backend):
    if backend == "squidpy":
        pytest.importorskip("squidpy")
    result = ss.pl.signature(tissue, "sig", raw=False, backend=backend, show=False)
    axes = _as_axes(result)
    assert axes, f"{backend} returned no axes to inspect"
    for ax in axes:
        assert ax.yaxis_inverted(), f"{backend}: spatial y-axis must increase downward"
        assert _blob_is_at_top(ax), f"{backend}: the blob rendered in the wrong half"
    plt.close("all")


def test_backends_agree_on_orientation(tissue):
    """The two backends must not disagree -- the bug was that they did."""
    pytest.importorskip("squidpy")
    inverted = {}
    for backend in ("scanpy", "squidpy"):
        result = ss.pl.signature(tissue, "sig", raw=False, backend=backend, show=False)
        axes = _as_axes(result)
        inverted[backend] = [ax.yaxis_inverted() for ax in axes]
        plt.close("all")
    assert set(inverted["scanpy"]) == set(inverted["squidpy"]) == {True}


def test_spatial_plot_has_equal_aspect(tissue):
    """Anisotropic scaling of physical coordinates distorts anatomy."""
    result = ss.pl.signature(tissue, "sig", raw=False, backend="scanpy", show=False)
    for ax in _as_axes(result):
        assert ax.get_aspect() == 1.0
    plt.close("all")


def test_non_spatial_basis_keeps_cartesian_orientation(tissue):
    """A UMAP is not an image: leave it alone."""
    result = ss.pl.signature(
        tissue, "sig", raw=False, backend="scanpy", basis="X_umap", show=False
    )
    for ax in _as_axes(result):
        assert not ax.yaxis_inverted()
    plt.close("all")


def test_show_true_still_renders_and_returns_none(tissue):
    """The invert-then-show path must not break the default `show=True` contract."""
    out = ss.pl.signature(tissue, "sig", raw=False, backend="scanpy")
    assert out is None
    plt.close("all")


def test_a_spatial_result_refuses_to_be_drawn_on_a_umap(tissue):
    """R7: dropping the coordinates must not silently move the field onto another embedding."""
    del tissue.obsm["spatial"]
    with pytest.raises(KeyError, match="no longer present"):
        ss.pl.signature(tissue, "sig", backend="scanpy", show=False)


def test_missing_raw_panel_warns_in_both_entry_points(tissue):
    """R7: `signature` dropped the panel silently while `compare` raised. They must agree."""
    del tissue.obs["sig_raw"]
    with pytest.warns(UserWarning, match="only the smoothed panel"):
        ss.pl.signature(tissue, "sig", raw=True, backend="scanpy", show=False)
    plt.close("all")
    with pytest.warns(UserWarning, match="only the smoothed panel"):
        ss.pl.compare(tissue, ["sig"], raw=True, backend="scanpy", show=False)
    plt.close("all")
