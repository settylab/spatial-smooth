"""Plotting is a wrapper, not a reimplementation: kwargs must reach the backend verbatim."""
from __future__ import annotations

import pytest

import spatial_smooth as ss
from spatial_smooth import plot as pl


class _Recorder:
    """Stands in for a scanpy/squidpy plotting function and records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, adata, **kwargs):
        self.calls.append(kwargs)
        return "plotted"

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def smoothed(adata, signature):
    ss.smooth(adata, signature, "sig", steps="spatial")
    return adata


def _patch_scanpy(monkeypatch, recorder, attr="embedding"):
    scanpy = pytest.importorskip("scanpy")
    monkeypatch.setattr(scanpy.pl, attr, recorder)
    return scanpy


def test_kwargs_are_forwarded_verbatim(smoothed, monkeypatch):
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder)

    out = pl.signature(
        smoothed, "sig", backend="scanpy", show=False, size=12, title=["raw", "smooth"],
        frameon=False, ncols=2,
    )
    assert out == "plotted"
    call = recorder.last
    assert call["size"] == 12
    assert call["title"] == ["raw", "smooth"]
    assert call["frameon"] is False
    assert call["ncols"] == 2
    assert call["show"] is False


def test_color_is_set_from_provenance(smoothed, monkeypatch):
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder)

    pl.signature(smoothed, "sig", backend="scanpy", show=False)
    assert recorder.last["color"] == ["sig_raw", "sig"]

    pl.signature(smoothed, "sig", raw=False, backend="scanpy", show=False)
    assert recorder.last["color"] == ["sig"]


def test_defaults_are_injected_but_the_caller_wins(smoothed, monkeypatch):
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder)

    pl.signature(smoothed, "sig", backend="scanpy", show=False)
    assert recorder.last["cmap"] == "viridis"
    assert recorder.last["vmin"] == "p1"
    assert recorder.last["vmax"] == "p99"

    pl.signature(smoothed, "sig", backend="scanpy", show=False, cmap="magma", vmin=0.0)
    assert recorder.last["cmap"] == "magma"
    assert recorder.last["vmin"] == 0.0
    assert recorder.last["vmax"] == "p99"  # untouched default survives


def test_basis_defaults_to_the_last_spatial_step(smoothed, monkeypatch):
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder)

    pl.signature(smoothed, "sig", backend="scanpy", show=False)
    assert recorder.last["basis"] == "spatial"

    pl.signature(smoothed, "sig", backend="scanpy", show=False, basis="DM_EigenVectors")
    assert recorder.last["basis"] == "DM_EigenVectors"


def test_scanpy_spatial_backend_targets_pl_spatial(smoothed, monkeypatch):
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder, attr="spatial")
    pl.signature(smoothed, "sig", backend="scanpy-spatial", show=False, spot_size=30)
    assert recorder.last["spot_size"] == 30
    assert "basis" not in recorder.last  # sc.pl.spatial has no `basis`


def test_squidpy_backend_targets_spatial_scatter(smoothed, monkeypatch):
    squidpy = pytest.importorskip("squidpy")
    recorder = _Recorder()
    monkeypatch.setattr(squidpy.pl, "spatial_scatter", recorder)
    pl.signature(smoothed, "sig", backend="squidpy", size=4)
    call = recorder.last
    assert call["color"] == ["sig_raw", "sig"]
    assert call["shape"] is None
    assert call["spatial_key"] == "spatial"
    assert call["size"] == 4
    assert "vmin" not in call  # squidpy forwards unknown keys to matplotlib; do not smuggle them


def test_compare_stacks_results(smoothed, signature, monkeypatch):
    ss.smooth(smoothed, signature, "other", steps=[ss.KnnGaussian(k=8)])
    recorder = _Recorder()
    _patch_scanpy(monkeypatch, recorder)

    pl.compare(smoothed, ["sig", "other"], raw=True, backend="scanpy", show=False)
    assert recorder.last["color"] == ["sig_raw", "sig", "other"]


def test_unknown_backend_rejected(smoothed):
    with pytest.raises(ValueError, match="unknown backend"):
        pl.signature(smoothed, "sig", backend="ggplot")


def test_available_backends_is_honest():
    backends = pl.available_backends()
    assert set(backends) <= set(pl.BACKENDS)


def test_caller_supplied_color_raises_instead_of_being_ignored(smoothed):
    """Silently dropping `color` is the wrong failure mode under 'full control of the plot'."""
    with pytest.raises(TypeError, match="cannot be overridden"):
        pl.signature(smoothed, "sig", backend="scanpy", show=False, color=["something_else"])
