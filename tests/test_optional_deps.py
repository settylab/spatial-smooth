"""A missing optional backend must say which package, what for, and how to install it."""
from __future__ import annotations

import sys

import pytest

import spatial_smooth as ss
from spatial_smooth._deps import have, require


def _block(monkeypatch, *modules):
    for module in modules:
        monkeypatch.setitem(sys.modules, module, None)


def test_require_message_names_the_pip_line(monkeypatch):
    _block(monkeypatch, "KDEpy")
    with pytest.raises(ImportError) as excinfo:
        require("KDEpy")
    message = str(excinfo.value)
    assert "spatial_smooth needs `KDEpy`" in message
    assert 'pip install "KDEpy>=1.1"' in message
    assert "FFT-KDE" in message  # says what it is for


def test_kde_step_without_kdepy(adata, signature, monkeypatch):
    _block(monkeypatch, "KDEpy")
    with pytest.raises(ImportError, match=r'pip install "KDEpy>=1\.1"'):
        ss.smooth(adata, signature, "sig", steps="spatial-kde")


def test_gp_step_without_kompot(adata, signature, monkeypatch):
    _block(monkeypatch, "kompot")
    with pytest.raises(ImportError, match=r'pip install "kompot>=0\.7\.0"'):
        ss.smooth(adata, signature, "sig", steps="dm", auto_embed=False)


def test_diffusion_map_without_palantir(adata, monkeypatch):
    _block(monkeypatch, "palantir")
    del adata.obsm["DM_EigenVectors"]
    with pytest.raises(ImportError, match=r'pip install "palantir>=1\.4"'):
        ss.compute_diffusion_map(adata)


def test_auto_embed_reports_the_missing_embedding_backend(adata, signature, monkeypatch):
    _block(monkeypatch, "palantir")
    del adata.obsm["DM_EigenVectors"]
    with pytest.raises(ImportError, match="palantir"):
        ss.smooth(adata, signature, "sig", steps="dm", auto_embed=True)


def test_plotting_without_any_backend(adata, signature, monkeypatch):
    ss.smooth(adata, signature, "sig")
    _block(monkeypatch, "scanpy", "squidpy")
    with pytest.raises(ImportError, match="needs a plotting backend"):
        ss.pl.signature(adata, "sig")


def test_gp_step_rejects_a_kompot_without_smooth_expression(adata, signature, monkeypatch):
    """An old kompot imports fine but lacks the entry point; say so, do not AttributeError."""
    import types

    stub = types.ModuleType("kompot")
    stub.__version__ = "0.6.0"
    monkeypatch.setitem(sys.modules, "kompot", stub)
    with pytest.raises(ImportError, match=r"older releases do not provide"):
        ss.smooth(adata, signature, "sig", steps="dm", auto_embed=False)


def test_check_dependencies_reports_status(capsys):
    status = ss.check_dependencies()
    captured = capsys.readouterr().out
    assert "dependency check" in captured
    assert status["numpy"] is not None
    assert set(status) >= {"numpy", "scipy", "anndata", "kompot", "scanpy", "KDEpy"}


def test_have_never_raises(monkeypatch):
    _block(monkeypatch, "squidpy")
    assert have("squidpy") is False
    assert have("numpy") is True
