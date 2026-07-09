"""Shared fixtures: a small synthetic spatial dataset with a planted signature."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import anndata as ad
import numpy as np
import pandas as pd
import pytest

N_SIDE = 24
N_GENES = 8
SIGNATURE = ["Sig1", "Sig2", "Sig3"]


def _make_adata(seed: int = 0, scale: float = 1.0):
    """Jittered grid of cells; three signature genes light up one corner blob."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(N_SIDE), np.arange(N_SIDE))
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    coords += rng.normal(scale=0.12, size=coords.shape)
    n = coords.shape[0]

    # A blob comfortably wider than the default bandwidth (~6 grid spacings), as a real
    # signature domain is: otherwise the default smoother is expected to blur it away.
    centre = np.array([N_SIDE * 0.3, N_SIDE * 0.7])
    field = np.exp(-((coords - centre) ** 2).sum(1) / (2 * (N_SIDE * 0.25) ** 2))

    var_names = SIGNATURE + [f"Bg{i}" for i in range(N_GENES - len(SIGNATURE))]
    X = rng.normal(loc=0.4, scale=0.8, size=(n, N_GENES))
    for j in range(len(SIGNATURE)):
        X[:, j] += 3.0 * field + rng.normal(scale=1.5, size=n)  # strong signal, strong noise
    X = np.clip(X, 0, None).astype(np.float32)

    obs = pd.DataFrame(
        {
            "region": pd.Categorical(np.where(coords[:, 0] < N_SIDE / 2, "left", "right")),
            "condition": pd.Categorical(rng.choice(["a", "b"], size=n)),
            "truth": field,  # the noiseless signature field the smoothers should recover
        },
        index=[f"cell{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.obsm["spatial"] = coords * scale
    # A stand-in cell-state embedding, so GP steps do not drag in palantir.
    adata.obsm["DM_EigenVectors"] = np.column_stack(
        [field, X[:, :4] @ rng.normal(size=(4, 3))]
    ).astype(np.float64)
    adata.layers["logcounts"] = X.copy()
    return adata


@pytest.fixture
def adata():
    return _make_adata()


@pytest.fixture
def adata_scaled():
    """The same object with coordinates blown up 1000x -- for scale-invariance checks."""
    return _make_adata(scale=1000.0)


@pytest.fixture
def signature():
    return list(SIGNATURE)


def has_kompot() -> bool:
    try:
        import kompot  # noqa: F401
    except ImportError:
        return False
    return hasattr(kompot, "smooth_expression")


needs_kompot = pytest.mark.skipif(
    not has_kompot(), reason="kompot (with smooth_expression) not installed"
)
