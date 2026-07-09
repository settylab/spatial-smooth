#!/usr/bin/env python
"""Build ``tutorial.ipynb`` from source, so the notebook is reviewable as plain text.

Run ``python build_tutorial.py`` to regenerate the (unexecuted) notebook, then execute it with
``jupyter nbconvert --to notebook --execute --inplace tutorial.ipynb``.
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.rstrip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.rstrip()))


# --------------------------------------------------------------------------------------- #
md(
    """\
# spatial-smooth: a tutorial

> ## ⚠️ This package is for looking, not for measuring
>
> `spatial-smooth` makes spatial regions **easier to see**. What it produces is a picture.
>
> Smoothing works by making each cell look more like its neighbours. That is exactly what you
> want when you are trying to spot where a gene programme is switched on — and exactly what you
> must not feed into a statistical test. Once cells have been made to resemble their neighbours
> they are no longer independent measurements, so **differential expression, cluster
> comparisons, correlations and p-values computed on smoothed values are badly over-confident**.
> They will report strong, convincing structure in data that contains none.
>
> Every call writes the unsmoothed score next to the smoothed one, as
> `adata.obs["<name>_raw"]`. **Look at the smoothed one. Do your statistics on the raw one.**

A per-cell signature score is noisy — each cell is measured independently, so dropout and
sampling variance dominate, and a real anatomical region can be genuinely hard to pick out of
the speckle. **Smoothing** lets neighbouring cells borrow statistical strength.
The scientific choice is *which* neighbours count:

| smoothing | neighbours are… | recovers |
|---|---|---|
| **spatial** | physically adjacent cells | tissue architecture: niches, layers, gradients |
| **cell state** | transcriptionally similar cells | biology, independent of position |
| **both, composed** | first the manifold, then the tissue | denoised expression laid out in space |

This notebook walks three levels of control:

1. **One line.** Defaults do everything.
2. **Parameterized.** Choose the pipeline; forward plotting kwargs to scanpy/squidpy.
3. **Fully modular.** Compute → store → write to disk → reload → **plot without recomputing.**

The data is a public 10x Genomics Xenium mouse-brain section (CC-BY), downloaded on first run
(~4.5 MB). Nothing here needs a cluster."""
)

md(
    """\
## Setup

```bash
pip install "spatial-smooth[all]"
```"""
)

code(
    '''\
%matplotlib inline
import numpy as np
import pandas as pd
import scanpy as sc

import spatial_smooth as ss'''
)

# --------------------------------------------------------------------------------------- #
md(
    """\
## 1. The data

### Using your own data instead

**`spatial-smooth` needs exactly two things** from an `AnnData`, and nothing else:

1. `adata.X` (or a layer you name via `layer=`) holding **log-normalised** expression, and
2. `adata.obsm["spatial"]` holding the cells' physical coordinates, as an `(n_obs, 2)` array.

So if you already have a prepared object, skip the download entirely — this is the whole of
section 1 for you:

```python
import anndata as ad
adata = ad.read_h5ad("my_section.h5ad")
assert "spatial" in adata.obsm          # (n_obs, 2) coordinates
# adata.X must be log-normalised; if it holds raw counts:
#   adata.layers["counts"] = adata.X.copy()
#   sc.pp.normalize_total(adata); sc.pp.log1p(adata)
```

Then jump to section 2. Cell-state smoothing (`steps="dm"`) additionally wants
`obsm["DM_EigenVectors"]`, which `ss.smooth(..., auto_embed=True)` computes for you if absent.

### The example dataset

The rest of this notebook uses a public 10x Xenium mouse-brain coronal subset: ~36,000 cells, a
248-gene panel, one cell per row with physical centroids. We fetch the two small loose outputs
(cached, so re-running is free), assemble an `AnnData`, and put the centroids in
`obsm["spatial"]`."""
)

code(
    '''\
import pathlib, urllib.request

BASE = ("https://cf.10xgenomics.com/samples/xenium/1.0.2/"
        "Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP")
NAME = "Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP"
DATA = pathlib.Path("data/xenium_mousebrain")
DATA.mkdir(parents=True, exist_ok=True)

# Point this at your own .h5ad to run the whole notebook on your data instead.
PREPARED = pathlib.Path("data/prepared.h5ad")

for fname in (f"{NAME}_cell_feature_matrix.h5", f"{NAME}_cells.csv.gz"):
    dest = DATA / fname
    if PREPARED.exists():
        break
    if not dest.exists():
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(f"{BASE}/{fname}", dest)
    print(f"  {fname}  ({dest.stat().st_size / 1e6:.1f} MB)")'''
)

code(
    '''\
import anndata as ad

if PREPARED.exists():
    # --- alternative path: load an object you prepared earlier -------------------
    adata = ad.read_h5ad(PREPARED)
    print(f"loaded {PREPARED}")
else:
    # --- example path: assemble the public Xenium section ------------------------
    adata = sc.read_10x_h5(DATA / f"{NAME}_cell_feature_matrix.h5")
    adata.var_names_make_unique()

    cells = pd.read_csv(DATA / f"{NAME}_cells.csv.gz").set_index("cell_id")
    cells.index = cells.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs = adata.obs.join(cells, how="left")
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    sc.pp.filter_cells(adata, min_counts=10)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

# The only two preconditions, checked explicitly.
assert "spatial" in adata.obsm, "spatial-smooth needs obsm['spatial']"
assert adata.X.max() < 100, "adata.X should be log-normalised, not raw counts"

print(f"{adata.n_obs:,} cells x {adata.n_vars} genes")'''
)

md(
    """\
A small **hippocampal** program from the panel — dentate-gyrus and CA markers. Four genes, each
sparse and noisy on its own."""
)

code(
    '''\
HIPPOCAMPUS = ["Prox1", "Neurod6", "Wfs1", "Fibcd1"]
assert set(HIPPOCAMPUS) <= set(adata.var_names)
HIPPOCAMPUS'''
)

# --------------------------------------------------------------------------------------- #
md(
    """\
## 2. Level one — one line, defaults do everything

`ss.smooth` with no `steps` argument smooths over `obsm["spatial"]` with a Gaussian kernel across
each cell's 400 nearest spatial neighbours. The bandwidth is inferred from the data (six median
nearest-neighbour distances), so you do not pick a number in microns.

`ss.pl.signature` then plots the raw score next to the smoothed one."""
)

code(
    '''\
ss.smooth(adata, HIPPOCAMPUS, "hippocampus")

ss.pl.signature(adata, "hippocampus")'''
)

md(
    """\
The raw panel is a speckle of individual cells; the smoothed panel resolves the dentate-gyrus
C-shape, the CA fields, and the cortical layers. Two columns appeared in `obs`, and a record of
what was run in `uns`."""
)

code(
    '''\
print(adata.obs[["hippocampus_raw", "hippocampus"]].describe().T)
print()

prov = ss.provenance(adata, "hippocampus")
print("genes    :", prov["genes"])
print("score    :", prov["score"])
print("pipeline :", [s["kind"] for s in prov["steps"]])
res = prov["steps"][0]["resolved"]
print("bandwidth:", round(res["sigma_used"], 1), "um nominal;",
      round(res["sigma_effective"], 1), "um effective",
      f"({res['kernel_mass_retained']:.0%} of the kernel kept)")'''
)

# --------------------------------------------------------------------------------------- #
md(
    """\
## 3. Level two — choose the pipeline, control the plot

### 3a. Composition: spatial, cell state, or both

`steps` selects what you smooth over:

| `steps` | pipeline | meaning |
|---|---|---|
| `"spatial"` (default) | `[KnnGaussian()]` | spatial only |
| `"dm"` | `[KompotGP()]` | cell state only |
| `"dm+spatial"` | `[KompotGP(), KnnGaussian()]` | both, cell state first |

Doing *just one of the two* is the ordinary case, not a special one — a one-element pipeline.
Composing runs the steps left to right: the spatial step smooths the expression the
cell-state step already denoised.

The cell-state step is a Gaussian-process regression over a diffusion map of the expression
manifold (`kompot.smooth_expression`, built on `mellon`). It needs `obsm["DM_EigenVectors"]`;
with `auto_embed=True` (the default) `spatial-smooth` computes it with Palantir if absent.

Everything below runs on the **full section** — every cell, no subsampling. The Gaussian process
is the slow step (a few minutes); the two spatial smoothers take about a second each."""
)

code(
    '''\
ss.compute_diffusion_map(adata)         # Palantir -> obsm["DM_EigenVectors"]
adata.obsm["DM_EigenVectors"].shape'''
)

code(
    '''\
# cell state only: GP over the diffusion map
ss.smooth(adata, HIPPOCAMPUS, "dm_only", steps="dm")

# spatial only: Gaussian kNN over tissue coordinates
ss.smooth(adata, HIPPOCAMPUS, "spatial_only", steps="spatial")

# both, composed: manifold first, then tissue
ss.smooth(adata, HIPPOCAMPUS, "composed", steps="dm+spatial")

ss.list_results(adata)'''
)

code(
    '''\
ss.pl.compare(
    adata, ["spatial_only", "dm_only", "composed"], raw=True,
    backend="scanpy", ncols=4, frameon=False,
)'''
)

md(
    """\
Read the four panels left to right: the raw score, then each pipeline. Spatial smoothing produces
the cleanest tissue field. Cell-state smoothing denoises without using position at all. Composing
does both, and is the smoothest of the three."""
)

md(
    """\
### 3b. Plot control: kwargs go straight through

`ss.pl.signature` is a wrapper, not a reimplementation. Everything after `name` is forwarded
**verbatim** to the backend:

| `backend` | underlying call |
|---|---|
| `"squidpy"` | `squidpy.pl.spatial_scatter` |
| `"scanpy"` | `scanpy.pl.embedding` |
| `"scanpy-spatial"` | `scanpy.pl.spatial` |
| `"auto"` (default) | squidpy if installed, else scanpy |

`color` is set for you from the stored provenance. Defaults (`cmap`, percentile colour limits,
a grey `na_color`) are injected only for keys you did not pass."""
)

code(
    '''\
ss.pl.signature(
    adata, "hippocampus", raw=False,
    backend="scanpy",          # -> scanpy.pl.embedding
    cmap="magma", vmax="p99.5", frameon=False,
    title="hippocampal signature, smoothed",
)'''
)

code(
    '''\
# The same result through squidpy, which knows about tissue images and library ids.
ss.pl.signature(adata, "hippocampus", backend="squidpy", cmap="magma", figsize=(6, 6))'''
)

md(
    """\
### 3c. Bandwidth is scale-invariant

Every default bandwidth is a multiple of the median nearest-neighbour distance, so the same
factor smooths the same amount whether coordinates are microns or millimetres. Rescale the
coordinates a thousandfold and the field is unchanged."""
)

code(
    '''\
rescaled = adata.copy()
rescaled.obsm["spatial"] = rescaled.obsm["spatial"] * 1000.0
ss.smooth(rescaled, HIPPOCAMPUS, "hippocampus")

a = adata.obs["hippocampus"].to_numpy()
b = rescaled.obs["hippocampus"].to_numpy()
print("max |difference| :", np.abs(a - b).max())
print("sigma (um)  :", round(ss.provenance(adata, "hippocampus")["steps"][0]["resolved"]["sigma_used"], 2))
print("sigma (nm)  :", round(ss.provenance(rescaled, "hippocampus")["steps"][0]["resolved"]["sigma_used"], 2))'''
)

md(
    """\
> **One caveat worth internalising.** The Gaussian process infers its length scale the same way,
> via `ls_factor`. Over a diffusion map kompot's native `ls_factor=10` is right; over *physical*
> coordinates it is ~200x the cell spacing and washes the field into a single global gradient.
> Use `ls_factor≈0.3` there — which is exactly what the `"spatial-gp"` shorthand does."""
)

# --------------------------------------------------------------------------------------- #
md(
    """\
## 4. Level three — fully modular: compute, store, plot later

Pass `Step` objects instead of a shorthand for complete control. Each step is a frozen dataclass:
a *specification*, not a fitted object, so it can be reused and is recorded verbatim."""
)

code(
    '''\
pipeline = [
    ss.KompotGP(basis="DM_EigenVectors", ls_factor=10.0, n_landmarks=5000),
    ss.KnnGaussian(basis="spatial", k=64, sigma_factor=4.0),
]
pipeline'''
)

code(
    '''\
ss.smooth(adata, HIPPOCAMPUS, "custom", steps=pipeline, store_genes=True)

print("smoothed score      :", adata.obs["custom"].shape)
print("smoothed expression :", adata.obsm["custom_smoothed"].shape)   # store_genes=True'''
)

md(
    """\
### The persistence contract

Everything is in the `AnnData`:

| key | contents |
|---|---|
| `adata.obs[name]` | smoothed score |
| `adata.obs[f"{name}_raw"]` | unsmoothed score, same genes and combiner |
| `adata.obsm[f"{name}_smoothed"]` | `(n_obs, n_genes)` smoothed expression (`store_genes=True`) |
| `adata.uns["spatial_smooth"][name]` | provenance: genes, pipeline, resolved bandwidths, version |

Write it out, and a later plotting call reads those keys. **Nothing is recomputed** — no `kompot`,
no `palantir`, no GP solve. That is what makes an expensive smoothing worth doing once."""
)

code(
    '''\
import anndata as ad

adata.write_h5ad("smoothed.h5ad")

reloaded = ad.read_h5ad("smoothed.h5ad")
print("stored results:", ss.list_results(reloaded))

prov = ss.provenance(reloaded, "custom")
for step in prov["steps"]:
    print(f"  {step['kind']:<14} basis={step['basis']:<18} resolved={step['resolved']}")'''
)

code(
    '''\
# Nothing is recomputed here: the smoothed values are read straight from the file.
import time
start = time.time()
ss.pl.signature(reloaded, "custom", backend="scanpy", frameon=False)
print(f"drawing the saved result took {time.time() - start:.2f} seconds")'''
)

md(
    """\
Smoothing this signature took minutes. Drawing it back from the saved file took a fraction of a
second, because the field was never recomputed — `spatial_smooth.plot` reads `obs` and `uns` and
hands them to scanpy. That is the whole point of saving: do the expensive step once."""
)

# --------------------------------------------------------------------------------------- #
md(
    """\
## 5. Odds and ends

### Restrict to a subset of cells

Cells filtered out neither train the smoother nor receive the field. The call returns a **new,
smaller** `AnnData` — use the return value."""
)

code(
    '''\
# A coarse annotation to filter on (this public subset ships none).
adata.obs["half"] = np.where(
    adata.obsm["spatial"][:, 0] < np.median(adata.obsm["spatial"][:, 0]), "left", "right"
)

left = ss.smooth(adata, HIPPOCAMPUS, "hippocampus", subset_key="half", include=["left"])
print(f"{adata.n_obs:,} cells -> {left.n_obs:,} after the filter")
print("provenance n_obs:", ss.provenance(left, "hippocampus")["n_obs"])'''
)

md(
    """\
### Fit the GP on one condition, evaluate everywhere

`KompotGP(groupby=..., condition=...)` trains on one group and imputes the field for all cells —
useful when one arm of an experiment is the reference.

### Other engines

`Kde` (a fine-grid FFT Nadaraya-Watson estimator, via `KDEpy`) renders a field rather than a
neighbour average; `"spatial-gp"` puts the Gaussian process on tissue coordinates with a sensible
`ls_factor`. Both are one `steps` argument away.

```python
ss.smooth(adata, HIPPOCAMPUS, "kde",        steps="spatial-kde")
ss.smooth(adata, HIPPOCAMPUS, "spatial_gp", steps="spatial-gp")
```

### Where to go next

* `ss.provenance(adata, name)` — exactly what was run, with the bandwidths it resolved.
* The **[Concepts](https://settylab.github.io/spatial-smooth/concepts.html)** page — composition
  semantics, the scoring contract, and why gene-level smoothing costs nothing in correctness.

And once more, because it is the thing that matters: **these smoothed values are for looking at.**
Do your statistics on `adata.obs["hippocampus_raw"]`."""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = pathlib.Path(__file__).parent / "tutorial.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
