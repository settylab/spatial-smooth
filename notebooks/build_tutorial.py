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
| **both, blended** | space and cell state, independently | a symmetric mean of the two views, on the raw score's scale |

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
A **hippocampal** program from the panel — dentate-gyrus and CA markers spanning a wide range of
abundance. We rank the candidate genes by **detection rate** — the fraction of cells with any
counts — because the *sparse* markers, seen in only a small minority of cells, are the real test
of smoothing: can it rescue a domain the raw speckle barely shows? The sparsest present marker
becomes our rescue stress-test later on."""
)

code(
    '''\
# Hippocampal markers this panel may carry; keep whichever are present.
HIPPO_MARKERS = ["Prox1", "Neurod6", "Wfs1", "Fibcd1", "Ascl1"]
present = [g for g in HIPPO_MARKERS if g in adata.var_names]
assert present, "none of the hippocampal markers are in this panel"

def detection_rate(gene):
    """Fraction of cells with any signal (log1p(0)=0, so X>0 reads it pre- or post-log)."""
    col = adata[:, gene].X
    col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
    return float((col > 0).mean())

rates = sorted((detection_rate(g), g) for g in present)
print("hippocampal markers, sparsest first:")
for r, g in rates:
    print(f"  {g:9s} detected in {r:6.1%} of cells")

HIPPOCAMPUS = [g for _, g in sorted(rates, reverse=True)]   # densest first
SPARSE_GENE = rates[0][1]                                    # the sparsest present marker
print(f"\\nsignature               : {HIPPOCAMPUS}")
print(f"sparse member to rescue : {SPARSE_GENE} ({rates[0][0]:.1%} of cells)")'''
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
| `"dm+spatial"` | `[KompotGP(), KnnGaussian()]` | both, cell state first (composed) |
| `"blend"` | `Blend("spatial", "dm")` | both, independent + symmetric (§3b) |

Doing *just one of the two* is the ordinary case, not a special one — a one-element pipeline.
Composing runs the steps left to right: the spatial step smooths the expression the
cell-state step already denoised. Blending (`"blend"`, covered in 3b) instead keeps the two views
independent.

The cell-state step is a Gaussian-process regression over a diffusion map of the expression
manifold (`kompot.smooth_expression`, built on `mellon`). It needs `obsm["DM_EigenVectors"]`;
with `auto_embed=True` (the default) `spatial-smooth` computes it with Palantir if absent.

### Smooth every gene **once**, then derive any signature for free

The Gaussian process is slow — a diffusion-map GP over the whole panel is the one expensive thing
this notebook does. So do it **once**. `ss.smooth_all(adata, steps=...)` smooths *every* gene
through a view and caches the result; every later `ss.smooth(..., all_genes=True)` — for a
signature, a single gene, or a `"blend"` — reads those pre-smoothed layers instead of recomputing.
The smoothers are per-gene operations, so a signature's smoothed columns gathered from the
all-genes layer are identical to smoothing that signature alone (bit-for-bit for the neighbour
smoothers; to floating-point precision for the GP). Two passes up front, then nothing recomputes.

Everything below runs on the **full section** — every cell, no subsampling."""
)

code(
    '''\
ss.compute_diffusion_map(adata)         # Palantir -> obsm["DM_EigenVectors"]
adata.obsm["DM_EigenVectors"].shape'''
)

code(
    '''\
# Smooth EVERY gene once through each view. This is the whole cost of the section: one spatial
# pass and one diffusion-map GP solve, each over the full panel.
import time
t0 = time.time()
ss.smooth_all(adata, steps="spatial")   # fast: Gaussian kNN over tissue coordinates
ss.smooth_all(adata, steps="dm")        # slow: the GP over the diffusion map -- every gene, ONCE
print(f"smoothed all {adata.n_vars} genes through both views in {time.time() - t0:.1f}s")'''
)

code(
    '''\
# Every mode is now *derived* from the two pre-smoothed layers -- a cache hit, no re-smoothing.
t0 = time.time()
ss.smooth(adata, HIPPOCAMPUS, "spatial_only", steps="spatial",    all_genes=True)
ss.smooth(adata, HIPPOCAMPUS, "dm_only",      steps="dm",         all_genes=True)
ss.smooth(adata, HIPPOCAMPUS, "composed",     steps="dm+spatial", all_genes=True)
print(f"derived three modes from the cached layers in {time.time() - t0:.2f}s")
ss.list_results(adata)'''
)

code(
    '''\
# The cheap check: how many distinct smooths did the cache store, and of what kind? The GP is the
# one that matters -- it must appear exactly once, however many modes read from it.
import json
from collections import Counter

entries = json.loads(adata.uns[ss.CACHE_KEY])["entries"].values()
kinds = Counter(json.loads(e["params"])["kind"] for e in entries)
print("distinct smooths cached:", dict(kinds))
print(f"kompot_gp ran {kinds['kompot_gp']}x -- the diffusion-map GP solved every gene exactly once")'''
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
does both, and is the smoothest of the three — and all three were *derived*, not recomputed, from
the two layers we smoothed up front. `composed`'s cell-state step reused the same GP solve; only
its final spatial pass over the GP output was new (the second `knn_gaussian` above)."""
)

# --------------------------------------------------------------------------------------- #
md(
    """\
### 3b. Blending: a symmetric alternative to composition

Composition (`"dm+spatial"`) is **not symmetric**: the spatial step smooths whatever the
cell-state step handed it, so the result inherits the spatial footprint and leans toward that
parent. `steps="blend"` takes the other route — it smooths the raw expression **independently**
over space and over cell state, then returns a *symmetric mean* of the two, staying roughly
equidistant from both rather than collapsing onto either.

The two views sit on different scales (spatial smoothing suppresses more variance than the GP), so
a blend standardises each score before averaging. That average lives in **z-units**, which would
not share a colour bar with the other modes — so the final, load-bearing step **range-calibrates**
it back onto the raw score's scale: it matches the raw score's mean and standard deviation
(`calibrate="std"`, the default), or the median and inter-quartile range
(`ss.Blend(calibrate="iqr")`, robust to outlier cells). The calibration is a single affine,
monotone map, so it never reorders cells — it only places the numbers where they belong."""
)

code(
    '''\
# both, blended: independent spatial + cell-state views, symmetric, range-calibrated.
# all_genes=True -> both branches read the layers we already smoothed; the GP does NOT run again.
gp_before = Counter(json.loads(e["params"])["kind"]
                    for e in json.loads(adata.uns[ss.CACHE_KEY])["entries"].values())["kompot_gp"]
ss.smooth(adata, HIPPOCAMPUS, "blended", steps="blend", all_genes=True)
gp_after = Counter(json.loads(e["params"])["kind"]
                   for e in json.loads(adata.uns[ss.CACHE_KEY])["entries"].values())["kompot_gp"]
print(f"GP smooths cached before blend: {gp_before}, after: {gp_after}  (unchanged -> reused)")

# the point of the calibration, in one table: the blended field lands on the raw score's
# scale (mean/std matched), not in z-units -- so it shares a colour bar with the rest.
cols = ["hippocampus_raw", "spatial_only", "composed", "blended"]
adata.obs[cols].describe().loc[["mean", "std", "min", "max"]].T'''
)

code(
    '''\
ss.pl.compare(
    adata, ["spatial_only", "dm_only", "composed", "blended"], raw=True,
    backend="scanpy", ncols=5, frameon=False,
)'''
)

md(
    """\
The blended panel carries cell-state structure and spatial coherence at once without collapsing
onto either parent, and — because it was calibrated — its numbers sit on the same scale as the raw
score beside it. The provenance records both branches and the affine calibration that was
applied:"""
)

code(
    '''\
blend_step = ss.provenance(adata, "blended")["steps"][0]
print("left      :", [s["kind"] for s in blend_step["left"]])
print("right     :", [s["kind"] for s in blend_step["right"]])
print("calibrate :", blend_step["calibrate"])
r = blend_step["resolved"]
print(f"blend std {r['blend_std']:.3f} vs raw std {r['raw_std']:.3f}  "
      f"(rescaled x{r['scale']:.3f} from the z-unit average)")'''
)

md(
    """\
### 3c. Rescuing a sparse gene

This is what smoothing is *for*. A sparse marker — detected in only a small minority of cells — is
almost invisible in the raw score: a scatter of isolated positive cells with no legible shape. But
if those cells cluster in a tissue domain, smoothing lets them reinforce each other and the domain
emerges. We already smoothed every gene above, so pulling out the sparse member costs nothing: it
is a single-gene signature read straight from the pre-smoothed spatial layer (`all_genes=True` →
a cache hit, no new smoothing)."""
)

code(
    '''\
print(f"{SPARSE_GENE}: detected in {detection_rate(SPARSE_GENE):.1%} of cells "
      f"-- raw is mostly zeros")

# A one-gene signature, derived from the all-genes spatial layer we already computed.
ss.smooth(adata, [SPARSE_GENE], f"{SPARSE_GENE}_smoothed", steps="spatial", all_genes=True)

ss.pl.signature(
    adata, f"{SPARSE_GENE}_smoothed", raw=True,
    backend="scanpy", cmap="magma", frameon=False,
)'''
)

md(
    """\
Left, the raw gene: a sparse speckle you could not annotate. Right, the smoothed field: the same
handful of positive cells, now summed over their neighbourhoods, resolve the hippocampal domain
they belong to. Nothing about the measurement changed — the raw score is still there in
`obs[f"{SPARSE_GENE}_smoothed_raw"]` for any statistics — but the *picture* went from noise to a
region you can point at. (And, once more: the smoothed panel is for **looking**. A test run on it
would treat each cell's borrowed signal as independent evidence and badly overstate the domain.)"""
)

# --------------------------------------------------------------------------------------- #
md(
    """\
### 3d. Plot control: kwargs go straight through

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
### 3e. Bandwidth is scale-invariant

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
ra = ss.provenance(adata, "hippocampus")["steps"][0]["resolved"]
rb = ss.provenance(rescaled, "hippocampus")["steps"][0]["resolved"]
print("max |difference|      :", np.abs(a - b).max())
print("sigma_effective (um)  :", round(ra["sigma_effective"], 2))
print("sigma_effective (nm)  :", round(rb["sigma_effective"], 2))
print("(nominal was", round(ra["sigma_nominal"], 2), "um -- the number NOT to quote)")'''
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
# k=64 truncates the Gaussian. The package says so, on stderr, and the notebook keeps that
# message: it is the disclosure this section exists to teach.
ss.smooth(adata, HIPPOCAMPUS, "custom", steps=pipeline, store_genes=True)

print("smoothed score      :", adata.obs["custom"].shape)
print("smoothed expression :", adata.obsm["custom_smoothed"].shape)   # store_genes=True'''
)

md(
    """\
### Read the warning — it is doing its job

That pipeline chose `k=64`, and the package objected. Restricting the Gaussian to a cell's 64
nearest neighbours **cuts the kernel off** before it has faded: only ~69% of its weight lies
inside that radius. The bandwidth the data actually feels is therefore *narrower* than the
nominal `sigma`, and — because 64 neighbours reach further apart in sparse tissue than in dense
tissue — it is **not the same in every cell**.

Nothing here is broken. A truncated Gaussian is a perfectly respectable smoother. But if you were
to write "we smoothed with a Gaussian of σ = 52 µm" in a methods section, you would be reporting a
number the code never applied. That is what the warning is for, and it tells you exactly which
number to quote instead:

```python
res = ss.provenance(adata, "custom")["steps"][1]["resolved"]
res["sigma_used"]            # 52.1  <- nominal; do NOT quote this
res["sigma_effective"]       # 35.7  <- what the kernel behaves like; quote this
res["kernel_mass_retained"]  # 0.69  <- how much of the Gaussian survived
```

Raise `k` (the default, 400, keeps ~96% of the mass) and the warning goes away, `sigma_effective`
converges on `sigma_used`, and the smoother becomes effectively fixed-bandwidth.

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

# Compile every code cell before writing. A syntax error otherwise costs a whole cluster run --
# nbconvert only discovers it when the kernel reaches that cell, minutes in.
_bad = []
for _i, _cell in enumerate(cells):
    if _cell["cell_type"] != "code":
        continue
    _src = "\n".join(
        "pass" if _line.strip().startswith(("%", "!")) else _line
        for _line in _cell["source"].splitlines()
    )
    try:
        compile(_src, f"cell{_i}", "exec")
    except SyntaxError as _exc:  # pragma: no cover
        _bad.append(f"cell {_i}: {_exc}")
if _bad:
    raise SystemExit("refusing to write a notebook with syntax errors:\n  " + "\n  ".join(_bad))

out = pathlib.Path(__file__).parent / "tutorial.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells, all code cells compile)")
