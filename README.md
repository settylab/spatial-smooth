# spatial-smooth

**Composable smoothing of gene signatures over space and cell state.**

Every cell in a single-cell or spatial assay is measured independently, so a per-cell signature
score is dominated by dropout and sampling noise. Smoothing lets neighbouring cells borrow
statistical strength: a speckled score becomes a coherent field. *Which* neighbours count is the
scientific choice, and `spatial-smooth` makes it explicit.

| smoothing | neighbours are… | recovers |
|---|---|---|
| **spatial** | physically adjacent cells (`obsm["spatial"]`) | tissue architecture: niches, layers, gradients |
| **cell state** | transcriptionally similar cells (a diffusion map) | biological structure, independent of position |
| **both, composed** | first the manifold, then the tissue | denoised expression laid out in space |

The three are one argument apart:

```python
import spatial_smooth as ss

ss.smooth(adata, genes, "sig")                     # spatial only  (the default)
ss.smooth(adata, genes, "sig", steps="dm")         # cell state only
ss.smooth(adata, genes, "sig", steps="dm+spatial") # both, in that order
ss.pl.signature(adata, "sig")                      # raw vs smoothed, on tissue
```

Results are written into the `AnnData`. Save it, ship it, reload it — plotting **never
recomputes**.

---

## Install

```bash
pip install "spatial-smooth[all]"
```

Only `numpy`, `scipy`, `pandas` and `anndata` are hard requirements. Everything else is an
extra, imported lazily, and reported with the exact `pip install` line when missing:

| extra | brings | needed for |
|---|---|---|
| `dm` | [`kompot`](https://github.com/settylab/kompot) ≥ 0.8.0 | the Gaussian-process step (`KompotGP`) |
| `embedding` | [`palantir`](https://github.com/settylab/palantir) | computing the diffusion map |
| `plot` | `scanpy`, `matplotlib` | plotting |
| `squidpy` | `squidpy` | the squidpy plotting backend |
| `kde` | `KDEpy` | the fine-grid FFT smoother (`Kde`) |

```python
ss.check_dependencies()   # prints a table with a pip line for every gap
```

> `kompot ≥ 0.8.0` exposes `smooth_expression`; PyPI still carries 0.7.x. Until 0.8.0 ships,
> install it from source: `pip install "kompot @ git+https://github.com/settylab/kompot.git"`.

---

## Three levels of control

### 1 — one line, defaults do everything

```python
import scanpy as sc, spatial_smooth as ss

adata = sc.read_h5ad("xenium.h5ad")          # has obsm["spatial"], log-normalised
ss.smooth(adata, ["Prox1", "Neurod6", "Wfs1"], "hippocampus")
ss.pl.signature(adata, "hippocampus")
```

A Gaussian kernel over each cell's 100 nearest spatial neighbours, with a bandwidth inferred from
the data. `adata.obs["hippocampus"]` is the smoothed score, `adata.obs["hippocampus_raw"]` the
unsmoothed one, and the plot shows them side by side.

### 2 — pick the pipeline and pass plotting kwargs

```python
ss.smooth(adata, genes, "hippocampus", steps="dm+spatial")     # manifold, then tissue
ss.pl.signature(adata, "hippocampus", cmap="magma", size=6, vmax="p99.5", ncols=2)
```

Everything after `name` in `pl.signature` is forwarded **verbatim** to the backend
(`squidpy.pl.spatial_scatter`, `scanpy.pl.embedding`, or `scanpy.pl.spatial`).

Shorthands for `steps`:

| shorthand | pipeline |
|---|---|
| `"spatial"` (default) | `[KnnGaussian()]` |
| `"dm"` | `[KompotGP()]` |
| `"dm+spatial"` | `[KompotGP(), KnnGaussian()]` |
| `"spatial+dm"` | `[KnnGaussian(), KompotGP()]` |
| `"spatial-kde"` | `[Kde()]` |
| `"spatial-gp"` | `[KompotGP(basis="spatial", ls_factor=0.3)]` |
| `"none"` | `[]` — raw score only |

### 3 — fully modular: compute, store, plot later

```python
pipeline = [
    ss.KompotGP(basis="DM_EigenVectors", ls_factor=10.0, n_landmarks=8000),
    ss.KnnGaussian(basis="spatial", k=64, sigma_factor=4.0),
]
ss.smooth(adata, genes, "hippocampus", steps=pipeline, store_genes=True)
adata.write_h5ad("smoothed.h5ad")

# ... months later, on a laptop, without kompot or palantir installed:
import anndata as ad, spatial_smooth as ss
adata = ad.read_h5ad("smoothed.h5ad")
ss.provenance(adata, "hippocampus")["steps"]     # exactly what was run, with resolved bandwidths
ss.pl.signature(adata, "hippocampus")            # renders; recomputes nothing
```

---

## What lands in the `AnnData`

| key | contents |
|---|---|
| `adata.obs[name]` | smoothed signature score |
| `adata.obs[f"{name}_raw"]` | unsmoothed score, same genes and combiner |
| `adata.obsm[f"{name}_smoothed"]` | `(n_obs, n_genes)` smoothed expression (`store_genes=True`) |
| `adata.uns["spatial_smooth"][name]` | provenance: genes, pipeline, resolved bandwidths, version |

`ss.provenance(adata, name)` reads that back with the pipeline decoded; `ss.list_results(adata)`
lists every stored result. The plotting module reads **only** these keys — that is what makes
"plot without recomputing" a contract rather than a hope, and it is covered by a test that seals
off every compute path before rendering a reloaded file.

---

## Composition semantics

A pipeline is an ordered list of steps. Each step smooths an `(n_obs, n_genes)` expression matrix
over one embedding and hands the result to the next, so `[KompotGP(), KnnGaussian()]` smooths the
*already manifold-denoised* expression over physical space. The score is formed once, at the end.

The `mean_z` combiner standardises each gene with the **raw** matrix's mean and standard
deviation, for both the raw and the smoothed score. So the two share a colour scale — and,
because `KnnGaussian` and `Kde` are row-stochastic (their weights sum to one, hence they map
constants to themselves), smoothing the genes and then scoring gives *exactly* the same answer as
scoring and then smoothing the score. Gene-level is what the pipeline does, which keeps the
Gaussian-process step — which does not commute — meaningful in the same framework.

---

## Choosing a smoother

| step | engine | full slide (~1.6 × 10⁵ cells) | gives you |
|---|---|---|---|
| `KnnGaussian` | Gaussian kernel over `k` spatial neighbours | ~1 s | the default; fast, sharp |
| `Kde` | FFT Nadaraya-Watson on a fine grid (KDEpy) | ~1 s | a rendered field; resolution-bound |
| `KompotGP` | Gaussian-process regression (kompot/mellon) | minutes | a length scale, a posterior, fit-on-one-condition |

`KnnGaussian` and `KompotGP` produce visually equivalent spatial fields, with the GP marginally
sharper when given enough landmarks; the kNN kernel is roughly two orders of magnitude faster
per gene. Use the GP when you want its extras — an explicit length scale, uncertainty, or
`groupby`/`condition` (fit the GP on one condition, evaluate it everywhere) — and for smoothing
over a diffusion map, where it is the established choice.

### Bandwidths are scale-invariant

Every default bandwidth is a multiple of the median nearest-neighbour distance of the
coordinates, so the same factor smooths the same amount whether coordinates are in microns or
millimetres. `KnnGaussian(sigma_factor=6.0)` is ~6 cell spacings (≈ 50 µm on a section with 8 µm
spacing). `KompotGP` inherits mellon's empirical length scale
(`ls = ls_factor · geometric_mean(nn_distances) · e³`), which is scale-invariant for the same
reason.

**One caveat worth internalising.** Over a diffusion map, kompot's native `ls_factor=10` is
right. Over *physical* coordinates it is roughly 200× the cell spacing and collapses the field
into a single global gradient. Use `ls_factor≈0.3` there — that is exactly what the
`"spatial-gp"` shorthand does.

---

## Tutorial

[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) walks the three levels on a public 10x
Xenium mouse-brain section (downloaded on first run, ~4.5 MB), ending with a reload-and-replot
that proves nothing is recomputed.

## Documentation

Built with Sphinx: `pip install -e ".[docs]" && make -C docs html`.

## License

MIT. See [LICENSE](LICENSE).
