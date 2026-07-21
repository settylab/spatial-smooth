# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-21

### Added
- **`blend` composition mode** — a symmetric alternative to `dm+spatial` composition. Smooths the raw
  expression independently over space and over cell state, then returns a range-calibrated symmetric
  mean of the two views (`steps="blend"`, or `ss.Blend(left=..., right=..., calibrate="std"|"iqr")`),
  so the result stays roughly equidistant from both parents instead of collapsing onto either.
- **Smoothing reuse cache** — each smoother's output is memoized on the `AnnData`, keyed by a stable
  hash of its input matrix, parameters and basis, so a repeated computation is served from the cache
  instead of recomputed. A four-mode figure that runs `spatial`, `dm` and `blend` pays for the
  expensive diffusion-map GP once, not twice. Bounded (LRU cap, `cache_max_entries=N`) and opt-out
  (`cache=False`); `ss.clear_smooth_cache(adata)` drops cache artifacts while leaving results intact.
- **`smooth_all` / `all_genes=True`** — smooth every gene once through a view (`ss.smooth_all(adata,
  steps=...)`), then derive any signature, single gene or `blend` from the pre-smoothed layers for free
  (`ss.smooth(..., all_genes=True)`). Two passes up front, then nothing recomputes.

### Fixed
- The tutorial notebook now **ships executed, with its figures** — both the committed
  `notebooks/tutorial.ipynb` and the rendered documentation. (A prior rebuild committed the notebook
  unexecuted, so the docs rendered no outputs.)
- The tutorial's dataset download now sends a browser `User-Agent`; the 10x CDN rejected the default
  `urllib` one with HTTP 403.
- **PyPI project-page images** — the README (the PyPI long-description) now references its figures and
  links by absolute URL, so they render on pypi.org instead of 404'ing against relative paths.

## [0.1.0] - 2026-07-10

### Added
- Initial release: composable spatial and cell-state smoothing of gene-set signatures
  (`ss.smooth`, `ss.smooth_signature`), the `spatial` / `dm` / `dm+spatial` pipelines, the
  `KnnGaussian` / `KompotGP` / `Kde` smoothers, provenance recording, the compute-once/plot-forever
  persistence contract, and the `ss.pl` plotting namespace wrapping scanpy and squidpy.

[0.2.0]: https://github.com/settylab/spatial-smooth/releases/tag/v0.2.0
[0.1.0]: https://github.com/settylab/spatial-smooth/releases/tag/v0.1.0
