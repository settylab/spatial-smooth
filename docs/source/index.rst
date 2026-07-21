spatial-smooth
==============

**Composable smoothing of gene-set signatures over space and cell state.**

.. danger::

   **This package is for visualization only.** It exists to make spatial regions easier to see.
   Its output is a picture, not data.

   Smoothing deliberately makes neighbouring cells look like one another. That is exactly what
   you want when you are trying to *see* where a program is active, and exactly what you must
   not hand to a statistical test. A smoothed score is spatially autocorrelated by construction:
   the cells are no longer independent observations, so differential expression, differential
   abundance, clustering, correlations and p-values computed on smoothed values will find
   "significant" structure in pure noise.

   **Look at the smoothed values. Do the statistics on the raw ones**
   (``adata.obs[f"{name}_raw"]``, which every call writes for you), using a method that accounts
   for spatial dependence.

Every cell in a single-cell or spatial assay is measured independently, so a per-cell signature
score is dominated by dropout and sampling noise -- a speckle of dots in which a real anatomical
region is genuinely hard to spot. Smoothing lets neighbouring cells borrow statistical strength,
turning that speckle into a coherent field you can read at a glance. *Which* neighbours count is
the scientific choice, and this package makes it explicit.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - smoothing
     - neighbours are...
     - recovers
   * - **spatial**
     - physically adjacent cells (``obsm["spatial"]``)
     - tissue architecture: niches, layers, gradients
   * - **cell state**
     - transcriptionally similar cells (a diffusion map)
     - biological structure, independent of position
   * - **both, composed**
     - first the manifold, then the tissue
     - denoised expression laid out in space
   * - **both, blended**
     - space and cell state, independently
     - a symmetric mean of the two views, on the raw score's scale

Each is one argument apart, and **blend** -- space and cell state on equal footing -- is the mode
to reach for by default:

.. code-block:: python

   import spatial_smooth as ss

   ss.smooth(adata, genes, "sig", steps="blend")      # both, symmetric  (recommended)
   ss.smooth(adata, genes, "sig")                     # spatial only     (the fast default engine)
   ss.smooth(adata, genes, "sig", steps="dm")         # cell state only
   ss.smooth(adata, genes, "sig", steps="dm+spatial") # both, composed
   ss.pl.signature(adata, "sig")                      # raw vs smoothed, on tissue

Results are written into the :class:`~anndata.AnnData`. Save it, ship it, reload it -- plotting
**never recomputes**.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   concepts
   tutorial
   api

Links
-----

* **Source code:** `github.com/settylab/spatial-smooth <https://github.com/settylab/spatial-smooth>`_
  (private; ask the `Setty Lab <https://setty-lab.org>`_ for access)
* **Issue tracker:** `github.com/settylab/spatial-smooth/issues <https://github.com/settylab/spatial-smooth/issues>`_
* **Setty Lab:** `setty-lab.org <https://setty-lab.org>`_ -- Fred Hutchinson Cancer Center
* **Related:** `kompot <https://github.com/settylab/kompot>`_ (the GP backend),
  `mellon <https://github.com/settylab/mellon>`_ (its GP engine),
  `palantir <https://github.com/settylab/palantir>`_ (diffusion maps)

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
