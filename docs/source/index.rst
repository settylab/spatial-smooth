spatial-smooth
==============

**Composable smoothing of gene signatures over space and cell state.**

Every cell in a single-cell or spatial assay is measured independently, so a per-cell signature
score is dominated by dropout and sampling noise. Smoothing lets neighbouring cells borrow
statistical strength: a speckled score becomes a coherent field. *Which* neighbours count is the
scientific choice, and this package makes it explicit.

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

The three are one argument apart:

.. code-block:: python

   import spatial_smooth as ss

   ss.smooth(adata, genes, "sig")                     # spatial only  (the default)
   ss.smooth(adata, genes, "sig", steps="dm")         # cell state only
   ss.smooth(adata, genes, "sig", steps="dm+spatial") # both, in that order
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

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
