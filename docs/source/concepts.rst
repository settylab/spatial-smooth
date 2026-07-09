Concepts
========

Steps and pipelines
-------------------

A **step** smooths an ``(n_obs, n_genes)`` expression matrix over one embedding stored in
``adata.obsm``. A **pipeline** is an ordered list of steps: each consumes the previous step's
output. ``[KompotGP(), KnnGaussian()]`` therefore denoises along the expression manifold first,
then smooths the *already-denoised* expression over physical space.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - step
     - default basis
     - engine
   * - :class:`~spatial_smooth.steps.KnnGaussian`
     - ``spatial``
     - Gaussian kernel over ``k`` nearest neighbours
   * - :class:`~spatial_smooth.steps.Kde`
     - ``spatial``
     - FFT Nadaraya-Watson on a fine grid (KDEpy)
   * - :class:`~spatial_smooth.steps.KompotGP`
     - ``DM_EigenVectors``
     - Gaussian-process regression (kompot/mellon)

Steps are frozen dataclasses -- *specifications*, not fitted objects. They carry no data, so the
same pipeline can be reused across datasets and recorded verbatim in ``adata.uns``.

Shorthands cover the common pipelines, so the common cases need no imports:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - shorthand
     - pipeline
     - meaning
   * - ``"spatial"`` (default)
     - ``[KnnGaussian()]``
     - spatial smoothing only
   * - ``"dm"``
     - ``[KompotGP()]``
     - cell-state smoothing only
   * - ``"dm+spatial"``
     - ``[KompotGP(), KnnGaussian()]``
     - both, cell state first
   * - ``"spatial+dm"``
     - ``[KnnGaussian(), KompotGP()]``
     - both, spatial first
   * - ``"spatial-kde"``
     - ``[Kde()]``
     - spatial, KDE engine
   * - ``"spatial-gp"``
     - ``[KompotGP(basis="spatial", ls_factor=0.3)]``
     - spatial, GP engine
   * - ``"none"``
     - ``[]``
     - no smoothing; raw score only

Doing *just one of the two* is the ordinary case, not a special one: ``steps="spatial"`` and
``steps="dm"`` are single-element pipelines.


The storage contract
--------------------

Everything :func:`~spatial_smooth.core.smooth` produces is written into the ``AnnData``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - key
     - contents
   * - ``adata.obs[name]``
     - smoothed signature score, ``float32``
   * - ``adata.obs[f"{name}_raw"]``
     - unsmoothed score from the same genes and combiner
   * - ``adata.obsm[f"{name}_smoothed"]``
     - ``(n_obs, n_genes)`` smoothed expression (``store_genes=True``)
   * - ``adata.uns["spatial_smooth"][name]``
     - provenance: genes, pipeline, resolved bandwidths, package version

Nothing else is needed to render the result later. :mod:`spatial_smooth.plot` reads those keys
and never recomputes, so a smoothed object can be written to ``.h5ad``, shipped, reloaded, and
plotted in an environment without ``kompot``, ``KDEpy`` or ``palantir`` installed. The package's
test suite asserts this adversarially: it blocks those imports and replaces every compute entry
point with a function that raises, then renders a reloaded file.

:func:`~spatial_smooth.core.provenance` reads the record back with the pipeline decoded, including
the bandwidth each step actually resolved to:

.. code-block:: python

   >>> ss.provenance(adata, "sig")["steps"]
   [{'kind': 'knn_gaussian', 'basis': 'spatial', 'k': 100, 'sigma': None,
     'sigma_factor': 6.0, 'workers': -1,
     'resolved': {'sigma_used': 73.4, 'k_used': 100}}]


Scoring, and why gene-level smoothing is free
---------------------------------------------

The multi-gene score is ``mean_z`` by default: standardise each gene, then average. **The mean and
standard deviation always come from the raw matrix**, for both the raw and the smoothed score.
Two consequences, both intended.

First, raw and smoothed scores share one scale, so they belong on a common colour bar.

Second, for a *row-stochastic* smoother -- one whose weights sum to one, so it maps a constant
field to itself -- smoothing the genes and then scoring is **exactly** scoring and then smoothing
the score. Writing :math:`W` for the operator and :math:`x_j` for gene :math:`j`,

.. math::

   W\left(\frac{1}{g}\sum_j \frac{x_j - \mu_j}{\sigma_j}\right)
   = \frac{1}{g}\sum_j \frac{W x_j - \mu_j}{\sigma_j},

because :math:`W\mathbf{1} = \mathbf{1}` carries the constants through untouched.
:class:`~spatial_smooth.steps.KnnGaussian` and :class:`~spatial_smooth.steps.Kde` are both
row-stochastic. The pipeline therefore smooths per gene at no cost in correctness, which is what
keeps a Gaussian-process step -- linear, but *not* row-stochastic, so it does **not** commute --
meaningful inside the same framework.


Choosing a smoother
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 20 28

   * - step
     - engine
     - full slide (~1.6e5 cells)
     - gives you
   * - ``KnnGaussian``
     - Gaussian kernel over ``k`` spatial neighbours
     - ~1 s
     - the default; fast, sharp
   * - ``Kde``
     - FFT Nadaraya-Watson on a fine grid
     - ~1 s
     - a rendered field; resolution-bound
   * - ``KompotGP``
     - Gaussian-process regression
     - minutes
     - a length scale, a posterior, fit-on-one-condition

``KnnGaussian`` and ``KompotGP`` produce visually equivalent spatial fields, with the GP
marginally sharper when given enough landmarks; the kNN kernel is roughly two orders of magnitude
faster per gene. Reach for the GP when you want its extras -- an explicit length scale,
uncertainty, or ``groupby``/``condition`` (fit on one condition, evaluate everywhere) -- and for
smoothing over a diffusion map, where it is the established choice.


Bandwidths are scale-invariant
------------------------------

Every default bandwidth is a multiple of the median nearest-neighbour distance of the coordinates,
so the same factor smooths the same amount whether the coordinates are in microns or millimetres.
``KnnGaussian(sigma_factor=6.0)`` is about six cell spacings (~50 um on a section with 8 um
spacing).

``KompotGP`` inherits mellon's empirical length scale,
:math:`\ell = \texttt{ls\_factor} \cdot \operatorname{geomean}(d_\mathrm{NN}) \cdot e^{3}`
(``mellon.parameters.compute_ls``), which is scale-invariant for the same reason. Passing an
explicit ``ls`` (or an explicit ``sigma`` to ``KnnGaussian``) bypasses the empirical path and puts
you back in absolute coordinate units.

.. warning::

   Over a diffusion map, kompot's native ``ls_factor=10`` is right. Over **physical coordinates**
   it is roughly 200x the cell spacing and collapses the field into a single global gradient. Use
   ``ls_factor`` near ``0.3`` there -- which is precisely what the ``"spatial-gp"`` shorthand does.

A small effective length scale also needs enough landmarks to resolve it: keep ``n_landmarks``
large enough that the landmark spacing stays below the length scale.
