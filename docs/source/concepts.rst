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
   [{'kind': 'knn_gaussian', 'basis': 'spatial', 'k': 400, 'sigma': None,
     'sigma_factor': 6.0, 'workers': -1,
     'resolved': {'sigma_nominal': 78.3, 'sigma_used': 78.3, 'k_used': 400,
                  'kernel_mass_retained': 0.955,
                  'sigma_effective': 71.6,
                  'sigma_effective_p1': 52.2, 'sigma_effective_p99': 84.1}}]

``sigma_nominal`` is the Gaussian's width *before* truncation -- a bandwidth no cell actually
experiences. (``sigma_used`` is the same number under its older name.) ``sigma_effective`` is what
the ``k``-truncated kernel behaves like. **Quote the latter** -- see :ref:`truncation` below.


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
     - truncated Gaussian over ``k`` spatial neighbours
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


.. _truncation:

Truncation: what bandwidth the data actually sees
-------------------------------------------------

:class:`~spatial_smooth.steps.KnnGaussian` evaluates its Gaussian only over each cell's ``k``
nearest neighbours. That truncation is not free, and it is the number most likely to end up
misquoted in a methods section, so the package measures it.

Write :math:`d_{ij}` for the distance from cell :math:`i` to its neighbour :math:`j`, and
:math:`\mathcal{N}_k(i)` for the ``k`` nearest neighbours of :math:`i` (itself included). The
weights are

.. math::

   w_{ij} = \frac{\exp\!\left(-d_{ij}^{2} / 2\sigma^{2}\right)}
                 {\sum_{l \in \mathcal{N}_k(i)} \exp\!\left(-d_{il}^{2} / 2\sigma^{2}\right)},
   \qquad \sum_{j \in \mathcal{N}_k(i)} w_{ij} = 1 ,

so the operator is row-stochastic by construction. Let :math:`r_k(i)` be the distance to the
``k``-th neighbour. Two diagnostics follow, both recorded by
:func:`~spatial_smooth.core.provenance`.

**Retained kernel mass.** For a *continuous* isotropic 2-D Gaussian the mass inside radius
:math:`r` is :math:`1 - e^{-r^{2}/2\sigma^{2}}`, so an estimate of the fraction of the untruncated
kernel that survives the ``k``-neighbour cutoff is

.. math::

   m = \frac{1}{n}\sum_{i} \left[ 1 - \exp\!\left(-\,r_k(i)^{2} / 2\sigma^{2}\right) \right].

**Effective bandwidth.** A 2-D Gaussian satisfies :math:`\mathbb{E}[d^{2}] = 2\sigma^{2}`.
Inverting that on the *realised* weights gives the bandwidth the truncated kernel behaves like:

.. math::

   \sigma_{\mathrm{eff}}(i) = \sqrt{\tfrac{1}{2} \sum_{j \in \mathcal{N}_k(i)} w_{ij}\, d_{ij}^{2}} .

This :math:`m` is an **estimate, not an exact accounting**: it equals the discrete retained weight
fraction :math:`\sum_{j \in \mathcal{N}_k(i)} \tilde{w}_{ij} \big/ \sum_{j} \tilde{w}_{ij}`
(unnormalised :math:`\tilde{w}`) only where the local point density is uniform. Measured against
the discrete truth on a real section it is biased low by 0.006 at ``k=400`` and 0.022 at
``k=100`` -- i.e. it *under*-reports retention, so the warning below fires slightly early rather
than slightly late.

Because :math:`r_k(i)` is fixed by a neighbour *count*, it contracts where cells are dense and
expands where they are sparse: :math:`\sigma_{\mathrm{eff}}` varies from cell to cell, and the
smoother is **truncated-Gaussian and implicitly density-adaptive** rather than fixed-bandwidth.
``provenance()`` therefore stores ``kernel_mass_retained`` (:math:`m`), ``sigma_effective``
(:math:`\overline{\sigma_{\mathrm{eff}}}`) and its 1st/99th percentiles, and a ``UserWarning``
fires when :math:`m < 0.9`.

Measured on the tutorial's 10x Xenium mouse-brain section (:math:`n = 36{,}602`, median
nearest-neighbour distance 13.05 µm, hence nominal :math:`\sigma = 6 \times 13.05 = 78.3` µm):

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - quantity
     - ``k = 100``
     - ``k = 400`` (default)
   * - retained mass :math:`m`
     - 0.58
     - **0.96**
   * - :math:`\overline{\sigma_{\mathrm{eff}}}`
     - 47.3 µm
     - **71.6 µm**
   * - :math:`\sigma_{\mathrm{eff}}` p1 → p99
     - 24.9 → 68.6 µm
     - 52.2 → 84.1 µm

At ``k = 100`` the nominal :math:`\sigma` overstates the realised bandwidth by roughly 40%, which
is why the default is ``k = 400``. **Quote** ``sigma_effective``, **never** ``sigma_nominal``.

Implementation: :func:`spatial_smooth.smoothers.knn_gaussian_operator` (``return_info=True``).


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
