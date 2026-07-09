Installation
============

.. code-block:: bash

   pip install "spatial-smooth[all]"

Only ``numpy``, ``scipy``, ``pandas`` and ``anndata`` are hard requirements. Everything else is
an extra, imported lazily, and reported with the exact ``pip install`` line when missing.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - extra
     - brings
     - needed for
   * - ``dm``
     - `kompot <https://github.com/settylab/kompot>`_
     - the Gaussian-process step (:class:`~spatial_smooth.steps.KompotGP`)
   * - ``embedding``
     - `palantir <https://github.com/settylab/palantir>`_
     - computing the diffusion map
   * - ``plot``
     - ``scanpy``, ``matplotlib``
     - plotting
   * - ``squidpy``
     - ``squidpy``
     - the squidpy plotting backend
   * - ``kde``
     - ``KDEpy``
     - the fine-grid FFT smoother (:class:`~spatial_smooth.steps.Kde`)

Check where you stand at any time:

.. code-block:: python

   import spatial_smooth as ss
   ss.check_dependencies()

.. code-block:: text

   spatial_smooth v0.1.0 -- dependency check
     numpy       2.4.6       ok
     pandas      2.3.3       ok
     scipy       1.18.0      ok
     anndata     0.12.17     ok
     kompot      -           missing: pip install "kompot>=0.7.0" (optional)
     ...

Every missing optional dependency raises where it is first needed, naming the package, what it
is for, and how to install it -- never a bare ``ModuleNotFoundError`` from deep in a call stack.

Reading a smoothed object needs *none* of the extras: a colleague can open your ``.h5ad`` and
plot it with ``scanpy`` alone.
