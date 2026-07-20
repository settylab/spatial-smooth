API reference
=============

Compute
-------

.. autofunction:: spatial_smooth.smooth

.. autofunction:: spatial_smooth.compute_diffusion_map

.. autofunction:: spatial_smooth.select_cells


Steps
-----

.. automodule:: spatial_smooth.steps
   :no-members:

.. autoclass:: spatial_smooth.steps.KnnGaussian
   :members:
   :exclude-members: apply, to_dict

.. autoclass:: spatial_smooth.steps.Kde
   :members:
   :exclude-members: apply, to_dict

.. autoclass:: spatial_smooth.steps.KompotGP
   :members:
   :exclude-members: apply, to_dict

.. autoclass:: spatial_smooth.steps.Blend
   :members:

.. autoclass:: spatial_smooth.steps.Step
   :members:

.. autofunction:: spatial_smooth.steps.resolve_steps

.. autofunction:: spatial_smooth.steps.as_blend


Reading results back
--------------------

.. autofunction:: spatial_smooth.provenance

.. autofunction:: spatial_smooth.list_results


Plotting
--------

.. automodule:: spatial_smooth.plot
   :no-members:

.. autofunction:: spatial_smooth.plot.signature

.. autofunction:: spatial_smooth.plot.compare

.. autofunction:: spatial_smooth.plot.available_backends


Smoothing kernels
-----------------

.. automodule:: spatial_smooth.smoothers
   :no-members:

.. autofunction:: spatial_smooth.smoothers.knn_gaussian_operator

.. autofunction:: spatial_smooth.smoothers.smooth_matrix_knn_gaussian

.. autofunction:: spatial_smooth.smoothers.smooth_field_knn_gaussian

.. autofunction:: spatial_smooth.smoothers.smooth_matrix_kde

.. autofunction:: spatial_smooth.smoothers.smooth_field_kde

.. autofunction:: spatial_smooth.smoothers.median_nn_distance


Environment
-----------

.. autofunction:: spatial_smooth.check_dependencies
