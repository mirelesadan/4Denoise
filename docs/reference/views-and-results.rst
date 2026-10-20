Views and Results API
=====================

These objects retain the calibration or analysis context needed to interpret
their arrays. Shape-changing methods generally return new objects; inspect
the individual method contracts before relying on that behavior.

Real-space images
-----------------

.. autoclass:: fourdenoise.RealSpace

   .. automethod:: __init__

.. automethod:: fourdenoise.RealSpace.show

.. automethod:: fourdenoise.RealSpace.set_scale

.. automethod:: fourdenoise.RealSpace.crop

.. automethod:: fourdenoise.RealSpace.resize

.. automethod:: fourdenoise.RealSpace.copy

Diffraction patterns
--------------------

.. autoclass:: fourdenoise.ReciprocalSpace

   .. automethod:: __init__

.. automethod:: fourdenoise.ReciprocalSpace.show

.. automethod:: fourdenoise.ReciprocalSpace.set_scale

.. automethod:: fourdenoise.ReciprocalSpace.get_peaks

.. automethod:: fourdenoise.ReciprocalSpace.get_centers

.. automethod:: fourdenoise.ReciprocalSpace.get_intensities

.. automethod:: fourdenoise.ReciprocalSpace.block_direct_beam

.. automethod:: fourdenoise.ReciprocalSpace.inpaint_background

.. automethod:: fourdenoise.ReciprocalSpace.copy

Analysis results
----------------

.. autoclass:: fourdenoise.BraggnessResult
   :members: as_real_space

.. autoclass:: fourdenoise.RQCalibration
   :members: matrix, inverse_matrix, transform_vectors, to_dict, from_matrix

.. autoclass:: fourdenoise.RQComparison
   :members: parameters, calibration, image_matrix, transform_points, set_parameters, apply, reset, export, close

.. autoclass:: fourdenoise.PeakDetectionResult
   :members: observed_mask

.. autoclass:: fourdenoise.PeakScanResult
   :members: scan_shape, peak_counts, observed_peak_counts

.. automethod:: fourdenoise.PeakScanResult.load

.. automethod:: fourdenoise.PeakScanResult.copy

``PeakDetectionResult`` describes one pattern's detections. ``PeakScanResult``
contains scan-indexed ragged arrays and optional integrated intensities;
it is not itself a :class:`fourdenoise.RealSpace` image. A file produced by
``DiffractionSource.get_peaks(destination=...)`` uses a separate peak schema,
not the saved diffraction-array format. See the
:doc:`peak guide <../guides/peaks-and-strain>` for coordinates, provenance,
masked entries, and selective numeric-table reads.

.. autoclass:: fourdenoise.StrainResult
   :members: valid_mask, as_real_space
