HyperData API
=============

``HyperData`` wraps a 2D image, 3D pattern stack, or 4D scan. The public
methods below are selected entry points, not an inventory of implementation
helpers. See the :doc:`data model <../concepts/data-model>` and task guides
before using a method on a new dataset.

.. autoclass:: fourdenoise.HyperData

   .. automethod:: __init__

Loading and metadata
--------------------

.. automethod:: fourdenoise.HyperData.metadata_summary

.. automethod:: fourdenoise.HyperData.open

.. automethod:: fourdenoise.HyperData.open_hdf5

.. automethod:: fourdenoise.HyperData.to_polar_hdf5

.. automethod:: fourdenoise.HyperData.resize_hdf5

.. automethod:: fourdenoise.HyperData.to_cartesian_hdf5

.. automethod:: fourdenoise.HyperData.remove_bg_hdf5

.. automethod:: fourdenoise.HyperData.save

.. automethod:: fourdenoise.HyperData.copy

Disk-backed source
------------------

Obtain a read-only source inside ``with HyperData.open(path) as source:``.
Source reductions return independent in-memory views; disk-output transforms
return saved paths, not full tensors. ``get_peaks`` returns a
:class:`fourdenoise.PeakScanResult` unless a destination is supplied.
See the :doc:`streaming tutorial <../getting-started/streaming>` and
:doc:`I/O guide <../guides/io-and-calibration>` for ownership, memory budgets,
optional readers, and the distinction between acquisition records and active
calibration. Global tensor denoising is not implicitly streamed.

.. autoclass:: fourdenoise.DiffractionSource
   :members: shape, ndim, dtype, nbytes, scan_shape, pattern_shape, real_shape, backend, filename, dataset_path, metadata, reader_metadata, reader_chunk_nbytes, has_unfold_metadata, closed

.. automethod:: fourdenoise.DiffractionSource.get_dp

.. automethod:: fourdenoise.DiffractionSource.virtual_image

.. automethod:: fourdenoise.DiffractionSource.get_peaks

.. automethod:: fourdenoise.DiffractionSource.to_polar

.. automethod:: fourdenoise.DiffractionSource.to_cartesian

.. automethod:: fourdenoise.DiffractionSource.resize

.. automethod:: fourdenoise.DiffractionSource.remove_bg

.. automethod:: fourdenoise.DiffractionSource.iter_chunks

.. automethod:: fourdenoise.DiffractionSource.load

.. automethod:: fourdenoise.DiffractionSource.close

.. autofunction:: fourdenoise.unpack_reader_metadata

Viewing and geometry
--------------------

.. automethod:: fourdenoise.HyperData.compare_rq

.. automethod:: fourdenoise.HyperData.set_rq_calibration

.. automethod:: fourdenoise.HyperData.get_dp

.. automethod:: fourdenoise.HyperData.virtual_image

.. automethod:: fourdenoise.HyperData.visualize

.. automethod:: fourdenoise.HyperData.crop

.. automethod:: fourdenoise.HyperData.resize

.. automethod:: fourdenoise.HyperData.to_polar

.. automethod:: fourdenoise.HyperData.to_cartesian

Preprocessing and denoising
---------------------------

.. automethod:: fourdenoise.HyperData.clip

.. automethod:: fourdenoise.HyperData.remove_bg

.. automethod:: fourdenoise.HyperData.alignment

.. automethod:: fourdenoise.HyperData.block_direct_beam

.. automethod:: fourdenoise.HyperData.unfold

.. automethod:: fourdenoise.HyperData.reshape

.. automethod:: fourdenoise.HyperData.denoise

.. automethod:: fourdenoise.HyperData.denoise_info

.. automethod:: fourdenoise.HyperData.rank_scree

Peak and strain analysis
------------------------

.. automethod:: fourdenoise.HyperData.get_braggness

.. automethod:: fourdenoise.HyperData.get_peaks

.. automethod:: fourdenoise.HyperData.get_centers

.. automethod:: fourdenoise.HyperData.get_intensities

.. automethod:: fourdenoise.HyperData.get_strains

.. automethod:: fourdenoise.HyperData.get_clusters
