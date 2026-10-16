"""Read-only diffraction sources, independent of in-memory analysis objects.

Backends supply shape, dtype, basic slicing and a resource-closing callback.
Object factories keep this module independent of the main fourdenoise import.
Opening a source never converts its complete array or reads unfolding payloads.
"""

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from math import prod
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from fourdenoise_geometry import (
    _center_beam_metadata_from_pixels, _normalize_real_origin, _real_spacing_pair,
)
from fourdenoise_io import (
    _HYPERDATA_HDF5_FORMAT, _checked_hyperdata_format_version,
    _decode_hdf5_value, _migrate_hyperdata_metadata, _read_hdf5_value,
    _select_hdf5_dataset,
)
from fourdenoise_reductions import (
    _diffraction_shape, _get_dp_array, _working_memory_bytes,
)
from fourdenoise_traversals import _clone_unfold_metadata
from fourdenoise_detectors import _virtual_image


_SOURCE_METADATA_KEYS = (
    'real_units', 'real_conv_factor', 'real_origin', 'reciprocal_units',
    'reciprocal_conv_factor', 'polar_metadata', 'cepstral_metadata',
    'center_beam_metadata', 'rq_calibration', 'acquisition_metadata',
)


def _source_metadata(metadata, overrides, hyperdata_type, pattern_shape):
    """Validate calibrations without constructing an array-backed HyperData."""
    result = deepcopy(metadata)
    for domain in ('real', 'reciprocal'):
        unit_key, scale_key = f'{domain}_units', f'{domain}_conv_factor'
        if overrides.get(unit_key) is not None or overrides.get(scale_key) is not None:
            result[unit_key] = overrides.get(unit_key)
            result[scale_key] = overrides.get(scale_key)
        units, scale = result.get(unit_key), result.get(scale_key)
        if units is not None or scale is not None:
            result[unit_key], result[scale_key] = hyperdata_type._validate_scale(
                units, scale, domain,
            )
    origin = overrides.get('real_origin')
    result['real_origin'] = _normalize_real_origin(
        result.get('real_origin', (0.0, 0.0)) if origin is None else origin,
    )
    if result.get('polar_metadata') is not None and result.get('cepstral_metadata') is not None:
        raise ValueError('Cepstral and polar coordinate metadata cannot be combined.')
    reciprocal_override = (
        overrides.get('reciprocal_units') is not None
        or overrides.get('reciprocal_conv_factor') is not None
    )
    if reciprocal_override:
        if result.get('polar_metadata') is not None or result.get('cepstral_metadata') is not None:
            raise ValueError(
                'Reciprocal calibration overrides require Cartesian diffraction axes; '
                'omit them for saved polar/cepstral data so its coordinate metadata remains valid.'
            )
        beam = result.get('center_beam_metadata')
        if beam is not None:
            if not isinstance(beam, dict) or not {'center_px', 'radius_px'} <= beam.keys():
                raise ValueError('A beam calibration override requires center_px and radius_px metadata.')
            if tuple(beam.get('shape', pattern_shape)) != tuple(pattern_shape):
                raise ValueError('Cannot override calibration with stale beam metadata.')
            beam.update(_center_beam_metadata_from_pixels(
                beam['radius_px'], beam['center_px'], pattern_shape,
                units=result['reciprocal_units'], conv_factor=result['reciprocal_conv_factor'],
            ))
    return result


def _chunk_lengths(shape, chunk_shape, dtype, working_memory_mb):
    """Bound requested scan chunks by two explicit read/copy buffers."""
    n_scan_axes = len(shape) - 2
    if isinstance(chunk_shape, Integral) and not isinstance(chunk_shape, (bool, np.bool_)):
        lengths = (int(chunk_shape),) * n_scan_axes
    else:
        try:
            lengths = tuple(chunk_shape)
        except TypeError as exc:
            raise ValueError('chunk_shape must contain positive integers.') from exc
    if len(lengths) != n_scan_axes or any(
        not isinstance(length, Integral) or isinstance(length, (bool, np.bool_)) or length < 1
        for length in lengths
    ):
        raise ValueError(f'chunk_shape must contain {n_scan_axes} positive integer(s).')
    budget = _working_memory_bytes(working_memory_mb)
    pattern_bytes = prod(shape[-2:]) * np.dtype(dtype).itemsize
    max_patterns = budget // (2 * pattern_bytes)
    if max_patterns < 1:
        raise ValueError(
            'working_memory_mb is too small for one pattern and its copy; '
            f'need at least {2 * pattern_bytes} bytes.'
        )
    lengths = [min(int(length), size) for length, size in zip(lengths, shape[:-2])]
    # Preserve rectangular blocks, reducing the largest requested axis first.
    while prod(lengths) > max_patterns:
        axis = max(range(n_scan_axes), key=lengths.__getitem__)
        other_size = prod(lengths[:axis] + lengths[axis + 1:])
        lengths[axis] = max(1, min(lengths[axis] - 1, max_patterns // other_size))
    return tuple(lengths)


def _source_chunk_slices(shape, lengths):
    """Yield basic scan slices without selecting or reshaping source pixels."""
    scan_shape = shape[:-2]
    grid = tuple((size + length - 1) // length for size, length in zip(scan_shape, lengths))
    for index in np.ndindex(*grid):
        yield tuple(
            slice(i * length, min((i + 1) * length, size))
            for i, length, size in zip(index, lengths, scan_shape)
        )


class DiffractionSource:
    """Read-only, on-demand access to a nonempty 3D stack or 4D scan.

    Obtain a source with ``HyperData.open(...)`` inside a ``with`` block.
    Shapes follow ``(N, Ky, Kx)`` or ``(Ry, Rx, Ky, Kx)``. ``get_dp`` and
    ``iter_chunks`` return independent in-memory objects; ``load`` explicitly
    loads the complete dataset. There is deliberately no public ``array``
    property or implicit NumPy conversion. This is disk-backed reading, not
    a promise that all analysis or denoising methods execute lazily.

    ``metadata`` returns a defensive copy of descriptive/calibration records.
    Large unfolding undo payloads are deferred until ``load`` and are not
    attached to partial chunks. ``has_unfold_metadata`` reports their presence
    without reading them. Generic HDF5 and NumPy files have no assumed
    calibration; provide calibration explicitly when opening them.

    The source owns its file handle or memory map. Closing it is idempotent,
    prevents further reads, and does not invalidate already returned objects.
    Shape, dtype and copied metadata remain inspectable after closing.
    """

    def __init__(self, dataset, hyperdata_type, reciprocal_type, *, metadata=None,
                 backend, filename, dataset_path=None, close_callback,
                 unfold_loader=None, real_type=None):
        self._shape = _diffraction_shape(dataset)
        self._dtype = np.dtype(dataset.dtype)
        self._dataset = dataset
        self._hyperdata_type = hyperdata_type
        self._reciprocal_type = reciprocal_type
        self._real_type = real_type
        self._metadata = deepcopy(metadata or {})
        self._backend = backend
        self._filename = str(Path(filename).resolve())
        self._dataset_path = dataset_path
        self._close_callback = close_callback
        self._unfold_loader = unfold_loader
        self._has_unfold_metadata = unfold_loader is not None
        self._closed = False

    @property
    def shape(self):
        """Complete stored shape, available without reading pixels."""
        return self._shape

    @property
    def ndim(self):
        """Number of array axes: three for a stack, four for a scan."""
        return len(self.shape)

    @property
    def dtype(self):
        """Stored NumPy dtype, including its byte order."""
        return self._dtype

    @property
    def nbytes(self):
        """Uncompressed numerical-array size, excluding metadata and overhead."""
        return prod(self.shape) * self.dtype.itemsize

    @property
    def scan_shape(self):
        """Leading scan axes, without inventing a 2D layout for 3D stacks."""
        return self.shape[:-2]

    @property
    def pattern_shape(self):
        """Detector axes ``(Ky, Kx)``."""
        return self.shape[-2:]

    @property
    def real_shape(self):
        """2D scan shape for 4D data; None for a 3D stack."""
        return self.scan_shape if self.ndim == 4 else None

    @property
    def backend(self):
        """Storage backend: ``'hdf5'``, ``'numpy'`` or ``'rosetta'``."""
        return self._backend

    @property
    def filename(self):
        """Absolute source-file path."""
        return self._filename

    @property
    def dataset_path(self):
        """HDF5 path, Rosetta result index label, or None for NumPy files."""
        return self._dataset_path

    @property
    def reader_chunk_nbytes(self):
        """Largest declared lazy reader chunk; zero for native direct slicing.

        RosettaSciIO may read a complete Dask chunk even for a small selection.
        Its size is reserved separately from 4Denoise's read/reduction buffers.
        Task graphs, compression/library workspaces and OS caches are not
        covered by this estimate or by a process-RAM guarantee.
        """
        return int(getattr(self._dataset, 'reader_chunk_nbytes', 0))

    @property
    def reader_metadata(self):
        """Independent decoded Rosetta axes/metadata, or None for other records."""
        from fourdenoise_rosetta import unpack_reader_metadata
        return unpack_reader_metadata(self._metadata.get('acquisition_metadata') or {})

    @property
    def metadata(self):
        """Independent metadata copy, excluding deferred unfolding payloads."""
        return deepcopy(self._metadata)

    @property
    def real_units(self):
        """Stored or explicitly supplied scan-axis units."""
        return self._metadata.get('real_units')

    @property
    def real_conv_factor(self):
        """Scan spacing, either scalar or signed ``(y, x)`` units per pixel."""
        return self._metadata.get('real_conv_factor')

    @property
    def real_origin(self):
        """Calibrated scan coordinate at pixel ``(0, 0)``."""
        return self._metadata.get('real_origin', (0.0, 0.0))

    @property
    def reciprocal_units(self):
        """Stored or explicitly supplied detector-axis units."""
        return self._metadata.get('reciprocal_units')

    @property
    def reciprocal_conv_factor(self):
        """Positive detector spacing in units per pixel."""
        return self._metadata.get('reciprocal_conv_factor')

    @property
    def has_unfold_metadata(self):
        """Whether a full load can restore saved unfolding undo metadata."""
        return self._has_unfold_metadata

    @property
    def closed(self):
        """Whether the owned disk resource has been closed."""
        return self._closed

    def _require_open(self):
        if self.closed:
            raise RuntimeError('The diffraction source is closed; use it inside the with block.')

    def _reader_working_memory(self, working_memory_mb):
        """Reserve one serial lazy IO chunk before budgeting managed arrays."""
        if not self.reader_chunk_nbytes:
            return working_memory_mb
        budget = _working_memory_bytes(working_memory_mb)
        remaining = budget - self.reader_chunk_nbytes
        if remaining < 1:
            raise ValueError(
                f'working_memory_mb cannot hold the reader chunk ({self.reader_chunk_nbytes} bytes). '
                'Increase the budget or use smaller reader_kwargs chunks if the plugin supports them.'
            )
        return remaining / 1024 ** 2

    def __array__(self, dtype=None, copy=None):
        """Reject implicit materialization; call load explicitly instead."""
        raise TypeError('A DiffractionSource cannot be converted implicitly; call source.load() explicitly.')

    def close(self):
        """Close the owned resource once; returned data remains independent."""
        if not self.closed:
            self._closed = True
            self._unfold_loader = None
            self._close_callback()

    def get_dp(self, y=None, x=None, mask=None, operation=None,
               selection_units='pixels', *, working_memory_mb=None):
        """Read a pattern or perform a bounded scan reduction on disk.

        Accepts the same pixel/calibrated scan selections and masks as
        ``HyperData.get_dp``. Supported reductions are mean, std, min and max;
        random selection is supported too. Exact median is rejected because
        it is not bounded-memory. ``working_memory_mb`` budgets reduction
        arrays, not OS caches or process RAM. The returned ReciprocalSpace
        owns its data, so edits never change the source file.
        """
        self._require_open()
        effective_memory = self._reader_working_memory(working_memory_mb)
        selected_operation = y if isinstance(y, str) and x is None and operation is None else operation
        if selected_operation == 'random':
            effective_memory = working_memory_mb
        pattern = _get_dp_array(
            self._dataset, y, x, mask, operation, selection_units=selection_units,
            real_units=self.real_units, real_conv_factor=self.real_conv_factor,
            real_origin=self.real_origin, working_memory_mb=effective_memory,
            allow_median=False,
        )
        return self._reciprocal_type(
            np.array(pattern, copy=True), units=self.reciprocal_units,
            conv_factor=self.reciprocal_conv_factor,
            polar_metadata=self._metadata.get('polar_metadata'),
            cepstral_metadata=self._metadata.get('cepstral_metadata'),
            center_beam_metadata=self._metadata.get('center_beam_metadata'),
        )

    def virtual_image(self, annulus=None, *, radius=None, centers=None,
                      mask=None, theta_range=None, detector_units='auto',
                      axis_units='auto', vmin=None, vmax=None, grid=True,
                      num_div=10, plot_mask=False, grid_color='black', axes=True,
                      return_detector=False, title='Virtual Detector Image',
                      units='auto', mask_cmap='turbo', mask_vmin=None,
                      mask_vmax=None, mask_power=1, mask_log_scale=True,
                      mask_show_kwargs=None, ring_color='green', ring_alpha=0.3,
                      show=True, working_memory_mb=None):
        """Integrate a virtual detector without loading the complete 4D scan.

        Detector definitions, inclusive boundaries, units and display options
        are the same as :meth:`fourdenoise.HyperData.virtual_image`: specify exactly one
        of ``annulus=(inner, outer)``, ``radius`` with ``centers``, or a real
        Boolean/signed/fractional detector ``mask``. Overlapping disks count
        once. Polar data supports annuli, wrapped ``theta_range`` or masks,
        but not Cartesian disks. Radii in polar pixel mode refer to original
        Cartesian-pixel distances, not row indices.

        ``detector_units='auto'`` uses reciprocal calibration if available;
        choose ``'pixels'`` explicitly for pixel radii/positions. ``axis_units``
        chooses real-space display units, while ``units`` chooses detector
        preview units. All other visualization options follow HyperData's
        method, including the optional unmasked mean-pattern overlay.

        Return a fully owned :class:`fourdenoise.RealSpace` image with scan calibration,
        spacing and origin. It remains usable after the source closes. With
        ``return_detector=True``, also return the masked mean detector image
        (ReciprocalSpace, or polar HyperData). Mean patterns are accumulated
        in the same read pass only when requested by this option or plot_mask.

        ``working_memory_mb`` budgets numerical input/work buffers (default
        256 MiB), reserving one declared Rosetta reader chunk too. It excludes
        the returned scan-sized image, detector construction, metadata,
        library/OS caches and plotting; it is not a process-RAM limit. A budget
        too small for one pattern raises before pixel reads. Products and sums
        use at least float64 precision; source data/dtype is never modified.
        NaN/Inf propagate according to ordinary NumPy multiplication/sums.
        Complex amplitudes are rejected; supply real-valued intensities.
        3D stacks are rejected because they have no defined 2D scan grid.

        Examples
        --------
        >>> with HyperData.open('scan.hdf5') as source:
        ...     image = source.virtual_image(
        ...         annulus=(10, 35), detector_units='pixels',
        ...         working_memory_mb=64, show=False,
        ...     )
        >>> image.show()
        """
        self._require_open()
        if self._real_type is None:
            raise RuntimeError('Obtain virtual-image sources through HyperData.open().')
        geometry = {
            key: self._metadata.get(key) for key in (
                'reciprocal_units', 'reciprocal_conv_factor', 'polar_metadata',
            )
        }
        return _virtual_image(
            self._dataset, annulus, radius=radius, centers=centers, mask=mask,
            theta_range=theta_range, detector_units=detector_units,
            axis_units=axis_units, vmin=vmin, vmax=vmax, grid=grid,
            num_div=num_div, plot_mask=plot_mask, grid_color=grid_color,
            axes=axes, return_detector=return_detector, title=title, units=units,
            mask_cmap=mask_cmap, mask_vmin=mask_vmin, mask_vmax=mask_vmax,
            mask_power=mask_power, mask_log_scale=mask_log_scale,
            mask_show_kwargs=mask_show_kwargs, ring_color=ring_color,
            ring_alpha=ring_alpha, show=show,
            working_memory_mb=self._reader_working_memory(working_memory_mb),
            geometry=geometry,
            real_factory=partial(
                self._real_type, units=self.real_units,
                conv_factor=self.real_conv_factor, origin=self.real_origin,
                quantity='Virtual detector signal',
            ),
            reciprocal_factory=partial(
                self._reciprocal_type, units=self.reciprocal_units,
                conv_factor=self.reciprocal_conv_factor,
                polar_metadata=self._metadata.get('polar_metadata'),
                cepstral_metadata=self._metadata.get('cepstral_metadata'),
                center_beam_metadata=self._metadata.get('center_beam_metadata'),
            ),
            polar_factory=partial(
                self._hyperdata_type,
                polar_metadata=self._metadata.get('polar_metadata'),
            ),
        )

    def _make_block(self, array, scan_slices):
        metadata = self.metadata
        if self.ndim == 4:
            spacing = _real_spacing_pair(self.real_conv_factor)
            metadata['real_origin'] = tuple(
                origin + selection.start * step
                for origin, selection, step in zip(self.real_origin, scan_slices, spacing)
            )
        return self._hyperdata_type(array, **metadata)

    def iter_chunks(self, chunk_shape=16, *, working_memory_mb=None):
        """Yield ``(scan_slices, HyperData)`` blocks with independent storage.

        ``chunk_shape`` is a maximum number of scan positions per axis:
        one positive integer, a one-tuple for 3D, or a two-tuple for 4D.
        ``working_memory_mb`` defaults to 256 MiB and may reduce those sizes.
        It budgets two numerical read/copy buffers, excluding metadata, OS/
        HDF5 caches and chunks retained by the caller. Do not collect all
        chunks into a list when the full dataset is too large for RAM.

        Detector geometry is preserved and 4D scan origins are offset for
        each block. Unfolding undo metadata is deliberately absent from
        partial blocks. Iterator reads fail clearly after source closure.
        """
        self._require_open()
        lengths = _chunk_lengths(self.shape, chunk_shape, self.dtype,
                                 self._reader_working_memory(working_memory_mb))
        for scan_slices in _source_chunk_slices(self.shape, lengths):
            self._require_open()
            selection = scan_slices + (slice(None), slice(None))
            array = np.array(self._dataset[selection], copy=True)
            block = self._make_block(array, scan_slices)
            yield scan_slices, block
            del block, array

    def load(self, *, max_memory_mb=None, working_memory_mb=None):
        """Explicitly load all pixels and saved metadata into HyperData.

        The returned object owns independent, writable storage and remains
        usable after closing this source. Dtype, calibration, acquisition,
        R/Q, beam and transformed-coordinate metadata are preserved. Saved
        unfolding undo data is read only here and restored on the full object.

        ``max_memory_mb`` optionally caps the full numerical output allocation;
        an insufficient cap raises MemoryError before any pixel reads.
        None permits that allocation, which may require many GiB.
        ``working_memory_mb`` separately bounds temporary input blocks
        (default 256 MiB). Neither limit includes metadata/undo payloads,
        library caches, allocator overhead, or other caller-owned arrays.
        """
        self._require_open()
        if max_memory_mb is not None:
            try:
                limit = _working_memory_bytes(max_memory_mb)
            except ValueError as exc:
                raise ValueError(str(exc).replace('working_memory_mb', 'max_memory_mb')) from exc
            if self.nbytes > limit:
                raise MemoryError(
                    f'Loading {self.shape} ({self.dtype}) requires {self.nbytes} bytes '
                    f'for pixels, exceeding max_memory_mb={max_memory_mb}. '
                    'Use get_dp/iter_chunks or explicitly increase the limit.'
                )
        lengths = _chunk_lengths(self.shape, self.scan_shape, self.dtype,
                                 self._reader_working_memory(working_memory_mb))
        array = np.empty(self.shape, dtype=self.dtype)
        for scan_slices in _source_chunk_slices(self.shape, lengths):
            self._require_open()
            selection = scan_slices + (slice(None), slice(None))
            array[selection] = self._dataset[selection]
        result = self._hyperdata_type(array, **self.metadata)
        if self._unfold_loader is not None:
            result.unfold_metadata = _clone_unfold_metadata(
                self._unfold_loader(), share_payload=True,
            )
        return result


@contextmanager
def _open_diffraction_source(filename, hyperdata_type, reciprocal_type, *, real_type=None,
                             hdf5_dataset=None, real_units=None,
                             real_conv_factor=None, reciprocal_units=None,
                             reciprocal_conv_factor=None, real_origin=None,
                             backend='auto', reader=None, dataset=None,
                             axis_order=None, reader_kwargs=None):
    """Select a backend and own its resource for the source's entire lifetime."""
    path = Path(filename).expanduser()
    suffix = path.suffix.lower()
    if not isinstance(backend, str) or backend not in ('auto', 'native', 'rosetta'):
        raise ValueError("backend must be 'auto', 'native' or 'rosetta'.")
    native_suffixes = {'.npy', '.h5', '.hdf5', '.hdf', '.4denoise'}
    rosetta_requested = reader is not None or backend == 'rosetta' or (
        backend == 'auto' and suffix not in native_suffixes
    )
    if backend == 'native' and (rosetta_requested or suffix not in native_suffixes):
        raise ValueError('Native disk-backed open supports .npy, .h5, .hdf5, .hdf and .4denoise files.')
    if not rosetta_requested and any(item is not None for item in (dataset, axis_order, reader_kwargs)):
        raise ValueError('dataset, axis_order and reader_kwargs require a RosettaSciIO reader.')
    if not path.is_file():
        raise FileNotFoundError(f"'{path}' does not exist or is not a file.")
    if rosetta_requested and hdf5_dataset is not None:
        raise ValueError('For RosettaSciIO use dataset or reader_kwargs, not hdf5_dataset.')
    if suffix == '.npy' and hdf5_dataset is not None:
        raise ValueError('hdf5_dataset applies only to HDF5 files, not .npy files.')
    overrides = {
        'real_units': real_units, 'real_conv_factor': real_conv_factor,
        'reciprocal_units': reciprocal_units,
        'reciprocal_conv_factor': reciprocal_conv_factor, 'real_origin': real_origin,
    }
    source = None
    if rosetta_requested:
        from fourdenoise_rosetta import _open_rosetta_data
        with _open_rosetta_data(
            path, reader=reader, dataset=dataset, axis_order=axis_order,
            reader_kwargs=reader_kwargs,
        ) as (array, metadata, label, close_callback):
            try:
                metadata = _source_metadata(metadata, overrides, hyperdata_type, array.shape[-2:])
                source = DiffractionSource(
                    array, hyperdata_type, reciprocal_type, metadata=metadata,
                    backend='rosetta', filename=path, dataset_path=label,
                    close_callback=close_callback, real_type=real_type,
                )
                yield source
            finally:
                if source is not None:
                    source.close()
    elif suffix == '.npy':
        dataset = np.load(path, mmap_mode='r', allow_pickle=False)
        if not isinstance(dataset, np.memmap):
            if hasattr(dataset, 'close'):
                dataset.close()
            raise ValueError('Expected one memory-mappable .npy array, not a NumPy archive.')
        close_callback = dataset._mmap.close
        try:
            metadata = _source_metadata({}, overrides, hyperdata_type, dataset.shape[-2:])
            source = DiffractionSource(
                dataset, hyperdata_type, reciprocal_type, metadata=metadata,
                backend='numpy', filename=path, close_callback=close_callback,
                real_type=real_type,
            )
            yield source
        finally:
            source.close() if source is not None else close_callback()
    else:
        file = h5py.File(path, 'r')
        try:
            metadata, unfold_loader = {}, None
            file_format = _decode_hdf5_value(file.attrs.get('fourdenoise_format'))
            if file_format == _HYPERDATA_HDF5_FORMAT:
                version = _checked_hyperdata_format_version(file, path)
                if hdf5_dataset is not None and str(hdf5_dataset).strip('/\\') != 'array':
                    raise ValueError(
                        "Saved HyperData files contain their data at '/array'; "
                        "omit hdf5_dataset or use '/array'."
                    )
                if 'array' not in file or not isinstance(file['array'], h5py.Dataset):
                    raise ValueError(f"'{path}' does not contain a saved data array.")
                dataset = file['array']
                if 'metadata' in file:
                    group = file['metadata']
                    metadata = {
                        key: _read_hdf5_value(group, key)
                        for key in _SOURCE_METADATA_KEYS if key in group
                    }
                    if 'unfold_metadata' in group and _decode_hdf5_value(
                        group['unfold_metadata'].attrs.get('kind')
                    ) != 'none':
                        def unfold_loader():
                            saved = {'unfold_metadata': _read_hdf5_value(group, 'unfold_metadata')}
                            return _migrate_hyperdata_metadata(saved, version)['unfold_metadata']
                metadata = _migrate_hyperdata_metadata(metadata, version)
            elif suffix == '.4denoise':
                raise ValueError(f"'{path}' is not a valid 4Denoise HyperData save file.")
            else:
                dataset = _select_hdf5_dataset(file, hdf5_dataset)
            metadata = _source_metadata(metadata, overrides, hyperdata_type, dataset.shape[-2:])
            source = DiffractionSource(
                dataset, hyperdata_type, reciprocal_type, metadata=metadata,
                backend='hdf5', filename=path, dataset_path=dataset.name,
                close_callback=file.close, unfold_loader=unfold_loader,
                real_type=real_type,
            )
            yield source
        finally:
            source.close() if source is not None else file.close()
