"""Budgeted, per-pattern transforms from an HDF5 source to an HDF5 file.

This is not a chunking adapter for global denoisers or scan-axis operations.
The transform must preserve the leading scan axes and act independently on
each diffraction pattern. Callers supply the reader factory and transform;
this module deliberately never imports ``fourdenoise``.
"""

from copy import deepcopy
from math import prod
from numbers import Integral
from pathlib import Path
import os
import tempfile

import h5py
import numpy as np
from tqdm import tqdm

from fourdenoise_geometry import _real_spacing_pair
from fourdenoise_io import (
    _HYPERDATA_HDF5_FORMAT, _HYPERDATA_HDF5_VERSION, _write_hdf5_value,
)
from fourdenoise_reductions import _working_memory_bytes


# Separate ordinary source workspace from angular padding, whose cost scales
# with radial rows, not the full polar image. Twelve columns cover order 0..5.
_SOURCE_WORKSPACE_BYTES_PER_PIXEL = 64
_OUTPUT_WORKSPACE_BYTES_PER_PIXEL = 128
_MAX_ANGULAR_PADDING_COLUMNS = 12
_ANGULAR_PADDING_BYTES_PER_PIXEL = 64
_PROBE_OUTPUT_ITEMSIZE = 32
_METADATA_KEYS = (
    'acquisition_metadata', 'real_units', 'real_conv_factor', 'real_origin',
    'reciprocal_units', 'reciprocal_conv_factor', 'polar_metadata',
    'cepstral_metadata', 'center_beam_metadata',
)


def _pattern_shape(value):
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise ValueError('output_pattern_shape must contain two positive integers.') from exc
    if len(shape) != 2 or any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral)
        or size < 1 for size in shape
    ):
        raise ValueError('output_pattern_shape must contain two positive integers.')
    return tuple(map(int, shape))


def _stream_transform_plan(shape, dtype, output_pattern_shape,
                           working_memory_mb=None, *, output_dtype=None):
    """Bound explicit buffers; this is not a process-wide resident RAM limit.

    Reserve 64 bytes/source pixel for conversions and spline workspace, and
    128 bytes/output pixel for coordinate maps, masks and interpolation
    temporaries. Separately reserve 64 bytes for each of 12 extra angular
    columns per radial row, covering padding and spline buffers even for
    narrow polar patterns. Add two input and two output block buffers at
    their actual dtypes. Before the probe, reserve 32 bytes/output element
    (the largest supported numeric NumPy element). Allocator overhead, HDF5
    caches, caller-owned data and metadata are excluded. Custom transforms
    must stay within this workspace contract and must not retain blocks.
    """
    shape = tuple(shape)
    if len(shape) not in (3, 4) or any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral)
        or size < 1 for size in shape
    ):
        raise ValueError('source must be a nonempty 3D stack or 4D scan.')
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biufc' or dtype.itemsize > _PROBE_OUTPUT_ITEMSIZE:
        raise TypeError('source must contain supported numeric or Boolean data.')
    output_pattern_shape = _pattern_shape(output_pattern_shape)
    output_itemsize = _PROBE_OUTPUT_ITEMSIZE
    if output_dtype is not None:
        output_dtype = np.dtype(output_dtype)
        if output_dtype.kind not in 'biufc' or output_dtype.itemsize > _PROBE_OUTPUT_ITEMSIZE:
            raise TypeError('transform output must contain supported numeric or Boolean data.')
        output_itemsize = output_dtype.itemsize
    budget = _working_memory_bytes(working_memory_mb)
    source_pixels, output_pixels = prod(shape[-2:]), prod(output_pattern_shape)
    source_workspace_bytes = _SOURCE_WORKSPACE_BYTES_PER_PIXEL * source_pixels
    output_workspace_bytes = _OUTPUT_WORKSPACE_BYTES_PER_PIXEL * output_pixels
    angular_padding_bytes = (
        _ANGULAR_PADDING_BYTES_PER_PIXEL * _MAX_ANGULAR_PADDING_COLUMNS * int(shape[-2])
    )
    fixed_bytes = source_workspace_bytes + output_workspace_bytes + angular_padding_bytes
    pattern_bytes = 2 * (source_pixels * dtype.itemsize + output_pixels * output_itemsize)
    minimum_bytes = fixed_bytes + pattern_bytes
    if budget < minimum_bytes:
        raise ValueError(
            'working_memory_mb is too small for one pattern and transform workspace; '
            f'need at least {minimum_bytes / 1024 ** 2:.6g} MiB ({minimum_bytes} bytes).'
        )
    scan_shape = tuple(map(int, shape[:-2]))
    max_patterns = min(prod(scan_shape), (budget - fixed_bytes) // pattern_bytes)
    if len(scan_shape) == 1:
        chunk_shape = (min(scan_shape[0], max_patterns),)
    else:
        width = min(scan_shape[1], max_patterns)
        chunk_shape = (min(scan_shape[0], max_patterns // width), width)
    return {
        'budget_bytes': budget, 'fixed_bytes': fixed_bytes,
        'source_workspace_bytes': source_workspace_bytes,
        'output_workspace_bytes': output_workspace_bytes,
        'angular_padding_bytes': angular_padding_bytes,
        'pattern_bytes': pattern_bytes, 'max_patterns': max_patterns,
        'chunk_shape': chunk_shape,
        'estimated_peak_bytes': fixed_bytes + prod(chunk_shape) * pattern_bytes,
    }


def _remaining_scan_slices(scan_shape, chunk_shape):
    """Cover the scan exactly once, excluding the already-written origin probe."""
    if len(scan_shape) == 1:
        for start in range(1, scan_shape[0], chunk_shape[0]):
            yield (slice(start, min(start + chunk_shape[0], scan_shape[0])),)
        return
    height, width = scan_shape
    rows, columns = chunk_shape
    for start in range(1, width, columns):
        yield slice(0, 1), slice(start, min(start + columns, width))
    for y in range(1, height, rows):
        for x in range(0, width, columns):
            yield slice(y, min(y + rows, height)), slice(x, min(x + columns, width))


def _transformed_block(reader, scan_slices, transform, output_pattern_shape,
                       output_dtype=None):
    """Read one basic slice and construct the same calibrated block as the reader.

    The reader protocol is HyperData.open_hdf5's context-managed reader:
    shape, dtype, _dataset, _metadata and _hyperdata_type. Direct slices avoid
    the iter_chunks generator retaining a previous input during the next read.
    """
    metadata = dict(reader._metadata)
    origin = metadata.get('real_origin', (0.0, 0.0))
    if len(scan_slices) == 2:
        steps = _real_spacing_pair(metadata.get('real_conv_factor'))
        origin = tuple(
            value + selection.start * step
            for value, selection, step in zip(origin, scan_slices, steps)
        )
    metadata['real_origin'] = origin
    metadata.pop('unfold_metadata', None)
    block = reader._hyperdata_type(
        reader._dataset[scan_slices + (slice(None), slice(None))], **metadata,
    )
    transformed = transform(block)
    if not isinstance(transformed, reader._hyperdata_type):
        raise TypeError('transform must return a HyperData object.')
    array = transformed.array
    scan_shape = tuple(selection.stop - selection.start for selection in scan_slices)
    expected_shape = scan_shape + output_pattern_shape
    if not isinstance(array, np.ndarray) or array.shape != expected_shape:
        raise ValueError(
            f'transform must preserve scan shape and return shape {expected_shape}; '
            f'got {getattr(array, "shape", None)}.'
        )
    if array.dtype.kind not in 'biufc' or array.dtype.itemsize > _PROBE_OUTPUT_ITEMSIZE:
        raise TypeError('transform output must contain supported numeric or Boolean data.')
    if output_dtype is not None and array.dtype != output_dtype:
        raise ValueError(
            f'transform output dtype changed from {output_dtype} to {array.dtype}.'
        )
    return transformed


def _write_block(dataset, reader, scan_slices, transform, output_pattern_shape):
    # Keep all block references local so they disappear before the next read.
    transformed = _transformed_block(
        reader, scan_slices, transform, output_pattern_shape, dataset.dtype,
    )
    dataset[scan_slices + (slice(None), slice(None))] = transformed.array


def _stream_transform(source, destination, *, reader_factory, transform,
                      output_pattern_shape, working_memory_mb=None,
                      hdf5_dataset=None, compression='gzip', compression_opts=4,
                      overwrite=False, atomic=True, progress=True):
    """Write an independent diffraction-pattern transform to disk and return its path.

    ``reader_factory`` is ``HyperData.open_hdf5``. ``transform(block)`` must
    return HyperData with unchanged scan axes and the supplied diffraction
    shape, a stable numeric dtype, and no internal progress bar. Global scan
    operations and denoisers are not supported. A one-pattern probe discovers
    dtype and output metadata; it is written once and released before the next
    read. The memory plan is checked BEFORE any probe read or allocation.

    All metadata follows the first transformed block, including the global
    real origin, acquisition, R/Q, reciprocal, polar, cepstral and center-beam
    fields. Partial-block unfolding undo metadata is always cleared.

    The budget follows _stream_transform_plan, not an arbitrary callable's
    total RAM use. Source/output arrays stay on disk; only current scan blocks
    are held. The default budget is _working_memory_bytes(None)'s 256 MiB.
    Atomic output uses a same-directory temporary file and publishes only
    after the input reader and output file close. Failed atomic writes remove
    the temporary file and preserve an existing destination. ``atomic=False``
    explicitly allows an incomplete destination on failure. Existing files
    require ``overwrite=True``; source aliases (including hardlinks) are never
    valid destinations. No eager output object is returned.
    """
    for name, value in (('overwrite', overwrite), ('atomic', atomic), ('progress', progress)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean.')
    if compression is not None and not isinstance(compression, str):
        raise ValueError('compression must be a string or None.')
    if not callable(reader_factory) or not callable(transform):
        raise TypeError('reader_factory and transform must be callable.')
    output_pattern_shape = _pattern_shape(output_pattern_shape)
    _working_memory_bytes(working_memory_mb)
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    if not destination_path.suffix:
        destination_path = destination_path.with_suffix('.4denoise')
    if destination_path.suffix.lower() not in {'.4denoise', '.h5', '.hdf5', '.hdf'}:
        raise ValueError('destination must be a HyperData/HDF5 file path.')
    if source_path.resolve() == destination_path.resolve() or (
        source_path.exists() and destination_path.exists()
        and source_path.samefile(destination_path)
    ):
        raise ValueError('source and destination must be different files.')
    if (destination_path.exists() or destination_path.is_symlink()) and not overwrite:
        raise FileExistsError(f"'{destination_path}' exists; use overwrite=True to replace it.")
    if destination_path.is_dir():
        raise IsADirectoryError(f"'{destination_path}' is a directory.")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory '{destination_path.parent}' does not exist.")

    temporary_path = None
    try:
        with reader_factory(source_path, hdf5_dataset=hdf5_dataset) as reader:
            _stream_transform_plan(
                reader.shape, reader.dtype, output_pattern_shape, working_memory_mb,
            )
            scan_shape = tuple(map(int, reader.shape[:-2]))
            first_slices = (slice(0, 1),) * len(scan_shape)
            first = _transformed_block(reader, first_slices, transform, output_pattern_shape)
            output_dtype = first.array.dtype
            plan = _stream_transform_plan(
                reader.shape, reader.dtype, output_pattern_shape, working_memory_mb,
                output_dtype=output_dtype,
            )
            metadata = {key: deepcopy(getattr(first, key)) for key in _METADATA_KEYS}
            metadata['unfold_metadata'] = None
            metadata['rq_calibration'] = (
                deepcopy(first.rq_calibration.to_dict())
                if first.rq_calibration is not None else None
            )
            if atomic:
                descriptor, name = tempfile.mkstemp(
                    prefix=f'.{destination_path.name}.', suffix='.tmp',
                    dir=destination_path.parent,
                )
                os.close(descriptor)
                temporary_path = Path(name)
            write_path = temporary_path or destination_path
            with h5py.File(write_path, 'w' if atomic or overwrite else 'x') as file:
                full_shape = scan_shape + output_pattern_shape
                file.attrs['fourdenoise_format'] = _HYPERDATA_HDF5_FORMAT
                file.attrs['format_version'] = _HYPERDATA_HDF5_VERSION
                file.attrs['saved_class'] = 'HyperData'
                file.attrs['array_ndim'] = len(full_shape)
                file.attrs['array_dtype'] = str(output_dtype)
                options = {} if compression is None else {'compression': compression}
                if compression is not None and compression_opts is not None:
                    options['compression_opts'] = compression_opts
                dataset = file.create_dataset(
                    'array', shape=full_shape, dtype=output_dtype,
                    chunks=(1,) * len(scan_shape) + output_pattern_shape, **options,
                )
                group = file.create_group('metadata')
                group.attrs['kind'] = 'dict'
                for key, value in metadata.items():
                    _write_hdf5_value(
                        group, key, value, compression=compression,
                        compression_opts=compression_opts,
                    )
                del metadata
                dataset[first_slices + (slice(None), slice(None))] = first.array
                del first
                with tqdm(total=prod(scan_shape), desc='Diffraction patterns',
                          disable=not progress) as bar:
                    bar.update(1)
                    for scan_slices in _remaining_scan_slices(scan_shape, plan['chunk_shape']):
                        _write_block(dataset, reader, scan_slices, transform, output_pattern_shape)
                        bar.update(prod(selection.stop - selection.start for selection in scan_slices))
        if atomic:
            if overwrite:
                os.replace(temporary_path, destination_path)
            else:
                # Link publication fails rather than clobbering a racing writer.
                os.link(temporary_path, destination_path)
        return str(destination_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
