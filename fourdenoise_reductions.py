"""Bounded scan-block iteration and numerical diffraction reductions.

These helpers consume NumPy arrays, memory maps, or sliceable HDF5 datasets.
They never convert or flatten the complete source. They do not attach metadata;
the public data objects remain responsible for constructing calibrated results.
"""

from numbers import Real

import numpy as np

from fourdenoise_geometry import (
    _normalize_unit_mode, _parse_real_selection, _real_spacing_pair, _resolve_unit_mode,
)


class _ScanRegion:
    """A basic-slice source adapter; constructing an HDF5 region reads nothing."""

    def __init__(self, source, ranges):
        self.source, self.ranges = source, ranges
        self.shape = tuple(b - a for a, b in ranges) + tuple(source.shape[-2:])
        self.dtype = source.dtype

    def __getitem__(self, selection):
        mapped = list(selection)
        for axis, (start, _) in enumerate(self.ranges):
            s = mapped[axis]
            mapped[axis] = slice(start + s.start, start + s.stop, s.step)
        return self.source[tuple(mapped)]


def _get_dp_array(source, y=None, x=None, mask=None, operation=None, *,
                  selection_units='pixels', real_units=None, real_conv_factor=None,
                  real_origin=(0.0, 0.0), working_memory_mb=None, allow_median=True):
    """Shared scan selection/reduction coordinator for RAM and HDF5 readers."""
    shape = _diffraction_shape(source)
    ndim = len(shape)
    operations = ('mean', 'std', 'min', 'max', 'median', 'random')
    if isinstance(y, str) and x is None and operation is None:
        operation, y = y, None
    operation = 'mean' if operation is None else operation
    if not isinstance(operation, str) or operation not in operations:
        raise ValueError(f'operation must be one of {operations}.')
    if operation == 'median' and not allow_median:
        raise ValueError('Exact median is not bounded-memory; load an explicit selection into HyperData first.')
    if working_memory_mb is not None:
        _working_memory_bytes(working_memory_mb)
        if operation in ('median', 'random'):
            raise ValueError(f'working_memory_mb is unsupported for {operation}; omit it or use mean, std, min, or max.')
    mode = _normalize_unit_mode(selection_units, label='selection_units')
    factor = None
    if ndim == 4:
        _, factor, mode = _resolve_unit_mode(selection_units, real_units, real_conv_factor, label='selection_units')
    elif mode == 'calibrated':
        raise ValueError("selection_units='calibrated' is only supported for 4D data.")
    if mask is not None:
        if ndim != 4:
            raise ValueError('mask requires a 4D dataset (Ry, Rx, Ky, Kx).')
        mask = _validate_scan_mask(shape[:-2], mask)
    ranges = tuple((0, size) for size in shape[:-2])
    selected_source = source
    if mask is None and (y is not None or x is not None):
        if ndim == 3:
            index = y if x is None else x if y is None else None
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                raise ValueError('A 3D stack selection requires 1 scan index: one integer index as y or x.')
            return source[int(index), :, :]
        steps = _real_spacing_pair(factor)
        y0, y1, y_kind = _parse_real_selection(y, shape[0], 'y', mode, steps[0], real_origin[0])
        x0, x1, x_kind = _parse_real_selection(x, shape[1], 'x', mode, steps[1], real_origin[1])
        if y_kind == x_kind == 'index':
            return source[y0, x0, :, :]
        ranges = ((y0, y1), (x0, x1))
        selected_source = _ScanRegion(source, ranges)
    if operation == 'random':
        if mask is not None:
            ys, xs = np.nonzero(mask)
            selected = np.random.randint(ys.size)
            indices = (int(ys[selected]), int(xs[selected]))
        else:
            indices = tuple(int(np.random.randint(a, b)) for a, b in ranges)
        print(f'Acquired diffraction pattern at position {indices}...\n')
        return source[indices + (slice(None), slice(None))]
    if operation == 'median':
        selected = source[mask] if mask is not None else source[tuple(slice(a, b) for a, b in ranges) + (slice(None), slice(None))]
        return np.median(selected, axis=tuple(range(selected.ndim - 2)))
    return _reduce_diffraction(selected_source, operation, mask=mask, working_memory_mb=working_memory_mb)


_DEFAULT_WORKING_MEMORY_MB = 256.0
_STREAMING_REDUCTIONS = frozenset(('mean', 'std', 'min', 'max'))


def _working_memory_bytes(working_memory_mb):
    """Resolve a positive MiB budget; None selects the automatic default."""
    value = _DEFAULT_WORKING_MEMORY_MB if working_memory_mb is None else working_memory_mb
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('working_memory_mb must be a positive finite number or None.')
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('working_memory_mb must be a positive finite number or None.') from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError('working_memory_mb must be a positive finite number or None.')
    byte_count = value * 1024 ** 2
    if not np.isfinite(byte_count) or byte_count < 1:
        raise ValueError('working_memory_mb must represent at least one byte and be finite.')
    return int(byte_count)


def _diffraction_shape(source):
    """Validate a numeric stack/scan without materializing its data."""
    shape = tuple(source.shape)
    if len(shape) not in (3, 4) or any(size < 1 for size in shape):
        raise ValueError('Diffraction reduction requires a nonempty 3D stack or 4D scan.')
    if np.dtype(source.dtype).kind not in 'biufc':
        raise TypeError('Diffraction reduction requires numeric or Boolean data.')
    return shape


def _validate_scan_mask(scan_shape, mask):
    """Accept only a Boolean mask with the exact leading-axis shape."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype.kind != 'b':
        raise TypeError('mask must be Boolean; integer masks are not scan selections.')
    if mask.shape != tuple(scan_shape):
        raise ValueError(f'mask must have shape {tuple(scan_shape)}, not {mask.shape}.')
    if not np.any(mask):
        raise ValueError('mask selects no diffraction patterns (no True pixels).')
    return mask


def _diffraction_reduction_plan(shape, dtype, operation, working_memory_mb=None):
    """Account conservatively for explicit arrays and choose a block limit.

    Include two input buffers (a disk-read buffer plus a possible masked copy),
    the variance workspace, and detector-sized accumulator/merge/output arrays.
    The source, caller's mask, allocator overhead and library caches are outside
    this working-memory budget; it is not a process-RAM limit.
    """
    if operation not in _STREAMING_REDUCTIONS:
        raise ValueError('Bounded diffraction reduction supports mean, std, min, and max.')
    if len(shape) not in (3, 4) or any(size < 1 for size in shape):
        raise ValueError('Diffraction reduction requires a nonempty 3D stack or 4D scan.')
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biufc':
        raise TypeError('Diffraction reduction requires numeric or Boolean data.')
    budget = _working_memory_bytes(working_memory_mb)
    pixels = int(shape[-2]) * int(shape[-1])
    accumulator_dtype = np.dtype(np.result_type(dtype, np.float64))
    real_dtype = np.empty((), dtype=accumulator_dtype).real.dtype
    if operation in ('min', 'max'):
        result_dtype = dtype
        fixed_bytes = 3 * pixels * dtype.itemsize
    elif operation == 'mean':
        result_dtype = dtype if dtype.kind in 'fc' else np.dtype('float64')
        fixed_bytes = pixels * (4 * accumulator_dtype.itemsize + result_dtype.itemsize)
    else:
        result_dtype = np.empty((), dtype=dtype).real.dtype if dtype.kind in 'fc' else np.dtype('float64')
        fixed_bytes = pixels * (12 * accumulator_dtype.itemsize + 4 * real_dtype.itemsize + result_dtype.itemsize)
    pattern_bytes = 2 * pixels * dtype.itemsize
    if operation == 'std':
        pattern_bytes += pixels * accumulator_dtype.itemsize
        if dtype.kind == 'c':
            pattern_bytes += pixels * real_dtype.itemsize
    minimum_bytes = fixed_bytes + pattern_bytes
    if budget < minimum_bytes:
        raise ValueError(
            f'working_memory_mb is too small for {operation} on {tuple(shape[-2:])}; '
            f'need at least {minimum_bytes / 1024 ** 2:.6g} MiB '
            f'({minimum_bytes} bytes) for one pattern and its accumulators.'
        )
    max_patterns = min(int(np.prod(shape[:-2])), (budget - fixed_bytes) // pattern_bytes)
    return {
        'max_patterns': max_patterns,
        'accumulator_dtype': accumulator_dtype,
        'real_dtype': real_dtype,
        'result_dtype': result_dtype,
        'budget_bytes': budget,
        'fixed_bytes': fixed_bytes,
        'pattern_bytes': pattern_bytes,
        'estimated_peak_bytes': fixed_bytes + max_patterns * pattern_bytes,
    }


def _scan_block_slices(shape, max_patterns):
    """Yield rectangular basic scan slices with at most max_patterns entries."""
    if isinstance(max_patterns, (bool, np.bool_)) or not isinstance(max_patterns, (int, np.integer)) or max_patterns < 1:
        raise ValueError('max_patterns must be a positive integer.')
    scan_shape = shape[:-2]
    if len(scan_shape) == 1:
        lengths = (min(scan_shape[0], max_patterns),)
    else:
        width = min(scan_shape[1], max_patterns)
        lengths = (min(scan_shape[0], max_patterns // width), width)
    grid = tuple((size + length - 1) // length for size, length in zip(scan_shape, lengths))
    for index in np.ndindex(*grid):
        yield tuple(
            slice(i * length, min((i + 1) * length, size))
            for i, length, size in zip(index, lengths, scan_shape)
        )


def _iter_diffraction_blocks(source, max_patterns, mask=None):
    """Yield (block, selected_count), never gathering the full selected stack.

    Basic rectangular slices work even for non-contiguous arrays. An optional
    Boolean selection copies at most the current block. Empty masked blocks
    are skipped before reading the source, which matters for disk-backed data.
    ``mask`` must already be validated with :func:`_validate_scan_mask`.
    """
    shape = _diffraction_shape(source)
    for slices in _scan_block_slices(shape, max_patterns):
        block_count = int(np.prod([s.stop - s.start for s in slices]))
        local_mask = None if mask is None else mask[slices]
        count = block_count if local_mask is None else int(np.count_nonzero(local_mask))
        if count == 0:
            continue
        block = source[slices + (slice(None), slice(None))]
        if local_mask is not None and count != block_count:
            block = block[local_mask]
        yield block, count
        del block


def _reduce_diffraction(source, operation='mean', *, mask=None, working_memory_mb=None):
    """Reduce leading scan axes using stable, memory-budgeted block statistics.

    Floating/complex input retains NumPy's result dtype, integer means/stds
    return float64, and extrema retain the source dtype. Mean and variance
    accumulation use at least float64 precision. Population standard deviation
    (ddof=0) uses block-centered deviations and Chan's variance merge, never
    the cancellation-prone E[x**2] - E[x]**2 formula. NaN/Inf propagation is
    retained rather than silently applying nan-aware reductions.
    """
    shape = _diffraction_shape(source)
    mask = _validate_scan_mask(shape[:-2], mask)
    plan = _diffraction_reduction_plan(shape, source.dtype, operation, working_memory_mb)
    count = 0
    aggregate = None
    moment = None
    for block, batch_count in _iter_diffraction_blocks(source, plan['max_patterns'], mask):
        axes = tuple(range(block.ndim - 2))
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            if operation in ('min', 'max'):
                batch = np.min(block, axis=axes) if operation == 'min' else np.max(block, axis=axes)
                if aggregate is None:
                    aggregate = batch
                else:
                    combine = np.minimum if operation == 'min' else np.maximum
                    combine(aggregate, batch, out=aggregate)
                del batch
            elif operation == 'mean':
                batch = np.sum(block, axis=axes, dtype=plan['accumulator_dtype'])
                if aggregate is None:
                    aggregate = batch
                else:
                    np.add(aggregate, batch, out=aggregate)
                del batch
            else:
                batch_mean = np.mean(block, axis=axes, dtype=plan['accumulator_dtype'])
                residual = np.empty(block.shape, dtype=plan['accumulator_dtype'])
                np.subtract(block, batch_mean, out=residual)
                if np.iscomplexobj(residual):
                    power = np.empty(block.shape, dtype=plan['real_dtype'])
                    np.absolute(residual, out=power)
                    np.square(power, out=power)
                    batch_moment = np.sum(power, axis=axes, dtype=plan['real_dtype'])
                    del power
                else:
                    np.square(residual, out=residual)
                    batch_moment = np.sum(residual, axis=axes, dtype=plan['real_dtype'])
                del residual
                if aggregate is None:
                    aggregate, moment = batch_mean, batch_moment
                else:
                    total = count + batch_count
                    delta = batch_mean - aggregate
                    aggregate += delta * (batch_count / total)
                    moment += batch_moment + np.abs(delta) ** 2 * (count * batch_count / total)
                    del delta
                del batch_mean, batch_moment
        count += batch_count
        del block
    if count == 0:
        raise ValueError('The selection contains no diffraction patterns.')
    if operation == 'mean':
        aggregate /= count
    elif operation == 'std':
        moment /= count
        np.sqrt(moment, out=moment)
        aggregate = moment
    return aggregate.astype(plan['result_dtype'], copy=False)
