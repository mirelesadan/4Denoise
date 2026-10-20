"""Memory-budgeted, selected-axis resampling without full-source conversion.

Output tiles use global coordinates: fractional area bins and interpolation
halos are identical regardless of the chosen tile size. Sources need only
shape, dtype and basic slicing, so the engine also consumes HDF5 datasets.
"""

from math import ceil, prod
from numbers import Integral

import numpy as np
from scipy.ndimage import gaussian_filter

from fourdenoise_geometry import _polar_grid
from fourdenoise_reductions import _working_memory_bytes


def _normalize_resize_shape(shape, ndim, label='shape'):
    """Validate positive integer sizes, including NumPy arrays/scalars."""
    values = (shape,) if np.ndim(shape) == 0 and ndim == 1 else shape
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f'{label} must contain {ndim} positive integer(s).') from exc
    if len(values) != ndim or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
        for v in values
    ):
        raise ValueError(f'{label} must contain {ndim} positive integer(s).')
    return tuple(int(v) for v in values)


def _resize_spec(shape, requested, domain):
    """Return selected axes and full output shape for a 3D/4D dataset."""
    if len(shape) not in (3, 4) or any(size < 1 for size in shape):
        raise ValueError('resize requires a nonempty 3D stack or 4D scan.')
    if not isinstance(domain, str) or domain.lower() not in ('real', 'reciprocal'):
        raise ValueError("domain must be 'real' or 'reciprocal'.")
    domain = domain.lower()
    axes = tuple(range(len(shape) - 2)) if domain == 'real' else (len(shape) - 2, len(shape) - 1)
    selected = _normalize_resize_shape(requested, len(axes))
    output = list(shape)
    for axis, size in zip(axes, selected):
        output[axis] = size
    return axes, tuple(output)


def _resize_plan(shape, dtype, output_shape, axes, method='area', *,
                 working_memory_mb=None, polar_metadata=None):
    """Budget input reads, area intermediates, interpolation halos and tiles.

    Resident source/destination arrays, allocator overhead and HDF5 caches are
    excluded. Estimates conservatively include a second native read buffer,
    three accumulator buffers, output tiles, filter work and coordinate grids.
    """
    shape, output_shape, axes = tuple(shape), tuple(output_shape), tuple(axes)
    if not isinstance(method, str) or method.lower() not in ('area', 'linear', 'nearest'):
        raise ValueError("method must be 'area', 'linear', or 'nearest'.")
    method = method.lower()
    if not shape or len(shape) != len(output_shape) or any(v < 1 for v in shape + output_shape):
        raise ValueError('resize requires nonempty source and output dimensions.')
    if not axes or tuple(sorted(set(axes))) != axes or any(a < 0 or a >= len(shape) for a in axes):
        raise ValueError('Resize axes must be distinct, increasing, valid axis indices.')
    if any(shape[a] != output_shape[a] for a in range(len(shape)) if a not in axes):
        raise ValueError('Only the selected axes may change size.')
    if method == 'area' and any(output_shape[a] > shape[a] for a in axes):
        raise ValueError("method='area' supports downsampling only.")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biufc':
        raise TypeError('resize requires numeric or Boolean data.')
    unchanged = shape == output_shape
    result_dtype = dtype if unchanged or method == 'nearest' or dtype.kind in 'fc' else np.dtype('float64')
    # Interpolating float16 is supported by working in double precision.
    accumulator_dtype = np.dtype(np.result_type(dtype, np.float64))
    periodic = set()
    polar = polar_metadata is not None and axes == (len(shape) - 2, len(shape) - 1)
    if polar:
        _, _, theta, _, _, _ = _polar_grid(polar_metadata, shape[-2:])
        if method != 'area' and np.isclose(theta[1] - theta[0], 360):
            periodic.add(axes[-1])
    scales = {a: shape[a] / output_shape[a] for a in axes}
    sigma = {a: max(0.0, (scales[a] - 1) / 2) if method == 'linear' else 0.0 for a in axes}
    halo = {a: int(ceil(4 * sigma[a])) + 1 for a in axes}

    def estimate(tile):
        input_shape = list(tile)
        for a in axes:
            if shape[a] == output_shape[a]:
                input_shape[a] = tile[a]
            elif method == 'area':
                input_shape[a] = min(shape[a], ceil(tile[a] * scales[a]) + 2)
            else:
                step = ((shape[a] - 1) / (output_shape[a] - 1) if output_shape[a] > 1 else 0) if polar and a == axes[0] else scales[a]
                extent = ceil(max(0, tile[a] - 1) * step) + 2 * halo[a] + 3
                input_shape[a] = extent if a in periodic else min(shape[a], extent)
        inputs, outputs = prod(input_shape), prod(tile)
        if unchanged:
            return 2 * inputs * dtype.itemsize + outputs * result_dtype.itemsize
        coordinates = len(axes) * prod(tile[a] for a in axes) * 8
        filter_work = 3 * prod(input_shape[a] for a in axes) * accumulator_dtype.itemsize
        return (2 * inputs * dtype.itemsize + 3 * inputs * accumulator_dtype.itemsize
                + 2 * outputs * max(result_dtype.itemsize, accumulator_dtype.itemsize)
                + (coordinates + filter_work if method != 'area' else 0))

    budget = _working_memory_bytes(working_memory_mb)
    minimum = estimate((1,) * len(shape))
    if budget < minimum:
        raise ValueError(f'working_memory_mb is too small for resize; need at least {minimum / 1024**2:.6g} MiB ({minimum} bytes).')
    tile = output_shape
    while estimate(tile) > budget:
        candidates = []
        for a, size in enumerate(tile):
            if size > 1:
                candidate = list(tile)
                candidate[a] = (size + 1) // 2
                candidates.append(tuple(candidate))
        tile = min(candidates, key=estimate)
    return {
        'shape': shape, 'output_shape': output_shape, 'axes': axes, 'method': method,
        'tile_shape': tile, 'dtype': dtype, 'result_dtype': result_dtype,
        'accumulator_dtype': accumulator_dtype, 'periodic_axes': periodic,
        'polar': polar, 'sigma': sigma, 'halo': halo,
        'budget_bytes': budget, 'estimated_peak_bytes': estimate(tile),
    }


def _sample_coordinates(plan, axis, start, stop):
    """Pixel-center Cartesian sampling; polar radius preserves endpoints."""
    old, new = plan['shape'][axis], plan['output_shape'][axis]
    positions = np.arange(start, stop, dtype=float)
    if plan['polar']:
        if axis == plan['axes'][0]:
            return positions * ((old - 1) / (new - 1) if new > 1 else 0)
        return positions * old / new
    return (positions + 0.5) * old / new - 0.5


def _read_resize_window(source, windows, periodic_axes):
    """Read basic slices, assembling only local angular wrap-around halos."""
    if not periodic_axes:
        return source[tuple(slice(a, b) for a, b in windows)]
    axis = next(iter(periodic_axes))
    output = np.empty(tuple(b - a for a, b in windows), dtype=source.dtype)
    start, stop = windows[axis]
    cursor = start
    while cursor < stop:
        local_start = cursor % source.shape[axis]
        length = min(stop - cursor, source.shape[axis] - local_start)
        selection = [slice(a, b) for a, b in windows]
        selection[axis] = slice(local_start, local_start + length)
        destination = [slice(None)] * len(windows)
        destination[axis] = slice(cursor - start, cursor - start + length)
        block = source[tuple(selection)]
        output[tuple(destination)] = block
        del block
        cursor += length
    return output


def _area_axis(block, axis, old_size, new_size, output_slice, input_start, dtype):
    """Average globally positioned bins with exact fractional overlap weights."""
    moved = np.moveaxis(block, axis, 0)
    factor, remainder = divmod(old_size, new_size)
    if remainder == 0:
        grouped = moved.reshape((output_slice.stop - output_slice.start, factor) + moved.shape[1:])
        return np.moveaxis(np.mean(grouped, axis=1, dtype=dtype), 0, axis)
    output = np.zeros((output_slice.stop - output_slice.start,) + moved.shape[1:], dtype=dtype)
    scratch = np.empty(moved.shape[1:], dtype=dtype)
    scale = old_size / new_size
    for local, j in enumerate(range(output_slice.start, output_slice.stop)):
        start, stop = j * scale, (j + 1) * scale
        for i in range(max(0, int(np.floor(start))), min(old_size, int(np.ceil(stop)))):
            weight = max(0.0, min(stop, i + 1) - max(start, i)) / scale
            if weight:
                np.multiply(moved[i - input_start], weight, out=scratch, dtype=dtype)
                np.add(output[local], scratch, out=output[local])
    return np.moveaxis(output, 0, axis)


def _interpolate_tile(block, windows, output_slices, plan):
    """Vectorized local interpolation without mixing scan/stack coordinates."""
    axes = tuple(a for a in plan['axes'] if plan['shape'][a] != plan['output_shape'][a])
    vectors = [_sample_coordinates(plan, a, output_slices[a].start, output_slices[a].stop) - windows[a][0] for a in axes]
    tile_shape = tuple(s.stop - s.start for s in output_slices)
    selectors = [np.arange(size) for size in block.shape]
    if plan['method'] == 'nearest':
        for a, vector in zip(axes, vectors):
            selectors[a] = np.clip(np.floor(vector + 0.5).astype(int), 0, block.shape[a] - 1)
        return block[np.ix_(*selectors)]
    working = block.astype(plan['accumulator_dtype'], copy=False)
    sigmas = tuple(plan['sigma'].get(a, 0.0) for a in range(block.ndim))
    modes = tuple('nearest' if plan['polar'] and a == plan['axes'][-1] else 'mirror' for a in range(block.ndim))
    if any(sigmas):
        working = gaussian_filter(working, sigmas, mode=modes)
    lower = [np.floor(vector).astype(int) for vector in vectors]
    fractions = [vector - index for vector, index in zip(vectors, lower)]
    result = np.zeros(tile_shape, dtype=plan['accumulator_dtype'])
    for corner in np.ndindex(*(2,) * len(axes)):
        weights = []
        for a, index, fraction, upper in zip(axes, lower, fractions, corner):
            index = index + upper
            if plan['polar'] or block.shape[a] == 1:
                selectors[a] = np.clip(index, 0, block.shape[a] - 1)
            else:
                period = 2 * (block.shape[a] - 1)
                reflected = index % period
                selectors[a] = np.minimum(reflected, period - reflected)
            weight_shape = [1] * block.ndim
            weight_shape[a] = len(fraction)
            weights.append((fraction if upper else 1 - fraction).reshape(weight_shape))
        term = working[np.ix_(*selectors)]
        for weight in weights:
            np.multiply(term, weight, out=term)
        np.add(result, term, out=result)
        del term
    return result.astype(plan['result_dtype'], copy=False)


def _iter_resized_tiles(source, plan):
    """Yield output slices and results, with all reads bounded by the plan."""
    grid = tuple(ceil(size / length) for size, length in zip(plan['output_shape'], plan['tile_shape']))
    for index in np.ndindex(*grid):
        output_slices = tuple(slice(i * length, min((i + 1) * length, size)) for i, length, size in zip(index, plan['tile_shape'], plan['output_shape']))
        windows = [(s.start, s.stop) for s in output_slices]
        if plan['shape'] != plan['output_shape']:
            for a in plan['axes']:
                if plan['shape'][a] == plan['output_shape'][a]:
                    continue
                selection = output_slices[a]
                if plan['method'] == 'area':
                    scale = plan['shape'][a] / plan['output_shape'][a]
                    start, stop = int(np.floor(selection.start * scale)), int(np.ceil(selection.stop * scale))
                else:
                    coords = _sample_coordinates(plan, a, selection.start, selection.stop)
                    start = int(np.floor(coords[0])) - plan['halo'][a]
                    stop = int(np.ceil(coords[-1])) + plan['halo'][a] + 1
                if a not in plan['periodic_axes']:
                    start, stop = max(0, start), min(plan['shape'][a], stop)
                windows[a] = (start, stop)
        block = _read_resize_window(source, windows, plan['periodic_axes'])
        if plan['shape'] == plan['output_shape']:
            result = np.array(block, copy=True)
        elif plan['method'] == 'area':
            result = block
            for a in plan['axes']:
                if plan['shape'][a] != plan['output_shape'][a]:
                    result = _area_axis(result, a, plan['shape'][a], plan['output_shape'][a], output_slices[a], windows[a][0], plan['accumulator_dtype'])
            result = result.astype(plan['result_dtype'], copy=False)
        else:
            result = _interpolate_tile(block, windows, output_slices, plan)
        yield output_slices, result
        del block, result


def _resize_array(source, output_shape, axes, method='area', *, working_memory_mb=None,
                  polar_metadata=None, out=None, on_tile=None):
    """Resize into RAM or a supplied array/HDF5 destination without mutation.

    Explicit destinations must match planned shape/dtype and cannot overlap a
    NumPy source. The default result owns its storage, including no-op resizes.
    """
    plan = _resize_plan(source.shape, source.dtype, output_shape, axes, method,
                        working_memory_mb=working_memory_mb, polar_metadata=polar_metadata)
    if out is None:
        out = np.empty(plan['output_shape'], dtype=plan['result_dtype'])
    elif tuple(out.shape) != plan['output_shape'] or np.dtype(out.dtype) != plan['result_dtype']:
        raise ValueError('Resize destination shape and dtype must match the planned output.')
    if isinstance(source, np.ndarray) and isinstance(out, np.ndarray) and np.shares_memory(source, out):
        raise ValueError('Resize destination cannot overlap the source.')
    for selection, tile in _iter_resized_tiles(source, plan):
        out[selection] = tile
        if on_tile is not None:
            on_tile(prod(tile.shape))
        del tile
    return out
