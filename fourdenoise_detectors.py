"""Shared detector definitions, bounded projection and virtual-image rendering.

Numerical helpers accept basic-slice sources, not HyperData instances. Injected
factories preserve the public result types without importing the main module.
"""

from math import prod

import numpy as np
from matplotlib.colors import to_rgba

from fourdenoise_geometry import _normalize_unit_mode, _polar_grid, _resolve_unit_mode
from fourdenoise_reductions import (
    _diffraction_shape, _scan_block_slices, _working_memory_bytes,
)


def _detector_scale(mode, geometry):
    """Resolve radii against the original Cartesian calibration for polar data."""
    polar = geometry.get('polar_metadata')
    if polar is None:
        return _resolve_unit_mode(
            mode, geometry.get('reciprocal_units'),
            geometry.get('reciprocal_conv_factor'), label='detector_units',
        )
    return _resolve_unit_mode(
        mode, polar.get('cartesian_reciprocal_units'),
        polar.get('cartesian_reciprocal_conv_factor'), label='detector_units',
    )


def _build_detector_weights(shape, annulus=None, *, radius=None, centers=None,
                            mask=None, theta_range=None, detector_units='auto',
                            geometry=None):
    """Build one real detector weight image; overlaps of disks are a union.

    Cartesian boundaries include both radii. Polar radii are sampled original
    Cartesian-pixel distances, and angular intervals include both endpoints.
    Signed/fractional masks are used unchanged, without area normalization.
    """
    geometry = geometry or {}
    polar = geometry.get('polar_metadata')
    has_disk = radius is not None or centers is not None
    if has_disk and (radius is None or centers is None):
        raise ValueError('Circular detectors require both radius and centers.')
    if sum((annulus is not None, has_disk, mask is not None)) != 1:
        raise ValueError('Specify exactly one detector: annulus, radius with centers, or mask.')
    if has_disk and polar is not None:
        raise ValueError('radius and centers define Cartesian circles; use annulus or mask for polar data.')
    if theta_range is not None and (annulus is None or polar is None):
        raise ValueError('theta_range is only valid with annulus on polar data.')
    ky, kx = shape
    if mask is not None:
        _normalize_unit_mode(detector_units, label='detector_units')
        weights = np.asarray(mask)
        if weights.shape != tuple(shape):
            raise ValueError(f'mask must match the reciprocal-space shape {tuple(shape)}; got {weights.shape}.')
        if weights.dtype.kind not in 'biuf':
            raise TypeError('mask must contain Boolean or real-valued numeric weights.')
        if not np.all(np.isfinite(weights)):
            raise ValueError('mask must contain only finite values.')
        weights = weights.astype(np.result_type(weights.dtype, np.float32), copy=False)
    else:
        _, scale, mode = _detector_scale(detector_units, geometry)
        scale = scale if mode == 'calibrated' else 1.0
        if annulus is not None:
            radii = np.asarray(annulus, dtype=float)
            if radii.shape != (2,) or not np.all(np.isfinite(radii)):
                raise ValueError('annulus must contain exactly two finite radii (inner, outer).')
            inner, outer = radii / scale
            if not 0 <= inner < outer:
                raise ValueError('annulus must satisfy 0 <= inner_radius < outer_radius.')
            if polar is not None:
                radial_range, _, _, dr, t0, dt = _polar_grid(polar, shape)
                radial = radial_range[0] + np.arange(ky) * dr
                radial_mask = (radial >= inner) & (radial <= outer)
                angular_mask = np.ones(kx, dtype=bool)
                if theta_range is not None:
                    angles = np.asarray(theta_range, dtype=float)
                    if angles.shape != (2,) or not np.all(np.isfinite(angles)):
                        raise ValueError('theta_range must contain exactly two finite angles.')
                    start, stop = angles
                    theta = np.mod(t0 + np.arange(kx) * dt, 360.0)
                    angular_mask = ((theta >= start) & (theta <= stop) if start <= stop
                                    else (theta >= start) | (theta <= stop))
                weights = (radial_mask[:, None] & angular_mask[None, :]).astype(np.float32)
            else:
                y, x = np.ogrid[:ky, :kx]
                dist_sq = (y - (ky - 1) / 2) ** 2 + (x - (kx - 1) / 2) ** 2
                weights = ((dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)).astype(np.float32)
        else:
            positions = np.asarray(centers, dtype=float)
            if positions.shape == (2,):
                positions = positions.reshape(1, 2)
            if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
                raise ValueError('centers must be one (ky, kx) pair or a nonempty (N, 2) array.')
            if not np.all(np.isfinite(positions)):
                raise ValueError('centers must contain only finite values.')
            if mode == 'calibrated':
                positions = np.column_stack(((ky - 1) / 2 - positions[:, 0] / scale,
                                             (kx - 1) / 2 + positions[:, 1] / scale))
            radii = np.asarray(radius, dtype=float)
            if radii.ndim == 0:
                radii = np.full(len(positions), float(radii))
            if radii.shape != (len(positions),):
                raise ValueError('radius must be a scalar or contain one value per center.')
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
                raise ValueError('All detector radii must be positive and finite.')
            radii = radii / scale
            weights = np.zeros(shape, dtype=np.float32)
            y, x = np.ogrid[:ky, :kx]
            for (cy, cx), r in zip(positions, radii):
                weights[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = 1
    if not np.any(weights != 0):
        raise ValueError('The detector selects no reciprocal-space pixels.')
    return weights


def _virtual_projection_plan(shape, dtype, weights, working_memory_mb, need_mean):
    """Budget detector state, two read buffers and a promoted weighted block."""
    budget = _working_memory_bytes(working_memory_mb)
    dtype = np.dtype(dtype)
    accumulator = np.dtype(np.result_type(dtype, weights.dtype, np.float64))
    pixels = prod(shape[-2:])
    fixed = weights.nbytes + (4 * pixels * accumulator.itemsize if need_mean else 0)
    per_pattern = pixels * (2 * dtype.itemsize + accumulator.itemsize) + 2 * accumulator.itemsize
    minimum = fixed + per_pattern
    if budget < minimum:
        raise ValueError(
            'working_memory_mb is too small for virtual_image; '
            f'need at least {minimum} bytes for one pattern and detector workspace.'
        )
    return min(prod(shape[:-2]), (budget - fixed) // per_pattern), accumulator


def _project_virtual_detector(source, weights, *, working_memory_mb=None, need_mean=False):
    """Project basic scan blocks once; never convert the complete source.

    Float64 (or wider/complex) products and sums avoid integer overflow and
    float32 accumulation loss. NaN/Inf propagation, including zero times NaN,
    follows NumPy multiplication rather than silently repairing source values.
    The compact returned scan image is outside the temporary-workspace budget.
    """
    shape = _diffraction_shape(source)
    if len(shape) != 4:
        raise ValueError('virtual_image requires a 4D scan (Ry, Rx, Ky, Kx).')
    max_patterns, dtype = _virtual_projection_plan(
        shape, source.dtype, weights, working_memory_mb, need_mean,
    )
    image = np.empty(shape[:-2], dtype=dtype)
    mean = np.zeros(shape[-2:], dtype=dtype) if need_mean else None
    for scan_slices in _scan_block_slices(shape, max_patterns):
        block = source[scan_slices + (slice(None), slice(None))]
        weighted = np.empty(block.shape, dtype=dtype)
        with np.errstate(invalid='ignore', over='ignore'):
            np.multiply(block, weights, out=weighted, dtype=dtype)
            image[scan_slices] = np.sum(weighted, axis=(-2, -1), dtype=dtype)
            if need_mean:
                mean += np.sum(block, axis=(0, 1), dtype=dtype)
        del weighted, block
    if need_mean:
        mean /= prod(shape[:-2])
    return image, mean


def _virtual_image(source, annulus=None, *, radius=None, centers=None,
                   mask=None, theta_range=None, detector_units='auto', axis_units='auto',
                   vmin=None, vmax=None, grid=True, num_div=10, plot_mask=False,
                   grid_color='black', axes=True, return_detector=False,
                   title='Virtual Detector Image', units='auto', mask_cmap='turbo',
                   mask_vmin=None, mask_vmax=None, mask_power=1, mask_log_scale=True,
                   mask_show_kwargs=None, ring_color='green', ring_alpha=0.3,
                   show=True, working_memory_mb=None, geometry=None,
                   real_factory, reciprocal_factory, polar_factory):
    """Shared public-method coordinator with injected calibrated result factories."""
    shape = _diffraction_shape(source)
    if len(shape) != 4:
        raise ValueError('virtual_image requires a 4D scan (Ry, Rx, Ky, Kx).')
    if mask_show_kwargs is not None and not isinstance(mask_show_kwargs, dict):
        raise TypeError('mask_show_kwargs must be a dictionary or None.')
    if np.dtype(source.dtype).kind == 'c':
        raise TypeError('virtual_image requires real-valued diffraction intensities, not complex amplitudes.')
    units = _normalize_unit_mode(units, label='units')
    _normalize_unit_mode(axis_units, label='axis_units')
    if not np.isscalar(ring_alpha) or not np.isfinite(ring_alpha) or not 0 <= ring_alpha <= 1:
        raise ValueError('ring_alpha must be a finite number between 0 and 1.')
    to_rgba(ring_color)
    geometry = geometry or {}
    weights = _build_detector_weights(
        shape[-2:], annulus, radius=radius, centers=centers, mask=mask,
        theta_range=theta_range, detector_units=detector_units, geometry=geometry,
    )
    image, mean = _project_virtual_detector(
        source, weights, working_memory_mb=working_memory_mb,
        need_mean=plot_mask or return_detector,
    )
    result = real_factory(image)
    if show:
        if vmin is None:
            positives = image[image > 0]
            vmin = np.min(image) if np.any(weights < 0) or not positives.size else np.min(positives)
        if vmax is None:
            vmax = np.max(image)
        result.show(title=title, cmap='gray', vmin=vmin, vmax=vmax, axes=axes,
                    grid=grid, num_div=num_div, grid_color=grid_color, axis_units=axis_units)
    if plot_mask:
        options = dict(title='Mean Diffraction Pattern and Virtual Detector',
                       cmap=mask_cmap, vmin=mask_vmin, vmax=mask_vmax, power=mask_power,
                       logScale=mask_log_scale, axis_units=units, overlay_mask=weights != 0,
                       overlay_color=ring_color, overlay_alpha=ring_alpha)
        if geometry.get('polar_metadata') is not None:
            options['aspect'] = 'auto'
        options.update(mask_show_kwargs or {})
        reciprocal_factory(mean).show(**options)
    if return_detector:
        array = mean * weights
        detector = (polar_factory(array) if geometry.get('polar_metadata') is not None
                    else reciprocal_factory(array))
        return result, detector
    return result
