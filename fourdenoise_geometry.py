"""Pure coordinate and calibration helpers shared by 4Denoise objects."""

import numpy as np
from copy import deepcopy
from numbers import Integral
from scipy.ndimage import gaussian_filter, map_coordinates

def _center_beam_metadata_from_pixels(radius_px, center_px, shape, *,
                                      units=None, conv_factor=None,
                                      **extra_metadata):
    """Build direct-beam metadata from pixel-space beam parameters."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2:
        raise ValueError("Direct-beam metadata requires a 2D reciprocal shape.")

    radius_px = float(radius_px)
    center_px = tuple(float(v) for v in center_px)
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Direct-beam radius must be positive and finite.")
    if len(center_px) != 2 or not np.all(np.isfinite(center_px)):
        raise ValueError("Direct-beam center must contain two finite values.")

    metadata = {
        'radius_px': radius_px,
        'center_px': center_px,
        'shape': shape,
        'coordinate_convention': (
            "center_px is an array-coordinate (ky, kx) pair. "
            "center_calibrated, when present, is relative to the diffraction "
            "origin with positive ky upward and positive kx to the right."
        ),
    }

    has_units = units is not None
    has_factor = conv_factor is not None
    if has_units != has_factor:
        raise ValueError(
            "Direct-beam metadata cannot use a partial reciprocal-space "
            "calibration. Define both units and conv_factor, or clear both."
        )
    if has_units:
        if not np.isscalar(conv_factor) or not np.isfinite(conv_factor) or conv_factor <= 0:
            raise ValueError("conv_factor must be a positive scalar.")
        metadata.update({
            'calibrated_units': str(units).strip(),
            'conv_factor': float(conv_factor),
            'radius_calibrated': radius_px * float(conv_factor),
            'center_calibrated': _center_to_calibrated(
                center_px,
                shape,
                conv_factor,
            ),
        })

    for key, value in extra_metadata.items():
        if value is not None:
            metadata[key] = deepcopy(value)

    return metadata


def _resampled_center_beam_metadata(metadata, old_shape, new_shape, *,
                                    crop_origin=(0, 0), crop_shape=None,
                                    units=None, conv_factor=None):
    """Map a circular beam through an integer crop and isotropic resampling.

    Return None when the old geometry is stale, the beam center is cropped out,
    or unequal axis scales would turn the circle into an ellipse. Historical
    alignment coordinates are not copied because they describe the old image.
    """
    if metadata is None:
        return None
    old_shape = tuple(old_shape)
    new_shape = tuple(new_shape)
    crop_shape = tuple(crop_shape) if crop_shape is not None else old_shape
    if tuple(metadata.get('shape', ())) != old_shape:
        return None
    if 'center_px' not in metadata or 'radius_px' not in metadata:
        return None

    scales = np.asarray(new_shape, dtype=float) / np.asarray(crop_shape, dtype=float)
    if not np.isclose(scales[0], scales[1]):
        return None
    center = np.asarray(metadata['center_px'], dtype=float)
    origin = np.asarray(crop_origin, dtype=float)
    new_center = (center - origin + 0.5) * scales - 0.5
    if not np.all(np.isfinite(new_center)) or np.any(new_center < 0) or np.any(
        new_center > np.asarray(new_shape) - 1
    ):
        return None
    return _center_beam_metadata_from_pixels(
        float(metadata['radius_px']) * scales[0], new_center, new_shape,
        units=units, conv_factor=conv_factor,
        source='resampling', parent_source=metadata.get('source'),
    )


def _resized_data_metadata(metadata, old_shape, new_shape, domain, method):
    """Share global resize geometry between in-memory and streamed outputs."""
    keys = ('real_units', 'real_conv_factor', 'real_origin', 'reciprocal_units',
            'reciprocal_conv_factor', 'polar_metadata', 'cepstral_metadata',
            'center_beam_metadata')
    result = {key: deepcopy(metadata.get(key)) for key in keys}
    result['real_origin'] = tuple(metadata.get('real_origin', (0.0, 0.0)))
    if tuple(old_shape) == tuple(new_shape):
        return result
    axes = tuple(range(len(old_shape) - 2)) if domain == 'real' else (len(old_shape) - 2, len(old_shape) - 1)
    factors = tuple(old_shape[a] / new_shape[a] for a in axes)
    if domain == 'real':
        factor = result['real_conv_factor']
        if factor is not None:
            scales = factors if len(factors) == 2 else factors * 2
            result['real_conv_factor'] = _scaled_real_spacing(factor, scales)
        if len(old_shape) == 4:
            steps = _real_spacing_pair(factor)
            result['real_origin'] = tuple(v + 0.5 * (f - 1) * s for v, f, s in zip(result['real_origin'], factors, steps))
    else:
        factor = result['reciprocal_conv_factor']
        if factor is not None:
            if np.allclose(factors, factors[0]):
                result['reciprocal_conv_factor'] = factor * factors[0]
            else:
                result['reciprocal_units'] = result['reciprocal_conv_factor'] = None
        result['polar_metadata'] = _resampled_polar_metadata(
            result['polar_metadata'], old_shape[-2:], new_shape[-2:], endpoint_radial=method != 'area',
        )
        result['cepstral_metadata'] = _resampled_cepstral_metadata(
            result['cepstral_metadata'], old_shape[-2:], new_shape[-2:],
        )
        result['center_beam_metadata'] = _resampled_center_beam_metadata(
            result['center_beam_metadata'], old_shape[-2:], new_shape[-2:],
            units=result['reciprocal_units'], conv_factor=result['reciprocal_conv_factor'],
        ) if result['polar_metadata'] is None else None
    return result


def _beam_center_pixels(shape, metadata=None, conv_factor=None):
    """Resolve a canonical beam center once for rotations, strain and plotting.

    Prefer current center_px over historical mean-fit coordinates. Reject
    stale geometry rather than rotating about a believable but wrong pivot.
    The fallback is the subpixel image midpoint, never a rounded pixel.
    """
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError('center_beam_metadata must be a dictionary or None.')
        if 'shape' in metadata:
            try:
                stored_shape = tuple(metadata['shape'])
            except TypeError as exc:
                raise ValueError('center_beam_metadata has an invalid shape.') from exc
            if stored_shape != tuple(shape):
                raise ValueError('center_beam_metadata has a stale pattern shape; update it first.')
        for key in ('center_px', 'mean_fit_center_px'):
            if metadata.get(key) is not None:
                center = np.asarray(metadata[key], dtype=float)
                if center.shape != (2,) or not np.isfinite(center).all():
                    raise ValueError(f'{key} must contain a finite coordinate pair.')
                return tuple(center), key
        if metadata.get('center_calibrated') is not None:
            factor = metadata.get('conv_factor', conv_factor)
            if factor is None or not np.isscalar(factor) or not np.isfinite(factor) or factor <= 0:
                raise ValueError('center calibration factor must be positive and finite.')
            return _calibrated_center_to_pixels(metadata['center_calibrated'], factor, shape), 'center_calibrated'
    return tuple((size - 1) / 2 for size in shape), 'image midpoint'


def _cropped_data_metadata(metadata, old_shape, new_shape, *, real_bounds=None,
                           reciprocal_bounds=None):
    """One crop rule for scan origins, calibrated patterns and transformed axes."""
    result = _resized_data_metadata(metadata, old_shape, old_shape, 'real', 'area')
    if real_bounds is not None:
        steps = _real_spacing_pair(result['real_conv_factor'])
        factors = tuple((b - a) / size for (a, b), size in zip(real_bounds, new_shape[:2]))
        result['real_origin'] = tuple(v + a * s for v, (a, _), s in zip(result['real_origin'], real_bounds, steps))
        if result['real_conv_factor'] is not None:
            result['real_conv_factor'] = _scaled_real_spacing(result['real_conv_factor'], factors)
            result['real_origin'] = tuple(v + 0.5 * (f - 1) * s for v, f, s in zip(result['real_origin'], factors, steps))
    if reciprocal_bounds is None:
        return result
    old_pattern, new_pattern = old_shape[-2:], new_shape[-2:]
    factors = tuple((b - a) / size for (a, b), size in zip(reciprocal_bounds, new_pattern))
    factor = result['reciprocal_conv_factor']
    if factor is not None:
        if np.isclose(*factors):
            result['reciprocal_conv_factor'] = factor * factors[0]
        else:
            result['reciprocal_units'] = result['reciprocal_conv_factor'] = None
    result['polar_metadata'] = _resampled_polar_metadata(
        result['polar_metadata'], old_pattern, new_pattern, bounds=reciprocal_bounds,
    )
    result['cepstral_metadata'] = _resampled_cepstral_metadata(
        result['cepstral_metadata'], old_pattern, new_pattern, bounds=reciprocal_bounds,
    )
    changed = tuple(old_pattern) != tuple(new_pattern) or reciprocal_bounds != ((0, old_pattern[0]), (0, old_pattern[1]))
    if changed:
        fractional = any(not float(v).is_integer() for bounds in reciprocal_bounds for v in bounds)
        result['center_beam_metadata'] = None if fractional or metadata.get('polar_metadata') is not None else _resampled_center_beam_metadata(
            result['center_beam_metadata'], old_pattern, new_pattern,
            crop_origin=tuple(a for a, _ in reciprocal_bounds),
            crop_shape=tuple(b - a for a, b in reciprocal_bounds),
            units=result['reciprocal_units'], conv_factor=result['reciprocal_conv_factor'],
        )
    return result


def _flipped_data_metadata(metadata, shape, *, flip_axes):
    """Update signed scan steps, beam/cepstral centers and the R/Q frame."""
    result = _resized_data_metadata(metadata, shape, shape, 'real', 'area')
    q_axes = (len(shape) - 2, len(shape) - 1)
    if result['polar_metadata'] is not None and any(a in q_axes for a in flip_axes):
        raise ValueError('Flipping polar radius or angle axes would invalidate polar metadata; convert to Cartesian first.')
    if len(shape) == 4 and result['real_conv_factor'] is not None:
        steps = list(_real_spacing_pair(_normalize_real_spacing(result['real_conv_factor'])))
        origin = list(_normalize_real_origin(result['real_origin']))
        for a in (0, 1):
            if a in flip_axes:
                origin[a] += (shape[a] - 1) * steps[a]
                steps[a] *= -1
        result['real_origin'] = tuple(origin)
        result['real_conv_factor'] = steps[0] if np.isclose(*steps) else tuple(steps)
    for field in ('center_beam_metadata', 'cepstral_metadata'):
        record = result[field]
        if record is None:
            continue
        for key in ('center_px', 'mean_fit_center_px', 'std_fit_center_px', 'target_center_px'):
            if key in record:
                center = np.asarray(record[key], dtype=float)
                if center.shape == (2,):
                    record[key] = tuple(shape[-2 + i] - 1 - c if q_axes[i] in flip_axes else c for i, c in enumerate(center))
        if field == 'center_beam_metadata' and 'center_calibrated' in record:
            center = np.asarray(record['center_calibrated'], dtype=float)
            if center.shape == (2,):
                record['center_calibrated'] = tuple(-c if q_axes[i] in flip_axes else c for i, c in enumerate(center))
    if metadata.get('rq_calibration') is not None:
        from fourdenoise_rq import _reframe_rq_calibration
        real, reciprocal = np.eye(2), np.eye(2)
        for a in flip_axes:
            if a in q_axes:
                reciprocal[1 - (a - q_axes[0]), 1 - (a - q_axes[0])] = -1
            elif len(shape) == 4:
                real[1 - a, 1 - a] = -1
        result['rq_calibration'] = _reframe_rq_calibration(metadata['rq_calibration'], real=real, reciprocal=reciprocal)
    else:
        result['rq_calibration'] = None
    return result


def _rotated_data_metadata(metadata, old_shape, new_shape, *, angle_degrees):
    """Map beam coordinates through image-centered rotation and reframe R/Q."""
    if metadata.get('cepstral_metadata') is not None:
        raise ValueError('Diffraction rotation cannot retain cepstral coordinates.')
    result = _resized_data_metadata(metadata, old_shape, old_shape, 'real', 'area')
    from fourdenoise_rq import _reframe_rq_calibration, _rotation_matrix
    result['rq_calibration'] = _reframe_rq_calibration(
        metadata.get('rq_calibration'), reciprocal=_rotation_matrix(angle_degrees),
    )
    if metadata.get('polar_metadata') is not None:
        _, _, theta, _, _, _ = _polar_grid(metadata['polar_metadata'], old_shape[-2:])
        if tuple(old_shape) != tuple(new_shape) or not np.isclose(theta[1] - theta[0], 360):
            raise ValueError('Polar rotation requires unchanged shape and full-circle angular support.')
        result['center_beam_metadata'] = None
    elif result['center_beam_metadata'] is not None:
        beam = result['center_beam_metadata']
        if 'radius_px' not in beam:
            result['center_beam_metadata'] = None
        else:
            center, _ = _beam_center_pixels(old_shape[-2:], beam, result['reciprocal_conv_factor'])
            old_mid = (np.asarray(old_shape[-2:]) - 1) / 2
            new_mid = (np.asarray(new_shape[-2:]) - 1) / 2
            # Array rows point down; positive SciPy angles are Cartesian CCW.
            new_center = _rotation_matrix(angle_degrees) @ (np.asarray(center) - old_mid) + new_mid
            result['center_beam_metadata'] = _center_beam_metadata_from_pixels(
                beam['radius_px'], new_center, new_shape[-2:],
                units=result['reciprocal_units'], conv_factor=result['reciprocal_conv_factor'],
                source='rotation', parent_source=beam.get('source'),
            )
    return result


_GEOMETRY_OPERATIONS = {'resize': _resized_data_metadata, 'crop': _cropped_data_metadata,
                        'flip': _flipped_data_metadata, 'rotation': _rotated_data_metadata}


def _transform_geometry(metadata, old_shape, new_shape, operation, **kwargs):
    """Dispatch operation-specific metadata rules; unsupported operations fail."""
    if operation == 'values':
        if tuple(old_shape) != tuple(new_shape):
            raise ValueError('Value-only geometry preservation requires unchanged shape.')
        return _resized_data_metadata(metadata, old_shape, old_shape, 'real', 'area')
    if operation not in _GEOMETRY_OPERATIONS:
        raise ValueError(f'Unsupported geometry operation: {operation}.')
    if operation == 'flip':
        if tuple(old_shape) != tuple(new_shape):
            raise ValueError('Axis flips cannot change shape.')
        return _flipped_data_metadata(metadata, old_shape, **kwargs)
    return _GEOMETRY_OPERATIONS[operation](metadata, old_shape, new_shape, **kwargs)


def _polar_spec(shape, center=None, r_max=None, output_shape=None):
    """Resolve a Cartesian-to-polar sampling domain without allocating pixels.

    The default center is the subpixel midpoint, and the default radius fits
    inside the canvas. Returned sizes match HyperData.to_polar defaults.
    """
    height, width = shape
    center = ((height - 1) / 2, (width - 1) / 2) if center is None else center
    values = np.asarray(center)
    if values.shape != (2,) or values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)):
        raise ValueError('center must contain two finite (y, x) pixel coordinates.')
    cy, cx = map(float, values)
    if not (0 <= cy <= height - 1 and 0 <= cx <= width - 1):
        raise ValueError('center must lie inside the diffraction pattern bounds.')
    if r_max is None:
        radius = min(cy, cx, height - 1 - cy, width - 1 - cx)
    else:
        value = np.asarray(r_max)
        if value.ndim != 0 or value.dtype.kind not in 'iuf':
            raise ValueError('r_max must be a positive finite value.')
        radius = float(value)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('r_max must be a positive finite value.')
    if output_shape is None:
        sizes = (max(1, int(np.ceil(radius))), max(4, int(np.ceil(2 * np.pi * radius))))
    else:
        try:
            sizes = tuple(output_shape)
        except TypeError as exc:
            raise ValueError('output_shape must contain two positive integers.') from exc
        if len(sizes) != 2 or any(
            isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size <= 0
            for size in sizes
        ):
            raise ValueError('output_shape must contain two positive integers.')
        sizes = tuple(map(int, sizes))
    return (cy, cx), radius, sizes


def _polar_grid(metadata, shape):
    """Resolve uniform sample coordinates separately from polar support."""
    nr, nt = shape
    if tuple(metadata.get('axis_order', ('radius', 'theta'))) != ('radius', 'theta'):
        raise ValueError("Polar axes must be ordered as ('radius', 'theta').")
    if tuple(metadata.get('output_shape', shape)) != tuple(shape):
        raise ValueError('Polar metadata output_shape does not match the data.')
    radius = np.asarray(metadata.get(
        'radius_range_pixels', (0.0, metadata.get('r_max', float(nr - 1))),
    ), dtype=float)
    theta = np.asarray(metadata.get('theta_range', (0.0, 360.0)), dtype=float)
    support = np.asarray(metadata.get('radius_support_range_pixels', radius), dtype=float)
    for name, values in (('radius range', radius), ('radius support', support), ('theta range', theta)):
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f'Polar {name} must contain two finite numbers.')
    if radius[0] < 0 or radius[1] < radius[0] or support[0] < 0 or support[1] < support[0]:
        raise ValueError('Polar radius ranges must be nonnegative and increasing.')
    span = theta[1] - theta[0]
    if not 0 < span <= 360 + 1e-9:
        raise ValueError('Polar angular support must be increasing and at most 360 degrees.')
    dr = float(metadata.get(
        'radius_sample_step_pixels', (radius[1] - radius[0]) / (nr - 1) if nr > 1 else 0,
    ))
    dt = float(metadata.get('theta_step', span / nt))
    t0 = float(metadata.get('theta_sample_start', theta[0]))
    if not np.isfinite(dr) or dr < 0 or not np.isfinite(t0):
        raise ValueError('Polar sample origin and radial step must be finite.')
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, span / nt):
        raise ValueError('Polar theta_step is inconsistent with angular support and shape.')
    if nr > 1 and not np.isclose(dr * (nr - 1), radius[1] - radius[0]):
        raise ValueError('Polar radial sample step is inconsistent with radial range and shape.')
    return radius, support, theta, dr, t0, dt


def _resampled_polar_metadata(metadata, old_shape, new_shape, *, bounds=None,
                              endpoint_radial=False):
    """Map polar coordinates through a crop/area resize or endpoint resize.

    Sample ranges identify actual row centers. Support identifies the radial
    annulus and angular sector available for inverse reconstruction. Angular
    coverage and the first angular sample are distinct after area averaging.
    """
    if metadata is None:
        return None
    radius, support, theta, dr, t0, dt = _polar_grid(metadata, old_shape)
    if bounds is None:
        bounds = ((0, old_shape[0]), (0, old_shape[1]))
    y_limits, x_limits = bounds
    sy = (y_limits[1] - y_limits[0]) / new_shape[0]
    sx = (x_limits[1] - x_limits[0]) / new_shape[1]
    if endpoint_radial:
        r0, r1 = radius
        new_dr = (r1 - r0) / (new_shape[0] - 1) if new_shape[0] > 1 else 0.0
        if new_shape[0] == 1:
            r1 = r0
        new_t0 = t0
        new_support = support
    else:
        first_row = y_limits[0] + (sy - 1) / 2
        last_row = y_limits[1] - (sy + 1) / 2
        r0, r1 = radius[0] + np.array([first_row, last_row]) * dr
        if r0 < -1e-12 or r1 > radius[1] + 1e-12:
            raise ValueError(
                'This polar crop would sample beyond the radial grid. Crop without '
                'radial upsampling, or use resize(method="linear" or "nearest").'
            )
        r0, r1 = max(0.0, r0), min(radius[1], r1)
        new_dr = dr * sy
        new_t0 = t0 + (x_limits[0] + (sx - 1) / 2) * dt
        new_support = np.array([
            max(support[0], radius[0] + (y_limits[0] - 0.5) * dr),
            min(support[1], radius[0] + (y_limits[1] - 0.5) * dr),
        ]) if dr > 0 else support
    new_theta = theta[0] + np.asarray(x_limits) * dt
    scale = metadata.get('cartesian_reciprocal_conv_factor')
    if scale is None:
        scale = metadata.get('radius_conv_factor', 1.0)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Polar radial conversion factor must be positive and finite.')
    result = deepcopy(metadata)
    result.update({
        'is_polar': True,
        'axis_order': ('radius', 'theta'),
        'output_shape': tuple(new_shape),
        'radius_range': (float(r0), float(r1)),
        'radius_range_pixels': (float(r0), float(r1)),
        'radius_support_range_pixels': tuple(float(value) for value in new_support),
        'radius_display_range': (r0 * scale, r1 * scale),
        'radius_conv_factor': scale,
        'radius_step_pixels': (r1 - r0) / new_shape[0],
        'radius_step': (r1 - r0) / new_shape[0] * scale,
        'radius_sample_step_pixels': new_dr,
        'radius_sample_step': new_dr * scale,
        'r_max': float(new_support[1]),
        'theta_range': tuple(float(value) for value in new_theta),
        'theta_sample_start': float(new_t0),
        'theta_step': dt * sx,
        'theta_units': 'deg',
    })
    return result




def _resampled_cepstral_metadata(metadata, old_shape, new_shape, *, bounds=None):
    """Retain cepstral identity and map its zero through index resampling."""
    if metadata is None:
        return None
    if tuple(metadata.get('shape', old_shape)) != tuple(old_shape):
        raise ValueError('Cepstral metadata shape does not match the data.')
    center = np.asarray(metadata['center_px'], dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError('Cepstral center_px must contain finite (y, x) coordinates.')
    if bounds is None:
        bounds = ((0, old_shape[0]), (0, old_shape[1]))
    start = np.array([values[0] for values in bounds])
    factors = np.array([b - a for a, b in bounds]) / new_shape
    result = deepcopy(metadata)
    result['shape'] = tuple(new_shape)
    result['center_px'] = tuple((center - start + 0.5) / factors - 0.5)
    return result


def _parse_reciprocal_crop_limits(limits, length, *, name, mode='pixels',
                                  conv_factor=None, axis=0,
                                  allow_fractional_index=False):
    """Return finite half-open pixel bounds, including calibrated limits."""
    if limits is None:
        return (0, length)

    def to_pixel(value):
        if mode != 'calibrated':
            return value
        sign = -1 if axis == 0 else 1
        return (length - 1) / 2.0 + sign * value / conv_factor

    try:
        values = np.asarray(limits, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric index or a length-2 range.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite limits.")
    if values.ndim == 0:
        start = float(values)
        if mode == 'calibrated':
            start = float(np.rint(to_pixel(start)))
        elif not allow_fractional_index and not start.is_integer():
            raise ValueError(f"{name}: a single pixel index must be an integer.")
        end = start + 1
    elif values.shape == (2,):
        start, end = (to_pixel(float(value)) for value in values)
        if mode == 'calibrated':
            start, end = sorted((start, end))
    else:
        raise ValueError(f"{name} must be an index or a length-2 range.")

    # Remove conversion roundoff without discarding meaningful subpixel offsets.
    bounds = tuple(
        int(np.rint(value)) if np.isclose(value, np.rint(value), rtol=0, atol=1e-12)
        else float(value)
        for value in (start, end)
    )
    if not (0 <= bounds[0] < bounds[1] <= length):
        raise ValueError(
            f"{name}: range {bounds} must satisfy 0 <= start < end <= {length}."
        )
    return bounds


def _reciprocal_crop_shape(y_limits, x_limits, output_shape):
    """Infer crop size from actual extents, or validate an explicit size."""
    if output_shape is None:
        return tuple(
            max(1, int(round(stop - start)))
            for start, stop in (y_limits, x_limits)
        )
    try:
        shape = np.asarray(output_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("kshape must contain two positive integers.") from exc
    if (
        shape.shape != (2,) or shape.dtype.kind not in 'iuf'
        or not np.all(np.isfinite(shape)) or np.any(shape <= 0)
        or np.any(shape != np.floor(shape))
    ):
        raise ValueError("kshape must contain two positive integers.")
    return tuple(int(value) for value in shape)


def _resample_reciprocal_crop(array, y_limits, x_limits, output_shape):
    """Crop the final two axes using the actual half-open pixel bounds.

    Output pixel ``j`` samples ``start + (j + 0.5) * step - 0.5``, where
    ``step = (stop - start) / output_size``. Thus an integer crop of unchanged
    size is an exact slice, while a shifted four-pixel window ``(1.2, 5.2)``
    samples ``1.2, 2.2, 3.2, 4.2``. No leading (scan/stack) axis is interpolated.

    Bilinear interpolation uses neighboring source pixels, not a resized
    integer approximation of the requested window. Downsampling is preceded
    by a Gaussian anti-alias filter with sigma ``max(0, (step - 1) / 2)``.
    Only the enclosing region and required interpolation/filter halo are
    converted and filtered for each pattern. Detector edges use reflection
    about the boundary pixel center (SciPy ``mirror`` mode).
    """
    array = np.asarray(array)
    extents = np.array([stop - start for start, stop in (y_limits, x_limits)])
    integer_bounds = all(
        float(value).is_integer() for value in y_limits + x_limits
    )
    if integer_bounds and tuple(extents) == tuple(output_shape):
        y0, y1 = (int(value) for value in y_limits)
        x0, x1 = (int(value) for value in x_limits)
        return array[..., y0:y1, x0:x1]

    steps = extents / np.asarray(output_shape)
    sigma = np.maximum(0, (steps - 1) / 2)
    # Four sigma covers the Gaussian kernel; one more pixel covers interpolation.
    halo = np.ceil(4 * sigma).astype(int) + 1
    origin = np.maximum(
        0, np.floor([y_limits[0], x_limits[0]]).astype(int) - halo,
    )
    stop = np.minimum(
        array.shape[-2:],
        np.ceil([y_limits[1], x_limits[1]]).astype(int) + halo,
    )
    sample_axes = [
        start + (np.arange(size) + 0.5) * step - 0.5 - offset
        for (start, _), size, step, offset in zip(
            (y_limits, x_limits), output_shape, steps, origin,
        )
    ]
    coordinates = np.array(np.meshgrid(*sample_axes, indexing='ij'))
    output_dtype = np.result_type(array.dtype, np.float32)
    output = np.empty(array.shape[:-2] + tuple(output_shape), dtype=output_dtype)
    region = array[..., origin[0]:stop[0], origin[1]:stop[1]]
    for index in np.ndindex(array.shape[:-2]):
        pattern = region[index].astype(output_dtype, copy=False)
        if np.any(sigma > 0):
            pattern = gaussian_filter(pattern, sigma, mode='mirror')
        map_coordinates(
            pattern, coordinates, output=output[index], order=1,
            mode='mirror', prefilter=False,
        )
    return output


def _normalize_real_spacing(spacing):
    """Validate signed ``(y, x)`` units per pixel for oriented scan axes."""
    if isinstance(spacing, (bool, np.bool_)):
        raise ValueError("Real-space pixel spacing must be nonzero and finite.")
    try:
        values = np.asarray(spacing, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Real-space pixel spacing must be numeric.") from exc
    if values.ndim == 0:
        if not np.isfinite(values) or values == 0:
            raise ValueError("Real-space pixel spacing must be nonzero and finite.")
        return float(values)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values == 0):
        raise ValueError("Real-space pixel spacing must be a nonzero finite (y, x) pair.")
    return tuple(float(value) for value in values)


def _real_spacing_pair(spacing):
    """Return ``(y, x)`` spacing without changing the public scalar form."""
    if spacing is None:
        return (1.0, 1.0)
    return (spacing, spacing) if np.isscalar(spacing) else tuple(spacing)


def _scaled_real_spacing(spacing, factors):
    """Scale each pixel axis, retaining scalar form for isotropic spacing."""
    result = tuple(
        float(value) * float(factor)
        for value, factor in zip(_real_spacing_pair(spacing), factors)
    )
    return result[0] if np.isclose(result[0], result[1]) else result


def _normalize_real_origin(origin):
    """Validate the physical coordinate of the first real-space pixel center."""
    try:
        values = np.asarray(origin, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("real_origin must be a finite (y, x) pair.") from exc
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("real_origin must be a finite (y, x) pair.")
    return tuple(float(value) for value in values)


def _parse_real_selection(value, max_len, name, mode, conv_factor,
                          origin=0.0):
    """Parse real-space scalar/range selections into half-open pixel slices."""
    if value is None:
        return 0, max_len, 'all'

    is_pair = (
        isinstance(value, (tuple, list, np.ndarray))
        and np.asarray(value).shape == (2,)
    )
    if is_pair:
        a, b = np.asarray(value, dtype=float)
        if not np.all(np.isfinite((a, b))):
            raise ValueError(f"{name} range must contain finite values.")
        if mode == 'calibrated':
            a = (a - origin) / conv_factor
            b = (b - origin) / conv_factor
            if conv_factor > 0:
                start = int(np.floor(a))
                stop = int(np.ceil(b))
            else:
                # Physical [low, high) runs opposite to pixel index order.
                start = int(np.floor(b)) + 1
                stop = int(np.floor(a)) + 1
        else:
            if not float(a).is_integer() or not float(b).is_integer():
                raise ValueError(
                    f"{name} range must contain integer pixel indices when "
                    "selection_units='pixels'."
                )
            start = int(a)
            stop = int(b)
        if not (0 <= start < stop <= max_len):
            raise ValueError(
                f"Invalid {name} range ({value[0]}, {value[1]}) for "
                f"length {max_len} using {mode} units."
            )
        return start, stop, 'range'

    if np.isscalar(value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} index must be finite.")
        if mode == 'calibrated':
            index = int(np.rint((value - origin) / conv_factor))
        else:
            if not value.is_integer():
                raise ValueError(
                    f"{name} index must be an integer when "
                    "selection_units='pixels'."
                )
            index = int(value)
        if not (0 <= index < max_len):
            raise ValueError(
                f"{name} index {value:g} out of bounds for length "
                f"{max_len} using {mode} units."
            )
        return index, index + 1, 'index'

    raise ValueError(f"{name} must be a scalar, a length-2 range, or None.")


def _normalize_unit_mode(unit_mode, label='axis_units'):
    """Normalize a unit-selection mode used by plotting/selection helpers."""
    if unit_mode is None:
        unit_mode = 'auto'
    if not isinstance(unit_mode, str):
        raise ValueError(f"{label} must be 'auto', 'pixels', or 'calibrated'.")

    normalized = unit_mode.strip().lower().replace('_', '-')
    if normalized in {'auto', 'default'}:
        return 'auto'
    if normalized in {'pixel', 'pixels', 'px'}:
        return 'pixels'
    if normalized in {'calibrated', 'calibration', 'physical', 'data', 'units'}:
        return 'calibrated'
    raise ValueError(f"{label} must be 'auto', 'pixels', or 'calibrated'.")


def _resolve_unit_mode(unit_mode, units, conv_factor, label='axis_units'):
    """Resolve ``auto``/pixel/calibrated mode to units and a pixel step."""
    mode = _normalize_unit_mode(unit_mode, label=label)
    if mode == 'pixels':
        return None, None, 'pixels'

    has_units = units is not None
    has_factor = conv_factor is not None
    if has_units != has_factor:
        raise ValueError(
            f"{label} cannot use a partial calibration. Define both units and "
            "conv_factor, or clear both."
        )

    if not has_units:
        if mode == 'calibrated':
            raise ValueError(
                f"{label}='calibrated' requires stored units and conv_factor."
            )
        return None, None, 'pixels'

    factor = _normalize_real_spacing(conv_factor)
    return str(units).strip(), factor, 'calibrated'


def _center_to_calibrated(center_px, shape, conv_factor):
    """Convert pixel ``(ky, kx)`` to reciprocal coords, positive ky upward."""
    center_y, center_x = tuple(float(v) for v in center_px)
    origin_y = (int(shape[0]) - 1) / 2.0
    origin_x = (int(shape[1]) - 1) / 2.0
    return (
        (origin_y - center_y) * float(conv_factor),
        (center_x - origin_x) * float(conv_factor),
    )


def _calibrated_center_to_pixels(center, conv_factor, shape):
    """Convert calibrated reciprocal ``(ky, kx)`` to pixel coordinates."""
    center = np.asarray(center, dtype=float)
    origin_y = (int(shape[0]) - 1) / 2.0
    origin_x = (int(shape[1]) - 1) / 2.0
    return (
        origin_y - center[0] / float(conv_factor),
        origin_x + center[1] / float(conv_factor),
    )
