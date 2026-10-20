"""Peak-scan results and bounded source routing around the existing detector.

No detection algorithm is duplicated here. Object factories supplied by the
source keep this module independent of the public fourdenoise import.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import prod
from numbers import Integral
from pathlib import Path
import os
import tempfile

import h5py
import numpy as np
from scipy.fft import next_fast_len
from tqdm import tqdm

from fourdenoise_geometry import _resolve_unit_mode
from fourdenoise_io import (
    _PEAK_SCAN_HDF5_FORMAT, _decode_hdf5_value, _read_hdf5_value, _write_hdf5_value,
)
from fourdenoise_reductions import _working_memory_bytes
from fourdenoise_streaming import _stream_destination


def _finite_peak_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate scalar detector parameters identically for eager/streamed paths."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a finite scalar.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar.') from exc
    if not np.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive.')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be non-negative.')
    return value


def _validate_peak_options(radius, min_distance, trench_width=1.0,
                           kernel_amp=1.0, trench_amp=-0.5, threshold_abs=1.0,
                           threshold_rel=None, r_range=None, n_fold=None,
                           sym_mode='none', sym_tolerance_px=2.0,
                           center_tolerance_px=2.0, orbit_min_fraction=0.5,
                           reorder=False, return_details=False):
    """Normalize options without touching source pixels or allocating a template."""
    if not isinstance(return_details, (bool, np.bool_)):
        raise ValueError('return_details must be a Boolean value.')
    radius = _finite_peak_scalar(radius, 'radius', positive=True)
    trench_width = _finite_peak_scalar(trench_width, 'trench_width', positive=True)
    kernel_amp = _finite_peak_scalar(kernel_amp, 'kernel_amp')
    trench_amp = _finite_peak_scalar(trench_amp, 'trench_amp')
    if kernel_amp <= trench_amp:
        raise ValueError('kernel_amp must exceed trench_amp.')
    if isinstance(min_distance, (bool, np.bool_)) or not isinstance(min_distance, Integral) or min_distance < 1:
        raise ValueError('min_distance must be a positive integer.')
    min_distance = int(min_distance)
    if threshold_abs is not None:
        threshold_abs = _finite_peak_scalar(threshold_abs, 'threshold_abs')
    if threshold_rel is not None:
        threshold_rel = _finite_peak_scalar(threshold_rel, 'threshold_rel', nonnegative=True)
        if threshold_rel > 1:
            raise ValueError('threshold_rel must be between 0 and 1.')
    if r_range is not None:
        try:
            limits = np.asarray(r_range, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('r_range must be a finite (r_min, r_max) pair.') from exc
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or not 0 <= limits[0] < limits[1]:
            raise ValueError('r_range must satisfy 0 <= r_min < r_max.')
        r_range = tuple(map(float, limits))
    if n_fold is not None:
        if isinstance(n_fold, (bool, np.bool_)) or not isinstance(n_fold, Integral) or n_fold < 1:
            raise ValueError('n_fold must be a positive integer or None.')
        n_fold = int(n_fold)
    if not isinstance(sym_mode, str) or sym_mode not in ('none', 'repair', 'prune', 'both'):
        raise ValueError("sym_mode must be 'none', 'repair', 'prune', or 'both'.")
    if sym_mode != 'none' and (n_fold is None or n_fold < 2):
        raise ValueError('sym_mode requires n_fold >= 2.')
    if not isinstance(reorder, (bool, np.bool_)):
        raise ValueError('reorder must be a Boolean value.')
    if reorder and (n_fold is None or n_fold < 2):
        raise ValueError('reorder=True requires n_fold >= 2.')
    sym_tolerance_px = _finite_peak_scalar(sym_tolerance_px, 'sym_tolerance_px', positive=True)
    center_tolerance_px = _finite_peak_scalar(center_tolerance_px, 'center_tolerance_px', nonnegative=True)
    orbit_min_fraction = _finite_peak_scalar(orbit_min_fraction, 'orbit_min_fraction', positive=True)
    if orbit_min_fraction > 1:
        raise ValueError('orbit_min_fraction must be in (0, 1].')
    return dict(radius=radius, min_distance=min_distance, trench_width=trench_width,
                kernel_amp=kernel_amp, trench_amp=trench_amp, threshold_abs=threshold_abs,
                threshold_rel=threshold_rel, r_range=r_range, n_fold=n_fold,
                sym_mode=sym_mode, sym_tolerance_px=sym_tolerance_px,
                center_tolerance_px=center_tolerance_px, orbit_min_fraction=orbit_min_fraction,
                reorder=bool(reorder), return_details=bool(return_details))


_PEAK_FIELDS = ('coords', 'scores', 'synthetic_mask', 'orbit_ids')
_PEAK_SCAN_VERSION = '1.0'


@dataclass(slots=True, repr=False, eq=False)
class PeakScanResult:
    """Ragged peak arrays indexed by the original scan position.

    ``coords``, ``scores``, ``synthetic_mask`` and ``orbit_ids`` are object
    arrays with shape (Ry, Rx) for a 4D scan or (N,) for a 3D stack. Each
    ``coords[index]`` has shape (n_peaks, 2), in (y, x) order. The other entries
    have matching row order. ``intensities`` has the same ragged structure when
    requested, otherwise None. These are raw circular-window sums, not template
    correlation scores or background-corrected estimates. Symmetry-generated
    rows have NaN scores/intensities and True synthetic_mask.

    ``processed_mask`` distinguishes masked-out positions from evaluated
    patterns containing zero peaks. Masked-out entries are empty arrays.
    ``metadata`` records coordinate units/convention, pixel detection settings,
    resolved beam/search center, scan geometry and original source metadata.
    Calibrated coordinates are y-up/x-right about the *geometric* detector
    midpoint; choosing an off-center search origin does not shift this frame.

    The result owns its arrays and remains usable after source closure. It is
    a peak collection, not a RealSpace image; peak_counts is a scan-shaped array
    suitable for plotting. Its size grows with the number of scan positions and
    peaks, independently of the temporary-workspace budget. For larger results,
    supply destination to DiffractionSource.get_peaks instead.
    """

    coords: np.ndarray
    scores: np.ndarray
    synthetic_mask: np.ndarray
    orbit_ids: np.ndarray
    processed_mask: np.ndarray
    metadata: dict
    intensities: np.ndarray | None = None

    @property
    def scan_shape(self):
        """Original leading scan-axis shape, without inventing a 2D layout."""
        return self.coords.shape

    @property
    def peak_counts(self):
        """Scan-shaped counts of all returned peaks, including repaired rows."""
        return np.array([len(a) for a in self.coords.flat], dtype=np.int64).reshape(self.scan_shape)

    @property
    def observed_peak_counts(self):
        """Scan-shaped counts excluding symmetry-generated peaks."""
        return np.array([np.count_nonzero(~a) for a in self.synthetic_mask.flat],
                        dtype=np.int64).reshape(self.scan_shape)

    def copy(self):
        """Return independent numerical arrays, ragged entries and metadata."""
        return deepcopy(self)

    @classmethod
    def load(cls, filename, *, max_memory_mb=None):
        """Explicitly load a saved peak collection, not the diffraction pixels.

        File mode stores append-only numeric tables and per-scan starts/counts,
        not pickle or HDF5 variable-length object arrays. This loader validates
        schema, table lengths, ranges and scan indexing before exposing entries.
        max_memory_mb optionally caps stored numeric arrays and object pointer
        tables before reading them. Metadata, Python/NumPy object headers and
        allocator/library overhead are excluded; this is not a RAM guarantee.
        This operation materializes the whole collection. For selected entries,
        read /starts, /counts and the corresponding /coords/... slices with h5py.
        """
        with h5py.File(filename, 'r') as file:
            if _decode_hdf5_value(file.attrs.get('fourdenoise_format')) != _PEAK_SCAN_HDF5_FORMAT:
                raise ValueError('Not a saved 4Denoise peak collection.')
            if _decode_hdf5_value(file.attrs.get('format_version')) != _PEAK_SCAN_VERSION:
                raise ValueError('Unsupported peak-collection format_version.')
            try:
                scan_shape = tuple(file.attrs.get('scan_shape', ()))
            except TypeError as exc:
                raise ValueError('Peak file has an invalid scan_shape.') from exc
            if len(scan_shape) not in (1, 2) or any(not isinstance(n, Integral) or n < 1 for n in scan_shape):
                raise ValueError('Peak file has an invalid scan_shape.')
            scan_shape = tuple(map(int, scan_shape))
            names = _PEAK_FIELDS + ('starts', 'counts', 'processed_mask')
            has_intensities = bool(file.attrs.get('has_intensities', False))
            if has_intensities:
                names += ('intensities',)
            if any(name not in file or not isinstance(file[name], h5py.Dataset) for name in names):
                raise ValueError('Peak file is missing a required numeric dataset.')
            total = file['coords'].shape[0] if file['coords'].ndim == 2 else -1
            for name in names:
                expected = scan_shape if name in ('starts', 'counts', 'processed_mask') else ((total, 2) if name == 'coords' else (total,))
                kinds = 'iu' if name in ('starts', 'counts', 'orbit_ids') else 'b' if name in ('processed_mask', 'synthetic_mask') else 'iuf'
                if file[name].shape != expected or file[name].dtype.kind not in kinds:
                    raise ValueError(f'Peak file has incompatible shape/dtype for {name}.')
            numeric_bytes = sum(file[name].size * file[name].dtype.itemsize for name in names)
            numeric_bytes += prod(scan_shape) * np.dtype(object).itemsize * (4 + has_intensities)
            if max_memory_mb is not None:
                try:
                    limit = _working_memory_bytes(max_memory_mb)
                except ValueError as exc:
                    raise ValueError(str(exc).replace('working_memory_mb', 'max_memory_mb')) from exc
                if numeric_bytes > limit:
                    raise MemoryError('Peak collection exceeds max_memory_mb; read selected numeric slices with h5py.')
            arrays = {name: file[name][()] for name in names}
            if not np.all(np.isfinite(arrays['coords'])):
                raise ValueError('Peak file coordinates must be finite.')
            starts, counts = arrays['starts'], arrays['counts']
            if np.any(starts < 0) or np.any(counts < 0) or np.any(starts > total) or np.any(counts > total - starts):
                raise ValueError('Peak file has out-of-range starts/counts.')
            if np.any(counts[~arrays['processed_mask']] != 0) or sum(map(int, counts.flat)) != total:
                raise ValueError('Peak file counts disagree with the processed mask or total peak count.')
            # Nonempty intervals must partition the table, regardless of block traversal.
            ranges = sorted((int(s), int(s) + int(c)) for s, c in zip(starts.flat, counts.flat) if c)
            end = 0
            for start, stop in ranges:
                if start != end:
                    raise ValueError('Peak file contains overlapping or missing peak-table ranges.')
                end = stop
            if 'metadata' not in file:
                raise ValueError('Peak file is missing metadata.')
            metadata = _read_hdf5_value(file, 'metadata')
            if not isinstance(metadata, dict) or tuple(metadata.get('scan_shape', ())) != scan_shape:
                raise ValueError('Peak file metadata has an incompatible scan shape.')
        ragged = {name: np.empty(scan_shape, dtype=object) for name in _PEAK_FIELDS}
        if has_intensities:
            ragged['intensities'] = np.empty(scan_shape, dtype=object)
        for index in np.ndindex(scan_shape):
            start, count = int(starts[index]), int(counts[index])
            for name in ragged:
                ragged[name][index] = arrays[name][start:start + count]
        return cls(**ragged, processed_mask=arrays['processed_mask'], metadata=metadata)


def _peak_source_plan(shape, dtype, options, working_memory_mb, chunk_shape):
    """Reserve conservative one-pattern FFT/candidate workspace and raw buffers.

    Estimates include template construction, padded FFT arrays, full detector
    masks, candidate/provenance arrays, integration work and symmetry matching.
    Existing caches, metadata and retained output are outside this estimate.
    """
    budget = _working_memory_bytes(working_memory_mb)
    radius = options['radius'] + options['trench_width']
    if not np.isfinite(radius):
        raise ValueError('radius + trench_width must be finite.')
    side = 2 * int(np.ceil(radius)) + 1
    pixels = prod(shape[-2:])
    fft_pixels = prod(next_fast_len(int(n) + side - 1) for n in shape[-2:])
    symmetry = options['n_fold'] or 1
    symmetry_bytes = 512 * pixels * symmetry + 64 * symmetry ** 2 if (options['sym_mode'] != 'none' or options['reorder']) else 0
    fixed = 512 * pixels + 256 * fft_pixels + 64 * side ** 2 + symmetry_bytes
    per_pattern = 2 * pixels * np.dtype(dtype).itemsize
    if budget < fixed + per_pattern:
        raise ValueError(f'working_memory_mb is too small for peak detection; need at least {fixed + per_pattern} bytes.')
    max_patterns = min(prod(shape[:-2]), (budget - fixed) // per_pattern)
    if chunk_shape is not None:
        if isinstance(chunk_shape, (bool, np.bool_)) or not isinstance(chunk_shape, Integral) or chunk_shape < 1:
            raise ValueError('chunk_size must be a positive integer or None.')
        max_patterns = min(max_patterns, int(chunk_shape))
    return {'max_patterns': max_patterns, 'fixed_bytes': fixed, 'pattern_bytes': per_pattern,
            'estimated_peak_bytes': fixed + max_patterns * per_pattern}


def _empty_peak_entry(calibrated, with_intensities):
    entry = dict(coords=np.empty((0, 2), dtype=float if calibrated else np.int64),
                 scores=np.empty(0, dtype=float), synthetic_mask=np.empty(0, dtype=bool),
                 orbit_ids=np.empty(0, dtype=np.int64))
    if with_intensities:
        entry['intensities'] = np.empty(0, dtype=float)
    return entry


def _create_peak_tables(file, scan_shape, metadata, with_intensities):
    file.attrs['fourdenoise_format'] = _PEAK_SCAN_HDF5_FORMAT
    file.attrs['format_version'] = _PEAK_SCAN_VERSION
    file.attrs['scan_shape'] = scan_shape
    file.attrs['has_intensities'] = with_intensities
    options = {'compression': 'gzip', 'compression_opts': 4}
    for name, dtype in (('starts', np.int64), ('counts', np.int64), ('processed_mask', bool)):
        file.create_dataset(name, shape=scan_shape, dtype=dtype, chunks=True, **options)
    fields = dict(coords=np.float64 if metadata['coordinate_unit_mode'] == 'calibrated' else np.int64,
                  scores=np.float64, synthetic_mask=bool, orbit_ids=np.int64)
    if with_intensities:
        fields['intensities'] = np.float64
    for name, dtype in fields.items():
        extra = (2,) if name == 'coords' else ()
        file.create_dataset(name, shape=(0,) + extra, maxshape=(None,) + extra,
                            chunks=(4096,) + extra, dtype=dtype, **options)
    _write_hdf5_value(file, 'metadata', metadata, compression='gzip', compression_opts=4)


def _append_peak_entry(file, index, entry, processed):
    """Append aligned numeric fields and record the original scan index."""
    start, count = file['coords'].shape[0], len(entry['coords'])
    for name, values in entry.items():
        dataset = file[name]
        dataset.resize(start + count, axis=0)
        if count:
            dataset[start:start + count] = values
    file['starts'][index] = start
    file['counts'][index] = count
    file['processed_mask'][index] = processed


def _get_source_peaks(source, options, *, center=None, real_mask=None,
                      intensity_radius=None, coords_units='pixels', working_memory_mb=None,
                      chunk_size=None, destination=None, overwrite=False, atomic=True, progress=True):
    """Read bounded row-major scan blocks and reuse one ReciprocalSpace detector."""
    source._require_open()
    if source.dtype.kind not in 'biuf':
        raise TypeError('get_peaks requires real-valued diffraction intensities.')
    if source._metadata.get('polar_metadata') is not None or source._metadata.get('cepstral_metadata') is not None:
        raise ValueError('Streamed get_peaks requires Cartesian diffraction intensities.')
    for name, value in (('progress', progress), ('overwrite', overwrite), ('atomic', atomic)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean.')
    if destination is None and (overwrite or not atomic):
        raise ValueError('overwrite/atomic file controls require destination.')
    if intensity_radius is not None:
        intensity_radius = _finite_peak_scalar(intensity_radius, 'intensity_radius', positive=True)
    scan_shape = source.scan_shape
    if real_mask is not None:
        if source.ndim != 4:
            raise ValueError('real_mask is only supported for 4D scans.')
        real_mask = np.asarray(real_mask)
        if real_mask.shape != scan_shape or real_mask.dtype.kind != 'b':
            raise ValueError(f'real_mask must be Boolean with shape {scan_shape}.')
    units, scale, mode = _resolve_unit_mode(coords_units, source.reciprocal_units,
                                            source.reciprocal_conv_factor, label='coords_units')
    plan = _peak_source_plan(source.shape, source.dtype, options,
                             source._reader_working_memory(working_memory_mb), chunk_size)
    # A zero-stride placeholder supplies shape/calibration but allocates one scalar.
    detector = source._reciprocal_type(
        np.broadcast_to(np.zeros((), dtype=source.dtype), source.pattern_shape),
        units=source.reciprocal_units, conv_factor=source.reciprocal_conv_factor,
        center_beam_metadata=source._metadata.get('center_beam_metadata'),
    )
    center = detector._peak_search_center(center)
    options = dict(options, center=center, return_details=True)
    kernel = detector._peak_template(options['radius'], options['trench_width'],
                                      options['kernel_amp'], options['trench_amp'])
    detector._peak_search_geometry(source.pattern_shape, center, options['r_range'], kernel.shape[0] // 2)
    metadata = dict(scan_shape=scan_shape, pattern_shape=source.pattern_shape,
                    source_dtype=str(source.dtype), source_filename=source.filename,
                    source_backend=source.backend, source_dataset=source.dataset_path,
                    coordinate_order=('y', 'x'), coordinate_unit_mode=mode,
                    coordinate_units=units if mode == 'calibrated' else 'pixels',
                    coordinate_reference='geometric midpoint, y-up/x-right' if mode == 'calibrated' else 'array pixels, y-down/x-right',
                    search_center_pixels=center, detector_options=options,
                    intensity_radius_pixels=intensity_radius,
                    intensity_policy='raw float64 circular-window sums; synthetic entries NaN',
                    source_metadata=source.metadata)
    fields = _PEAK_FIELDS + (('intensities',) if intensity_radius is not None else ())
    result = None
    file = temporary = None
    if destination is None:
        result = PeakScanResult(
            **{name: np.empty(scan_shape, dtype=object) for name in fields},
            processed_mask=np.zeros(scan_shape, dtype=bool), metadata=metadata,
        )
    else:
        path = Path(destination).expanduser()
        if not path.suffix:
            path = path.with_suffix('.peaks.h5')
        path = _stream_destination(source.filename, path, overwrite=overwrite,
                                   atomic=atomic, progress=progress, compression='gzip')
        if path.suffix.lower() not in ('.h5', '.hdf5', '.hdf'):
            raise ValueError('Peak collections must use .h5/.hdf5/.hdf, not .4denoise.')
    try:
        if destination is not None:
            if atomic:
                descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
                os.close(descriptor)
                temporary = Path(name)
            file = h5py.File(temporary or path, 'w' if atomic or overwrite else 'x')
            _create_peak_tables(file, scan_shape, metadata, intensity_radius is not None)
        width = scan_shape[-1]
        max_columns = min(width, plan['max_patterns'])
        row_indices = range(scan_shape[0]) if source.ndim == 4 else (None,)
        with tqdm(total=prod(scan_shape), desc='Detecting peaks', disable=not progress) as bar:
            for row in row_indices:
                for start in range(0, width, max_columns):
                    source._require_open()
                    stop = min(start + max_columns, width)
                    scan_slices = (slice(row, row + 1), slice(start, stop)) if row is not None else (slice(start, stop),)
                    local_mask = None if real_mask is None else real_mask[scan_slices]
                    block = None
                    if local_mask is None or np.any(local_mask):
                        block = np.array(source._dataset[scan_slices + (slice(None), slice(None))], copy=True)
                    for j in range(stop - start):
                        index = (row, start + j) if row is not None else (start + j,)
                        local_index = (0, j) if row is not None else (j,)
                        processed = local_mask is None or bool(local_mask[local_index])
                        entry = _empty_peak_entry(mode == 'calibrated', intensity_radius is not None)
                        if processed:
                            dp = block[local_index]
                            try:
                                detected = detector.get_peaks(**options, _array=dp)
                            except (ValueError, TypeError, RuntimeError) as exc:
                                raise type(exc)(f'Peak detection failed at scan index {index}: {exc}') from exc
                            pixels = detected.coords
                            coords = pixels
                            if mode == 'calibrated':
                                coords = np.column_stack((((source.pattern_shape[0] - 1) / 2 - pixels[:, 0]) * scale,
                                                          (pixels[:, 1] - (source.pattern_shape[1] - 1) / 2) * scale))
                            entry = dict(coords=coords, scores=detected.scores,
                                         synthetic_mask=detected.synthetic_mask, orbit_ids=detected.orbit_ids)
                            if intensity_radius is not None:
                                intensities = np.full(len(pixels), np.nan)
                                observed = ~detected.synthetic_mask
                                if np.any(observed):
                                    intensities[observed] = detector.get_intensities(
                                        intensity_radius, centers=pixels[observed], _array=np.asarray(dp, dtype=np.float64),
                                    )
                                entry['intensities'] = intensities
                            del dp, detected, pixels, coords
                        if file is not None:
                            _append_peak_entry(file, index, entry, processed)
                        else:
                            for name, values in entry.items():
                                getattr(result, name)[index] = values
                            result.processed_mask[index] = processed
                        del entry
                        bar.update(1)
                    del block
        if file is not None:
            file.close()
            file = None
            if atomic:
                if overwrite:
                    os.replace(temporary, path)
                else:
                    os.link(temporary, path)
            return str(path)
        return result
    finally:
        if file is not None:
            file.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
