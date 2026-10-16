"""Optional RosettaSciIO adapter and portable reader-metadata archives.

No optional libraries are imported until a reader is requested. Pixel arrays
are only sliced, never flattened or converted in full. Reader axes describe
storage coordinates; their zero is not assumed to be the direct-beam center.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime
import importlib
import inspect
from math import prod
from numbers import Integral, Real
from pathlib import Path
import warnings

import numpy as np


# Curate lazy-capable plugins rather than risk an eager-only reader ignoring lazy.
_LAZY_READER_APIS = frozenset({
    'rsciio.blockfile', 'rsciio.digitalmicrograph', 'rsciio.emd', 'rsciio.empad',
    'rsciio.tia', 'rsciio.hspy', 'rsciio.mrc', 'rsciio.numpy',
    'rsciio.quantumdetector', 'rsciio.ripple', 'rsciio.tiff', 'rsciio.tvips',
})
_ROLE_NAMES = {
    'ry': {'ry', 'scan_y', 'scan y', 'real_y', 'real y'},
    'rx': {'rx', 'scan_x', 'scan x', 'real_x', 'real x'},
    'ky': {'ky', 'k_y', 'qy', 'q_y', 'detector_y', 'detector y', 'reciprocal_y', 'reciprocal y'},
    'kx': {'kx', 'k_x', 'qx', 'q_x', 'detector_x', 'detector x', 'reciprocal_x', 'reciprocal x'},
    'y': {'y', 'height'}, 'x': {'x', 'width'},
}
_UNITS = {
    'm': ('length', 1.0), 'meter': ('length', 1.0), 'metre': ('length', 1.0),
    'mm': ('length', 1e-3), 'um': ('length', 1e-6),
    '\u00b5m': ('length', 1e-6), '\u03bcm': ('length', 1e-6),
    'nm': ('length', 1e-9), 'nanometer': ('length', 1e-9), 'nanometre': ('length', 1e-9),
    'pm': ('length', 1e-12), 'angstrom': ('length', 1e-10),
    'angstroms': ('length', 1e-10), '\u00e5': ('length', 1e-10),
    'rad': ('angle', 1.0), 'radian': ('angle', 1.0),
    'mrad': ('angle', 1e-3), 'milliradian': ('angle', 1e-3),
    'deg': ('angle', np.pi / 180), 'degree': ('angle', np.pi / 180),
    'degrees': ('angle', np.pi / 180),
}


def _unit_description(units):
    """Recognize microscopy lengths, inverse lengths and angular pixel scales."""
    if not isinstance(units, str):
        return None
    key = units.strip().lower().replace(' ', '').replace('\u2212', '-')
    inverse = False
    if key.startswith('1/'):
        key, inverse = key[2:].strip('()'), True
    elif key.endswith(('^-1', '**-1', '-1')):
        for suffix in ('**-1', '^-1', '-1'):
            if key.endswith(suffix):
                key, inverse = key[:-len(suffix)].strip('()'), True
                break
    if key == 'a' and inverse:
        key = 'angstrom'
    description = _UNITS.get(key)
    if description is None or (inverse and description[0] != 'length'):
        return None
    return ('inverse_length', 1 / description[1]) if inverse else description


def _pack_reader_value(value, depth=0):
    """Archive metadata without pickle or unsafe HDF5 dictionary key names."""
    if depth > 64:
        raise ValueError('RosettaSciIO metadata exceeds 64 levels of nesting.')
    if isinstance(value, Mapping):
        return {'kind': 'mapping', 'items': [
            (_pack_reader_value(key, depth + 1), _pack_reader_value(item, depth + 1))
            for key, item in value.items()
        ]}
    if isinstance(value, (tuple, list)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [_pack_reader_value(item, depth + 1) for item in value]}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'O':
            return {'kind': 'object_array', 'shape': value.shape,
                    'items': [_pack_reader_value(item, depth + 1) for item in value.flat]}
        if value.dtype.kind not in 'biufcUS':
            raise TypeError(f'Unsupported RosettaSciIO metadata array dtype {value.dtype}.')
        if value.dtype.kind == 'S':
            return {'kind': 'byte_array', 'shape': value.shape,
                    'dtype': value.dtype.str, 'value': np.frombuffer(value.tobytes(), dtype=np.uint8).copy()}
        return {'kind': 'array', 'value': value.copy()}
    if isinstance(value, np.generic):
        return _pack_reader_value(value.item(), depth + 1)
    if isinstance(value, bytes):
        return {'kind': 'bytes', 'value': np.frombuffer(value, dtype=np.uint8).copy()}
    if isinstance(value, (datetime, date)):
        return {'kind': 'datetime' if isinstance(value, datetime) else 'date', 'value': value.isoformat()}
    if isinstance(value, Path):
        return {'kind': 'path', 'value': str(value)}
    if isinstance(value, int) and not -(1 << 63) <= value < (1 << 64):
        return {'kind': 'large_int', 'value': str(value)}
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return {'kind': 'scalar', 'value': value}
    raise TypeError(f'Unsupported RosettaSciIO metadata type {type(value).__name__}; no pickle fallback is used.')


def _unpack_reader_value(value, depth=0):
    """Restore a portable metadata archive without executing stored content."""
    if depth > 64 or not isinstance(value, dict):
        raise ValueError('Invalid RosettaSciIO metadata archive.')
    kind = value.get('kind')
    if kind == 'mapping':
        return {_unpack_reader_value(key, depth + 1): _unpack_reader_value(item, depth + 1)
                for key, item in value['items']}
    if kind in ('list', 'tuple', 'object_array'):
        items = [_unpack_reader_value(item, depth + 1) for item in value['items']]
        if kind == 'object_array':
            array = np.empty(len(items), dtype=object)
            array[:] = items
            return array.reshape(value['shape'])
        return tuple(items) if kind == 'tuple' else items
    if kind == 'array':
        return np.array(value['value'], copy=True)
    if kind in ('bytes', 'byte_array'):
        payload = np.asarray(value['value'], dtype=np.uint8).tobytes()
        return payload if kind == 'bytes' else np.frombuffer(payload, dtype=np.dtype(value['dtype'])).copy().reshape(value['shape'])
    if kind == 'datetime':
        return datetime.fromisoformat(value['value'])
    if kind == 'date':
        return date.fromisoformat(value['value'])
    if kind == 'path':
        return Path(value['value'])
    if kind == 'large_int':
        return int(value['value'])
    if kind == 'scalar':
        return value['value']
    raise ValueError(f'Unknown RosettaSciIO metadata archive kind {kind!r}.')


def unpack_reader_metadata(acquisition_metadata):
    """Return independent RosettaSciIO axes and metadata from an acquisition record.

    Accept ``data.acquisition_metadata`` or
    ``source.metadata['acquisition_metadata']``. Returns a dictionary with
    ``axes``, ``metadata`` and ``original_metadata``; returns None when the
    record has no RosettaSciIO archive. Vendor keys containing slashes, bytes,
    arrays and timestamps survive HyperData save/load without pickle. Source
    axis descriptions remain in original storage order; the acquisition
    record separately stores the applied ``axis_order``.
    """
    if not isinstance(acquisition_metadata, dict):
        raise TypeError('acquisition_metadata must be a dictionary.')
    archive = acquisition_metadata.get('rosetta_metadata')
    if archive is None:
        return None
    if archive.get('version') != 1:
        raise ValueError('Unsupported RosettaSciIO metadata archive version.')
    return {key: _unpack_reader_value(archive[key])
            for key in ('axes', 'metadata', 'original_metadata')}


def _axes_by_index(record, shape):
    """Validate axis descriptors using index_in_array, not dictionary order."""
    axes = record.get('axes', [])
    if not isinstance(axes, (list, tuple)):
        raise ValueError('RosettaSciIO axes must be a list of dictionaries.')
    indexed = {}
    for axis in axes:
        if not isinstance(axis, Mapping):
            raise ValueError('RosettaSciIO axes must contain dictionaries.')
        index = axis.get('index_in_array')
        if not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(shape):
            raise ValueError('Each RosettaSciIO axis requires a valid integer index_in_array.')
        if int(index) in indexed:
            raise ValueError('RosettaSciIO axes have duplicate index_in_array values.')
        size = axis.get('size')
        if size is not None and (not isinstance(size, Integral) or isinstance(size, (bool, np.bool_)) or size != shape[index]):
            raise ValueError(f'Axis {index} size does not match the RosettaSciIO data shape.')
        indexed[int(index)] = dict(axis)
    return indexed


def _name_matches(axis, role):
    return str(axis.get('name', '')).strip().lower() in _ROLE_NAMES[role]


def _resolve_axis_order(record, shape, axis_order):
    """Infer y/x only from explicit roles or navigation flags plus y/x labels."""
    axes = _axes_by_index(record, shape)
    ndim = len(shape)
    if axis_order is not None:
        try:
            order = tuple(axis_order)
        except TypeError as exc:
            raise ValueError('axis_order must be a permutation of source axis indices.') from exc
        if len(order) != ndim or any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_)) for i in order):
            raise ValueError(f'axis_order must contain {ndim} integer source axis indices.')
        order = tuple(int(i) for i in order)
        if set(order) != set(range(ndim)):
            raise ValueError(f'axis_order must be a permutation of range({ndim}).')
        return order, axes
    roles = ('ry', 'rx', 'ky', 'kx') if ndim == 4 else ('ky', 'kx')
    matched = [[i for i, axis in axes.items() if _name_matches(axis, role)] for role in roles]
    if all(len(items) == 1 for items in matched):
        order = tuple(items[0] for items in matched)
        if ndim == 3:
            remaining = tuple(i for i in range(ndim) if i not in order)
            order = remaining + order
        if len(set(order)) == ndim:
            return order, axes
    navigation = [i for i, axis in axes.items() if isinstance(axis.get('navigate'), (bool, np.bool_)) and axis['navigate']]
    signal = [i for i, axis in axes.items() if isinstance(axis.get('navigate'), (bool, np.bool_)) and not axis['navigate']]
    if len(navigation) == ndim - 2 and len(signal) == 2:
        def yx(group):
            y = [i for i in group if _name_matches(axes[i], 'y')]
            x = [i for i in group if _name_matches(axes[i], 'x')]
            return (y[0], x[0]) if len(y) == len(x) == 1 else None
        detector_order = yx(signal)
        scan_order = tuple(navigation) if ndim == 3 else yx(navigation)
        if detector_order is not None and scan_order is not None:
            return scan_order + detector_order, axes
    labels = ', '.join(f"{i}: {axis.get('name')!r}, navigate={axis.get('navigate')!r}" for i, axis in axes.items())
    raise ValueError(
        'Cannot determine scan/detector y/x axes unambiguously. '
        f'Supply axis_order for target {"(Ry, Rx, Ky, Kx)" if ndim == 4 else "(N, Ky, Kx)"}. '
        f'Source axes: {labels or "not provided"}.'
    )


def _uniform_axis(axis, size):
    """Extract regular axis spacing and origin without accepting nonlinear axes."""
    if not axis:
        return None
    if 'axis' in axis:
        values = np.asarray(axis['axis'])
        if values.ndim != 1 or values.size != size or values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)):
            return None
        if size < 2:
            return None
        scale = float(values[1] - values[0])
        if scale == 0 or not np.allclose(np.diff(values), scale, rtol=1e-7, atol=abs(scale) * 1e-9):
            return None
        return scale, float(values[0]), axis.get('units')
    scale, offset = axis.get('scale'), axis.get('offset', 0.0)
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) or not np.isfinite(v) for v in (scale, offset)) or scale == 0:
        return None
    return float(scale), float(offset), axis.get('units')


def _calibration_from_axes(axes, order, raw_shape):
    """Import supported pixel spacings; retain nonrepresentable axes descriptively."""
    result, notes = {}, []
    domains = [('reciprocal', order[-2:])]
    if len(order) == 4:
        domains.insert(0, ('real', order[:2]))
    for domain, indices in domains:
        descriptions = [_uniform_axis(axes.get(i), raw_shape[i]) for i in indices]
        if any(item is None for item in descriptions):
            if any(axes.get(i) for i in indices):
                notes.append(f'{domain} axes are incomplete or nonuniform; automatic calibration unavailable, use pixel units or explicit overrides.')
            continue
        y_axis, x_axis = descriptions
        y_unit, x_unit = _unit_description(y_axis[2]), _unit_description(x_axis[2])
        allowed = ('length',) if domain == 'real' else ('inverse_length', 'angle')
        if y_unit is None or x_unit is None or y_unit[0] != x_unit[0] or y_unit[0] not in allowed:
            if any(units not in (None, '', 'pixels', 'pixel', 'px') for _, _, units in descriptions):
                notes.append(f'{domain} axis units are unknown or incompatible; automatic calibration unavailable, use pixel units or explicit overrides.')
            continue
        conversion = x_unit[1] / y_unit[1]
        y_scale, x_scale = y_axis[0], x_axis[0] * conversion
        if domain == 'real':
            result.update(real_units=y_axis[2].strip(), real_conv_factor=(y_scale, x_scale),
                          real_origin=(y_axis[1], x_axis[1] * conversion))
        elif np.isclose(abs(y_scale), abs(x_scale), rtol=1e-7, atol=0):
            result.update(reciprocal_units=y_axis[2].strip(), reciprocal_conv_factor=abs(y_scale))
        else:
            notes.append('reciprocal axes have unequal pixel-size magnitudes; automatic scalar calibration cannot represent them, use pixel units or explicit overrides.')
    return result, notes


def _select_record(records, dataset):
    """Select one diffraction record, reporting every returned dataset on ambiguity."""
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, (list, tuple)) or any(not isinstance(record, Mapping) for record in records):
        raise TypeError('RosettaSciIO file_reader must return dictionaries or a list of dictionaries.')
    headers = []
    candidates = []
    for index, record in enumerate(records):
        array = record.get('data')
        shape = tuple(getattr(array, 'shape', ()))
        dtype = np.dtype(array.dtype) if hasattr(array, 'dtype') else None
        title = record.get('metadata', {}).get('General', {}).get('title', '')
        headers.append(f'  {index}: title={title!r}, shape={shape}, ndim={len(shape)}, dtype={dtype}')
        if len(shape) in (3, 4) and all(isinstance(n, Integral) and n > 0 for n in shape) and dtype is not None and dtype.kind in 'biufc':
            candidates.append(index)
    listing = '\n'.join(headers) or '  (no datasets returned)'
    if dataset is None:
        if len(candidates) != 1:
            raise ValueError(f'{len(candidates)} suitable 3D/4D datasets found. Select one with dataset=index or a unique title.\nDatasets:\n{listing}')
        index = candidates[0]
    elif isinstance(dataset, Integral) and not isinstance(dataset, (bool, np.bool_)):
        index = int(dataset)
        if index not in candidates:
            raise ValueError(f'dataset={index} does not select a suitable 3D/4D numeric array.\nDatasets:\n{listing}')
    elif isinstance(dataset, str):
        matches = [i for i in candidates if records[i].get('metadata', {}).get('General', {}).get('title') == dataset]
        if len(matches) != 1:
            raise ValueError(f'dataset title {dataset!r} must select exactly one suitable array; use an index.\nDatasets:\n{listing}')
        index = matches[0]
    else:
        raise TypeError('dataset must be a nonnegative integer index, unique title, or None.')
    return records[index], index, records


def _resolve_reader(path, reader):
    """Use the official plugin registry and import only the selected plugin."""
    try:
        rsciio = importlib.import_module('rsciio')
    except ImportError as exc:
        raise ImportError('RosettaSciIO is optional. Install it with python -m pip install -e ".[rosetta]" from the repository root.') from exc
    registry = rsciio.IO_PLUGINS
    if reader is None:
        extension = path.suffix.lstrip('.').lower()
        matches = [item for item in registry if extension in {ext.lower().lstrip('.') for ext in item.get('file_extensions', [])}]
    else:
        if not isinstance(reader, str) or not reader.strip():
            raise ValueError('reader must name a registered RosettaSciIO plugin.')
        key = reader.strip().lower()
        matches = [item for item in registry if key in {
            item['name'].lower(), item['api'].lower(), item['api'].split('.')[-1].lower(),
            *(alias.lower() for alias in item.get('name_aliases', [])),
        }]
    if len(matches) != 1:
        options = ', '.join(item['name'] for item in (matches or registry))
        raise ValueError(f'Cannot select one RosettaSciIO reader for {path.suffix!r}; use reader="plugin_name". Candidates: {options}.')
    plugin = matches[0]
    if plugin['api'] not in _LAZY_READER_APIS:
        raise NotImplementedError(f"Reader {plugin['name']!r} is not in 4Denoise's curated lazy-capable plugin set; no eager fallback will be attempted.")
    module = importlib.import_module(plugin['api'])
    function = module.file_reader
    if 'lazy' not in inspect.signature(function).parameters:
        raise NotImplementedError(f"Reader {plugin['name']!r} does not advertise a lazy parameter; no eager fallback will be attempted.")
    return rsciio, plugin, function


def _close_reader_resources(records):
    """Build one deduplicated close action for all returned lazy records."""
    from rsciio.utils.file import get_file_handle
    resources = []
    seen = set()
    for record in records:
        data = record.get('data') if isinstance(record, Mapping) else None
        if data is None:
            continue
        handle = data._mmap if isinstance(data, np.memmap) else None
        if handle is None and hasattr(data, '__dask_graph__'):
            handle = get_file_handle(data, warn=False)
        if handle is not None and id(handle) not in seen:
            seen.add(id(handle))
            resources.append(handle)

    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        failures = []
        for resource in resources:
            try:
                resource.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError('Failed to close a RosettaSciIO file resource.') from failures[0]
    return close


class _RosettaArray:
    """Basic-slice adapter that computes only requested lazy storage blocks."""

    def __init__(self, array, order):
        if not isinstance(array, np.memmap) and not (
            hasattr(array, '__dask_graph__') and callable(getattr(array, 'compute', None))
        ):
            raise NotImplementedError('RosettaSciIO returned an eager array despite lazy=True; no eager fallback is supported by HyperData.open.')
        self._array = array
        self._order = order
        self.shape = tuple(int(array.shape[i]) for i in order)
        self.dtype = np.dtype(array.dtype)
        chunks = getattr(array, 'chunks', None)
        self.reader_chunk_nbytes = 0 if chunks is None else prod(max(axis) for axis in chunks) * self.dtype.itemsize

    def __array__(self, *args, **kwargs):
        raise TypeError('RosettaSciIO source arrays must be read through basic slices.')

    def __getitem__(self, selection):
        if not isinstance(selection, tuple) or len(selection) != len(self.shape) or any(
            not isinstance(item, (slice, Integral)) or isinstance(item, (bool, np.bool_)) for item in selection
        ):
            raise TypeError('RosettaSciIO source reads require one basic integer/slice per axis.')
        raw_selection = [None] * len(selection)
        for axis, item in zip(self._order, selection):
            raw_selection[axis] = item
        block = self._array[tuple(raw_selection)]
        if hasattr(block, 'compute'):
            block = block.compute(scheduler='synchronous')
        remaining_raw_axes = [i for i, item in enumerate(raw_selection) if isinstance(item, slice)]
        remaining_target_axes = [i for i, item in zip(self._order, selection) if isinstance(item, slice)]
        permutation = tuple(remaining_raw_axes.index(i) for i in remaining_target_axes)
        return np.transpose(block, permutation)


@contextmanager
def _open_rosetta_data(path, *, reader=None, dataset=None, axis_order=None, reader_kwargs=None):
    """Yield canonical sliceable data and imported metadata; close owned handles."""
    if reader_kwargs is None:
        reader_kwargs = {}
    if not isinstance(reader_kwargs, dict) or any(not isinstance(key, str) for key in reader_kwargs):
        raise TypeError('reader_kwargs must be a dictionary with string keys.')
    if {'lazy', 'filename'} & reader_kwargs.keys():
        raise ValueError('lazy and filename are managed by HyperData.open, not reader_kwargs.')
    for option in ('mode', 'mmap_mode'):
        if option in reader_kwargs and reader_kwargs[option] not in (None, 'r'):
            raise ValueError(f'{option} must be read-only (r) for HyperData.open.')
    rsciio, plugin, function = _resolve_reader(path, reader)
    import dask
    records, close = None, None
    try:
        with dask.config.set({'array.chunk-size': '16MiB'}):
            records = function(str(path), lazy=True, **reader_kwargs)
        record_list = [records] if isinstance(records, Mapping) else records
        if isinstance(record_list, (list, tuple)):
            close = _close_reader_resources(record_list)
        record, index, _ = _select_record(records, dataset)
        raw = record['data']
        order, axes = _resolve_axis_order(record, raw.shape, axis_order)
        array = _RosettaArray(raw, order)
        metadata, notes = _calibration_from_axes(axes, order, raw.shape)
        metadata['acquisition_metadata'] = {
            'format': 'rosettasciio', 'source_path': str(path.resolve()),
            'reader': plugin['api'], 'reader_version': str(rsciio.__version__),
            'dataset_index': index, 'acquisition_name': record.get('metadata', {}).get('General', {}).get('title', ''),
            'source_shape': tuple(raw.shape), 'axis_order': order,
            'calibration_notes': notes,
            'detector_coordinate_policy': (
                'Only detector pixel-size magnitude is imported. Original offsets/signs remain '
                'descriptive; no beam center or R/Q calibration is inferred.'
            ),
            'rosetta_metadata': {'version': 1, **{
                key: _pack_reader_value(record.get(key, [] if key == 'axes' else {}))
                for key in ('axes', 'metadata', 'original_metadata')
            }},
        }
        if len(order) == 4:
            metadata['acquisition_metadata']['scan_shape'] = array.shape[:2]
        for note in notes:
            warnings.warn(f'RosettaSciIO calibration: {note}', UserWarning, stacklevel=3)
        yield array, metadata, f'[{index}]', close
    finally:
        if close is not None:
            close()
