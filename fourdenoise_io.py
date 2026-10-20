"""Standalone HDF5 serialization, dataset selection, and schema reading."""

from pathlib import Path
import h5py
import numpy as np
from fourdenoise_traversals import _unfold_payload_arrays


_HYPERDATA_HDF5_FORMAT = '4denoise.hyperdata'
_PEAK_SCAN_HDF5_FORMAT = '4denoise.peak_scan'


_HYPERDATA_HDF5_VERSION = '1.1'


_HYPERDATA_HDF5_READABLE_VERSIONS = ('1.0', '1.1')


def _decode_hdf5_value(value):
    """Convert HDF5 byte strings and NumPy scalars to Python values."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.dtype == object:
        flat_values = value.ravel()
        if all(isinstance(item, (bytes, str)) for item in flat_values):
            decoded = [
                item.decode('utf-8') if isinstance(item, bytes) else item
                for item in flat_values
            ]
            return np.array(decoded, dtype=str).reshape(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hdf5_dataset_kwargs(value, compression, compression_opts):
    """Return compression kwargs only when HDF5 supports them for the value."""
    array = np.asarray(value)
    if compression is None or array.shape == ():
        return {}
    kwargs = {'compression': compression}
    if compression_opts is not None:
        kwargs['compression_opts'] = compression_opts
    return kwargs


def _write_hdf5_value(parent, name, value, compression=None,
                      compression_opts=None):
    """Recursively write a Python metadata value into an HDF5 group."""
    if value is None:
        group = parent.create_group(name)
        group.attrs['kind'] = 'none'
        return

    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs['kind'] = 'dict'
        for key, item in value.items():
            key_name = str(key)
            if '/' in key_name:
                raise ValueError(
                    "Metadata dictionary keys cannot contain '/'. "
                    f"Got key {key_name!r}."
                )
            _write_hdf5_value(
                group,
                key_name,
                item,
                compression=compression,
                compression_opts=compression_opts,
            )
        return

    if isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs['kind'] = 'tuple' if isinstance(value, tuple) else 'list'
        group.attrs['length'] = len(value)
        for idx, item in enumerate(value):
            _write_hdf5_value(
                group,
                f'item_{idx:08d}',
                item,
                compression=compression,
                compression_opts=compression_opts,
            )
        return

    if isinstance(value, str):
        dtype = h5py.string_dtype(encoding='utf-8')
        dataset = parent.create_dataset(name, data=value, dtype=dtype)
        dataset.attrs['kind'] = 'str'
        return

    if isinstance(value, bytes):
        dtype = h5py.string_dtype(encoding='utf-8')
        dataset = parent.create_dataset(
            name,
            data=value.decode('utf-8'),
            dtype=dtype,
        )
        dataset.attrs['kind'] = 'str'
        return

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            group = parent.create_group(name)
            group.attrs['kind'] = 'ndarray-object'
            group.attrs['shape'] = value.shape
            flat_values = value.ravel()
            group.attrs['length'] = flat_values.size
            for idx, item in enumerate(flat_values):
                _write_hdf5_value(
                    group,
                    f'item_{idx:08d}',
                    item,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            return

        if value.dtype.kind in {'U', 'S'}:
            dtype = h5py.string_dtype(encoding='utf-8')
            dataset = parent.create_dataset(
                name,
                # h5py accepts Python strings, not NumPy's fixed-width Unicode dtype.
                data=value.astype(str).astype(object),
                dtype=dtype,
            )
        else:
            dataset = parent.create_dataset(
                name,
                data=value,
                **_hdf5_dataset_kwargs(value, compression, compression_opts),
            )
        dataset.attrs['kind'] = 'ndarray'
        return

    if np.isscalar(value):
        dataset = parent.create_dataset(name, data=value)
        dataset.attrs['kind'] = 'scalar'
        return

    raise TypeError(
        f"Cannot save metadata value {name!r} with unsupported type "
        f"{type(value).__name__}."
    )


def _read_hdf5_value(parent, name):
    """Recursively read a Python metadata value from an HDF5 group."""
    obj = parent[name]
    kind = _decode_hdf5_value(obj.attrs.get('kind', 'ndarray'))

    if kind == 'none':
        return None

    if isinstance(obj, h5py.Dataset):
        value = obj[()]
        return _decode_hdf5_value(value)

    if kind == 'dict':
        return {
            key: _read_hdf5_value(obj, key)
            for key in obj.keys()
        }

    if kind in {'list', 'tuple'}:
        length = int(obj.attrs.get('length', len(obj.keys())))
        values = [
            _read_hdf5_value(obj, f'item_{idx:08d}')
            for idx in range(length)
        ]
        return tuple(values) if kind == 'tuple' else values

    if kind == 'ndarray-object':
        shape = tuple(int(v) for v in obj.attrs['shape'])
        length = int(obj.attrs.get('length', np.prod(shape)))
        values = [
            _read_hdf5_value(obj, f'item_{idx:08d}')
            for idx in range(length)
        ]
        return np.array(values, dtype=object).reshape(shape)

    raise ValueError(f"Unsupported HDF5 metadata kind {kind!r}.")


def _is_hyperdata_hdf5_file(filename):
    """Return True if filename is a 4Denoise HyperData HDF5 file."""
    path = Path(filename).expanduser()
    if not path.exists() or not path.is_file():
        return False
    try:
        with h5py.File(path, 'r') as file:
            file_format = _decode_hdf5_value(
                file.attrs.get('fourdenoise_format', None)
            )
            return file_format == _HYPERDATA_HDF5_FORMAT
    except OSError:
        return False


def _checked_hyperdata_format_version(file, path):
    """Validate the on-disk schema before reading saved HyperData metadata."""
    version = _decode_hdf5_value(file.attrs.get('format_version'))
    if version is None:
        raise ValueError(
            f"'{path}' is missing its 4Denoise format_version; the saved "
            "metadata schema cannot be determined safely."
        )
    if version not in _HYPERDATA_HDF5_READABLE_VERSIONS:
        raise ValueError(
            f"'{path}' uses unsupported 4Denoise format version "
            f"{version!r}; this installation can read versions "
            f"{', '.join(_HYPERDATA_HDF5_READABLE_VERSIONS)}."
        )
    return version


def _migrate_hyperdata_metadata(metadata, version):
    """Normalize each supported file schema to the current in-memory form."""
    if version in _HYPERDATA_HDF5_READABLE_VERSIONS:
        migrated = dict(metadata)
        migrated.setdefault('real_origin', (0.0, 0.0))
        if version == '1.0' and isinstance(migrated.get('unfold_metadata'), dict):
            unfolded = dict(migrated['unfold_metadata'])
            unfolded.setdefault(
                'preserved_values_nbytes',
                sum(values.nbytes for values in _unfold_payload_arrays(unfolded)),
            )
            migrated['unfold_metadata'] = unfolded
        return migrated
    raise ValueError(f"No metadata migration is defined for version {version!r}.")


def _load_hyperdata_hdf5(filename):
    """Load array and metadata from a 4Denoise HyperData HDF5 file."""
    path = Path(filename).expanduser()
    with h5py.File(path, 'r') as file:
        file_format = _decode_hdf5_value(
            file.attrs.get('fourdenoise_format', None)
        )
        if file_format != _HYPERDATA_HDF5_FORMAT:
            raise ValueError(
                f"'{path}' is not a 4Denoise HyperData file."
            )
        if 'array' not in file:
            raise ValueError(f"'{path}' does not contain a saved data array.")

        version = _checked_hyperdata_format_version(file, path)
        array = file['array'][()]
        metadata = {}
        if 'metadata' in file:
            metadata_group = file['metadata']
            metadata = {
                key: _read_hdf5_value(metadata_group, key)
                for key in metadata_group.keys()
            }
    return array, _migrate_hyperdata_metadata(metadata, version)


def _format_hdf5_dataset_listing(dataset_info):
    """Format discovered HDF5 datasets for selection and error messages."""
    if not dataset_info:
        return '  (no datasets found)'

    return '\n'.join(
        f"  {info['path']}: shape={info['shape']}, "
        f"ndim={info['ndim']}, dtype={info['dtype']}"
        for info in dataset_info
    )


def _select_hdf5_dataset(file, dataset_path=None, *,
                         selector_name='hdf5_dataset', exclude_internal=False):
    """Select a numeric dataset without loading it into memory."""
    if _decode_hdf5_value(file.attrs.get('fourdenoise_format')) == _PEAK_SCAN_HDF5_FORMAT:
        raise ValueError('This file contains a peak collection; use PeakScanResult.load(...) instead of HyperData.')
    path = Path(file.filename)
    dataset_info = []

    def _collect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_info.append({
                'path': f'/{name}', 'shape': tuple(obj.shape),
                'ndim': obj.ndim, 'dtype': str(obj.dtype),
                'is_candidate': obj.ndim >= 2 and (
                    np.issubdtype(obj.dtype, np.number)
                    or np.issubdtype(obj.dtype, np.bool_)
                ),
            })

    file.visititems(_collect_dataset)
    if dataset_path is not None:
        if not isinstance(dataset_path, (str, Path)):
            raise TypeError(
                f"{selector_name} must be a string or pathlib.Path."
            )
        selected_path = str(dataset_path).replace('\\', '/')
        if not selected_path.strip('/'):
            raise ValueError(f"{selector_name} must name a dataset.")
        if not selected_path.startswith('/'):
            selected_path = f'/{selected_path}'
        try:
            selected = file[selected_path]
        except KeyError as exc:
            listing = _format_hdf5_dataset_listing(dataset_info)
            raise KeyError(
                f"HDF5 dataset '{selected_path}' was not found in "
                f"'{path}'.\nAvailable datasets:\n{listing}"
            ) from exc
        if not isinstance(selected, h5py.Dataset):
            raise ValueError(
                f"HDF5 path '{selected_path}' refers to a group, not a dataset."
            )
        if selected.ndim < 2:
            raise ValueError(
                f"HDF5 dataset '{selected_path}' has {selected.ndim} "
                "dimension(s); HyperData requires at least 2."
            )
        if not (
            np.issubdtype(selected.dtype, np.number)
            or np.issubdtype(selected.dtype, np.bool_)
        ):
            raise TypeError(
                f"HDF5 dataset '{selected_path}' has non-numeric dtype "
                f"{selected.dtype}."
            )
        return selected

    candidates = [
        info for info in dataset_info
        if info['is_candidate'] and (
            not exclude_internal
            or not any(
                part.startswith('#') for part in info['path'].split('/')
            )
        )
    ]
    if len(candidates) == 1:
        return file[candidates[0]['path']]
    if not candidates:
        listing = _format_hdf5_dataset_listing(dataset_info)
        raise ValueError(
            f"No numeric dataset with at least 2 dimensions was found "
            f"in '{path}'.\nAvailable datasets:\n{listing}"
        )
    listing = _format_hdf5_dataset_listing(candidates)
    raise ValueError(
        f"{len(candidates)} numeric multidimensional datasets were "
        f"found in '{path}'. Select one with "
        f"HyperData(filename, {selector_name}='/path/to/dataset').\n"
        f"Candidate datasets:\n{listing}"
    )


def _read_hdf5_file(filename, dataset_path=None):
    """Load a selected generic HDF5 dataset into a NumPy array."""
    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file.")
    try:
        with h5py.File(path, 'r') as file:
            return _select_hdf5_dataset(file, dataset_path)[()]
    except OSError as exc:
        raise ValueError(f"Could not open '{path}' as an HDF5 file.") from exc
