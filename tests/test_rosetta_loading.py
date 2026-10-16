"""Optional RosettaSciIO reader contracts, axes, ownership and real plugins."""

from contextlib import nullcontext
from datetime import date, datetime
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_rosetta as ros


class FakeHandle:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class LazyArray:
    """Track lazy reads while rejecting conversion of the full mock source."""

    def __init__(self, values, *, chunks=None, handle=None, calls=None):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.chunks = chunks or tuple((n,) for n in self.shape)
        self.handle = handle
        self.calls = calls if calls is not None else []

    def __dask_graph__(self):
        return {}

    def __array__(self, *args, **kwargs):
        raise AssertionError('Do not convert the complete lazy source.')

    def __getitem__(self, selection):
        self.calls.append(('slice', selection))
        return LazyArray(self.values[selection], handle=self.handle, calls=self.calls)

    def compute(self, scheduler=None):
        self.calls.append(('compute', scheduler))
        return np.array(self.values, copy=True)


def axes(shape, names=None, navigation=None):
    names = names or (['Ry', 'Rx', 'Ky', 'Kx'] if len(shape) == 4 else ['frame', 'Ky', 'Kx'])
    return [dict(index_in_array=i, size=n, name=name, scale=1.0, offset=0.0,
                 units='pixels', **({} if navigation is None else {'navigate': navigation[i]}))
            for i, (n, name) in enumerate(zip(shape, names))]


def record(values, title='scan', *, descriptors=None, lazy=True, handle=None):
    return {'data': LazyArray(values, handle=handle) if lazy else values,
            'axes': descriptors if descriptors is not None else axes(values.shape),
            'metadata': {'General': {'title': title}}, 'original_metadata': {}}


class RosettaMockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'scan.dm4'
        self.path.touch()
        self.values = np.arange(3 * 4 * 5 * 6, dtype=np.int16).reshape(3, 4, 5, 6)
        self.handle = FakeHandle()
        self.records = [record(self.values, handle=self.handle)]
        self.reader_calls = []

        def file_reader(filename, lazy=False, chunks='auto', **kwargs):
            self.reader_calls.append((filename, lazy, chunks, kwargs))
            return self.records

        self.rsciio = types.ModuleType('rsciio')
        self.rsciio.__version__ = 'test'
        self.rsciio.IO_PLUGINS = [
            {'name': 'Digital Micrograph', 'name_aliases': ['DM'],
             'api': 'rsciio.digitalmicrograph', 'file_extensions': ['dm3', 'dm4']},
            {'name': 'Arina', 'name_aliases': [], 'api': 'rsciio.arina', 'file_extensions': ['h5']},
        ]
        self.plugin = types.ModuleType('rsciio.digitalmicrograph')
        self.plugin.file_reader = file_reader
        file_utils = types.ModuleType('rsciio.utils.file')
        file_utils.get_file_handle = lambda data, warn=False: data.handle
        dask = types.ModuleType('dask')
        dask.config = types.SimpleNamespace(set=lambda config: nullcontext())
        self.modules = {'rsciio': self.rsciio, 'rsciio.digitalmicrograph': self.plugin,
                        'rsciio.utils.file': file_utils, 'dask': dask}
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_automatic_reader_no_computation_until_selection(self):
        array = self.records[0]['data']
        with fd.HyperData.open(self.path) as source:
            self.assertEqual(source.backend, 'rosetta')
            self.assertEqual(source.shape, self.values.shape)
            self.assertEqual(source.dataset_path, '[0]')
            self.assertEqual(source.reader_chunk_nbytes, self.values.nbytes)
            self.assertEqual(array.calls, [])
            self.assertTrue(self.reader_calls[0][1])
            dp = source.get_dp(1, 2)
            np.testing.assert_array_equal(dp.array, self.values[1, 2])
            self.assertEqual(array.calls, [('slice', (1, 2, slice(None), slice(None))),
                                           ('compute', 'synchronous')])
        self.assertEqual(self.handle.close_count, 1)
        np.testing.assert_array_equal(dp.array, self.values[1, 2])

    def test_registry_names_aliases_and_explicit_backend(self):
        for name in ('DM', 'digitalmicrograph', 'rsciio.digitalmicrograph', 'Digital Micrograph'):
            with self.subTest(name=name), fd.HyperData.open(self.path, reader=name,
                                                           backend='rosetta') as source:
                np.testing.assert_array_equal(source.get_dp(0, 0).array, self.values[0, 0])
        with self.assertRaisesRegex(ValueError, 'Native'):
            with fd.HyperData.open(self.path, backend='native'):
                pass

    def test_ambiguous_plugins_and_lazy_only_preflight(self):
        self.rsciio.IO_PLUGINS.append(dict(self.rsciio.IO_PLUGINS[0], name='Another DM'))
        with self.assertRaisesRegex(ValueError, 'Candidates'):
            with fd.HyperData.open(self.path):
                pass
        self.assertEqual(self.reader_calls, [])
        self.rsciio.IO_PLUGINS.pop()
        with self.assertRaisesRegex(NotImplementedError, 'eager fallback'):
            with fd.HyperData.open(self.path, reader='Arina'):
                pass
        self.assertEqual(self.reader_calls, [])
        self.plugin.file_reader = lambda filename: self.records
        with self.assertRaisesRegex(NotImplementedError, 'lazy parameter'):
            with fd.HyperData.open(self.path, reader='DM'):
                pass

    def test_missing_dependency_has_installation_guidance(self):
        original = ros.importlib.import_module

        def missing(name):
            if name == 'rsciio':
                raise ImportError('deliberately absent')
            return original(name)

        with patch.object(ros.importlib, 'import_module', side_effect=missing):
            with self.assertRaisesRegex(ImportError, r'\[rosetta\]'):
                with fd.HyperData.open(self.path):
                    pass

    def test_reader_options_are_forwarded_but_managed_options_rejected(self):
        with fd.HyperData.open(self.path, reader_kwargs={'chunks': (1, 1, 5, 6), 'option': 4}):
            pass
        self.assertEqual(self.reader_calls[0][2:], ((1, 1, 5, 6), {'option': 4}))
        for options in ({'lazy': False}, {'filename': 'other'}, {'lazy': True}):
            with self.assertRaisesRegex(ValueError, 'managed'):
                with fd.HyperData.open(self.path, reader_kwargs=options):
                    pass
        for options in ([], {1: 2}):
            with self.assertRaisesRegex(TypeError, 'reader_kwargs'):
                with fd.HyperData.open(self.path, reader_kwargs=options):
                    pass
        with self.assertRaisesRegex(ValueError, 'hdf5_dataset'):
            with fd.HyperData.open(self.path, hdf5_dataset='/array'):
                pass

    def test_writable_reader_modes_are_rejected_before_opening(self):
        for option in ('mode', 'mmap_mode'):
            for mode in ('a', 'w', 'r+', 'c'):
                with self.assertRaisesRegex(ValueError, 'read-only'):
                    with fd.HyperData.open(self.path, reader_kwargs={option: mode}):
                        pass
        self.assertEqual(self.reader_calls, [])

    def test_thumbnail_is_not_selected_and_multiple_arrays_are_reported(self):
        self.records.insert(0, record(np.ones((2, 2)), 'thumbnail', handle=self.handle))
        with fd.HyperData.open(self.path) as source:
            self.assertEqual(source.dataset_path, '[1]')
        self.records.append(record(self.values + 1, 'second', handle=self.handle))
        with self.assertRaisesRegex(ValueError, r'2 suitable.*') as caught:
            with fd.HyperData.open(self.path):
                pass
        self.assertIn('thumbnail', str(caught.exception))
        self.assertIn(str(self.values.shape), str(caught.exception))
        for selector in (1, np.int64(1), 'scan'):
            with fd.HyperData.open(self.path, dataset=selector) as source:
                np.testing.assert_array_equal(source.load().array, self.values)

    def test_dataset_invalid_indices_types_and_duplicate_titles(self):
        self.records.append(record(self.values, handle=self.handle))
        for selector in (-1, 3, True, 1.2, ['scan'], 'missing', 'scan'):
            with self.subTest(selector=selector), self.assertRaises((ValueError, TypeError)):
                with fd.HyperData.open(self.path, dataset=selector):
                    pass

    def test_only_numeric_nonempty_3d_or_4d_records_are_candidates(self):
        for array in (np.ones((3, 4)), np.ones((2, 2, 2, 2, 2)),
                      np.zeros((0, 3, 4)), np.full((2, 3, 4), 'text')):
            self.records = [record(array, handle=self.handle)]
            with self.assertRaisesRegex(ValueError, '0 suitable'):
                with fd.HyperData.open(self.path):
                    pass

    def test_explicit_axis_permutation_is_applied_to_every_read_path(self):
        raw = np.transpose(self.values, (3, 1, 0, 2))
        self.records = [record(raw, descriptors=axes(raw.shape, names=['a', 'b', 'c', 'd']), handle=self.handle)]
        with self.assertRaisesRegex(ValueError, 'axis_order'):
            with fd.HyperData.open(self.path):
                pass
        with fd.HyperData.open(self.path, axis_order=(2, 1, 3, 0)) as source:
            np.testing.assert_array_equal(source.get_dp(1, 2).array, self.values[1, 2])
            np.testing.assert_allclose(source.get_dp(operation='mean').array, self.values.mean(axis=(0, 1)))
            mask = np.zeros(self.values.shape[:2], dtype=bool)
            mask[1, 2] = True
            np.testing.assert_array_equal(source.get_dp(mask=mask).array, self.values[1, 2])
            for slices, block in source.iter_chunks((2, 3)):
                np.testing.assert_array_equal(block.array, self.values[slices])
            np.testing.assert_array_equal(source.load().array, self.values)

    def test_navigation_and_yx_labels_infer_nonstandard_storage_order(self):
        raw = np.transpose(self.values, (3, 1, 0, 2))
        descriptors = axes(raw.shape, ['x', 'x', 'y', 'y'], [False, True, True, False])
        self.records = [record(raw, descriptors=descriptors, handle=self.handle)]
        with fd.HyperData.open(self.path) as source:
            np.testing.assert_array_equal(source.load().array, self.values)
            self.assertEqual(source.metadata['acquisition_metadata']['axis_order'], (2, 1, 3, 0))

    def test_three_dimensional_stack_permutation(self):
        values = self.values.reshape(12, 5, 6)
        raw = values.transpose(2, 0, 1)
        self.records = [record(raw, descriptors=axes(raw.shape, ['width', 'depth', 'height'], [False, True, False]), handle=self.handle)]
        with fd.HyperData.open(self.path) as source:
            self.assertEqual(source.scan_shape, (12,))
            self.assertIsNone(source.real_shape)
            np.testing.assert_array_equal(source.get_dp(8).array, values[8])
            np.testing.assert_array_equal(source.load().array, values)

    def test_invalid_axis_orders_close_handles_before_exposing_a_source(self):
        for order in ((0, 1), (0, 1, 2, 2), (0, 1, 2, 4), (False, 1, 2, 3), 0, (0, 1, 2, 3.0)):
            before = self.handle.close_count
            with self.assertRaisesRegex(ValueError, 'axis_order'):
                with fd.HyperData.open(self.path, axis_order=order):
                    pass
            self.assertEqual(self.handle.close_count, before + 1)

    def test_eager_only_result_rejected_and_owning_handles_closed(self):
        self.records = [record(self.values, lazy=False)]
        with self.assertRaisesRegex(NotImplementedError, 'eager array'):
            with fd.HyperData.open(self.path):
                pass

    def test_record_and_axes_validation(self):
        for bad_axes in ('not axes', [0], [dict(index_in_array=9)],
                         [dict(index_in_array=0), dict(index_in_array=0)],
                         [dict(index_in_array=0, size=100)]):
            self.records[0]['axes'] = bad_axes
            with self.assertRaises(ValueError):
                with fd.HyperData.open(self.path, axis_order=(0, 1, 2, 3)):
                    pass
        self.records = None
        with self.assertRaisesRegex(TypeError, 'file_reader'):
            with fd.HyperData.open(self.path):
                pass

    def test_no_fake_beam_or_rq_calibration_is_imported(self):
        self.records[0]['original_metadata'] = {'scan_rotation': 87.3, 'center': [1, 2]}
        with fd.HyperData.open(self.path) as source:
            data = source.load()
            self.assertIsNone(data.center_beam_metadata)
            self.assertIsNone(data.rq_calibration)
            self.assertEqual(source.reader_metadata['original_metadata']['scan_rotation'], 87.3)

    def test_imported_scales_offsets_units_and_signed_scan_spacing(self):
        descriptors = self.records[0]['axes']
        for axis, unit, scale, offset in zip(descriptors,
                                            ['nm', 'angstrom', 'mrad', 'rad'],
                                            [2, -30, 0.5, 0.0005], [10, 200, -2, 100]):
            axis.update(units=unit, scale=scale, offset=offset)
        with fd.HyperData.open(self.path) as source:
            self.assertEqual(source.real_units, 'nm')
            np.testing.assert_allclose(source.real_conv_factor, (2, -3))
            np.testing.assert_allclose(source.real_origin, (10, 20))
            self.assertEqual(source.reciprocal_units, 'mrad')
            self.assertEqual(source.reciprocal_conv_factor, 0.5)
            np.testing.assert_array_equal(source.get_dp(12, 14, selection_units='calibrated').array,
                                          self.values[1, 2])
            slices, block = list(source.iter_chunks((2, 3)))[-1]
            np.testing.assert_allclose(block.real_origin, (14, 11))
            self.assertIsNone(block.center_beam_metadata)

    def test_unsupported_calibration_warns_and_preserves_original_axes(self):
        descriptors = self.records[0]['axes']
        descriptors[2].update(units='mrad', scale=0.5)
        descriptors[3].update(units='mrad', scale=0.8)
        with self.assertWarnsRegex(UserWarning, 'unequal'), fd.HyperData.open(self.path) as source:
            self.assertIsNone(source.reciprocal_conv_factor)
            self.assertEqual(source.reader_metadata['axes'][3]['scale'], 0.8)
        descriptors[3].update(units='eV')
        with self.assertWarnsRegex(UserWarning, 'incompatible'), fd.HyperData.open(self.path):
            pass

    def test_reader_chunk_reserved_before_any_pixel_work(self):
        raw = self.records[0]['data']
        with fd.HyperData.open(self.path) as source:
            size = source.reader_chunk_nbytes
            for action in (source.get_dp, lambda **kw: next(source.iter_chunks(**kw)), source.load):
                with self.assertRaisesRegex(ValueError, 'reader chunk'):
                    action(working_memory_mb=size / 1024 ** 2)
            self.assertEqual(raw.calls, [])
            with self.assertRaisesRegex(MemoryError, 'max_memory_mb'):
                source.load(max_memory_mb=(source.nbytes - 1) / 1024 ** 2)
            self.assertEqual(raw.calls, [])
            np.testing.assert_array_equal(source.load(working_memory_mb=0.01).array, self.values)

    def test_random_selection_and_stored_data_lifetime(self):
        with fd.HyperData.open(self.path) as source:
            with patch.object(np.random, 'randint', side_effect=[1, 2]):
                dp = source.get_dp(operation='random')
            source.close()
            source.close()
            self.assertEqual(self.handle.close_count, 1)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                source.get_dp(0, 0)
        self.assertEqual(self.handle.close_count, 1)
        np.testing.assert_array_equal(dp.array, self.values[1, 2])

    def test_metadata_roundtrip_slash_keys_bytes_timestamps_and_arrays(self):
        original = {'Instrument/Detector': {'bytes': b'\xff\x00', 'array': np.arange(4),
                                          'date': date(2026, 10, 16), 'timestamp': datetime(2026, 10, 16, 12, 30),
                                          'huge': 2 ** 100}, 'path': Path('relative/file')}
        self.records[0]['original_metadata'] = original
        with fd.HyperData.open(self.path) as source:
            decoded = source.reader_metadata
            decoded['original_metadata']['Instrument/Detector']['array'][:] = -1
            self.assertTrue(np.all(source.reader_metadata['original_metadata']['Instrument/Detector']['array'] >= 0))
            data = source.load()
        destination = self.path.with_suffix('.4denoise')
        data.save(destination)
        reloaded = fd.HyperData(destination)
        decoded = fd.unpack_reader_metadata(reloaded.acquisition_metadata)['original_metadata']
        self.assertEqual(decoded['Instrument/Detector']['bytes'], original['Instrument/Detector']['bytes'])
        self.assertEqual(decoded['Instrument/Detector']['date'], original['Instrument/Detector']['date'])
        self.assertEqual(decoded['Instrument/Detector']['timestamp'], original['Instrument/Detector']['timestamp'])
        self.assertEqual(decoded['Instrument/Detector']['huge'], 2 ** 100)
        self.assertEqual(decoded['path'], original['path'])
        np.testing.assert_array_equal(decoded['Instrument/Detector']['array'], np.arange(4))
        with fd.HyperData.open(destination) as source:
            self.assertEqual(source.reader_metadata['original_metadata']['path'], Path('relative/file'))

    def test_selection_failure_and_unselected_records_share_one_close(self):
        self.records.append(record(self.values, 'other', handle=self.handle))
        with self.assertRaises(ValueError):
            with fd.HyperData.open(self.path):
                pass
        self.assertEqual(self.handle.close_count, 1)


class RosettaPureHelpersTests(unittest.TestCase):
    def test_optional_libraries_are_not_imported_with_core_metadata_helpers(self):
        script = (
            'import fourdenoise_rosetta, fourdenoise_sources, sys; '
            'assert not any(name in sys.modules for name in '
            '("rsciio", "dask", "fourdenoise", "tensorly"))'
        )
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_decoder_is_independent_of_optional_packages(self):
        self.assertIsNone(fd.unpack_reader_metadata({}))
        with self.assertRaises(TypeError):
            fd.unpack_reader_metadata(None)
        with self.assertRaisesRegex(ValueError, 'version'):
            fd.unpack_reader_metadata({'rosetta_metadata': {'version': 99}})
        with self.assertRaisesRegex(TypeError, 'pickle'):
            ros._pack_reader_value(object())
        with self.assertRaisesRegex(ValueError, 'kind'):
            ros._unpack_reader_value({'kind': 'execute'})

    def test_archive_object_arrays_and_fixed_width_binary_strings(self):
        objects = np.empty(2, dtype=object)
        objects[:] = [{'x/y': 3}, [1, 2]]
        value = {'object': objects, 'binary': np.array([b'\xff', b'\x00a'], dtype='S2'),
                 'unicode': np.array(['\u00e5', '\u03b2'])}
        archive = ros._pack_reader_value(value)
        with h5py.File('archive.h5', 'w', driver='core', backing_store=False) as file:
            fd._write_hdf5_value(file, 'archive', archive)
            restored = ros._unpack_reader_value(fd._read_hdf5_value(file, 'archive'))
        np.testing.assert_array_equal(restored['object'], objects)
        np.testing.assert_array_equal(restored['binary'], value['binary'])
        np.testing.assert_array_equal(restored['unicode'], value['unicode'])

    def test_axis_records_are_indexed_not_list_ordered(self):
        shape = (3, 4, 5, 6)
        descriptors = axes(shape)[::-1]
        order, indexed = ros._resolve_axis_order({'axes': descriptors}, shape, None)
        self.assertEqual(order, (0, 1, 2, 3))
        self.assertEqual(indexed[0]['name'], 'Ry')

    def test_explicit_order_allows_missing_axis_descriptions(self):
        order, indexed = ros._resolve_axis_order({}, (3, 4, 5, 6), [3, 2, 1, 0])
        self.assertEqual(order, (3, 2, 1, 0))
        self.assertEqual(indexed, {})
        with self.assertRaisesRegex(ValueError, 'unambiguously'):
            ros._resolve_axis_order({}, (3, 4, 5, 6), None)

    def test_numpy_navigation_flags(self):
        shape = (3, 4, 5, 6)
        descriptors = axes(shape, ['y', 'x', 'y', 'x'], [np.bool_(True), np.bool_(True), np.bool_(False), np.bool_(False)])
        self.assertEqual(ros._resolve_axis_order({'axes': descriptors}, shape, None)[0], (0, 1, 2, 3))

    def test_nonuniform_and_invalid_axes_are_not_imported_as_regular_scales(self):
        for values in ([0, 1, 4], [0, float('nan'), 2], [0, 0, 0]):
            self.assertIsNone(ros._uniform_axis({'axis': np.array(values), 'units': 'nm'}, 3))
        self.assertEqual(ros._uniform_axis({'axis': np.array([10, 8, 6]), 'units': 'nm'}, 3), (-2, 10, 'nm'))

    def test_inverse_angstrom_aliases_and_compatible_detector_units(self):
        for unit in ('1/A', 'A^-1', '\u00c5^-1', '1/angstrom', 'angstrom**-1'):
            self.assertEqual(ros._unit_description(unit), ('inverse_length', 1e10))
        shape = (2, 3, 4, 5)
        descriptors = axes(shape)
        descriptors[2].update(units='1/angstrom', scale=-0.01)
        descriptors[3].update(units='1/nm', scale=0.1)
        metadata, notes = ros._calibration_from_axes({i: a for i, a in enumerate(descriptors)}, (0, 1, 2, 3), shape)
        self.assertEqual(metadata['reciprocal_conv_factor'], 0.01)
        self.assertEqual(notes, [])


try:
    HAS_ROSETTA = importlib.util.find_spec('rsciio') is not None and importlib.util.find_spec('dask') is not None
except (ImportError, ValueError):
    HAS_ROSETTA = False


@unittest.skipUnless(HAS_ROSETTA, 'Optional RosettaSciIO/Dask packages not installed')
class RosettaRealPluginTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)

    def test_curated_plugins_match_actual_registry(self):
        from rsciio import IO_PLUGINS
        self.assertLessEqual(ros._LAZY_READER_APIS, {plugin['api'] for plugin in IO_PLUGINS})

    def test_emd_writer_reader_four_dimensional_axis_mapping(self):
        from rsciio.emd import file_writer
        values = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
        raw = values.transpose(3, 1, 0, 2)
        descriptors = axes(raw.shape, ['Kx', 'Rx', 'Ry', 'Ky'], [False, True, True, False])
        signal = {'data': raw, 'axes': descriptors,
                  'metadata': {'General': {'title': 'real-emd'}, 'Signal': {}}, 'original_metadata': {}}
        path = self.folder / 'scan.emd'
        file_writer(str(path), signal)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with fd.HyperData.open(path) as source:
                self.assertEqual(source.shape, values.shape)
                np.testing.assert_array_equal(source.get_dp(1, 2).array, values[1, 2])
                loaded = source.load()
                np.testing.assert_array_equal(loaded.array, values)
                self.assertEqual(source.metadata['acquisition_metadata']['reader'], 'rsciio.emd')
        np.testing.assert_array_equal(loaded.array, values)

    def test_hspy_preserves_calibration_and_closes_owned_hdf5_file(self):
        from rsciio.hspy import file_writer
        from rsciio.utils.file import get_file_handle
        values = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
        descriptors = axes(values.shape, navigation=[True, True, False, False])
        for axis, unit, scale in zip(descriptors, ['nm', 'nm', 'mrad', 'mrad'], [2, 3, 0.5, 0.5]):
            axis.update(units=unit, scale=scale)
        signal = {'data': values, 'axes': descriptors,
                  'metadata': {'General': {'title': 'hspy'}, 'Signal': {'signal_type': ''}},
                  'original_metadata': {}, 'tmp_parameters': {}, 'attributes': {'_lazy': False},
                  'package_info': {}, 'learning_results': {}, 'models': {}}
        path = self.folder / 'scan.hspy'
        file_writer(str(path), signal)
        with fd.HyperData.open(path) as source:
            self.assertEqual(source.real_conv_factor, (2, 3))
            self.assertEqual(source.reciprocal_conv_factor, 0.5)
            handle = get_file_handle(source._dataset._array, warn=False)
            self.assertIsNotNone(handle)
            np.testing.assert_array_equal(source.load().array, values)
        self.assertFalse(handle.id.valid)

    def test_ripple_stack_can_be_read_with_explicit_reader_chunks(self):
        from rsciio.ripple import file_writer
        values = np.arange(7 * 5 * 6, dtype=np.float32).reshape(7, 5, 6)
        descriptors = axes(values.shape, ['depth', 'height', 'width'], [True, False, False])
        signal = {'data': values, 'axes': descriptors,
                  'metadata': {'General': {'title': 'ripple'}, 'Signal': {'signal_type': ''}}, 'original_metadata': {}}
        path = self.folder / 'scan.rpl'
        file_writer(str(path), signal)
        with fd.HyperData.open(path, reader_kwargs={'chunks': (1, 5, 6)}) as source:
            self.assertEqual(source.shape, values.shape)
            self.assertEqual(source.reader_chunk_nbytes, values[0].nbytes)
            np.testing.assert_array_equal(source.get_dp(6).array, values[6])
            np.testing.assert_array_equal(source.load(working_memory_mb=0.001).array, values)


if __name__ == '__main__':
    unittest.main()
