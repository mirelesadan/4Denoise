"""Disk-backed source parity, resource ownership and bounded-reading contracts."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_sources as sources


class GuardedDataset:
    """Detect implicit conversion and record basic-slice read sizes."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape
        self.dtype = dataset.dtype
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('The complete disk source must never be converted.')

    def __getitem__(self, selection):
        if not isinstance(selection, tuple):
            raise AssertionError('Expected a basic tuple selection.')
        if any(not isinstance(item, (slice, int, np.integer)) for item in selection):
            raise AssertionError('Do not gather all masked patterns from disk.')
        result = self.dataset[selection]
        self.reads.append((selection, result.nbytes))
        return result


class DiffractionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.array = np.arange(5 * 7 * 3 * 4, dtype=np.int16).reshape(5, 7, 3, 4)
        self.npy = self.folder / 'scan.npy'
        self.hdf5 = self.folder / 'scan.h5'
        np.save(self.npy, self.array)
        with h5py.File(self.hdf5, 'w') as file:
            file.create_dataset('entry/data', data=self.array)

    def calibrated(self, path):
        return fd.HyperData.open(
            path, real_units='nm', real_conv_factor=(2, -3),
            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.25,
        )

    def test_same_source_type_and_read_only_header_properties(self):
        for path, backend in ((self.npy, 'numpy'), (self.hdf5, 'hdf5')):
            with self.subTest(backend=backend), fd.HyperData.open(path) as source:
                self.assertIsInstance(source, fd.DiffractionSource)
                self.assertEqual(source.shape, self.array.shape)
                self.assertEqual(source.ndim, 4)
                self.assertEqual(source.scan_shape, (5, 7))
                self.assertEqual(source.real_shape, (5, 7))
                self.assertEqual(source.pattern_shape, (3, 4))
                self.assertEqual(source.dtype, self.array.dtype)
                self.assertEqual(source.nbytes, self.array.nbytes)
                self.assertEqual(source.backend, backend)
                self.assertEqual(source.filename, str(path.resolve()))
                self.assertEqual(source.dataset_path, '/entry/data' if backend == 'hdf5' else None)
                self.assertIsNone(source.real_units)
                self.assertIsNone(source.reciprocal_conv_factor)
                self.assertFalse(source.closed)
                with self.assertRaises(AttributeError):
                    source.shape = (1,)
                with self.assertRaises(AttributeError):
                    source.array
                with self.assertRaisesRegex(TypeError, 'explicitly'):
                    np.asarray(source)
                with self.assertRaisesRegex(TypeError, 'explicitly'):
                    fd.HyperData(source)
                if backend == 'numpy':
                    self.assertIsInstance(source._dataset, np.memmap)
                    self.assertFalse(source._dataset.flags.writeable)
                else:
                    self.assertEqual(source._dataset.file.mode, 'r')

    def test_opening_hdf5_reads_no_detector_pixels(self):
        original = h5py.Dataset.__getitem__
        reads = []

        def tracked(dataset, selection):
            reads.append(dataset.name)
            return original(dataset, selection)

        with patch.object(h5py.Dataset, '__getitem__', tracked):
            with fd.HyperData.open(self.hdf5) as source:
                self.assertEqual(source.shape, self.array.shape)
                self.assertEqual(source.metadata['real_origin'], (0.0, 0.0))
                self.assertEqual(reads, [])
                source.get_dp(1, 2)
                self.assertEqual(reads, ['/entry/data'])

    def test_opening_numpy_requests_read_only_map_and_no_pickle(self):
        original = np.load
        with patch.object(sources.np, 'load', wraps=original) as loader:
            with fd.HyperData.open(self.npy) as source:
                self.assertEqual(source.shape, self.array.shape)
                loader.assert_called_once_with(self.npy, mmap_mode='r', allow_pickle=False)

    def test_pattern_reduction_range_and_mask_parity(self):
        mask = np.zeros(self.array.shape[:2], dtype=bool)
        mask[1:4, 2:6] = True
        expected = fd.HyperData(self.array)
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path), fd.HyperData.open(path) as source:
                np.testing.assert_array_equal(source.get_dp(2, 4).array, self.array[2, 4])
                for operation in ('mean', 'std', 'min', 'max'):
                    np.testing.assert_allclose(
                        source.get_dp(mask=mask, operation=operation, working_memory_mb=0.005).array,
                        expected.get_dp(mask=mask, operation=operation, working_memory_mb=0.005).array,
                    )
                    np.testing.assert_allclose(
                        source.get_dp((1, 4), (2, 6), operation=operation).array,
                        expected.get_dp((1, 4), (2, 6), operation=operation).array,
                    )
                with self.assertRaisesRegex(ValueError, 'median'):
                    source.get_dp(operation='median')
                with self.assertRaisesRegex(TypeError, 'real_mask'):
                    source.get_dp(real_mask=mask)

    def test_returned_data_is_independent_and_survives_close(self):
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path):
                with fd.HyperData.open(path) as source:
                    dp = source.get_dp(0, 0)
                    slices, block = next(source.iter_chunks((2, 3)))
                    loaded = source.load(working_memory_mb=0.001)
                    self.assertTrue(dp.array.flags.owndata)
                    self.assertTrue(block.array.flags.owndata)
                    self.assertTrue(loaded.array.flags.owndata)
                    dp.array[:] = -1
                    block.array[:] = -2
                    loaded.array[:] = -3
                    np.testing.assert_array_equal(source.get_dp(0, 0).array, self.array[0, 0])
                self.assertTrue(source.closed)
                self.assertEqual(source.shape, self.array.shape)
                self.assertTrue(np.all(dp.array == -1))
                self.assertTrue(np.all(block.array == -2))
                self.assertTrue(np.all(loaded.array == -3))
                for action in (lambda: source.get_dp(0, 0), source.load,
                               lambda: next(source.iter_chunks())):
                    with self.assertRaisesRegex(RuntimeError, 'closed'):
                        action()
                source.close()

    def test_explicit_close_and_exception_close_resources_once(self):
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path):
                with fd.HyperData.open(path) as source:
                    resource = source._dataset._mmap if path == self.npy else source._dataset.file
                    with patch.object(source, '_close_callback', wraps=source._close_callback) as closer:
                        source.close()
                        source.close()
                        self.assertEqual(closer.call_count, 1)
                    self.assertTrue(resource.closed if path == self.npy else not resource.id.valid)
                with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                    with fd.HyperData.open(path) as reader:
                        resource = reader._dataset._mmap if path == self.npy else reader._dataset.file
                        raise RuntimeError('deliberate')
                self.assertTrue(reader.closed)
                self.assertTrue(resource.closed if path == self.npy else not resource.id.valid)

    def test_iterator_cannot_continue_after_close(self):
        for path in (self.npy, self.hdf5):
            with fd.HyperData.open(path) as source:
                iterator = source.iter_chunks((1, 1))
                _, block = next(iterator)
            np.testing.assert_array_equal(block.array, self.array[:1, :1])
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                next(iterator)

    def test_signed_calibration_and_chunk_origins(self):
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path), self.calibrated(path) as source:
                self.assertEqual(source.real_units, 'nm')
                self.assertEqual(source.real_conv_factor, (2, -3))
                self.assertEqual(source.real_origin, (10, 20))
                dp = source.get_dp(14, 8, selection_units='calibrated')
                np.testing.assert_array_equal(dp.array, self.array[2, 4])
                self.assertEqual((dp.units, dp.conv_factor), ('mrad', 0.25))
                for slices, block in source.iter_chunks((2, 3)):
                    np.testing.assert_array_equal(block.array, self.array[slices])
                    self.assertEqual(block.real_origin, (10 + 2 * slices[0].start, 20 - 3 * slices[1].start))
                    self.assertEqual(block.real_conv_factor, (2, -3))
                    self.assertIsNone(block.unfold_metadata)
                loaded = source.load()
                self.assertEqual(loaded.real_origin, (10, 20))
                self.assertEqual(loaded.reciprocal_conv_factor, 0.25)

    def test_chunk_budget_shrinks_blocks_and_does_not_gather_masks(self):
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path), fd.HyperData.open(path) as source:
                tracker = GuardedDataset(source._dataset)
                source._dataset = tracker
                budget = 2 * self.array[0, 0].nbytes * 3
                restored = np.empty_like(self.array)
                for slices, block in source.iter_chunks((5, 7), working_memory_mb=budget / 1024 ** 2):
                    self.assertLessEqual(block.array.nbytes * 2, budget)
                    restored[slices] = block.array
                np.testing.assert_array_equal(restored, self.array)
                self.assertGreater(len(tracker.reads), 1)
                self.assertTrue(all(nbytes <= budget // 2 for _, nbytes in tracker.reads))
                tracker.reads.clear()
                mask = np.zeros(self.array.shape[:2], dtype=bool)
                mask[-1, -1] = True
                np.testing.assert_allclose(
                    source.get_dp(mask=mask, operation='mean', working_memory_mb=0.001).array,
                    self.array[-1, -1],
                )
                self.assertTrue(all(nbytes < self.array.nbytes for _, nbytes in tracker.reads))

    def test_full_load_is_explicit_and_reads_bounded_blocks(self):
        for path in (self.npy, self.hdf5):
            with self.subTest(path=path), fd.HyperData.open(path) as source:
                tracker = GuardedDataset(source._dataset)
                source._dataset = tracker
                with patch.object(sources.np, 'empty', wraps=np.empty) as allocator:
                    with self.assertRaisesRegex(MemoryError, 'max_memory_mb'):
                        source.load(max_memory_mb=(self.array.nbytes - 1) / 1024 ** 2)
                    allocator.assert_not_called()
                self.assertEqual(tracker.reads, [])
                loaded = source.load(max_memory_mb=self.array.nbytes / 1024 ** 2,
                                     working_memory_mb=0.001)
                np.testing.assert_array_equal(loaded.array, self.array)
                self.assertEqual(loaded.dtype, self.array.dtype)
                self.assertGreater(len(tracker.reads), 1)
                self.assertTrue(all(nbytes <= 0.001 * 1024 ** 2 / 2 for _, nbytes in tracker.reads))

    def test_validation_before_pixels_and_metadata_is_defensive(self):
        for path in (self.npy, self.hdf5):
            with fd.HyperData.open(path) as source:
                tracker = GuardedDataset(source._dataset)
                source._dataset = tracker
                for budget in (True, 0, -1, float('nan'), float('inf'), '64'):
                    with self.subTest(budget=budget):
                        with self.assertRaisesRegex(ValueError, 'max_memory_mb'):
                            source.load(max_memory_mb=budget)
                with self.assertRaisesRegex(ValueError, 'one pattern'):
                    list(source.iter_chunks(1, working_memory_mb=1 / 1024 ** 2))
                with self.assertRaisesRegex(ValueError, 'one pattern'):
                    source.load(working_memory_mb=1 / 1024 ** 2)
                for shape in (False, 0, (1, 0), (1,), 1.5, 'x', ((2, 3),)):
                    with self.assertRaises(ValueError):
                        list(source.iter_chunks(shape))
                self.assertEqual(tracker.reads, [])
                metadata = source.metadata
                metadata['real_origin'] = (-100, -100)
                self.assertEqual(source.real_origin, (0, 0))

    def test_three_dimensional_fortran_and_big_endian_numpy(self):
        array = np.asfortranarray(np.arange(7 * 3 * 4).reshape(7, 3, 4).astype('>i4'))
        np.save(self.npy, array)
        with fd.HyperData.open(self.npy) as source:
            self.assertEqual(source.scan_shape, (7,))
            self.assertIsNone(source.real_shape)
            self.assertEqual(source.dtype, array.dtype)
            np.testing.assert_array_equal(source.get_dp(6).array, array[6])
            chunks = list(source.iter_chunks(3))
            self.assertEqual(len(chunks), 3)
            for slices, block in chunks:
                self.assertEqual(block.dtype, array.dtype)
                np.testing.assert_array_equal(block.array, array[slices])
            loaded = source.load()
            self.assertEqual(loaded.dtype, array.dtype)
            np.testing.assert_array_equal(loaded.array, array)

    def test_invalid_dimensions_dtypes_and_numpy_archive(self):
        for array in (np.ones((2, 2)), np.ones((1, 1, 1, 1, 1)),
                      np.empty((0, 2, 2)), np.full((2, 3, 4), 'text')):
            for path in (self.npy, self.hdf5):
                with self.subTest(shape=array.shape, dtype=array.dtype, path=path):
                    if path == self.npy:
                        np.save(path, array)
                    else:
                        with h5py.File(path, 'w') as file:
                            file.create_dataset('data', data=array.astype('S') if array.dtype.kind == 'U' else array)
                    with self.assertRaises((ValueError, TypeError)):
                        with fd.HyperData.open(path):
                            pass
        np.save(self.npy, np.full((2, 3, 4), object(), dtype=object))
        with self.assertRaises(ValueError):
            with fd.HyperData.open(self.npy):
                pass
        with self.npy.open('wb') as stream:
            np.savez(stream, data=self.array)
        with self.assertRaisesRegex(ValueError, 'archive'):
            with fd.HyperData.open(self.npy):
                pass

    def test_format_and_calibration_errors(self):
        for path, kwargs, message in (
            (self.folder / 'missing.h5', {}, 'does not exist'),
            (self.npy, {'backend': 'unknown'}, 'backend'),
            (self.npy, {'hdf5_dataset': '/array'}, 'only to HDF5'),
            (self.npy, {'real_units': 'nm'}, 'both'),
            (self.npy, {'reciprocal_conv_factor': 2}, 'both'),
            (self.npy, {'real_units': 'nm', 'real_conv_factor': 0}, 'nonzero'),
            (self.npy, {'reciprocal_units': 'mrad', 'reciprocal_conv_factor': -1}, 'positive'),
            (self.npy, {'real_origin': (1,)}, 'real_origin'),
        ):
            with self.subTest(path=path, kwargs=kwargs), self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                with fd.HyperData.open(path, **kwargs):
                    pass
        with self.assertRaisesRegex(ValueError, 'HDF5'):
            with fd.HyperData.open_hdf5(self.npy):
                pass

    def test_multiple_hdf5_datasets_and_hdf_extensions(self):
        with h5py.File(self.hdf5, 'a') as file:
            file.create_dataset('other', data=np.zeros((2, 3, 4)))
        with self.assertRaisesRegex(ValueError, '2 numeric'):
            with fd.HyperData.open(self.hdf5):
                pass
        for suffix in ('.h5', '.hdf5', '.hdf'):
            path = self.folder / f'multiple{suffix}'
            self.hdf5.rename(path)
            with fd.HyperData.open(path, hdf5_dataset='/entry/data') as source:
                np.testing.assert_array_equal(source.load().array, self.array)
            path.rename(self.hdf5)

    def test_saved_metadata_preserved_and_overrides_do_not_modify_file(self):
        data = fd.HyperData(self.array, real_units='nm', real_conv_factor=(2, -3),
                            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.25)
        data.acquisition_metadata = {'experiment': {'label': 'original'}}
        data.set_rq_calibration(rotation_deg=17, mirror_axis='x')
        data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            1.2, (1, 1.5), data.pattern_shape, units='mrad', conv_factor=0.25,
        )
        path = self.folder / 'saved.4denoise'
        data.save(path)
        with fd.HyperData.open(path) as source:
            metadata = source.metadata
            metadata['acquisition_metadata']['experiment']['label'] = 'changed'
            self.assertEqual(source.metadata['acquisition_metadata']['experiment']['label'], 'original')
            _, block = next(source.iter_chunks((2, 3)))
            loaded = source.load()
            self.assertFalse(source.has_unfold_metadata)
        for obj in (block, loaded):
            self.assertEqual(obj.acquisition_metadata, data.acquisition_metadata)
            self.assertEqual(obj.rq_calibration, data.rq_calibration)
            self.assertEqual(obj.center_beam_metadata, data.center_beam_metadata)
        with fd.HyperData.open(path, real_units='angstrom', real_conv_factor=20,
                               real_origin=(0, 0)) as source:
            self.assertEqual(source.load().real_conv_factor, 20)
        saved = fd.HyperData(path)
        self.assertEqual(saved.real_units, 'nm')
        self.assertEqual(saved.real_origin, (10, 20))
        with fd.HyperData.open(path, reciprocal_units='inverse_nm',
                               reciprocal_conv_factor=0.5) as source:
            beam = source.metadata['center_beam_metadata']
            self.assertEqual(beam['center_px'], data.center_beam_metadata['center_px'])
            self.assertEqual(beam['radius_px'], 1.2)
            self.assertEqual(beam['radius_calibrated'], 0.6)
            self.assertEqual(beam['calibrated_units'], 'inverse_nm')
            self.assertEqual(source.get_dp(0, 0).center_beam_metadata, beam)

    def test_saved_unfold_payload_is_deferred_and_full_load_can_undo(self):
        unfolded = fd.HyperData(self.array).unfold(method='hilbert', warn_on_crop=False)
        path = self.folder / 'unfolded.4denoise'
        unfolded.save(path)
        original_read = sources._read_hdf5_value
        payload_reads = []

        def tracked(group, key):
            if key == 'unfold_metadata':
                payload_reads.append(key)
            return original_read(group, key)

        with patch.object(sources, '_read_hdf5_value', tracked):
            with fd.HyperData.open(path) as source:
                self.assertTrue(source.has_unfold_metadata)
                self.assertNotIn('unfold_metadata', source.metadata)
                source.get_dp(0)
                _, block = next(source.iter_chunks(2))
                self.assertIsNone(block.unfold_metadata)
                self.assertEqual(payload_reads, [])
                loaded = source.load()
                self.assertEqual(payload_reads, ['unfold_metadata'])
            self.assertTrue(source.has_unfold_metadata)
        np.testing.assert_array_equal(loaded.unfold(undo=True).array, self.array)

    def test_saved_polar_metadata_survives_pattern_chunk_and_full_load(self):
        polar = fd.HyperData(self.array.astype(np.float32)).to_polar(output_shape=(2, 8))
        path = self.folder / 'polar.4denoise'
        polar.save(path)
        with fd.HyperData.open(path) as source:
            dp = source.get_dp(1, 2)
            _, block = next(source.iter_chunks((2, 3)))
            loaded = source.load()
            self.assertEqual(dp.polar_metadata, polar.polar_metadata)
            self.assertEqual(block.polar_metadata, polar.polar_metadata)
            self.assertEqual(loaded.polar_metadata, polar.polar_metadata)
            np.testing.assert_array_equal(loaded.array, polar.array)
        with self.assertRaisesRegex(ValueError, 'Cartesian diffraction axes'):
            with fd.HyperData.open(path, reciprocal_units='mrad', reciprocal_conv_factor=0.5):
                pass

    def test_initialization_failure_closes_the_numpy_map(self):
        original = np.load
        datasets = []

        def tracked(*args, **kwargs):
            dataset = original(*args, **kwargs)
            datasets.append(dataset)
            return dataset

        with patch.object(sources.np, 'load', tracked):
            with self.assertRaises(ValueError):
                with fd.HyperData.open(self.npy, reciprocal_units='mrad'):
                    pass
        self.assertTrue(datasets[0]._mmap.closed)

    def test_initialization_failure_closes_hdf5_and_validates_saved_schema(self):
        original = h5py.File
        files = []

        def tracked(*args, **kwargs):
            file = original(*args, **kwargs)
            files.append(file)
            return file

        with patch.object(sources.h5py, 'File', tracked):
            with self.assertRaisesRegex(ValueError, 'both'):
                with fd.HyperData.open(self.hdf5, reciprocal_units='mrad'):
                    pass
        self.assertFalse(files[0].id.valid)
        path = self.folder / 'invalid.4denoise'
        with original(path, 'w') as file:
            file.create_dataset('array', data=self.array)
        with self.assertRaisesRegex(ValueError, 'valid'):
            with fd.HyperData.open(path):
                pass
        with original(path, 'a') as file:
            file.attrs['fourdenoise_format'] = fd._HYPERDATA_HDF5_FORMAT
        with self.assertRaisesRegex(ValueError, 'format_version'):
            with fd.HyperData.open(path):
                pass
        with original(path, 'a') as file:
            file.attrs['format_version'] = '99.0'
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            with fd.HyperData.open(path):
                pass

    def test_legacy_unfold_payload_migrates_on_full_load(self):
        unfolded = fd.HyperData(self.array).unfold(method='hilbert', warn_on_crop=False)
        unfolded.unfold_metadata.pop('preserved_values_nbytes')
        path = self.folder / 'legacy.4denoise'
        unfolded.save(path)
        with h5py.File(path, 'a') as file:
            file.attrs['format_version'] = '1.0'
        with fd.HyperData.open(path) as source:
            loaded = source.load()
        self.assertGreater(loaded.unfold_metadata['preserved_values_nbytes'], 0)
        np.testing.assert_array_equal(loaded.unfold(undo=True).array, self.array)

    def test_source_module_import_does_not_load_analysis_package(self):
        script = (
            'import fourdenoise_sources, sys; '
            'assert "fourdenoise" not in sys.modules; '
            'assert not any(x in sys.modules for x in ("tensorly", "sklearn", "numba", "cv2"))'
        )
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(fd.DiffractionSource, sources.DiffractionSource)


if __name__ == '__main__':
    unittest.main()
