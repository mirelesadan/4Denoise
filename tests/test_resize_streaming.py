"""Numerical, memory, disk-output and chained-geometry resampling contracts."""

from math import prod
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_streaming as streaming
import fourdenoise_resampling as resampling


def area_reference(array, output_shape, axes):
    """Independent dense overlap-matrix reference for small test arrays."""
    result = array.astype(np.result_type(array.dtype, np.float64))
    for axis in axes:
        old, new = result.shape[axis], output_shape[axis]
        scale = old / new
        i = np.arange(old)[None, :]
        j = np.arange(new)[:, None]
        weights = np.maximum(0, np.minimum((j + 1) * scale, i + 1) - np.maximum(j * scale, i)) / scale
        result = np.moveaxis(np.tensordot(weights, result, axes=(1, axis)), 0, axis)
    return result


class SliceOnlySource:
    """Reject eager conversion and record the engine's basic source reads."""

    def __init__(self, array):
        self.array, self.shape, self.dtype = array, array.shape, array.dtype
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('The full source was converted.')

    def __getitem__(self, selection):
        if not all(isinstance(s, slice) for s in selection):
            raise AssertionError('Only basic source slices are allowed.')
        result = self.array[selection]
        self.reads.append(result.nbytes)
        return result


class ResizeNumericalTests(unittest.TestCase):
    def test_integer_fractional_and_stack_area_match_global_weights(self):
        for shape, requested, domain in (
            ((6, 8, 5, 7), (3, 4), 'real'),
            ((5, 7, 9, 11), (3, 4), 'real'),
            ((5, 7, 9, 11), (4, 6), 'reciprocal'),
            ((13, 5, 7), 6, 'real'),
            ((13, 5, 7), (3, 4), 'reciprocal'),
        ):
            array = np.arange(prod(shape), dtype=np.int16).reshape(shape)
            axes, target = resampling._resize_spec(shape, requested, domain)
            expected = area_reference(array, target, axes)
            for budget in (0.004, 1):
                with self.subTest(shape=shape, domain=domain, budget=budget):
                    actual = fd.HyperData(array).resize(requested, domain, working_memory_mb=budget)
                    np.testing.assert_allclose(actual.array, expected, rtol=1e-12, atol=1e-11)

    def test_linear_nearest_match_cartesian_reference_with_halos(self):
        array = np.random.default_rng(2).normal(size=(5, 7, 9, 11)).astype(np.float32)
        for method in ('linear', 'nearest'):
            for domain, requested in (('real', (3, 4)), ('real', (8, 10)), ('reciprocal', (4, 6)), ('reciprocal', (13, 17))):
                axes, target = resampling._resize_spec(array.shape, requested, domain)
                order = int(method == 'linear')
                expected = transform.resize(array, target, order=order, mode='reflect', preserve_range=True,
                                            anti_aliasing=order > 0 and any(target[a] < array.shape[a] for a in axes))
                with self.subTest(method=method, domain=domain, shape=requested):
                    actual = fd.HyperData(array).resize(requested, domain, method, working_memory_mb=0.03)
                    np.testing.assert_allclose(actual.array, expected, rtol=2e-6, atol=2e-6)

    def test_stack_interpolation(self):
        array = np.arange(13 * 5 * 7, dtype=float).reshape(13, 5, 7)
        for method in ('linear', 'nearest'):
            expected = transform.resize(array, (8, 5, 7), order=int(method == 'linear'), mode='reflect', preserve_range=True,
                                        anti_aliasing=method == 'linear')
            actual = fd.HyperData(array).resize(8, 'real', method, working_memory_mb=0.003)
            np.testing.assert_allclose(actual.array, expected, atol=1e-12)

    def test_unchanged_axis_is_not_resampled(self):
        array = np.arange(2 * 9 * 11, dtype=float).reshape(2, 9, 11)
        array[:, 4] += 10000
        for method in ('area', 'linear', 'nearest'):
            target = (2, 9, 6)
            expected = area_reference(array, target, (2,)) if method == 'area' else transform.resize(
                array, target, order=int(method == 'linear'), preserve_range=True,
                mode='reflect', anti_aliasing=method == 'linear',
            )
            actual = fd.HyperData(array).resize((9, 6), 'reciprocal', method, working_memory_mb=0.002)
            np.testing.assert_allclose(actual.array, expected, atol=1e-10)

    def test_no_op_is_exact_independent_and_preserves_dtype(self):
        for dtype in (np.bool_, np.uint64, np.float16, np.float32, np.complex64):
            array = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6).astype(dtype)
            if dtype == np.uint64:
                array += np.uint64(2**63)
            for method in ('area', 'linear', 'nearest'):
                actual = fd.HyperData(array).resize((3, 4), 'real', method, working_memory_mb=0.001)
                self.assertEqual(actual.dtype, array.dtype)
                self.assertFalse(np.shares_memory(array, actual.array))
                np.testing.assert_array_equal(actual.array, array)

    def test_dtype_rules_constant_values_and_range(self):
        for dtype in (np.bool_, np.int8, np.uint16, np.float16, np.float32, np.float64, np.complex64):
            array = np.full((3, 5, 7), 1 if dtype == np.bool_ else 100, dtype=dtype)
            for method in ('area', 'linear', 'nearest'):
                result = fd.HyperData(array).resize((3, 4), 'reciprocal', method, working_memory_mb=0.01)
                expected_dtype = array.dtype if method == 'nearest' or array.dtype.kind in 'fc' else np.dtype('float64')
                self.assertEqual(result.dtype, expected_dtype)
                np.testing.assert_allclose(result.array, array.flat[0], atol=1e-12)

    def test_nearest_uint64_never_round_trips_through_float(self):
        array = (np.arange(2 * 5 * 7, dtype=np.uint64) + np.uint64(2**63)).reshape(2, 5, 7)
        result = fd.HyperData(array).resize((10, 14), 'reciprocal', 'nearest', working_memory_mb=0.01)
        np.testing.assert_array_equal(result.array, np.repeat(np.repeat(array, 2, axis=1), 2, axis=2))

    def test_complex_interpolation_is_linear(self):
        base = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
        real = fd.HyperData(base).resize((4, 6), 'reciprocal', 'linear')
        result = fd.HyperData(base + 2j * base).resize((4, 6), 'reciprocal', 'linear')
        np.testing.assert_allclose(result.array, real.array * (1 + 2j), rtol=1e-6)
        self.assertEqual(result.dtype, np.complex64)

    def test_noncontiguous_input_is_unmodified(self):
        array = np.arange(5 * 7 * 9 * 11, dtype=float).reshape(5, 7, 9, 11)[::-1, :, ::-1, :].swapaxes(0, 1)
        before = array.copy()
        result = fd.HyperData(array).resize((4, 3), 'real', working_memory_mb=0.01)
        np.testing.assert_allclose(result.array, area_reference(array, (4, 3, 9, 11), (0, 1)))
        np.testing.assert_array_equal(array, before)

    def test_realspace_uses_shared_engine_and_updates_signed_geometry(self):
        image = fd.RealSpace(np.arange(35, dtype=np.float32).reshape(5, 7), units='nm', conv_factor=(2, -3), origin=(10, 20))
        result = image.resize((3, 4), working_memory_mb=0.001)
        reference = fd.HyperData(image.array[None]).resize((3, 4), 'reciprocal', working_memory_mb=0.001)
        np.testing.assert_array_equal(result.array, reference.array[0])
        np.testing.assert_allclose(result.conv_factor, (10 / 3, -21 / 4))
        np.testing.assert_allclose(result.origin, (10 + (5 / 3 - 1), 20 - 1.5 * (7 / 4 - 1)))

    def test_validation_is_explicit(self):
        data = fd.HyperData(np.ones((3, 5, 7), dtype=np.float32))
        for kwargs in ({'working_memory_mb': 0}, {'working_memory_mb': 1e-8}, {'method': 'bad'}):
            with self.assertRaises(ValueError):
                data.resize((2, 3), 'reciprocal', **kwargs)
        for shape in ((True, 3), (2.5, 3), (0, 3), (2,), 'wrong'):
            with self.assertRaises(ValueError):
                data.resize(shape, 'reciprocal')
        with self.assertRaisesRegex(ValueError, 'downsampling'):
            data.resize((6, 8), 'reciprocal')
        np.testing.assert_equal(data.resize(np.array([2, 3]), 'reciprocal').shape, (3, 2, 3))


class ResizeMemoryTests(unittest.TestCase):
    def test_basic_reads_are_tiled_not_eager(self):
        array = np.arange(5 * 7 * 9 * 11, dtype=float).reshape(5, 7, 9, 11)
        for method in ('area', 'linear', 'nearest'):
            source = SliceOnlySource(array)
            plan = resampling._resize_plan(array.shape, array.dtype, (3, 4, 9, 11), (0, 1), method, working_memory_mb=0.02)
            result = resampling._resize_array(source, (3, 4, 9, 11), (0, 1), method, working_memory_mb=0.02)
            self.assertGreater(len(source.reads), 1)
            self.assertLess(max(source.reads), array.nbytes)
            self.assertLessEqual(plan['estimated_peak_bytes'], plan['budget_bytes'])
            if method == 'area':
                np.testing.assert_allclose(result, area_reference(array, result.shape, (0, 1)))

    def test_peak_arrays_fit_budget_apart_from_resident_output(self):
        array = np.ones((12, 13, 32, 32), dtype=np.float32)
        for method in ('area', 'linear', 'nearest'):
            tracemalloc.start()
            result = resampling._resize_array(array, (7, 8, 32, 32), (0, 1), method, working_memory_mb=0.5)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertLess(peak - result.nbytes, 0.5 * 1024**2 + 128 * 1024)

    def test_explicit_destinations_are_validated_and_memmaps_work(self):
        array = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
        with tempfile.TemporaryDirectory() as folder:
            out = np.memmap(Path(folder) / 'out.bin', mode='w+', dtype=np.float32, shape=(3, 3, 4))
            self.assertIs(resampling._resize_array(array, out.shape, (1, 2), out=out), out)
            np.testing.assert_allclose(out, area_reference(array, out.shape, (1, 2)), rtol=1e-6)
            out._mmap.close()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            resampling._resize_array(array, array.shape, (1, 2), out=array)
        with self.assertRaisesRegex(ValueError, 'shape and dtype'):
            resampling._resize_array(array, (3, 3, 4), (1, 2), out=np.empty((3, 3, 4), dtype=np.float64))


class DiskResizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.array = np.arange(5 * 7 * 9 * 11, dtype=np.float32).reshape(5, 7, 9, 11)
        self.data = fd.HyperData(self.array, real_units='nm', real_conv_factor=(2, -3), real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.5)
        self.data.set_rq_calibration(rotation_deg=17)
        self.data.acquisition_metadata = {'experiment': 'resizing', 'scan_shape': (5, 7)}
        self.source = self.folder / 'source.4denoise'
        self.data.save(self.source)

    def test_reader_global_masked_regional_and_calibrated_reductions(self):
        mask = np.indices((5, 7)).sum(axis=0) % 3 != 0
        with fd.HyperData.open_hdf5(self.source) as reader:
            for operation in ('mean', 'std', 'min', 'max'):
                for kwargs in ({}, {'mask': mask}, {'y': (1, 4), 'x': (2, 6)}, {'y': (12, 18), 'x': (2, 14), 'selection_units': 'calibrated'}):
                    with self.subTest(operation=operation, selection=kwargs):
                        expected = self.data.get_dp(operation=operation, **kwargs)
                        actual = reader.get_dp(operation=operation, working_memory_mb=0.02, **kwargs)
                        np.testing.assert_allclose(actual.array, expected.array)
                        self.assertEqual((actual.units, actual.conv_factor), ('mrad', 0.5))
            with self.assertRaisesRegex(ValueError, 'median'):
                reader.get_dp('median')
            with self.assertRaises(TypeError):
                reader.get_dp(real_mask=mask)
            after_close = reader.get_dp(2, 3)
        np.testing.assert_array_equal(after_close.array, self.array[2, 3])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            reader.get_dp()

    def test_disk_results_and_metadata_match_memory_for_both_domains(self):
        for method in ('area', 'linear', 'nearest'):
            for domain, shape in (('real', (3, 4)), ('reciprocal', (4, 6))):
                with self.subTest(method=method, domain=domain):
                    path = self.folder / f'{domain}-{method}.4denoise'
                    saved = fd.HyperData.resize_hdf5(self.source, path, shape, domain, method, working_memory_mb=0.02, progress=False)
                    actual = fd.HyperData(saved)
                    expected = self.data.resize(shape, domain, method, working_memory_mb=0.03)
                    np.testing.assert_allclose(actual.array, expected.array)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.real_conv_factor, expected.real_conv_factor)
                    self.assertEqual(actual.real_origin, expected.real_origin)
                    self.assertEqual(actual.reciprocal_conv_factor, expected.reciprocal_conv_factor)
                    self.assertEqual(actual.reciprocal_units, expected.reciprocal_units)
                    self.assertEqual(actual.rq_calibration.to_dict(), expected.rq_calibration.to_dict())
                    self.assertEqual(actual.acquisition_metadata, expected.acquisition_metadata)

    def test_disk_stack_and_generic_dataset_selection(self):
        source = self.folder / 'generic.h5'
        array = self.array.reshape(35, 9, 11)
        with h5py.File(source, 'w') as file:
            file.create_dataset('data', data=array)
            file.create_dataset('extra', data=np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, 'numeric'):
            fd.HyperData.resize_hdf5(source, self.folder / 'out', 13, 'real', progress=False)
        saved = fd.HyperData.resize_hdf5(source, self.folder / 'out', 13, 'real', hdf5_dataset='data', working_memory_mb=0.003, progress=False)
        result = fd.HyperData(saved)
        expected = fd.HyperData(array).resize(13, 'real')
        np.testing.assert_allclose(result.array, expected.array)
        self.assertIsNone(result.real_conv_factor)

    def test_atomic_failure_protects_existing_file_and_cleans_temporary(self):
        output = self.folder / 'output.h5'
        output.write_bytes(b'original destination')

        def fail(*args, **kwargs):
            kwargs['out'][0, 0, 0, 0] = -1
            raise RuntimeError('injected failure after starting output')

        with patch.object(streaming, '_resize_array', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'injected'):
            fd.HyperData.resize_hdf5(self.source, output, (3, 4), 'real', overwrite=True, progress=False)
        self.assertEqual(output.read_bytes(), b'original destination')
        self.assertEqual(list(self.folder.glob('.output.h5.*.tmp')), [])

    def test_disk_validation_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'different'):
            fd.HyperData.resize_hdf5(self.source, self.source, (3, 4), 'real')
        destination = self.folder / 'existing.h5'
        destination.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            fd.HyperData.resize_hdf5(self.source, destination, (3, 4), 'real')

    def test_tiny_budget_does_not_publish_destination(self):
        output = self.folder / 'not-created.h5'
        with self.assertRaisesRegex(ValueError, 'working_memory_mb'):
            fd.HyperData.resize_hdf5(self.source, output, (3, 4), 'real', working_memory_mb=1e-8, progress=False)
        self.assertFalse(output.exists())


class ResizeWorkflowTests(unittest.TestCase):
    def test_polar_tiles_match_independent_full_pattern_reference(self):
        data = fd.HyperData(np.ones((2, 3, 17, 17), dtype=np.float32), reciprocal_units='mrad', reciprocal_conv_factor=0.25)
        polar = data.to_polar(output_shape=(9, 24), progress=False)
        polar.array[:] = np.random.default_rng(4).random(polar.shape)
        for partial in (False, True):
            working = polar.crop(kxlim=(3, 20)) if partial else polar
            oldr, oldt = working.shape[-2:]
            for method in ('linear', 'nearest'):
                nr, nt = 7, 13
                coords = np.array(np.meshgrid(np.linspace(0, oldr - 1, nr), np.arange(nt) * oldt / nt + 1, indexing='ij'))
                sigma = np.maximum(0, (np.array([oldr, oldt]) / [nr, nt] - 1) / 2)
                pattern = working.array[0, 0].astype(float)
                if method == 'linear':
                    pattern = gaussian_filter(pattern, sigma, mode=('mirror', 'nearest' if partial else 'wrap'))
                padded = np.pad(pattern, ((0, 0), (1, 1)), mode='edge' if partial else 'wrap')
                expected = map_coordinates(padded, coords, order=int(method == 'linear'), mode='nearest', prefilter=False)
                actual = working.resize((nr, nt), 'reciprocal', method, working_memory_mb=0.015)
                np.testing.assert_allclose(actual.array[0, 0], expected, atol=1e-6)
                self.assertEqual(actual.polar_metadata['theta_step'], (255 if partial else 360) / nt)
                if not partial:
                    self.assertEqual(actual.rotate_dps(17).shape, actual.shape)
                    self.assertEqual(actual.to_cartesian(progress=False).ndim, 4)

    def test_disk_polar_metadata_rotation_and_inverse(self):
        polar = fd.HyperData(np.ones((2, 3, 17, 17), dtype=np.float32)).to_polar(output_shape=(9, 24), progress=False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'polar.4denoise'
            polar.save(source)
            path = fd.HyperData.resize_hdf5(source, Path(directory) / 'resized', (7, 13), 'reciprocal', 'linear', working_memory_mb=0.015, progress=False)
            actual = fd.HyperData(path)
        self.assertTrue(actual.is_polar)
        np.testing.assert_allclose(actual.array, polar.resize((7, 13), 'reciprocal', 'linear').array)
        self.assertEqual(actual.rotate_dps(10).shape, actual.shape)
        self.assertEqual(actual.to_cartesian(progress=False).shape, (2, 3, 14, 14))

    def test_curve_resize_undo_preserves_original_exactly(self):
        array = np.arange(5 * 7 * 4 * 6, dtype=np.float32).reshape(5, 7, 4, 6)
        data = fd.HyperData(array)
        unfolded = data.unfold(domain='real', method='hilbert', curve_shape_strategy='resize', resize_side=4, preserve_original=True)
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, array)
        self.assertEqual(unfolded.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
