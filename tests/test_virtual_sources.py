"""Disk-backed virtual detectors, bounded reads and calibrated owned results."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_detectors as detectors


class CountingArray:
    """Only basic block reads are allowed; implicit materialization fails."""

    def __init__(self, values):
        self.values = values
        self.shape, self.dtype = values.shape, values.dtype
        self.calls = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('A complete-source conversion is forbidden.')

    def __getitem__(self, selection):
        self.calls.append(selection)
        if not all(isinstance(s, slice) for s in selection):
            raise AssertionError('Projection must use basic scan slices.')
        return self.values[selection].copy()


class VirtualSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(plt.close, 'all')
        self.root = Path(self.directory.name)
        self.values = np.arange(5 * 7 * 8 * 9, dtype=np.uint16).reshape(5, 7, 8, 9)
        self.data = fd.HyperData(
            self.values, real_units='nm', real_conv_factor=(2.0, 3.0),
            real_origin=(11.0, -4.0), reciprocal_units='1/nm',
            reciprocal_conv_factor=0.25,
        )
        self.npy = self.root / 'scan.npy'
        self.h5 = self.root / 'scan.hdf5'
        self.saved = self.root / 'scan.4denoise'
        np.save(self.npy, self.values)
        with h5py.File(self.h5, 'w') as file:
            file['scan'] = self.values
        self.data.save(self.saved)

    def open(self, path):
        if path == self.saved:
            return fd.HyperData.open(path)
        return fd.HyperData.open(
            path, real_units=self.data.real_units,
            real_conv_factor=self.data.real_conv_factor, real_origin=self.data.real_origin,
            reciprocal_units=self.data.reciprocal_units,
            reciprocal_conv_factor=self.data.reciprocal_conv_factor,
        )

    def test_cartesian_detectors_match_eager_all_native_backends(self):
        cases = [
            dict(annulus=(1, 3), detector_units='pixels'),
            dict(annulus=(0.25, 0.75)),
            dict(radius=2, centers=(2.5, 4.5), detector_units='pixels'),
            dict(radius=(2, 1.5), centers=((2, 3), (3, 4)), detector_units='pixels'),
            dict(radius=0.5, centers=((0.25, -0.5), (-0.25, 0.5))),
            dict(mask=np.eye(8, 9, dtype=bool)),
            dict(mask=np.linspace(-0.75, 1.5, 72).reshape(8, 9)),
        ]
        for path in (self.npy, self.h5, self.saved):
            with self.open(path) as source:
                for args in cases:
                    with self.subTest(path=path.suffix, args=list(args)):
                        expected, expected_detector = self.data.virtual_image(
                            **args, show=False, return_detector=True, working_memory_mb=0.02,
                        )
                        image, detector = source.virtual_image(
                            **args, show=False, return_detector=True, working_memory_mb=0.02,
                        )
                        np.testing.assert_allclose(image.array, expected.array)
                        np.testing.assert_allclose(detector.array, expected_detector.array)
                        self.assertIsInstance(image, fd.RealSpace)
                        self.assertEqual(image.units, self.data.real_units)
                        self.assertEqual(image.conv_factor, self.data.real_conv_factor)
                        self.assertEqual(image.origin, self.data.real_origin)
                        self.assertEqual(image.quantity, 'Virtual detector signal')
                        self.assertEqual(detector.units, self.data.reciprocal_units)
            image.array[0, 0] = -1
            self.assertEqual(self.values[0, 0, 0, 0], 0)
            self.assertTrue(np.isfinite(detector.array).all())
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                source.virtual_image(annulus=(1, 3), show=False)

    def test_projection_matches_independent_numpy_reference(self):
        weights = np.linspace(-1.5, 2, 72).reshape(8, 9)
        expected = np.sum(self.values.astype(np.float64) * weights, axis=(-2, -1))
        for values in (self.values, self.values[:, ::-1, ::-1, ::-1]):
            with self.subTest(strides=values.strides):
                actual = fd.HyperData(values).virtual_image(mask=weights, show=False,
                                                          working_memory_mb=0.002)
                ref = np.sum(values.astype(np.float64) * weights, axis=(-2, -1))
                np.testing.assert_allclose(actual.array, ref)
        with self.open(self.npy) as source:
            actual = source.virtual_image(mask=weights, show=False, working_memory_mb=0.002)
        np.testing.assert_allclose(actual.array, expected)

    def test_integer_product_overflow_is_avoided(self):
        values = np.full((2, 3, 8, 9), 30000, dtype=np.int16)
        weights = np.full((8, 9), 100000, dtype=np.int32)
        actual = fd.HyperData(values).virtual_image(mask=weights, show=False)
        np.testing.assert_array_equal(actual.array, 30000.0 * 100000 * 72)
        self.assertEqual(actual.array.dtype, np.dtype('float64'))
        np.testing.assert_array_equal(values, 30000)
        np.testing.assert_array_equal(weights, 100000)

    def test_union_and_calibrated_center_convention(self):
        values = np.ones((1, 1, 9, 9))
        data = fd.HyperData(values, reciprocal_units='1/nm', reciprocal_conv_factor=0.5)
        first, dp = data.virtual_image(radius=0.1, centers=(1, 1),
                                      return_detector=True, show=False)
        expected = np.zeros((9, 9))
        expected[2, 6] = 1
        np.testing.assert_array_equal(dp.array, expected)
        both = data.virtual_image(radius=1, centers=((2, 6), (2, 6)),
                                  detector_units='pixels', show=False)
        single = data.virtual_image(radius=1, centers=(2, 6),
                                    detector_units='pixels', show=False)
        np.testing.assert_array_equal(single.array, both.array)
        np.testing.assert_array_equal(first.array, 1)

    def test_basic_read_sizes_single_pass_and_optional_mean(self):
        array = CountingArray(self.values)
        source = fd.DiffractionSource(
            array, fd.HyperData, fd.ReciprocalSpace, real_type=fd.RealSpace,
            backend='test', filename=self.npy, close_callback=lambda: None,
        )
        self.addCleanup(source.close)
        weights = np.ones((8, 9), dtype=np.float32)
        budget = 0.006
        max_patterns, _ = detectors._virtual_projection_plan(
            self.values.shape, self.values.dtype, weights, budget, True,
        )
        with patch.object(detectors, '_project_virtual_detector',
                          wraps=detectors._project_virtual_detector) as project:
            image = source.virtual_image(mask=weights, show=False, working_memory_mb=budget)
            self.assertFalse(project.call_args.kwargs['need_mean'])
        array.calls.clear()
        image, dp = source.virtual_image(mask=weights, return_detector=True,
                                         show=False, working_memory_mb=budget)
        visit_count = np.zeros(self.values.shape[:2], dtype=int)
        for selection in array.calls:
            self.assertEqual(selection[-2:], (slice(None), slice(None)))
            region = visit_count[selection[:2]]
            self.assertLessEqual(region.size, max_patterns)
            region += 1
        self.assertGreater(len(array.calls), 1)
        np.testing.assert_array_equal(visit_count, 1)
        np.testing.assert_array_equal(image.array, self.values.sum(axis=(-2, -1)))
        np.testing.assert_allclose(dp.array, self.values.mean(axis=(0, 1)))

    def test_invalid_detector_or_budget_fails_before_reads(self):
        cases = [dict(), dict(annulus=(3, 1)), dict(radius=2),
                 dict(radius=2, centers=[]), dict(mask=np.zeros((8, 9))),
                 dict(mask=np.ones((2, 2))), dict(mask=np.full((8, 9), np.nan)),
                 dict(mask=np.ones((8, 9), dtype=complex)),
                 dict(annulus=(1, 3), mask=np.ones((8, 9))),
                 dict(annulus=(1, 3), theta_range=(0, 30)),
                 dict(annulus=(1, 3), detector_units='unknown'),
                 dict(annulus=(1, 3), working_memory_mb=0.0001),
                 dict(annulus=(1, 3), working_memory_mb=-1),
                 dict(annulus=(1, 3), mask_show_kwargs=[]),
                 dict(annulus=(1, 3), ring_alpha=2)]
        with self.open(self.npy) as source:
            array = CountingArray(self.values)
            source._dataset = array
            for args in cases:
                with self.subTest(args=list(args)), self.assertRaises((ValueError, TypeError)):
                    source.virtual_image(**args, show=False)
            self.assertEqual(array.calls, [])
            with self.assertRaisesRegex(TypeError, 'unexpected keyword'):
                source.virtual_image(msak=np.ones((8, 9)))

    def test_3d_stack_rejected_without_inventing_grid(self):
        path = self.root / 'stack.npy'
        np.save(path, self.values.reshape(35, 8, 9))
        with fd.HyperData.open(path) as source:
            with self.assertRaisesRegex(ValueError, '4D scan'):
                source.virtual_image(annulus=(1, 3), show=False)

    def test_polar_annuli_sectors_and_return_metadata(self):
        polar = self.data.to_polar(output_shape=(8, 32), progress=False)
        for data in (polar, polar.resize((4, 16), domain='reciprocal'),
                     polar.crop(kylim=(2, 7), kxlim=(8, 24))):
            data.save(self.saved, overwrite=True)
            with fd.HyperData.open(self.saved) as source:
                angles_to_test = (None, (90, 220))
                if data.shape[-1] == 32:
                    angles_to_test += ((300, 40),)
                for angles in angles_to_test:
                    args = dict(annulus=(0.2, 0.8), theta_range=angles, show=False,
                                return_detector=True, working_memory_mb=0.02)
                    expected, exp_detector = data.virtual_image(**args)
                    actual, detector = source.virtual_image(**args)
                    np.testing.assert_allclose(actual.array, expected.array)
                    np.testing.assert_allclose(detector.array, exp_detector.array)
                    self.assertTrue(detector.is_polar)
                    self.assertEqual(detector.polar_metadata, data.polar_metadata)
                with self.assertRaisesRegex(ValueError, 'Cartesian'):
                    source.virtual_image(radius=1, centers=(2, 3), show=False)

    def test_display_options_and_unmasked_mean_preview(self):
        mask = np.eye(8, 9, dtype=bool)
        with self.open(self.saved) as source:
            with patch.object(fd.RealSpace, 'show') as show, \
                    patch.object(fd.ReciprocalSpace, 'show', autospec=True) as preview:
                source.virtual_image(mask=mask, title='Bright field', grid=False,
                                     axes=False, axis_units='pixels', units='pixels',
                                     plot_mask=True, mask_cmap='magma', mask_vmin=8,
                                     mask_log_scale=False, ring_color='red', ring_alpha=0.7,
                                     mask_show_kwargs={'title': 'Detector preview'})
            self.assertEqual(show.call_args.kwargs['title'], 'Bright field')
            self.assertEqual(show.call_args.kwargs['axis_units'], 'pixels')
            self.assertFalse(show.call_args.kwargs['axes'])
            np.testing.assert_allclose(preview.call_args.args[0].array,
                                       self.values.mean(axis=(0, 1)))
            np.testing.assert_array_equal(preview.call_args.kwargs['overlay_mask'], mask)
            self.assertEqual(preview.call_args.kwargs['cmap'], 'magma')
            self.assertEqual(preview.call_args.kwargs['overlay_color'], 'red')
            self.assertEqual(preview.call_args.kwargs['overlay_alpha'], 0.7)
            self.assertEqual(preview.call_args.kwargs['title'], 'Detector preview')

    def test_source_and_eager_public_detector_parameters_match(self):
        import inspect
        self.assertEqual(inspect.signature(fd.DiffractionSource.virtual_image),
                         inspect.signature(fd.HyperData.virtual_image))

    def test_nonfinite_and_complex_follow_numpy_sums(self):
        values = self.values.astype(float)
        values[0, 0, 0, 0] = np.nan
        weights = np.ones((8, 9))
        weights[0, 0] = 0
        with np.errstate(invalid='ignore'):
            expected = np.sum(values * weights, axis=(-2, -1))
            actual = fd.HyperData(values).virtual_image(mask=weights, show=False)
        np.testing.assert_allclose(actual.array, expected, equal_nan=True)
        complex_values = self.values.astype(complex) * (1 + 2j)
        with self.assertRaisesRegex(TypeError, 'real-valued diffraction'):
            fd.HyperData(complex_values).virtual_image(mask=weights, show=False)
        actual, _ = detectors._project_virtual_detector(complex_values, weights)
        np.testing.assert_allclose(actual, np.sum(complex_values * weights, axis=(-2, -1)))


if __name__ == '__main__':
    unittest.main()
