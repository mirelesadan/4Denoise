"""Coordinate-based crop regressions using ramps, peaks, and detector counts."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import resize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Tests imported the wrong fourdenoise.py: {fd.__file__}")


class FractionalCropTests(unittest.TestCase):
    def setUp(self):
        self.y, self.x = np.indices((16, 20))
        self.ramp = (100 * self.y + self.x).astype(np.float64)

    def crop_both(self, image, y_limits, x_limits, shape=None):
        hyper = fd.HyperData(image[None, None]).crop(
            kylim=y_limits, kxlim=x_limits, kshape=shape,
            reciprocal_limit_units='pixels',
        )
        pattern = fd.ReciprocalSpace(image).crop(
            kylim=y_limits, kxlim=x_limits, kshape=shape,
        )
        return hyper.array[0, 0], pattern.array

    @staticmethod
    def sample_axis(limits, size):
        return limits[0] + (np.arange(size) + 0.5) * (limits[1] - limits[0]) / size - 0.5

    def test_different_offsets_do_not_return_the_same_crop(self):
        first = self.crop_both(self.ramp, (1.2, 5.2), (1.2, 5.2), (4, 4))
        second = self.crop_both(self.ramp, (1.8, 5.8), (1.8, 5.8), (4, 4))
        for a, b in zip(first, second):
            self.assertFalse(np.array_equal(a, b))
            np.testing.assert_allclose(b - a, 60.6, atol=1e-12)

    def test_fractional_ramp_matches_analytical_sample_positions(self):
        y_limits, x_limits = (3.25, 7.25), (4.6, 8.6)
        y = self.sample_axis(y_limits, 4)
        x = self.sample_axis(x_limits, 4)
        expected = 100 * y[:, None] + x[None, :]
        for actual in self.crop_both(self.ramp, y_limits, x_limits, (4, 4)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_fractional_upsampling_retains_window_position_and_width(self):
        y_limits, x_limits = (3.2, 7.2), (5.8, 9.8)
        expected = (
            100 * self.sample_axis(y_limits, 8)[:, None]
            + self.sample_axis(x_limits, 8)[None, :]
        )
        for actual in self.crop_both(self.ramp, y_limits, x_limits, (8, 8)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_interpolation_uses_neighbors_outside_enclosing_slice(self):
        # The first upsampled point lies just below each integer slice boundary.
        y_limits, x_limits = (3.1, 7.1), (5.1, 9.1)
        expected = (
            100 * self.sample_axis(y_limits, 8)[:, None]
            + self.sample_axis(x_limits, 8)[None, :]
        )
        for actual in self.crop_both(self.ramp, y_limits, x_limits, (8, 8)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_small_offsets_are_not_rounded_away(self):
        for actual in self.crop_both(self.ramp, (8.000001, 12.000001), (8, 12), (4, 4)):
            np.testing.assert_allclose(actual - self.ramp[8:12, 8:12], 0.0001, atol=1e-12)

    def test_default_shape_uses_requested_extent_for_both_classes(self):
        for actual in self.crop_both(self.ramp, (1.2, 5.2), (1.8, 5.8)):
            self.assertEqual(actual.shape, (4, 4))
        for actual in self.crop_both(self.ramp, (1.1, 1.3), (2.2, 2.4)):
            self.assertEqual(actual.shape, (1, 1))

    def test_localized_peak_matches_independent_bilinear_interpolation(self):
        y_profile = np.exp(-0.5 * ((np.arange(16) - 6.3) / 1.1) ** 2)
        x_profile = np.exp(-0.5 * ((np.arange(20) - 8.2) / 1.3) ** 2)
        image = y_profile[:, None] * x_profile[None, :]
        y_limits, x_limits = (2.2, 10.2), (4.6, 12.6)
        expected = (
            np.interp(self.sample_axis(y_limits, 8), np.arange(16), y_profile)[:, None]
            * np.interp(self.sample_axis(x_limits, 8), np.arange(20), x_profile)[None, :]
        )
        for actual in self.crop_both(image, y_limits, x_limits, (8, 8)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_integer_crops_remain_exact_views_without_dtype_change(self):
        image = self.ramp.astype(np.uint16)
        for actual in self.crop_both(image, (3, 7), (5, 9), (4, 4)):
            np.testing.assert_array_equal(actual, image[3:7, 5:9])
            self.assertEqual(actual.dtype, image.dtype)
            self.assertTrue(np.shares_memory(actual, image))

    def test_integer_valued_float_limits_do_not_force_interpolation(self):
        image = self.ramp.astype(np.uint16)
        for actual in self.crop_both(image, (3.0, 7.0), (5.0, 9.0)):
            np.testing.assert_array_equal(actual, image[3:7, 5:9])
            self.assertEqual(actual.dtype, image.dtype)

    def test_fractional_integer_counts_are_promoted_and_input_is_unchanged(self):
        image = self.ramp.astype(np.uint16)
        before = image.copy()
        expected = 100 * (3.2 + np.arange(4))[:, None] + (5.6 + np.arange(4))[None, :]
        for actual in self.crop_both(image, (3.2, 7.2), (5.6, 9.6), (4, 4)):
            self.assertEqual(actual.dtype.kind, 'f')
            self.assertFalse(np.shares_memory(actual, image))
            np.testing.assert_allclose(actual, expected, atol=3e-5)
        np.testing.assert_array_equal(image, before)

    def test_hyperdata_scan_crop_does_not_mix_diffraction_patterns(self):
        offsets = np.arange(6).reshape(2, 3) * 10000
        source = self.ramp[None, None] + offsets[:, :, None, None]
        result = fd.HyperData(source).crop(
            ylim=(0, 2), xlim=(1, 3), kylim=(3.2, 7.2), kxlim=(5.6, 9.6),
            real_limit_units='pixels', reciprocal_limit_units='pixels',
        )
        expected = (
            100 * (3.2 + np.arange(4))[:, None]
            + (5.6 + np.arange(4))[None, :]
        )
        np.testing.assert_allclose(result.array, expected + offsets[:, 1:, None, None])

    def test_calibrated_limits_match_equivalent_pixel_bounds(self):
        factor = 0.2
        data = fd.HyperData(
            self.ramp[None, None], reciprocal_units='mrad', reciprocal_conv_factor=factor,
        )
        y_limits, x_limits = (3.2, 7.2), (5.6, 9.6)
        y_origin, x_origin = (np.array(self.ramp.shape) - 1) / 2
        calibrated_y = tuple((y_origin - value) * factor for value in reversed(y_limits))
        calibrated_x = tuple((value - x_origin) * factor for value in x_limits)
        actual = data.crop(
            kylim=calibrated_y, kxlim=calibrated_x, reciprocal_limit_units='calibrated',
        )
        expected = data.crop(
            kylim=y_limits, kxlim=x_limits, reciprocal_limit_units='pixels',
        )
        np.testing.assert_allclose(actual.array, expected.array, atol=1e-12)
        self.assertAlmostEqual(actual.reciprocal_conv_factor, factor)

    def test_fractional_crop_spacing_is_based_on_actual_window_extent(self):
        beam = fd._center_beam_metadata_from_pixels(1, (5, 7), self.ramp.shape)
        data = fd.HyperData(
            self.ramp[None, None], real_units='nm', real_conv_factor=0.5,
            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.2,
            center_beam_metadata=beam,
        )
        hyper = data.crop(
            kylim=(2.2, 8.2), kxlim=(4.6, 10.6), kshape=(3, 3),
            reciprocal_limit_units='pixels',
        )
        pattern = data.get_dp(0, 0).crop(
            kylim=(2.2, 8.2), kxlim=(4.6, 10.6), kshape=(3, 3),
        )
        self.assertAlmostEqual(hyper.reciprocal_conv_factor, 0.4)
        self.assertAlmostEqual(pattern.conv_factor, 0.4)
        self.assertEqual(hyper.real_units, data.real_units)
        self.assertEqual(hyper.real_origin, data.real_origin)
        self.assertIsNone(hyper.center_beam_metadata)
        self.assertIsNone(pattern.center_beam_metadata)

    def test_full_frame_downsampling_matches_existing_anti_aliasing(self):
        image = ((self.y + self.x) % 2).astype(np.float32) * 100
        expected = resize(
            image, (4, 5), order=1, mode='reflect', anti_aliasing=True, preserve_range=True,
        )
        for actual in self.crop_both(image, None, None, (4, 5)):
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_filter_halo_matches_full_image_filter_before_sampling(self):
        image = np.random.default_rng(42).normal(size=self.ramp.shape)
        y_limits, x_limits = (3.2, 15.2), (4.6, 16.6)
        filtered = gaussian_filter(image, (1.5, 1.5), mode='mirror')
        grid = np.array(np.meshgrid(
            self.sample_axis(y_limits, 3), self.sample_axis(x_limits, 3), indexing='ij',
        ))
        expected = map_coordinates(filtered, grid, order=1, mode='mirror', prefilter=False)
        for actual in self.crop_both(image, y_limits, x_limits, (3, 3)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_detector_edge_reflection_uses_actual_subpixel_coordinates(self):
        y_limits, x_limits = (0.1, 4.1), (0.2, 4.2)
        expected = (
            100 * np.abs(self.sample_axis(y_limits, 8))[:, None]
            + np.abs(self.sample_axis(x_limits, 8))[None, :]
        )
        for actual in self.crop_both(self.ramp, y_limits, x_limits, (8, 8)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_invalid_limits_and_shapes_raise_value_errors(self):
        hyper = fd.HyperData(self.ramp[None, None])
        pattern = fd.ReciprocalSpace(self.ramp)
        for y_limits in ((-1, 3), (2, 17), (4, 4), (5, 4), (np.nan, 4),
                         (2, np.inf), (1, 2, 3), 16):
            for obj in (hyper, pattern):
                with self.subTest(type=type(obj).__name__, y_limits=y_limits):
                    with self.assertRaisesRegex(ValueError, 'kylim'):
                        obj.crop(kylim=y_limits)
        for shape in ((0, 4), (-1, 4), (2.5, 4), (3,), (3, np.nan), ('a', 'b')):
            for obj in (hyper, pattern):
                with self.subTest(type=type(obj).__name__, shape=shape):
                    with self.assertRaisesRegex(ValueError, 'kshape'):
                        obj.crop(kshape=shape)

    def test_list_and_numpy_limits_and_shapes_are_supported(self):
        for actual in self.crop_both(
            self.ramp, [3.2, 7.2], np.array([5.6, 9.6]), np.array([4, 4]),
        ):
            self.assertEqual(actual.shape, (4, 4))


if __name__ == '__main__':
    unittest.main()
