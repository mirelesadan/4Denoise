"""Simulation peak adapters share the main analysis without requiring abTEM."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':
    raise ImportError(f"Tests imported the wrong fourdenoise.py: {fd.__file__}")


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        '_simulation_peak_helpers_test', ROOT / 'abTEM_helpers.py',
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'abtem': types.ModuleType('abtem')}):
        spec.loader.exec_module(module)
    return module


class SimulationPeakHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_helpers()

    def setUp(self):
        self.image = np.random.default_rng(37).random((41, 47)) + 0.1
        self.pattern = fd.ReciprocalSpace(self.image)
        self.coords = np.array([
            (row, column) for row in (7.2, 19.4, 32.1)
            for column in (5.3, 12.6, 19.1, 26.8, 33.3, 40.7)
        ])
        self.radii = np.linspace(1.5, 4.0, len(self.coords))

    def test_default_scalar_radius_matches_main_centers(self):
        actual = self.helpers.getCenters(self.image.tolist(), coords=self.coords.tolist())
        np.testing.assert_array_equal(actual, self.pattern.get_centers(6, self.coords))

    def test_per_peak_radii_match_main_centers_and_do_not_mutate_inputs(self):
        coords_before, radii_before, image_before = (
            self.coords.copy(), self.radii.copy(), self.image.copy(),
        )
        actual = self.helpers.getCenters(self.image, coords=self.coords, r=self.radii)
        np.testing.assert_array_equal(actual, self.pattern.get_centers(self.radii, self.coords))
        np.testing.assert_array_equal(self.coords, coords_before)
        np.testing.assert_array_equal(self.radii, radii_before)
        np.testing.assert_array_equal(self.image, image_before)

    def test_orders_select_matching_coordinates_and_radii(self):
        for order in (1, np.int64(2), 3, 4):
            selection = slice(None, 18) if order == 4 else slice((order - 1) * 6, order * 6)
            expected = self.pattern.get_centers(self.radii[selection], self.coords[selection])
            for radii in (self.radii, self.radii[selection]):
                with self.subTest(order=order, radii_count=len(radii)):
                    actual = self.helpers.getCenters(
                        self.image, spots_order=order, coords=self.coords, r=radii,
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_scalar_radius_and_custom_spot_count_for_orders(self):
        for order, selection in ((2, slice(3, 6)), (4, slice(0, 9))):
            with self.subTest(order=order):
                actual = self.helpers.getCenters(
                    self.image, order, coords=self.coords,
                    num_spots_per_order=np.int64(3),
                )
                np.testing.assert_array_equal(
                    actual, self.pattern.get_centers(6, self.coords[selection]),
                )

    def test_iterations_repeat_main_refinement(self):
        expected = self.coords
        for _ in range(3):
            expected = self.pattern.get_centers(self.radii, expected)
        actual = self.helpers.getCenters(
            self.image, coords=self.coords, r=self.radii, iterations=np.int64(3),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_get_centers_delegates_plotting_each_iteration(self):
        first, second = self.coords + 0.1, self.coords + 0.2
        with patch.object(
            fd.ReciprocalSpace, 'get_centers', autospec=True, side_effect=[first, second],
        ) as refine:
            actual = self.helpers.getCenters(
                self.image, coords=self.coords, iterations=2, plotSpots=True,
            )
        self.assertIs(actual, second)
        self.assertEqual(refine.call_count, 2)
        np.testing.assert_array_equal(refine.call_args_list[0].kwargs['ref_coords'], self.coords)
        self.assertIs(refine.call_args_list[1].kwargs['ref_coords'], first)
        for call in refine.call_args_list:
            np.testing.assert_array_equal(call.args[1], np.full(18, 6))
            self.assertTrue(call.kwargs['show'])

    def test_single_spot_matches_main_at_edges_and_fractional_coordinates(self):
        for ky, kx, radius in (
            (0, 0, 2), (0.3, 46.1, 3.25), (40.0, 0.8, 4),
            (40.6, 46.0, 2.5), (15.3, 21.7, 5),
        ):
            for iterations in (1, 3):
                with self.subTest(ky=ky, kx=kx, r=radius, iterations=iterations):
                    expected = (ky, kx)
                    for _ in range(iterations):
                        expected = self.pattern.get_spotCenter(*expected, radius)
                    actual = self.helpers.getSpotCenter(
                        ky, kx, radius, self.image, iterations=iterations,
                    )
                    self.assertIsInstance(actual, tuple)
                    np.testing.assert_array_equal(actual, expected)

    def test_single_spot_forwards_plotting_and_updated_coordinates(self):
        with patch.object(
            fd.ReciprocalSpace, 'get_spotCenter', autospec=True,
            side_effect=[(2.25, 3.5), (2.5, 3.75)],
        ) as refine:
            actual = self.helpers.getSpotCenter(2, 3, 1.5, self.image, True, 2)
        self.assertEqual(actual, (2.5, 3.75))
        self.assertEqual(refine.call_count, 2)
        self.assertEqual(refine.call_args_list[0].args[1:], (2, 3, 1.5))
        self.assertEqual(refine.call_args_list[1].args[1:], (2.25, 3.5, 1.5))
        self.assertEqual(refine.call_args_list[1].kwargs, {'plotSpot': True})

    def test_default_intensities_with_centers_do_not_require_coords(self):
        with patch.object(self.helpers, 'make_mask', side_effect=AssertionError('duplicate mask')):
            actual = self.helpers.getIntensities(self.image, center=self.coords.tolist())
        np.testing.assert_array_equal(actual, self.pattern.get_intensities(6, centers=self.coords))

    def test_missing_centers_are_refined_by_main_intensity_method(self):
        for radius in (6, self.radii):
            with self.subTest(radius=radius):
                actual = self.helpers.getIntensities(self.image, r=radius, coords=self.coords)
                expected = self.pattern.get_intensities(radius, ref_coords=self.coords)
                np.testing.assert_array_equal(actual, expected)

    def test_intensity_orders_support_refined_full_and_selected_centers(self):
        for order in (1, np.int64(2), 3, 4):
            selection = slice(0, 18) if order == 4 else slice((order - 1) * 6, order * 6)
            for centers in (self.coords, self.coords[selection]):
                with self.subTest(order=order, count=len(centers)):
                    radii = self.radii if len(centers) == 18 else self.radii[selection]
                    actual = self.helpers.getIntensities(
                        self.image, order, center=centers, r=radii,
                    )
                    np.testing.assert_array_equal(actual, self.pattern.get_intensities(
                        self.radii[selection], centers=self.coords[selection],
                    ))
            actual = self.helpers.getIntensities(
                self.image, order, coords=self.coords, r=self.radii,
            )
            np.testing.assert_array_equal(actual, self.pattern.get_intensities(
                self.radii[selection], ref_coords=self.coords[selection],
            ))

    def test_intensity_delegation_uses_centers_or_reference_coordinates(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                with patch.object(
                    fd.ReciprocalSpace, 'get_intensities', autospec=True,
                    return_value=np.array([42.0]),
                ) as integrate:
                    actual = self.helpers.getIntensities(
                        self.image, center=self.coords if explicit else None, coords=self.coords,
                    )
                np.testing.assert_array_equal(actual, [42.0])
                integrate.assert_called_once()
                np.testing.assert_array_equal(integrate.call_args.args[1], np.full(18, 6))
                key = 'centers' if explicit else 'ref_coords'
                self.assertEqual(set(integrate.call_args.kwargs), {key})
                np.testing.assert_array_equal(integrate.call_args.kwargs[key], self.coords)

    def test_intensities_use_clipped_main_windows(self):
        centers = np.array([[0, 0], [40.4, 46.1], [-1.1, 18.2], [80, 80]])
        radii = np.array([2, 3.25, 2.5, 4])
        expected = self.pattern.get_intensities(radii, centers=centers)
        actual = self.helpers.getIntensities(self.image, center=centers, r=radii)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[-1], 0)

    def test_zero_signal_preserves_main_nonfinite_center_semantics(self):
        image = np.zeros((11, 13))
        with np.errstate(divide='ignore', invalid='ignore'):
            actual = self.helpers.getCenters(image, coords=[[5, 6]])
            single = self.helpers.getSpotCenter(5, 6, 2, image)
            expected = fd.ReciprocalSpace(image).get_centers(6, np.array([[5, 6]]))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isnan(single).all())

    def test_empty_peak_arrays_match_main(self):
        empty = np.empty((0, 2))
        self.assertEqual(self.helpers.getCenters(self.image, coords=empty).shape, (0, 2))
        self.assertEqual(self.helpers.getIntensities(self.image, center=empty).shape, (0,))
        self.assertEqual(self.helpers.getIntensities(self.image, coords=empty).shape, (0,))

    def test_masks_delegate_and_preserve_legacy_dtype(self):
        center, radius, shape = (1.3, 2.7), 2.5, (9, 13)
        expected = fd.make_mask(center, radius, shape)
        with patch.object(fd, 'make_mask', wraps=fd.make_mask) as mask:
            actual = self.helpers.make_mask(center, radius, shape)
        mask.assert_called_once()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(float))
        with patch.object(fd, 'circular_mask', wraps=fd.circular_mask) as circular:
            actual = self.helpers.circular_mask(2, 3, 3.25)
        circular.assert_called_once_with(2, 3, 3.25)
        np.testing.assert_array_equal(actual, fd.circular_mask(2, 3, 3.25))

    def test_invalid_patterns_raise_clear_errors(self):
        for image in (np.zeros(3), np.zeros((2, 2, 2)), np.empty((0, 3)), [['bad']], [[1j]], [[1], [1, 2]]):
            for operation in (
                lambda: self.helpers.getCenters(image, coords=[[0, 0]]),
                lambda: self.helpers.getIntensities(image, center=[[0, 0]]),
                lambda: self.helpers.getSpotCenter(0, 0, 1, image),
            ):
                with self.subTest(image=image):
                    with self.assertRaisesRegex(ValueError, '2D diffraction pattern'):
                        operation()

    def test_invalid_coordinate_arrays_raise_clear_errors(self):
        for coords in (
            None, [1, 2], [[1]], [[1, 2, 3]], [[np.nan, 1]], [[1, np.inf]],
            [['bad', 1]], np.array([[1 + 2j, 3]]), [[1, 2], [3]],
        ):
            for operation in (
                lambda: self.helpers.getCenters(self.image, coords=coords),
                lambda: self.helpers.getIntensities(self.image, coords=coords),
                lambda: self.helpers.getIntensities(self.image, center=coords),
            ):
                with self.subTest(coords=coords):
                    with self.assertRaisesRegex(ValueError, 'shape.*finite'):
                        operation()
        for ky, kx in ((np.nan, 1), (1, np.inf), ([1], 2)):
            with self.subTest(ky=ky, kx=kx):
                with self.assertRaisesRegex(ValueError, 'shape.*finite'):
                    self.helpers.getSpotCenter(ky, kx, 2, self.image)

    def test_invalid_radii_raise_clear_errors(self):
        for radius in (-1, np.inf, np.nan, [2], [2] * 17, [[2] * 18], 'bad', np.array(2 + 1j)):
            for operation in (
                lambda: self.helpers.getCenters(self.image, coords=self.coords, r=radius),
                lambda: self.helpers.getIntensities(self.image, center=self.coords, r=radius),
            ):
                with self.subTest(radius=radius):
                    with self.assertRaisesRegex(ValueError, 'r must'):
                        operation()

    def test_invalid_iterations_and_counts_raise_clear_errors(self):
        for value in (0, -1, 1.5, True, np.bool_(True), '2'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'iterations.*positive integer'):
                    self.helpers.getCenters(self.image, coords=self.coords, iterations=value)
                with self.assertRaisesRegex(ValueError, 'iterations.*positive integer'):
                    self.helpers.getSpotCenter(7, 8, 2, self.image, iterations=value)
                with self.assertRaisesRegex(ValueError, 'num_spots_per_order.*positive integer'):
                    self.helpers.getCenters(self.image, coords=self.coords, num_spots_per_order=value)

    def test_invalid_orders_and_insufficient_peaks_raise_clear_errors(self):
        for order in (0, -1, 1.5, True, np.bool_(True), '2', 5):
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, 'spots_order'):
                    self.helpers.getCenters(self.image, order, coords=self.coords)
                with self.assertRaisesRegex(ValueError, 'spots_order'):
                    self.helpers.getIntensities(self.image, order, center=self.coords)
        for order, count in ((1, 5), (2, 11), (3, 17), (4, 17)):
            with self.subTest(order=order, count=count):
                with self.assertRaisesRegex(ValueError, 'peaks for spots_order'):
                    self.helpers.getCenters(self.image, order, coords=self.coords[:count])
                with self.assertRaisesRegex(ValueError, 'peaks for spots_order'):
                    self.helpers.getIntensities(self.image, order, center=self.coords[:count])

    def test_mask_validation_is_clear(self):
        for center, radius, shape in (
            ((np.nan, 1), 2, (5, 5)), ((1, 2, 3), 2, (5, 5)),
            ((1, 2), -1, (5, 5)), ((1, 2), 2, (0, 5)),
            ((1, 2), 2, (3.5, 5)), ((1, 2), 2, (5,)), ((1, 2), 2, np.array(5)),
        ):
            with self.subTest(center=center, radius=radius, shape=shape):
                with self.assertRaises(ValueError):
                    self.helpers.make_mask(center, radius, shape)


if __name__ == '__main__':
    unittest.main()
