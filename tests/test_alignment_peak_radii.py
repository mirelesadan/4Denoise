"""Regression tests for per-peak radii and center-of-mass alignment."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class PeakRadiusRegressionTests(unittest.TestCase):
    def setUp(self):
        image = np.zeros((37, 37), dtype=float)
        image[8, 8] = 10.0
        image[24, 24] = 10.0
        image[27, 24] = 15.0
        self.dp = fd.ReciprocalSpace(image)
        self.centers = np.array([[8.0, 8.0], [24.0, 24.0]])
        self.radii = np.array([2.0, 4.0])

    def test_get_centers_uses_each_peaks_radius(self):
        actual = self.dp.get_centers(self.radii.tolist(), self.centers)
        expected = np.array([
            self.dp.get_spotCenter(*center, radius + 1e-10)
            for center, radius in zip(self.centers, self.radii)
        ])
        np.testing.assert_allclose(actual, expected)
        self.assertGreater(actual[1, 0], 25.0)

    def test_get_intensities_uses_each_peaks_radius_and_pixel_count(self):
        actual = self.dp.get_intensities(self.radii, centers=self.centers)
        expected = np.array([
            self.dp.get_intensities(radius, centers=center[None, :])[0]
            for center, radius in zip(self.centers, self.radii)
        ])
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual, [10.0, 25.0])

        background = np.array([1.0, 2.0])
        with patch.object(self.dp, "get_residualBg", return_value=background):
            corrected = self.dp.get_intensities(
                self.radii,
                centers=self.centers,
                compute_resBg=True,
                residual_frac=0.8,
            )
        np.testing.assert_allclose(
            corrected,
            expected - background * fd.ReciprocalSpace(
                np.ones_like(self.dp.array)
            ).get_intensities(self.radii, centers=self.centers) * 0.8,
        )

    def test_background_subtraction_counts_clipped_and_empty_windows(self):
        image = np.full((11, 11), 5.0)
        dp = fd.ReciprocalSpace(image)
        centers = np.array([[0.0, 0.0], [5.0, 5.0], [30.0, 30.0]])
        with patch.object(
            dp, "get_residualBg", return_value=np.array([1.0, 2.0, np.nan])
        ):
            corrected = dp.get_intensities(
                1.6, centers=centers, compute_resBg=True, residual_frac=1.0
            )

        np.testing.assert_allclose(corrected, [16.0, 27.0, 0.0])

    def test_scalar_background_is_applied_per_integrated_pixel(self):
        dp = fd.ReciprocalSpace(np.full((11, 11), 5.0))
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        with patch.object(dp, "get_residualBg", return_value=2.0):
            corrected = dp.get_intensities(
                1.6, centers=centers, compute_resBg=True, residual_frac=0.5
            )

        np.testing.assert_allclose(corrected, [16.0, 36.0])

    def test_radius_count_must_match_peak_count(self):
        with self.assertRaisesRegex(ValueError, "one radius per peak"):
            self.dp.get_centers([2.0], self.centers)
        with self.assertRaisesRegex(ValueError, "one radius per peak"):
            self.dp.get_intensities([2.0], centers=self.centers)

    def test_hyperdata_routes_peak_radii_through_both_methods(self):
        data = np.stack((self.dp.array, self.dp.array * 2.0), axis=0)[None, ...]
        dataset = fd.HyperData(data)
        with patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable):
            centers = dataset.get_centers(self.radii, self.centers)
            intensities = dataset.get_intensities(self.radii, centers=centers)

        self.assertEqual(centers.shape, (1, 2, 2, 2))
        np.testing.assert_allclose(intensities[0, 0], [10.0, 25.0])
        np.testing.assert_allclose(intensities[0, 1], [20.0, 50.0])


class AlignmentRegressionTests(unittest.TestCase):
    def test_com_alignment_preserves_xy_axis_order(self):
        data = np.zeros((1, 1, 17, 17), dtype=float)
        data[0, 0, 6, 11] = 1.0

        with (
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
            redirect_stdout(StringIO()),
        ):
            aligned = fd.HyperData(data).alignment(
                method="com", center=(8, 8), r_center=6, iterations=1
            )

        peak = np.unravel_index(np.argmax(aligned.array[0, 0]), (17, 17))
        self.assertEqual(peak, (8, 8))


if __name__ == "__main__":
    unittest.main()
