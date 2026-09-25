"""Regression tests for direct peak processing in HyperData."""

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


class HyperPeakFastPathTests(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(19).random((2, 2, 33, 39)) + 0.1
        self.reference = np.array([[8.0, 9.0], [23.0, 27.0]])
        self.radii = np.array([3.0, 4.0])

    def test_four_dimensional_fixed_peaks_and_mask(self):
        dataset = fd.HyperData(self.data)
        real_mask = np.array([[True, False], [True, True]])
        expected_centers = np.zeros((2, 2, 2, 2))
        expected_intensities = np.zeros((2, 2, 2))
        for i, j in np.ndindex(real_mask.shape):
            if real_mask[i, j]:
                dp = fd.ReciprocalSpace(self.data[i, j])
                expected_centers[i, j] = dp.get_centers(
                    self.radii, self.reference
                )
                expected_intensities[i, j] = dp.get_intensities(
                    self.radii, centers=expected_centers[i, j]
                )

        with (
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
            patch.object(dataset, "get_dp", side_effect=AssertionError("get_dp called")),
        ):
            centers = dataset.get_centers(
                self.radii, self.reference, real_mask=real_mask
            )
            known = dataset.get_intensities(
                self.radii, centers=centers, real_mask=real_mask
            )
            with patch.object(
                dataset, "get_centers", side_effect=AssertionError("second pass")
            ):
                inferred = dataset.get_intensities(
                    self.radii, ref_coords=self.reference, real_mask=real_mask
                )

        np.testing.assert_allclose(centers, expected_centers)
        np.testing.assert_allclose(known, expected_intensities)
        np.testing.assert_allclose(inferred, expected_intensities)

    def test_four_dimensional_ragged_peaks_and_mask(self):
        refs = [
            [self.reference, np.array([[16.0, 17.0]])],
            [np.empty((0, 2)), np.array([[8.0, 9.0]])],
        ]
        real_mask = np.array([[True, False], [True, True]])
        dataset = fd.HyperData(self.data)
        with patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable):
            centers = dataset.get_centers(3.0, refs, real_mask=real_mask)
            known = dataset.get_intensities(
                3.0, centers=centers, real_mask=real_mask
            )
            inferred = dataset.get_intensities(
                3.0, ref_coords=refs, real_mask=real_mask
            )

        self.assertIsInstance(inferred, list)
        self.assertEqual(len(inferred), 2)
        for i, j in np.ndindex(real_mask.shape):
            np.testing.assert_allclose(inferred[i][j], known[i][j])
        self.assertEqual(inferred[0][1].tolist(), [0.0])
        self.assertEqual(inferred[1][0].size, 0)

    def test_three_dimensional_fixed_and_ragged_peaks(self):
        dataset = fd.HyperData(self.data.reshape(4, 33, 39))
        refs = [
            self.reference,
            np.array([[16.0, 17.0]]),
            np.empty((0, 2)),
            self.reference,
        ]
        with patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable):
            fixed_centers = dataset.get_centers(3.0, self.reference)
            fixed_known = dataset.get_intensities(3.0, centers=fixed_centers)
            fixed_inferred = dataset.get_intensities(
                3.0, ref_coords=self.reference
            )
            ragged_centers = dataset.get_centers(3.0, refs)
            ragged_known = dataset.get_intensities(3.0, centers=ragged_centers)
            ragged_inferred = dataset.get_intensities(3.0, ref_coords=refs)

        self.assertEqual(fixed_centers.shape, (4, 2, 2))
        np.testing.assert_allclose(fixed_known, fixed_inferred)
        for known, inferred in zip(ragged_known, ragged_inferred):
            np.testing.assert_allclose(known, inferred)
        self.assertEqual(ragged_inferred[2].size, 0)

    def test_background_subtraction_still_uses_existing_estimator(self):
        dataset = fd.HyperData(self.data.reshape(4, 33, 39)[:2])
        with patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable):
            centers = dataset.get_centers(self.radii, self.reference)
            raw = dataset.get_intensities(
                self.radii, ref_coords=self.reference
            )
            with patch.object(
                fd.ReciprocalSpace, "get_residualBg",
                return_value=np.array([0.2, 0.4]),
            ) as background:
                corrected = dataset.get_intensities(
                    self.radii,
                    ref_coords=self.reference,
                    compute_resBg=True,
                    residual_frac=0.5,
                )

        self.assertEqual(background.call_count, 2)
        ones = fd.ReciprocalSpace(np.ones(dataset.shape[-2:]))
        pixel_counts = np.array([
            ones.get_intensities(self.radii, centers=dp_centers)
            for dp_centers in centers
        ])
        np.testing.assert_allclose(
            corrected,
            raw - pixel_counts * np.array([0.2, 0.4]) * 0.5,
        )

    def test_four_dimensional_edge_background_respects_real_mask(self):
        dataset = fd.HyperData(np.full((2, 2, 11, 11), 5.0))
        centers = np.broadcast_to(
            [[0.0, 0.0], [5.0, 5.0]], (2, 2, 2, 2)
        ).copy()
        real_mask = np.array([[True, False], [True, True]])
        with (
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
            patch.object(
                fd.ReciprocalSpace, "get_residualBg",
                return_value=np.array([1.0, 2.0]),
            ) as background,
        ):
            corrected = dataset.get_intensities(
                1.6, centers=centers, compute_resBg=True,
                residual_frac=1.0, real_mask=real_mask,
            )

        self.assertEqual(background.call_count, 3)
        np.testing.assert_allclose(
            corrected[real_mask], np.tile([16.0, 27.0], (3, 1))
        )
        np.testing.assert_array_equal(corrected[~real_mask], [[0.0, 0.0]])

    def test_gaussian_fitters_keep_yx_order_and_use_selected_model(self):
        dp = fd.ReciprocalSpace(np.ones((21, 21)))
        with patch.object(fd, "fit_gaussian_2d", return_value=(2.5, 4.5)) as fit:
            center = dp.get_spotCenter(10, 12, 3, method="gaussian")
        fit.assert_called_once()
        np.testing.assert_allclose(center, (9.5, 13.5))

        with (
            patch.object(fd, "fit_gaussian_2d", side_effect=AssertionError("wrong model")),
            patch.object(fd, "fit_elliptical_gaussian_2d", return_value=(2.5, 4.5)) as fit,
        ):
            center = dp.get_spotCenter(10, 12, 3, method="elliptical_gaussian")
        fit.assert_called_once()
        np.testing.assert_allclose(center, (9.5, 13.5))

        with self.assertRaisesRegex(ValueError, "method must be"):
            dp.get_spotCenter(10, 12, 3, method="unknown")

    def test_gaussian_models_fit_an_off_axis_synthetic_peak(self):
        yy, xx = np.mgrid[:41, :41]
        params = (18.3, 24.6, 2.5, 3.7)
        cases = [
            ("gaussian", fd.gaussian_2d((yy, xx), *params, 100.0, 0.2)),
            (
                "elliptical_gaussian",
                fd.elliptical_gaussian_2d(
                    (yy, xx), *params, 0.4, 100.0, 0.2
                ),
            ),
        ]
        for method, image in cases:
            with self.subTest(method=method):
                actual = fd.ReciprocalSpace(image).get_spotCenter(
                    18, 25, 8, method=method
                )
                np.testing.assert_allclose(actual, params[:2], atol=0.1)


if __name__ == "__main__":
    unittest.main()
