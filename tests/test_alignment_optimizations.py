"""Regression coverage for alignment and local ring-background calculations."""

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


class AlignmentOptimizationTests(unittest.TestCase):
    def test_quick_com_matches_masked_reference_for_disk_and_ring(self):
        rng = np.random.default_rng(7)
        array = rng.random((2, 3, 17, 19))
        array[1, 2] = 0
        data = fd.HyperData(array)
        center = (7.5, 9.25)

        for radius in (4, (2, 5)):
            with self.subTest(radius=radius):
                mask = fd.make_mask(center, radius, mask_dim=array.shape[-2:])
                yy, xx = np.indices(mask.shape)
                weighted = array * mask
                mass = weighted.sum(axis=(-2, -1))
                expected_y = np.divide(
                    (weighted * yy).sum(axis=(-2, -1)), mass,
                    out=np.zeros(mass.shape), where=mass != 0,
                )
                expected_x = np.divide(
                    (weighted * xx).sum(axis=(-2, -1)), mass,
                    out=np.zeros(mass.shape), where=mass != 0,
                )
                actual_y, actual_x = data._quickCOM(r_mask=radius, center=center)
                np.testing.assert_allclose(actual_y, expected_y, atol=1e-12)
                np.testing.assert_allclose(actual_x, expected_x, atol=1e-12)

    def test_com_alignment_promotes_integer_data_before_subpixel_warp(self):
        array = np.zeros((1, 1, 17, 19), dtype=np.uint16)
        array[0, 0, 6, 11] = 100
        array[0, 0, 7, 11] = 200
        original = array.copy()

        with (
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
            redirect_stdout(StringIO()),
        ):
            aligned = fd.HyperData(array).alignment(
                method="com", center=(8, 9), r_center=5,
            )

        self.assertEqual(aligned.array.dtype.kind, "f")
        self.assertGreater(np.max(np.abs(aligned.array - np.rint(aligned.array))), 0)
        np.testing.assert_array_equal(array, original)

    def test_disk_alignment_keeps_center_metadata_and_input_unchanged(self):
        yy, xx = np.indices((33, 37))
        array = np.empty((2, 2, 33, 37), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                cy = 15 + i
                cx = 18 + j
                array[i, j] = 2 + 30 * (
                    (yy - cy) ** 2 + (xx - cx) ** 2 <= 4**2
                )
        original = array.copy()

        with (
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
            redirect_stdout(StringIO()),
        ):
            aligned = fd.HyperData(array).alignment(
                method="disk", center=(15.5, 18.5), r_center=4,
                search_radius=4, iterations=2,
            )

        np.testing.assert_array_equal(array, original)
        self.assertEqual(aligned.array.dtype.kind, "f")
        self.assertIn("center_px", aligned.center_beam_metadata)
        np.testing.assert_allclose(
            aligned.center_beam_metadata["center_px"],
            ((aligned.shape[-2] - 1) / 2, (aligned.shape[-1] - 1) / 2),
            atol=1.0,
        )


class RingBackgroundOptimizationTests(unittest.TestCase):
    def test_rings_match_full_mask_for_internal_edge_and_empty_windows(self):
        rng = np.random.default_rng(3)
        image = rng.random((31, 39))
        centers = np.array([(10.3, 9.6), (0.2, 1.1), (40.0, 50.0)])
        radius = (2.0, 5.0)

        actual = fd.ReciprocalSpace(image).get_residualBg(
            centers=centers, r_spots=radius,
        )
        expected = []
        for center in centers:
            mask = fd.make_mask(center, radius, mask_dim=image.shape)
            expected.append(image[mask].mean() if mask.any() else np.nan)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_show_displays_union_of_local_rings(self):
        image = np.ones((21, 23))
        centers = np.array([(8.5, 9.0), (12.0, 14.0)])
        radius = (1.0, 3.0)
        with patch.object(fd.ReciprocalSpace, "show", autospec=True) as show:
            fd.ReciprocalSpace(image).get_residualBg(
                centers=centers, r_spots=radius, show=True,
            )
        expected_mask = fd.make_mask(centers, radius, mask_dim=image.shape)
        np.testing.assert_array_equal(show.call_args[0][0].array, image * expected_mask)


if __name__ == "__main__":
    unittest.main()
