"""Regression tests for vectorized 4D median-filter routing."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import median_filter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class MedianDenoiseRegressionTests(unittest.TestCase):
    @staticmethod
    def deterministic_data():
        data = np.arange(5 * 6 * 4 * 3, dtype=np.float32).reshape(5, 6, 4, 3)
        data = np.ascontiguousarray(data)
        data[2, 3, 1, 2] = 100000.0
        return data

    def test_real_domain_matches_axis_restricted_scipy_bit_for_bit(self):
        data = self.deterministic_data()
        original = data.copy()
        actual = fd.HyperData(data).denoise(
            method="median",
            domain="real",
            window_size=3,
            mode="reflect",
            cval=0.0,
            origin=0,
            return_array=True,
        )
        expected = median_filter(
            data,
            size=3,
            mode="reflect",
            cval=0.0,
            origin=0,
            axes=(0, 1),
        )

        self.assertEqual(actual.shape, data.shape)
        self.assertEqual(actual.dtype, data.dtype)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(data.view(np.uint32), original.view(np.uint32))

    def test_reciprocal_domain_matches_axis_restricted_scipy_bit_for_bit(self):
        data = self.deterministic_data()
        actual = fd.HyperData(data).denoise(
            method="median",
            domain="reciprocal",
            window_size=3,
            mode="reflect",
            cval=0.0,
            origin=0,
            return_array=True,
        )
        expected = median_filter(
            data,
            size=3,
            mode="reflect",
            cval=0.0,
            origin=0,
            axes=(2, 3),
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_two_dimensional_behavior_and_boundary_arguments_are_preserved(self):
        image = np.arange(30, dtype=np.float64).reshape(5, 6)
        image[0, 0] = 999.0
        actual = fd.HyperData(image).denoise(
            method="median",
            window_size=3,
            mode="nearest",
            cval=-1.0,
            origin=0,
            return_array=True,
        )
        expected = median_filter(
            image,
            size=(3, 3),
            mode="nearest",
            cval=-1.0,
            origin=0,
        )
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_domain_owns_axes_for_four_dimensional_data(self):
        with self.assertRaisesRegex(TypeError, "axes are selected by domain"):
            fd.HyperData(self.deterministic_data()).denoise(
                method="median",
                domain="real",
                window_size=3,
                axes=(2, 3),
                return_array=True,
            )


if __name__ == "__main__":
    unittest.main()
