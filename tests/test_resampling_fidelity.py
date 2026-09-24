"""Regression tests for detector-count fidelity during spatial resampling."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class ResamplingFidelityTests(unittest.TestCase):
    def test_hyper_crop_preserves_integer_count_range_in_reciprocal_space(self):
        original = np.full((2, 2, 4, 4), 100, dtype=np.uint16)
        data = fd.HyperData(original)

        resized = data.crop(kshape=(2, 2))
        subpixel = data.crop(
            kylim=(0.5, 3.5), kxlim=(0.5, 3.5),
            reciprocal_limit_units="pixels",
        )

        self.assertEqual(resized.array.dtype.kind, "f")
        self.assertEqual(subpixel.array.dtype.kind, "f")
        np.testing.assert_allclose(resized.array, 100)
        np.testing.assert_allclose(subpixel.array, 100)
        np.testing.assert_array_equal(original, 100)

    def test_hyper_crop_keeps_fractional_block_means_and_interpolation(self):
        scan = np.array([[100, 101], [102, 103]], dtype=np.uint16)
        original = np.broadcast_to(scan[:, :, None, None], (2, 2, 2, 2)).copy()
        binned = fd.HyperData(original).crop(rshape=(1, 1))

        self.assertEqual(binned.array.dtype.kind, "f")
        np.testing.assert_allclose(binned.array, 101.5)

        constant = fd.HyperData(np.full((3, 3, 2, 2), 100, dtype=np.uint16))
        interpolated = constant.crop(rshape=(2, 2))
        self.assertEqual(interpolated.array.dtype.kind, "f")
        np.testing.assert_allclose(interpolated.array, 100)

    def test_reciprocal_crop_avoids_unneeded_resampling(self):
        original = np.full((4, 4), 100, dtype=np.uint16)
        pattern = fd.ReciprocalSpace(original)

        untouched = pattern.crop()
        resized = pattern.crop(kshape=(2, 2))
        subpixel = pattern.crop(kylim=(0.5, 3.5), kxlim=(0.5, 3.5))

        self.assertEqual(untouched.array.dtype, original.dtype)
        np.testing.assert_array_equal(untouched.array, original)
        for result in (resized, subpixel):
            self.assertEqual(result.array.dtype.kind, "f")
            np.testing.assert_allclose(result.array, 100)

    def test_polar_interpolation_keeps_subpixel_values_for_3d_and_4d(self):
        image = np.broadcast_to(np.arange(8, dtype=np.uint16), (8, 8))
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                data = fd.HyperData(image.reshape(stack_shape + (8, 8)))
                polar = data.to_polar(
                    output_shape=(4, 16), order=1, clip=False, progress=False
                )
                nearest = data.to_polar(
                    output_shape=(4, 16), order=0, clip=False, progress=False
                )

                self.assertEqual(polar.array.dtype.kind, "f")
                self.assertEqual(nearest.array.dtype, image.dtype)
                np.testing.assert_allclose(
                    polar.array[(0,) * len(stack_shape) + (0, 0)], 3.5
                )

    def test_cartesian_interpolation_keeps_subpixel_values_for_3d_and_4d(self):
        image = np.broadcast_to(
            np.arange(5, dtype=np.uint16)[:, None], (5, 16)
        )
        metadata = {
            "r_max": 5.0,
            "radius_display_range": (0.0, 5.0),
            "radius_units": "pixels",
        }
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                data = fd.HyperData(
                    image.reshape(stack_shape + (5, 16)),
                    polar_metadata=metadata,
                )
                cartesian = data.to_cartesian(
                    output_shape=(10, 10), order=1,
                    clip=False, progress=False,
                )
                nearest = data.to_cartesian(
                    output_shape=(10, 10), order=0,
                    clip=False, progress=False,
                )

                self.assertEqual(cartesian.array.dtype.kind, "f")
                self.assertEqual(nearest.array.dtype, image.dtype)
                value = cartesian.array[(0,) * len(stack_shape) + (4, 4)]
                self.assertGreater(value, 0)
                self.assertLess(value, 1)


if __name__ == "__main__":
    unittest.main()
