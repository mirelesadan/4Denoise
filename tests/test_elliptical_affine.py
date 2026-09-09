"""Regression tests for non-square elliptical-distortion correction."""

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


def expected_affine(image, angle, major_axis, minor_axis, interpolation):
    """Construct the intended NumPy/OpenCV affine transform independently."""
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2

    source_points = np.float32(
        [
            [
                major_axis * np.cos(angle) + center_x,
                major_axis * np.sin(angle) + center_y,
            ],
            [
                -major_axis * np.cos(angle) + center_x,
                -major_axis * np.sin(angle) + center_y,
            ],
            [
                minor_axis * np.sin(angle) + center_x,
                -minor_axis * np.cos(angle) + center_y,
            ],
        ]
    )
    target_radius = max(major_axis, minor_axis)
    target_points = np.float32(
        [
            [
                target_radius * np.cos(angle) + center_x,
                target_radius * np.sin(angle) + center_y,
            ],
            [
                -target_radius * np.cos(angle) + center_x,
                -target_radius * np.sin(angle) + center_y,
            ],
            [
                target_radius * np.sin(angle) + center_x,
                -target_radius * np.cos(angle) + center_y,
            ],
        ]
    )
    matrix = cv2.getAffineTransform(source_points, target_points)
    flags = cv2.INTER_CUBIC if interpolation == "cubic" else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (width, height), flags=flags)


class EllipticalAffineRegressionTests(unittest.TestCase):
    def assert_public_correction(self, detector_shape):
        height, width = detector_shape
        base = np.arange(height * width, dtype=np.float32).reshape(height, width)
        data = np.stack(
            (
                np.stack((base, base + 1000), axis=0),
                np.stack((base + 2000, base + 3000), axis=0),
            ),
            axis=0,
        )
        params = np.array([0.31, 4.5, 2.75], dtype=float)
        expected = np.empty_like(data)
        for scan_y in range(data.shape[0]):
            for scan_x in range(data.shape[1]):
                expected[scan_y, scan_x] = expected_affine(
                    data[scan_y, scan_x], *params, "linear"
                )

        with (
            patch.object(fd.HyperData, "_extract_ellipse", return_value=params),
            patch.object(fd, "tqdm", new=lambda iterable, **kwargs: iterable),
        ):
            corrected = fd.HyperData(data).fix_elliptical_distortions(
                r=1,
                R=5,
                interp_method="linear",
                return_fix=True,
            )

        self.assertEqual(corrected.shape, data.shape)
        self.assertEqual(corrected.array.dtype, data.dtype)
        np.testing.assert_array_equal(corrected.array, expected)

    def test_square_detector_preserves_existing_geometry(self):
        self.assert_public_correction((17, 17))

    def test_non_square_detector_preserves_shape_and_geometry(self):
        self.assert_public_correction((13, 19))


if __name__ == "__main__":
    unittest.main()
