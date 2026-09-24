"""Regression tests for local diffraction-spot windows."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import center_of_mass

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


def padded_reference_center(image, ky, kx, radius):
    """Compute the original full-padding result for an equivalence check."""
    pad = int(np.ceil(radius))
    padded = np.pad(image, pad_width=pad, mode="constant")
    area_size = int(np.ceil(radius * 2))
    half = area_size // 2
    mask = fd.circular_mask(half, half, radius)
    ymin, ymax = int(ky + pad - half), int(ky + pad + half) + 1
    xmin, xmax = int(kx + pad - half), int(kx + pad + half) + 1
    spot = padded[ymin:ymax, xmin:xmax]
    if spot.shape != mask.shape:
        side = min(*spot.shape, *mask.shape)
        spot, mask = spot[:side, :side], mask[:side, :side]
    cy, cx = center_of_mass(spot * mask)
    return cy + ymin - pad, cx + xmin - pad


class LocalPeakWindowTests(unittest.TestCase):
    def test_centers_match_full_padding_at_edges_and_subpixel_positions(self):
        image = np.random.default_rng(14).random((32, 39)) + 0.1
        dp = fd.ReciprocalSpace(image)
        cases = [
            (0.0, 0.0, 2.0),
            (0.3, 38.1, 3.25),
            (31.0, 0.8, 4.0),
            (31.6, 38.0, 2.5),
            (15.3, 21.7, 5.0),
        ]
        for ky, kx, radius in cases:
            with self.subTest(ky=ky, kx=kx, radius=radius):
                expected = padded_reference_center(image, ky, kx, radius)
                actual = dp.get_spotCenter(ky, kx, radius)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_intensities_match_full_masks_for_interior_peaks(self):
        image = np.random.default_rng(15).random((64, 72)).astype(np.float32)
        dp = fd.ReciprocalSpace(image)
        centers = np.array([[12.2, 18.6], [29.7, 34.1], [49.3, 56.8]])
        radii = np.array([2.5, 3.7, 5.0])

        expected = []
        for (cy, cx), radius in zip(centers, radii):
            full_mask = fd.make_mask((cy, cx), radius + 1e-10, image.shape)
            masked = image * full_mask
            expected.append(masked[
                round(cy - (radius + 0.5)):round(cy + (radius + 0.5)),
                round(cx - (radius + 0.5)):round(cx + (radius + 0.5)),
            ].sum())

        np.testing.assert_array_equal(
            dp.get_intensities(radii, centers=centers), expected
        )

    def test_intensities_clip_at_detector_edge(self):
        image = np.zeros((16, 19), dtype=float)
        image[0, 0], image[0, 1] = 3.0, 2.0
        image[-1, -1], image[-2, -1] = 7.0, 4.0
        dp = fd.ReciprocalSpace(image)

        actual = dp.get_intensities(
            2.0, centers=np.array([[0.0, 0.0], [15.0, 18.0]])
        )
        np.testing.assert_array_equal(actual, [5.0, 11.0])


if __name__ == "__main__":
    unittest.main()
