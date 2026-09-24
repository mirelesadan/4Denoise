"""Regression tests for lossless loading and geometry-aware metadata."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class LoadPreservationTests(unittest.TestCase):
    def test_npy_preserves_counts_and_nans_by_default(self):
        source = np.full((2, 3, 4), 5.0)
        source[0, 0, 0] = 0
        source[0, 0, 1] = -2
        source[0, 1, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.npy'
            np.save(path, source)
            np.testing.assert_array_equal(fd.read_4D(path), source)
            np.testing.assert_array_equal(fd.HyperData(path).array, source)

            cleaned = fd.HyperData(
                path, clip_on_load=True, repair_nans=True,
            )
            np.testing.assert_array_equal(cleaned.array[0], source[1])
            np.testing.assert_array_equal(np.load(path), source)

    def test_generic_hdf5_is_lossless(self):
        source = np.array([[[0.0, -3.0], [np.nan, 9.0]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.h5'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('images', data=source)
            np.testing.assert_array_equal(fd.HyperData(path).array, source)


class MetadataLifecycleTests(unittest.TestCase):
    def setUp(self):
        beam = fd._center_beam_metadata_from_pixels(
            1.5, (7.5, 8.0), (16, 16), units='mrad', conv_factor=0.2,
            source='alignment', mean_fit_center_px=(7.5, 8.0),
        )
        self.data = fd.HyperData(
            np.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16),
            reciprocal_units='mrad', reciprocal_conv_factor=0.2,
            center_beam_metadata=beam,
        )

    def test_integer_crop_updates_beam_and_real_crop_keeps_it(self):
        real_only = self.data.crop(ylim=(1, 2))
        self.assertEqual(real_only.center_beam_metadata, self.data.center_beam_metadata)
        cropped = self.data.crop(
            kylim=(4, 12), kxlim=(4, 12),
            reciprocal_limit_units='pixels',
        )
        beam = cropped.center_beam_metadata
        self.assertEqual(beam['shape'], (8, 8))
        np.testing.assert_allclose(beam['center_px'], (3.5, 4.0))
        self.assertEqual(beam['radius_px'], 1.5)
        self.assertNotIn('mean_fit_center_px', beam)
        self.assertEqual(cropped.get_dp(0, 0).center_beam_metadata, beam)
        self.assertEqual(self.data.center_beam_metadata['shape'], (16, 16))

    def test_isotropic_resize_updates_beam_and_anisotropic_clears_it(self):
        resized = self.data.resize((8, 8), domain='reciprocal')
        beam = resized.center_beam_metadata
        self.assertEqual(beam['shape'], (8, 8))
        np.testing.assert_allclose(beam['center_px'], (3.5, 3.75))
        self.assertAlmostEqual(beam['radius_px'], 0.75)
        self.assertAlmostEqual(beam['conv_factor'], 0.4)
        self.assertIsNone(
            self.data.resize((8, 16), domain='reciprocal').center_beam_metadata
        )
        self.assertIsNone(
            self.data.crop(
                kylim=(1.5, 14.5), reciprocal_limit_units='pixels',
            ).center_beam_metadata
        )

    def test_reciprocalspace_crop_updates_beam(self):
        dp = self.data.get_dp(0, 0)
        cropped = dp.crop(kylim=(4, 12), kxlim=(4, 12))
        np.testing.assert_allclose(cropped.center_beam_metadata['center_px'], (3.5, 4.0))
        self.assertEqual(cropped.center_beam_metadata['shape'], (8, 8))
        self.assertIsNone(dp.crop(kshape=(8, 16)).center_beam_metadata)

    def test_reshape_and_domain_swap_do_not_keep_stale_beam_geometry(self):
        stacked = self.data.reshape(6, 16, 16)
        self.assertEqual(stacked.center_beam_metadata, self.data.center_beam_metadata)
        self.assertIsNone(self.data.reshape(2, 3, 8, 32).center_beam_metadata)
        self.assertIsNone(self.data.swap_domains().center_beam_metadata)

    def test_value_only_changes_retain_unfold_undo(self):
        original = self.data.array[:2, :2, :4, :4]
        unfolded = fd.HyperData(original).unfold(
            domain='real', method='serpentine',
        )
        clipped = unfolded.clip(a_min=20)
        self.assertIsNotNone(clipped.unfold_metadata)
        np.testing.assert_array_equal(
            clipped.unfold(undo=True).array, np.clip(original, 20, None),
        )
        normalized = unfolded.normalize(method='global')
        self.assertIsNotNone(normalized.unfold_metadata)
        np.testing.assert_allclose(
            normalized.unfold(undo=True).array,
            (original - original.min()) / (original.max() - original.min()),
        )
        self.assertIsNone(
            unfolded.resize((2, 2), domain='reciprocal').unfold_metadata
        )

    def test_unfold_metadata_survives_save_after_clip(self):
        original = np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2)
        clipped = fd.HyperData(original).unfold(method='serpentine').clip(a_min=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unfolded.4denoise'
            clipped.save(path)
            restored = fd.HyperData(path).unfold(undo=True)
        np.testing.assert_array_equal(restored.array, np.clip(original, 4, None))


if __name__ == '__main__':
    unittest.main()
