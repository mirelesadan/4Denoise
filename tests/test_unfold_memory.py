"""Exact undo without unnecessary copies of preserved tensor values."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourdenoise as fd


class UnfoldMemoryTests(unittest.TestCase):
    def setUp(self):
        self.original = np.arange(5 * 7 * 4 * 6, dtype=np.int16).reshape(5, 7, 4, 6)
        self.data = fd.HyperData(self.original)

    def test_both_domain_crop_preserves_only_excluded_values(self):
        unfolded, meta = self.data.unfold(
            domain='both', method='morton', return_metadata=True,
        )
        self.assertEqual(meta['excess_values_encoding'], 'separated_domains')
        self.assertIsInstance(meta['excess_values'], dict)
        self.assertLess(meta['preserved_values_nbytes'], self.original.nbytes)
        self.assertEqual(
            meta['preserved_values_nbytes'] + unfolded.array.nbytes,
            self.original.nbytes,
        )
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)

    def test_compatible_square_needs_no_excess_values(self):
        data = fd.HyperData(np.arange(4 ** 4).reshape((4,) * 4))
        unfolded, meta = data.unfold(
            domain='both', method='morton', return_metadata=True,
        )
        self.assertEqual(meta['preserved_values_nbytes'], 0)
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, data.array)

    def test_shape_preserving_results_share_read_only_payload(self):
        unfolded = self.data.unfold(method='hilbert')
        clipped = unfolded.clip(a_min=10)
        self.assertIsNot(clipped.unfold_metadata, unfolded.unfold_metadata)
        original_values = unfolded.unfold_metadata['excess_values']
        clipped_values = clipped.unfold_metadata['excess_values']
        self.assertTrue(np.shares_memory(original_values, clipped_values))
        self.assertFalse(clipped_values.flags.writeable)
        with self.assertRaises(ValueError):
            clipped_values[0, 0, 0] = 0
        np.testing.assert_array_equal(
            clipped.unfold(undo=True).array[0, 0],
            self.original[0, 0],
        )

        independent = unfolded.copy()
        self.assertFalse(np.shares_memory(
            independent.unfold_metadata['excess_values'], original_values,
        ))

    def test_legacy_full_tensor_metadata_can_still_be_undone(self):
        unfolded, meta = self.data.unfold(
            domain='both', method='morton', return_metadata=True,
        )
        legacy = deepcopy(meta)
        legacy['excess_values'] = self.original.copy()
        legacy['excess_values_encoding'] = 'full_tensor'
        np.testing.assert_array_equal(
            unfolded.unfold(undo=True, metadata=legacy).array, self.original,
        )

    def test_v1_full_tensor_saved_metadata_migrates(self):
        unfolded, meta = self.data.unfold(
            domain='both', method='morton', return_metadata=True,
        )
        legacy = deepcopy(meta)
        legacy['excess_values'] = self.original.copy()
        legacy['excess_values_encoding'] = 'full_tensor'
        legacy.pop('preserved_values_nbytes')
        unfolded.unfold_metadata = legacy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v1.4denoise'
            unfolded.save(path)
            with h5py.File(path, 'r+') as file:
                file.attrs['format_version'] = '1.0'
            loaded = fd.HyperData(path)
        self.assertEqual(
            loaded.unfold_metadata['preserved_values_nbytes'],
            self.original.nbytes,
        )
        np.testing.assert_array_equal(loaded.unfold(undo=True).array, self.original)

    def test_separated_excess_survives_save_and_load(self):
        unfolded = self.data.unfold(domain='both', method='morton')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unfolded.4denoise'
            unfolded.save(path)
            loaded = fd.HyperData(path)
        self.assertEqual(
            loaded.unfold_metadata['excess_values_encoding'],
            'separated_domains',
        )
        np.testing.assert_array_equal(loaded.unfold(undo=True).array, self.original)

    def test_preserve_original_is_explicit_and_shared_only_in_derived_results(self):
        unfolded, meta = self.data.unfold(
            method='hilbert', curve_shape_strategy='resize',
            preserve_original=True, return_metadata=True,
        )
        self.assertEqual(meta['preserved_values_nbytes'], self.original.nbytes)
        self.assertFalse(np.shares_memory(meta['original_values'], self.original))
        self.assertFalse(meta['original_values'].flags.writeable)
        clipped = unfolded.clip(a_min=10)
        self.assertTrue(np.shares_memory(
            clipped.unfold_metadata['original_values'], meta['original_values'],
        ))
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)


if __name__ == '__main__':
    unittest.main()
