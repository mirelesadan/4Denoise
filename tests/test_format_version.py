"""Saved HyperData format-version validation and metadata migration."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourdenoise as fd


class SavedFormatVersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'dataset.4denoise'
        self.original = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        fd.HyperData(self.original).save(self.path)

    def test_current_version_loads_eagerly_and_in_chunks(self):
        with h5py.File(self.path, 'r') as file:
            self.assertEqual(file.attrs['format_version'], '1.1')
        np.testing.assert_array_equal(fd.HyperData(self.path).array, self.original)
        with fd.HyperData.open_hdf5(self.path) as source:
            np.testing.assert_array_equal(source.get_dp(0, 0).array, self.original[0, 0])

    def test_unknown_version_rejected_by_both_paths(self):
        with h5py.File(self.path, 'r+') as file:
            file.attrs['format_version'] = '2.0'
        with self.assertRaisesRegex(ValueError, 'unsupported.*2.0'):
            fd.HyperData(self.path)
        with self.assertRaisesRegex(ValueError, 'unsupported.*2.0'):
            with fd.HyperData.open_hdf5(self.path):
                pass

    def test_missing_version_rejected_by_both_paths(self):
        with h5py.File(self.path, 'r+') as file:
            del file.attrs['format_version']
        with self.assertRaisesRegex(ValueError, 'missing.*format_version'):
            fd.HyperData(self.path)
        with self.assertRaisesRegex(ValueError, 'missing.*format_version'):
            with fd.HyperData.open_hdf5(self.path):
                pass

    def test_v1_missing_optional_origin_gets_default(self):
        with h5py.File(self.path, 'r+') as file:
            file.attrs['format_version'] = '1.0'
            del file['metadata']['real_origin']
        loaded = fd.HyperData(self.path)
        self.assertEqual(loaded.real_origin, (0.0, 0.0))
        with fd.HyperData.open_hdf5(self.path) as source:
            _, block = next(source.iter_chunks((1, 1)))
        self.assertEqual(block.real_origin, (0.0, 0.0))

    def test_marked_file_without_data_is_not_treated_as_generic_hdf5(self):
        with h5py.File(self.path, 'r+') as file:
            del file['array']
        with self.assertRaisesRegex(ValueError, 'does not contain a saved data array'):
            fd.HyperData(self.path)


if __name__ == '__main__':
    unittest.main()
