"""Regression tests for atomic HyperData HDF5 saves."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class AtomicSaveTests(unittest.TestCase):
    def test_atomic_save_and_overwrite_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.4denoise'
            original = fd.HyperData(np.zeros((2, 2, 3, 3), dtype=np.float32))
            replacement = fd.HyperData(np.ones((2, 2, 3, 3), dtype=np.float32))

            self.assertEqual(original.save(path), str(path))
            np.testing.assert_array_equal(fd.HyperData(path).array, original.array)
            with self.assertRaises(FileExistsError):
                replacement.save(path)

            replacement.save(path, overwrite=True)
            np.testing.assert_array_equal(fd.HyperData(path).array, replacement.array)
            self.assertEqual(list(path.parent.glob(f'.{path.name}.*.tmp')), [])

    def test_failed_overwrite_leaves_existing_file_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.4denoise'
            fd.HyperData(np.zeros((2, 2, 3, 3))).save(path)
            original_bytes = path.read_bytes()
            replacement = fd.HyperData(np.ones((2, 2, 3, 3)))

            with patch.object(fd, '_write_hdf5_value', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    replacement.save(path, overwrite=True)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(path.parent.glob(f'.{path.name}.*.tmp')), [])

    def test_failed_new_save_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.4denoise'
            data = fd.HyperData(np.ones((2, 2, 3, 3)))

            with patch.object(fd, '_write_hdf5_value', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    data.save(path)

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(f'.{path.name}.*.tmp')), [])

    def test_direct_save_remains_available(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.4denoise'
            data = fd.HyperData(np.ones((2, 2, 3, 3)))
            data.save(path, atomic=False)
            np.testing.assert_array_equal(fd.HyperData(path).array, data.array)
            with self.assertRaisesRegex(ValueError, 'atomic'):
                data.save(path, overwrite=True, atomic='yes')


if __name__ == '__main__':
    unittest.main()
