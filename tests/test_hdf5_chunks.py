"""Regression tests for bounded-memory HDF5 diffraction access."""

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


class HDF5ChunkReaderTests(unittest.TestCase):
    def test_generic_four_dimensional_chunks_and_single_pattern(self):
        array = np.arange(5 * 7 * 3 * 4).reshape(5, 7, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generic.h5'
            with h5py.File(path, 'w') as file:
                file.create_dataset('entry/data', data=array)

            with fd.HyperData.open_hdf5(path) as source:
                self.assertIsInstance(source._dataset, h5py.Dataset)
                self.assertEqual(source.shape, array.shape)
                np.testing.assert_array_equal(
                    source.get_dp(np.int64(4), np.int64(6)).array, array[4, 6]
                )
                blocks = list(source.iter_chunks((np.int64(2), np.int64(3))))
                self.assertEqual(len(blocks), 9)
                for scan_slices, block in blocks:
                    np.testing.assert_array_equal(
                        block.array, array[scan_slices + (slice(None), slice(None))]
                    )
                    self.assertIsNone(block.unfold_metadata)

            np.testing.assert_array_equal(blocks[0][1].array, array[:2, :3])
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                source.get_dp(0, 0)

    def test_saved_hyperdata_chunks_preserve_calibration(self):
        array = np.ones((3, 4, 5, 6), dtype=np.uint16)
        data = fd.HyperData(
            array, real_units='nm', real_conv_factor=(2.0, 3.0),
            real_origin=(10.0, 20.0), reciprocal_units='mrad',
            reciprocal_conv_factor=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.4denoise'
            data.save(path)
            with fd.HyperData.open_hdf5(path) as source:
                dp = source.get_dp(0, 0)
                self.assertEqual((dp.units, dp.conv_factor), ('mrad', 0.5))
                blocks = list(source.iter_chunks((2, 2)))

        self.assertEqual(blocks[-1][0], (slice(2, 3), slice(2, 4)))
        self.assertEqual(blocks[-1][1].real_origin, (14.0, 26.0))
        self.assertEqual(blocks[-1][1].real_conv_factor, (2.0, 3.0))
        np.testing.assert_array_equal(blocks[-1][1].array, array[2:3, 2:4])

    def test_three_dimensional_stack_and_validation(self):
        array = np.arange(7 * 2 * 3).reshape(7, 2, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stack.hdf5'
            with h5py.File(path, 'w') as file:
                file.create_dataset('stack', data=array)
                file.create_dataset('other', data=np.ones((2, 2)))

            with self.assertRaisesRegex(ValueError, '2 numeric'):
                with fd.HyperData.open_hdf5(path):
                    pass
            with fd.HyperData.open_hdf5(path, hdf5_dataset='stack') as source:
                blocks = list(source.iter_chunks(3))
                self.assertEqual(len(blocks), 3)
                np.testing.assert_array_equal(source.get_dp(6).array, array[6])
                np.testing.assert_array_equal(blocks[-1][1].array, array[6:7])
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    list(source.iter_chunks(0))
                with self.assertRaisesRegex(ValueError, '1 scan'):
                    source.get_dp(0, 1)


if __name__ == '__main__':
    unittest.main()
