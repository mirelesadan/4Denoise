"""Explicit binary layouts and safe legacy EMPAD raw loading."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourdenoise as fd


class RawLoadingTests(unittest.TestCase):
    def test_rectangular_4d_scan_with_explicit_shape_and_dtype(self):
        original = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.raw'
            original.tofile(path)
            loaded = fd.HyperData(path, raw_shape=original.shape, raw_dtype='u2')
        np.testing.assert_array_equal(loaded.array, original)
        self.assertEqual(loaded.real_shape, (2, 3))
        self.assertEqual(loaded.k_shape, (4, 5))

    def test_3d_stack_with_big_endian_dtype(self):
        original = np.arange(3 * 4 * 5, dtype='>u2').reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stack.raw'
            original.tofile(path)
            loaded = fd.HyperData(path, raw_shape=original.shape, raw_dtype='>u2')
        np.testing.assert_array_equal(loaded.array, original)
        self.assertEqual(loaded.scan_shape, (3,))
        self.assertIsNone(loaded.real_shape)

    def test_fortran_storage_order(self):
        original = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.raw'
            path.write_bytes(original.tobytes(order='F'))
            loaded = fd.HyperData(
                path, raw_shape=original.shape, raw_order='F',
            )
        np.testing.assert_array_equal(loaded.array, original)

    def test_explicit_shape_trims_only_when_requested(self):
        original = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.raw'
            original.tofile(path)
            full = fd.HyperData(path, raw_shape=original.shape)
            trimmed = fd.HyperData(
                path, raw_shape=original.shape,
                raw_trim_meta=True, raw_trim_dims=(2, 4),
            )
        self.assertEqual(full.shape, original.shape)
        np.testing.assert_array_equal(trimmed.array, original[..., :2, :4])

    def test_legacy_square_scan_and_empad_trim_are_retained(self):
        original = np.arange(
            2 * 2 * 130 * 128, dtype=np.float32,
        ).reshape(2, 2, 130, 128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empad.raw'
            original.tofile(path)
            loaded = fd.HyperData(path)
        self.assertEqual(loaded.shape, (2, 2, 128, 128))
        np.testing.assert_array_equal(loaded.array, original[..., :128, :128])

    def test_small_legacy_detector_is_not_rejected_by_default_trim(self):
        original = np.arange(2 * 2 * 4 * 5, dtype=np.float32).reshape(2, 2, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.raw'
            original.tofile(path)
            loaded = fd.read_4D(path, dp_dims=(5, 4))
        np.testing.assert_array_equal(loaded, original)

    def test_non_square_legacy_scan_requests_explicit_shape(self):
        original = np.zeros((2, 3, 3, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.raw'
            original.tofile(path)
            with self.assertRaisesRegex(ValueError, 'not square'):
                fd.read_4D(path, dp_dims=(2, 3), trim_meta=False)

    def test_wrong_file_size_and_bad_layout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.raw'
            np.arange(10, dtype=np.uint16).tofile(path)
            with self.assertRaisesRegex(ValueError, 'require'):
                fd.HyperData(path, raw_shape=(2, 3, 4, 5), raw_dtype='u2')
            with self.assertRaisesRegex(ValueError, 'raw_shape'):
                fd.HyperData(path, raw_shape=(2, 3, 4.5, 5))
            with self.assertRaisesRegex(ValueError, 'raw_order'):
                fd.HyperData(path, raw_shape=(2, 5), raw_order='bad')


if __name__ == '__main__':
    unittest.main()
