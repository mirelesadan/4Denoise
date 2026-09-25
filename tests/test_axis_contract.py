"""Shape contracts for 2D images, 3D stacks, and 4D scans."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourdenoise as fd


class HyperDataAxisContractTests(unittest.TestCase):
    def test_2d_image_has_no_scan_grid(self):
        data = fd.HyperData(np.zeros((5, 7)))
        self.assertEqual(data.scan_shape, ())
        self.assertEqual(data.pattern_shape, (5, 7))
        self.assertEqual(data.k_shape, (5, 7))
        self.assertIsNone(data.real_shape)

    def test_3d_stack_is_not_a_2d_scan_grid(self):
        data = fd.HyperData(np.zeros((3, 5, 7)))
        self.assertEqual(data.scan_shape, (3,))
        self.assertEqual(data.pattern_shape, (5, 7))
        self.assertIsNone(data.real_shape)
        copied = data.copy()
        self.assertEqual(copied.scan_shape, (3,))
        self.assertIsNone(copied.real_shape)

    def test_4d_scan_has_real_grid(self):
        data = fd.HyperData(np.zeros((2, 3, 5, 7)))
        self.assertEqual(data.scan_shape, (2, 3))
        self.assertEqual(data.real_shape, (2, 3))
        self.assertEqual(data.pattern_shape, (5, 7))

    def test_reshape_and_resize_recompute_axis_shapes(self):
        stack = fd.HyperData(np.zeros((6, 5, 7)))
        folded = stack.reshape(2, 3, 5, 7)
        self.assertEqual(folded.scan_shape, (2, 3))
        self.assertEqual(folded.real_shape, (2, 3))
        resized = stack.resize((4, 6), domain='reciprocal')
        self.assertEqual(resized.scan_shape, (6,))
        self.assertEqual(resized.pattern_shape, (4, 6))
        self.assertIsNone(resized.real_shape)

    def test_shape_contract_survives_save_and_load(self):
        data = fd.HyperData(np.zeros((3, 5, 7)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stack.4denoise'
            data.save(path)
            loaded = fd.HyperData(path)
        self.assertEqual(loaded.scan_shape, (3,))
        self.assertIsNone(loaded.real_shape)

    def test_1d_array_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            fd.HyperData(np.zeros(5))


if __name__ == '__main__':
    unittest.main()
