"""Exercise the exact script included in the streaming documentation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fourdenoise as fd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "streaming_example", ROOT / "scripts" / "streaming_example.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class StreamingDocumentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.input = self.folder / "scan.npy"
        self.values = np.full((2, 3, 96, 96), 10, dtype=np.uint16)
        y, x = np.ogrid[:96, :96]
        for cy, cx in ((20, 48), (48, 84)):
            self.values[..., (y - cy) ** 2 + (x - cx) ** 2 <= 9] += 100
        np.save(self.input, self.values)

    def test_documented_workflow_runs_without_loading_a_diffraction_tensor(self):
        with patch.object(fd.DiffractionSource, "load", side_effect=AssertionError("No full load")):
            result = example.run_example(self.input, self.folder / "output")
        expected = fd.HyperData(self.values).virtual_image(
            annulus=(37, 50), detector_units="pixels", show=False,
        )
        np.testing.assert_array_equal(result["image"].array, expected.array)
        np.testing.assert_allclose(result["mean_dp"].array, self.values.mean(axis=(0, 1)))
        np.testing.assert_array_equal(result["peaks"].observed_peak_counts, np.full((2, 3), 2))
        self.assertIsNotNone(result["peaks"].intensities)
        for name in ("peak_path", "polar_path", "cartesian_path"):
            self.assertTrue(Path(result[name]).is_file())
        with fd.HyperData.open(result["cartesian_path"]) as source:
            self.assertEqual(source.shape, self.values.shape)
        np.testing.assert_array_equal(np.load(self.input), self.values)

    def test_existing_outputs_are_not_overwritten(self):
        output = self.folder / "output"
        output.mkdir()
        sentinel = output / "peaks.h5"
        sentinel.write_bytes(b"existing result")
        with self.assertRaises(FileExistsError):
            example.run_example(self.input, output)
        self.assertEqual(sentinel.read_bytes(), b"existing result")

    def test_stack_is_not_silently_treated_as_a_scan(self):
        np.save(self.input, self.values.reshape(6, 96, 96))
        with self.assertRaisesRegex(ValueError, "4D"):
            example.run_example(self.input, self.folder / "output")


if __name__ == "__main__":
    unittest.main()
