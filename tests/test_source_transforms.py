"""General-source disk transforms: parity, ownership, geometry and atomic safety."""

from math import prod
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_streaming as streaming


class TrackedArray:
    def __init__(self, array):
        self.array = array
        self.shape, self.dtype = array.shape, array.dtype
        self.reads = []
        self.reader_chunk_nbytes = getattr(array, 'reader_chunk_nbytes', 0)

    def __array__(self, *args, **kwargs):
        raise AssertionError('Full-source conversion is forbidden.')

    def __getitem__(self, selection):
        self.reads.append(selection)
        return self.array[selection]


class SourceTransformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.values = np.arange(4 * 5 * 7 * 9, dtype=np.int16).reshape(4, 5, 7, 9)
        self.calibration = dict(real_units='nm', real_conv_factor=(2.0, -3.0),
                                real_origin=(11.0, -4.0), reciprocal_units='mrad',
                                reciprocal_conv_factor=0.25)
        self.data = fd.HyperData(self.values, **self.calibration)
        self.data.acquisition_metadata = {'experiment': 'source-transforms', 'scan_shape': (4, 5)}
        self.data.set_rq_calibration(rotation_deg=17, mirror_axis='x')
        self.data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            1.5, (3, 4), (7, 9), units='mrad', conv_factor=0.25,
        )
        self.npy = self.folder / 'scan.npy'
        self.h5 = self.folder / 'scan.hdf5'
        self.saved = self.folder / 'scan.4denoise'
        self.output = self.folder / 'output.4denoise'
        np.save(self.npy, self.values)
        with h5py.File(self.h5, 'w') as file:
            file['entry/scan'] = self.values
            file['another'] = np.zeros((2, 3, 4))
        self.data.save(self.saved)

    def open(self, path):
        if path == self.saved:
            return fd.HyperData.open(path)
        return fd.HyperData.open(path, **self.calibration,
                                 **({'hdf5_dataset': '/entry/scan'} if path == self.h5 else {}))

    def assert_result(self, path, expected):
        actual = fd.HyperData(path)
        np.testing.assert_allclose(actual.array, expected.array, rtol=1e-6, atol=1e-5)
        self.assertEqual(actual.dtype, expected.dtype)
        for key in streaming._METADATA_KEYS:
            self.assertEqual(getattr(actual, key), getattr(expected, key), key)
        actual_rq = actual.rq_calibration.to_dict() if actual.rq_calibration else None
        expected_rq = expected.rq_calibration.to_dict() if expected.rq_calibration else None
        self.assertEqual(actual_rq, expected_rq)
        self.assertIsNone(actual.unfold_metadata)
        return actual

    def test_polar_defaults_and_explicit_shapes_match_all_native_backends(self):
        for path in (self.npy, self.h5, self.saved):
            for order, shape in ((0, None), (1, (5, 16)), (3, (7, 25))):
                with self.subTest(path=path.suffix, order=order), self.open(path) as source:
                    baseline = fd.HyperData(self.values, **source.metadata)
                    expected = baseline.to_polar(output_shape=shape, order=order, progress=False)
                    with patch.object(source, 'load', side_effect=AssertionError('No full load')):
                        result = source.to_polar(
                            output_shape=shape, order=order, destination=self.output,
                            working_memory_mb=0.07, chunk_shape=(1, 2),
                            overwrite=True, progress=False,
                        )
                    self.assert_result(result, expected)
                    self.assertFalse(source.closed)
                    np.testing.assert_array_equal(source.get_dp(1, 2).array, self.values[1, 2])

    def test_background_array_and_object_match_all_native_backends(self):
        background = np.linspace(0.5, 10, 63).reshape(7, 9)
        original_background = background.copy()
        for path in (self.npy, self.h5, self.saved):
            for bg in (background, fd.ReciprocalSpace(background)):
                with self.subTest(path=path.suffix, type=type(bg)), self.open(path) as source:
                    expected = fd.HyperData(self.values, **source.metadata).remove_bg(
                        bg, bg_frac=0.7, a_min=None, a_max=900,
                    )
                    result = source.remove_bg(
                        bg, bg_frac=0.7, a_min=None, a_max=900,
                        destination=self.output, working_memory_mb=0.04,
                        overwrite=True, progress=False,
                    )
                    self.assert_result(result, expected)
                    self.assertFalse(source.closed)
        np.testing.assert_array_equal(background, original_background)
        np.testing.assert_array_equal(np.load(self.npy), self.values)

    def test_global_resize_fractional_bins_and_upsampling_match_all_native_backends(self):
        cases = [((3, 4), 'real', 'area'), ((7, 8), 'real', 'linear'),
                 ((7, 8), 'real', 'nearest'), ((4, 5), 'reciprocal', 'area'),
                 ((10, 13), 'reciprocal', 'linear'), ((10, 13), 'reciprocal', 'nearest')]
        for path in (self.npy, self.h5, self.saved):
            with self.open(path) as source:
                baseline = fd.HyperData(self.values, **source.metadata)
                for shape, domain, method in cases:
                    with self.subTest(path=path.suffix, domain=domain, method=method):
                        expected = baseline.resize(shape, domain, method)
                        with patch.object(source, 'load', side_effect=AssertionError('No full load')):
                            result = source.resize(
                                shape, domain, method, destination=self.output,
                                working_memory_mb=0.015, overwrite=True, progress=False,
                            )
                        self.assert_result(result, expected)
                self.assertFalse(source.closed)

    def test_three_dimensional_stacks_all_transform_paths(self):
        values = self.values.reshape(20, 7, 9)
        np.save(self.npy, values)
        with fd.HyperData.open(self.npy) as source:
            expected = fd.HyperData(values).to_polar(output_shape=(5, 16), progress=False)
            path = source.to_polar(output_shape=(5, 16), destination=self.output,
                                   working_memory_mb=0.04, chunk_shape=2, progress=False)
            self.assert_result(path, expected)
            resized = source.resize(13, 'real', destination=self.folder / 'stack', progress=False)
            self.assert_result(resized, fd.HyperData(values).resize(13, 'real'))
        with fd.HyperData.open(path) as source:
            result = source.to_cartesian(destination=self.folder / 'cartesian', progress=False)
            self.assert_result(result, expected.to_cartesian(progress=False))

    def test_polar_resize_cartesian_canvas_and_missing_sector_parity(self):
        polar = self.data.to_polar(output_shape=(8, 24), progress=False)
        for data in (polar, polar.crop(kylim=(2, 8), kxlim=(6, 18))):
            data.save(self.saved, overwrite=True)
            with fd.HyperData.open(self.saved) as source:
                result = source.resize((4, 8), 'reciprocal', destination=self.output,
                                        working_memory_mb=0.02, overwrite=True, progress=False)
                self.assert_result(result, data.resize((4, 8), 'reciprocal'))
                for canvas in (None, 'original', (11, 13)):
                    expected = data.to_cartesian(output_shape=canvas, fill_value=-7, progress=False)
                    result = source.to_cartesian(
                        output_shape=canvas, fill_value=-7, destination=self.output,
                        working_memory_mb=0.07, overwrite=True, progress=False,
                    )
                    self.assert_result(result, expected)

    def test_per_pattern_reads_cover_every_scan_once_and_obey_chunk_limit(self):
        with self.open(self.npy) as source:
            tracker = TrackedArray(source._dataset)
            source._dataset = tracker
            source.to_polar(output_shape=(5, 16), destination=self.output,
                            chunk_shape=(1, 2), working_memory_mb=0.04, progress=False)
            visits = np.zeros(self.values.shape[:2], dtype=int)
            for selection in tracker.reads:
                self.assertTrue(all(isinstance(s, slice) for s in selection))
                self.assertEqual(selection[-2:], (slice(None), slice(None)))
                region = visits[selection[:2]]
                self.assertLessEqual(region.size, 2)
                region += 1
            np.testing.assert_array_equal(visits, 1)
            self.assertEqual(tracker.reads[0][:2], (slice(0, 1), slice(0, 1)))

    def test_probe_copy_is_writable_without_mutating_readonly_numpy_source(self):
        def mutate(block):
            block.array.fill(-3)
            return block
        with self.open(self.npy) as source:
            path = streaming._stream_transform(
                source, self.output, transform=mutate, output_pattern_shape=(7, 9),
                working_memory_mb=0.04, progress=False,
            )
            self.assertFalse(source.closed)
        np.testing.assert_array_equal(fd.HyperData(path).array, -3)
        np.testing.assert_array_equal(np.load(self.npy), self.values)

    def test_failed_atomic_transform_preserves_file_and_borrowed_source(self):
        self.output.write_bytes(b'keep existing destination')
        with self.open(self.npy) as source:
            with patch.object(streaming, '_write_block', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    source.remove_bg(np.zeros((7, 9)), destination=self.output,
                                     overwrite=True, working_memory_mb=0.04, progress=False)
            self.assertFalse(source.closed)
            np.testing.assert_array_equal(source.get_dp(0, 0).array, self.values[0, 0])
        self.assertEqual(self.output.read_bytes(), b'keep existing destination')
        self.assertEqual(list(self.folder.glob('.output.4denoise.*.tmp')), [])

    def test_failed_atomic_resize_preserves_file_and_borrowed_source(self):
        self.output.write_bytes(b'keep existing destination')
        with self.open(self.npy) as source:
            with patch.object(streaming, '_resize_array', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    source.resize((3, 4), 'real', destination=self.output,
                                  overwrite=True, progress=False)
            self.assertFalse(source.closed)
        self.assertEqual(self.output.read_bytes(), b'keep existing destination')
        self.assertEqual(list(self.folder.glob('.output.4denoise.*.tmp')), [])

    def test_insufficient_memory_invalid_geometry_and_shapes_fail_before_reads(self):
        with self.open(self.npy) as source:
            tracker = TrackedArray(source._dataset)
            source._dataset = tracker
            calls = [lambda: source.to_polar(destination=self.output, working_memory_mb=1e-5),
                     lambda: source.to_polar(output_shape=(100000, 100000), destination=self.output,
                                             working_memory_mb=1),
                     lambda: source.to_polar(center=(-1, 3), destination=self.output),
                     lambda: source.to_polar(output_shape=(2.5, 16), destination=self.output),
                     lambda: source.to_polar(order=8, destination=self.output),
                     lambda: source.to_polar(chunk_shape=(0, 2), destination=self.output),
                     lambda: source.to_cartesian(destination=self.output),
                     lambda: source.resize((3, 4), 'real', destination=self.output, working_memory_mb=1e-5),
                     lambda: source.resize((8, 8), 'real', 'area', destination=self.output),
                     lambda: source.remove_bg(np.zeros((3, 3)), destination=self.output),
                     lambda: source.remove_bg(np.zeros((7, 9)), destination=self.output, bg_frac=np.nan),
                     lambda: source.remove_bg(np.zeros((7, 9)), destination=self.output, working_memory_mb=1e-5)]
            for call in calls:
                with self.assertRaises((TypeError, ValueError)):
                    call()
            self.assertEqual(tracker.reads, [])
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.folder.glob('.*.tmp')), [])

    def test_alias_existing_destination_and_closed_source_rejected(self):
        with self.open(self.saved) as source:
            for call in (lambda: source.resize((3, 4), 'real', destination=self.saved, overwrite=True),
                         lambda: source.to_polar(destination=self.saved, overwrite=True),
                         lambda: source.remove_bg(np.zeros((7, 9)), destination=self.saved, overwrite=True)):
                with self.assertRaisesRegex(ValueError, 'different files'):
                    call()
            os.link(self.saved, self.output)
            with self.assertRaisesRegex(ValueError, 'different files'):
                source.to_polar(destination=self.output, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'different files'):
                source.resize((3, 4), 'real', destination=self.output, overwrite=True)
        for call in (lambda: source.to_polar(destination=self.output),
                     lambda: source.resize((3, 4), 'real', destination=self.output),
                     lambda: source.to_cartesian(destination=self.output),
                     lambda: source.remove_bg(np.zeros((7, 9)), destination=self.output)):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                call()

    def test_nonatomic_compression_and_overwrite_controls(self):
        with self.open(self.npy) as source:
            result = source.remove_bg(np.zeros((7, 9)), destination=self.output,
                                      compression=None, atomic=False, progress=False)
            with h5py.File(result) as file:
                self.assertIsNone(file['array'].compression)
            with self.assertRaises(FileExistsError):
                source.to_polar(destination=self.output, progress=False)
            result = source.resize((3, 4), 'real', destination=self.output, overwrite=True,
                                   compression='lzf', compression_opts=None, progress=False)
            with h5py.File(result) as file:
                self.assertEqual(file['array'].compression, 'lzf')


if __name__ == '__main__':
    unittest.main()
