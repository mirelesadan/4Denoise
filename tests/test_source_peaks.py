"""Streamed detector parity, ragged results, calibration and incremental files."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_peaks as peaks_module


def pattern(shape, points, radius=2):
    image = np.full(shape, 5.0)
    y, x = np.ogrid[:shape[0], :shape[1]]
    for cy, cx, intensity in points:
        image[(y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] += intensity
    return image


class TrackedArray:
    def __init__(self, source):
        self.source, self.shape, self.dtype = source, source.shape, source.dtype
        self.reads = []
        self.previous = None
        self.reader_chunk_nbytes = getattr(source, 'reader_chunk_nbytes', 0)

    def __array__(self, *args, **kwargs):
        raise AssertionError('Do not materialize the complete source.')

    def __getitem__(self, selection):
        if self.previous is not None and self.previous() is not None:
            raise AssertionError('Previous read retained during the next read.')
        if not all(isinstance(s, slice) for s in selection):
            raise AssertionError('Only basic slices are supported.')
        array = self.source[selection]
        self.previous = weakref.ref(array)
        self.reads.append(selection)
        return array


class SourcePeakTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.shape = (33, 39)
        patterns = [pattern(self.shape, p) for p in (
            [(10, 10, 50), (22, 27, 35)], [(16, 19, 60)], [],
            [(10, 10, 40)], [(8, 19, 50), (16, 27, 50), (24, 19, 50)],
            [(2, 3, 50), (21, 29, 60)],
        )]
        self.values = np.stack(patterns).reshape(2, 3, *self.shape)
        self.calibration = dict(real_units='nm', real_conv_factor=(2, -3),
                                real_origin=(4, 5), reciprocal_units='mrad',
                                reciprocal_conv_factor=0.25)
        self.data = fd.HyperData(self.values, **self.calibration)
        self.data.acquisition_metadata = {'experiment': 'peaks'}
        self.data.set_rq_calibration(rotation_deg=17)
        self.data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            2, (16, 19), self.shape, units='mrad', conv_factor=0.25,
        )
        self.npy, self.h5, self.saved = (self.folder / name for name in ('scan.npy', 'scan.h5', 'scan.4denoise'))
        self.output = self.folder / 'peaks.h5'
        np.save(self.npy, self.values)
        with h5py.File(self.h5, 'w') as file:
            file['entry/scan'] = self.values
            file['other'] = np.zeros((2, 2, 2))
        self.data.save(self.saved)
        self.options = dict(radius=2, min_distance=3, threshold_abs=1)

    def open(self, path):
        if path == self.saved:
            return fd.HyperData.open(path)
        return fd.HyperData.open(path, **self.calibration,
                                 **({'hdf5_dataset': '/entry/scan'} if path == self.h5 else {}))

    def assert_results_equal(self, actual, expected):
        self.assertEqual(actual.scan_shape, expected.scan_shape)
        self.assertEqual(actual.metadata, expected.metadata)
        np.testing.assert_array_equal(actual.processed_mask, expected.processed_mask)
        self.assertEqual(actual.intensities is None, expected.intensities is None)
        for index in np.ndindex(actual.scan_shape):
            for name in ('coords', 'scores', 'synthetic_mask', 'orbit_ids', 'intensities'):
                if getattr(actual, name) is not None:
                    np.testing.assert_allclose(getattr(actual, name)[index], getattr(expected, name)[index], equal_nan=True)

    def test_native_backends_match_eager_positions_scores_and_integrals(self):
        for path in (self.npy, self.h5, self.saved):
            with self.subTest(path=path.suffix), self.open(path) as source:
                with patch.object(source, 'load', side_effect=AssertionError('No full load')):
                    result = source.get_peaks(**self.options, intensity_radius=2.5,
                                              chunk_size=2, working_memory_mb=2, progress=False)
                self.assertIsInstance(result, fd.PeakScanResult)
                self.assertEqual(result.coords.dtype, object)
                self.assertEqual(result.scan_shape, (2, 3))
                for index in np.ndindex((2, 3)):
                    dp = fd.ReciprocalSpace(self.values[index])
                    expected = dp.get_peaks(**self.options, return_details=True)
                    np.testing.assert_array_equal(result.coords[index], expected.coords)
                    np.testing.assert_allclose(result.scores[index], expected.scores)
                    np.testing.assert_array_equal(result.synthetic_mask[index], expected.synthetic_mask)
                    np.testing.assert_array_equal(result.orbit_ids[index], expected.orbit_ids)
                    np.testing.assert_allclose(result.intensities[index], dp.get_intensities(2.5, centers=expected.coords))
                self.assertTrue(np.all(result.processed_mask))
                self.assertEqual(result.peak_counts[0, 2], 0)
                self.assertIsNone(source._metadata.get('unfold_metadata'))
                self.assertFalse(source.closed)
            self.assertEqual(result.metadata['source_metadata']['real_units'], 'nm')
            self.assertEqual(result.metadata['source_metadata']['real_origin'], (4, 5))
            result.coords[0, 0][0, 0] = 99
            np.testing.assert_array_equal(np.load(self.npy), self.values)

    def test_optional_intensity_absence_and_owned_copy(self):
        with self.open(self.saved) as source:
            result = source.get_peaks(**self.options, progress=False)
        self.assertIsNone(result.intensities)
        copied = result.copy()
        copied.coords[0, 0][0, 0] = -1
        copied.processed_mask[0, 0] = False
        copied.metadata['source_metadata']['real_origin'] = (99, 99)
        self.assertNotEqual(result.coords[0, 0][0, 0], -1)
        self.assertTrue(result.processed_mask[0, 0])
        self.assertEqual(result.metadata['source_metadata']['real_origin'], (4, 5))

    def test_integer_integration_promotes_before_large_count_sums(self):
        values = np.full((1, 1, 33, 39), 2 ** 62, dtype=np.uint64)
        np.save(self.npy, values)
        detected = fd.PeakDetectionResult(np.array([[16, 19]]), np.array([1.0]), np.array([False]), np.array([-1]))
        with fd.HyperData.open(self.npy) as source, patch.object(fd.ReciprocalSpace, 'get_peaks', return_value=detected):
            result = source.get_peaks(**self.options, intensity_radius=2, progress=False)
        self.assertEqual(result.intensities[0, 0][0], float(2 ** 62) * 13)

    def test_calibrated_coords_and_auto_match_centered_y_up_convention(self):
        with self.open(self.saved) as source:
            pixel = source.get_peaks(**self.options, intensity_radius=2, progress=False)
            physical = source.get_peaks(**self.options, intensity_radius=2, coords_units='calibrated', progress=False)
            automatic = source.get_peaks(**self.options, coords_units='auto', progress=False)
        self.assertEqual(physical.metadata['coordinate_units'], 'mrad')
        self.assertEqual(physical.metadata['coordinate_order'], ('y', 'x'))
        for index in np.ndindex(pixel.scan_shape):
            expected = pixel.coords[index].astype(float)
            expected[:, 0] = (16 - expected[:, 0]) * 0.25
            expected[:, 1] = (expected[:, 1] - 19) * 0.25
            np.testing.assert_allclose(physical.coords[index], expected)
            np.testing.assert_allclose(automatic.coords[index], expected)
            np.testing.assert_allclose(physical.intensities[index], pixel.intensities[index])
        with fd.HyperData.open(self.npy) as source:
            result = source.get_peaks(**self.options, coords_units='auto', progress=False)
            self.assertEqual(result.metadata['coordinate_units'], 'pixels')
            with self.assertRaisesRegex(ValueError, 'stored units'):
                source.get_peaks(**self.options, coords_units='calibrated', progress=False)

    def test_stored_off_center_beam_and_explicit_override_match_eager(self):
        values = pattern(self.shape, [(10, 15, 50)])[None, None]
        beam = fd._center_beam_metadata_from_pixels(1, (10, 10), self.shape)
        fd.HyperData(values, center_beam_metadata=beam, **self.calibration).save(self.saved, overwrite=True)
        options = dict(self.options, r_range=(4, 6))
        with self.open(self.saved) as source:
            result = source.get_peaks(**options, coords_units='calibrated', progress=False)
            self.assertEqual(result.metadata['search_center_pixels'], (10, 10))
            np.testing.assert_array_equal(result.coords[0, 0], [[1.5, -1.0]])
            overridden = source.get_peaks(**options, center=(20, 20), progress=False)
            self.assertEqual(len(overridden.coords[0, 0]), 0)

    def test_three_dimensional_scan_indexing_and_mask_contract(self):
        values = self.values.reshape(6, *self.shape)
        np.save(self.npy, values)
        with fd.HyperData.open(self.npy) as source:
            result = source.get_peaks(**self.options, intensity_radius=2, progress=False, chunk_size=2)
            self.assertEqual(result.scan_shape, (6,))
            for i in range(6):
                np.testing.assert_array_equal(result.coords[i], fd.ReciprocalSpace(values[i]).get_peaks(**self.options))
            with self.assertRaisesRegex(ValueError, '4D'):
                source.get_peaks(**self.options, real_mask=np.ones(6, dtype=bool))

    def test_mask_empty_entries_skips_fully_masked_reads(self):
        mask = np.array([[False, False, False], [True, False, True]])
        with self.open(self.npy) as source:
            tracked = TrackedArray(source._dataset)
            source._dataset = tracked
            result = source.get_peaks(**self.options, intensity_radius=2, real_mask=mask, chunk_size=1, progress=False)
            self.assertEqual(len(tracked.reads), 2)
            self.assertTrue(all(s[0].start == 1 for s in tracked.reads))
            np.testing.assert_array_equal(result.processed_mask, mask)
            self.assertEqual(result.coords[0, 0].shape, (0, 2))
            self.assertEqual(result.intensities[0, 0].shape, (0,))
            tracked.reads.clear()
            empty = source.get_peaks(**self.options, real_mask=np.zeros((2, 3), dtype=bool), progress=False)
            self.assertFalse(np.any(empty.processed_mask))
            self.assertEqual(empty.peak_counts.sum(), 0)
            self.assertEqual(tracked.reads, [])

    def test_bounded_basic_reads_and_one_reused_detector(self):
        with self.open(self.npy) as source:
            tracked = TrackedArray(source._dataset)
            source._dataset = tracked
            init = fd.ReciprocalSpace.__init__
            created = []
            def counted(detector, *args, **kwargs):
                created.append(detector)
                init(detector, *args, **kwargs)
            with patch.object(fd.ReciprocalSpace, '__init__', new=counted):
                source.get_peaks(**self.options, intensity_radius=2, chunk_size=2, progress=False)
            self.assertEqual(len(created), 1)
            self.assertLessEqual(created[0].array.base.nbytes, source.dtype.itemsize)
            visited = np.zeros((2, 3), dtype=int)
            for selection in tracked.reads:
                self.assertEqual(selection[-2:], (slice(None), slice(None)))
                self.assertLessEqual(visited[selection[:2]].size, 2)
                visited[selection[:2]] += 1
            np.testing.assert_array_equal(visited, 1)

    def test_symmetry_provenance_and_synthetic_integrals_are_not_fabricated(self):
        options = dict(self.options, n_fold=4, sym_mode='repair', orbit_min_fraction=0.5, reorder=True)
        with self.open(self.saved) as source:
            result = source.get_peaks(**options, intensity_radius=2, progress=False)
        self.assertTrue(np.any(result.synthetic_mask[1, 1]))
        self.assertTrue(np.isnan(result.intensities[1, 1][result.synthetic_mask[1, 1]]).all())
        self.assertTrue(np.isnan(result.scores[1, 1][result.synthetic_mask[1, 1]]).all())
        np.testing.assert_array_less(result.observed_peak_counts[1, 1], result.peak_counts[1, 1])

    def test_incremental_numeric_file_roundtrip_and_partial_slice(self):
        with self.open(self.saved) as source:
            expected = source.get_peaks(**self.options, intensity_radius=2, coords_units='auto', progress=False)
            path = source.get_peaks(**self.options, intensity_radius=2, coords_units='auto', progress=False,
                                    destination=self.output, chunk_size=1)
            self.assertIsInstance(path, str)
            self.assertFalse(source.closed)
        actual = fd.PeakScanResult.load(path)
        self.assert_results_equal(actual, expected)
        with h5py.File(path, 'r') as file:
            self.assertEqual(file['coords'].dtype, np.dtype('float64'))
            self.assertNotEqual(file['coords'].dtype.kind, 'O')
            index = (1, 1)
            start, count = int(file['starts'][index]), int(file['counts'][index])
            np.testing.assert_allclose(file['coords'][start:start + count], expected.coords[index])
        with self.assertRaisesRegex(MemoryError, 'max_memory_mb'):
            fd.PeakScanResult.load(path, max_memory_mb=1e-6)
        with self.assertRaisesRegex(ValueError, 'PeakScanResult.load'):
            fd.HyperData(path)
        with self.assertRaisesRegex(ValueError, 'PeakScanResult.load'):
            with fd.HyperData.open(path):
                pass

    def test_file_output_does_not_allocate_ragged_result(self):
        with self.open(self.saved) as source, patch.object(peaks_module, 'PeakScanResult', side_effect=AssertionError('No full collection')):
            path = source.get_peaks(**self.options, destination=self.output, progress=False)
        result = fd.PeakScanResult.load(path)
        self.assertIsNone(result.intensities)
        self.assertEqual(result.scan_shape, (2, 3))

    def test_empty_file_and_mask_roundtrip(self):
        with self.open(self.npy) as source:
            path = source.get_peaks(**self.options, intensity_radius=2, destination=self.output,
                                    real_mask=np.zeros((2, 3), dtype=bool), progress=False)
        result = fd.PeakScanResult.load(path)
        self.assertEqual(result.peak_counts.sum(), 0)
        self.assertFalse(np.any(result.processed_mask))

    def test_failure_is_indexed_atomic_and_does_not_close_source(self):
        values = self.values.copy()
        values[0, 1] = np.nan
        np.save(self.npy, values)
        self.output.write_bytes(b'preserve output')
        with self.open(self.npy) as source:
            with self.assertRaisesRegex(ValueError, r'scan index \(0, 1\).*no finite'):
                source.get_peaks(**self.options, destination=self.output, overwrite=True, progress=False)
            self.assertFalse(source.closed)
        self.assertEqual(self.output.read_bytes(), b'preserve output')
        self.assertEqual(list(self.folder.glob('.peaks.h5.*.tmp')), [])

    def test_write_failure_atomicity_and_source_aliases(self):
        self.output.write_bytes(b'preserve output')
        with self.open(self.saved) as source:
            with patch.object(peaks_module, '_append_peak_entry', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    source.get_peaks(**self.options, destination=self.output, overwrite=True, progress=False)
            self.assertFalse(source.closed)
            alias = self.folder / 'alias.h5'
            os.link(self.saved, alias)
            with self.assertRaisesRegex(ValueError, 'different files'):
                source.get_peaks(**self.options, destination=alias, overwrite=True, progress=False)
        self.assertEqual(self.output.read_bytes(), b'preserve output')
        self.assertEqual(list(self.folder.glob('.*.tmp')), [])

    def test_invalid_options_and_memory_raise_before_pixel_reads(self):
        cases = [dict(radius=0), dict(min_distance=0), dict(threshold_rel=2),
                 dict(kernel_amp=-1, trench_amp=1), dict(n_fold=0), dict(sym_mode='invalid'),
                 dict(sym_mode='repair'), dict(reorder=True), dict(center=(np.nan, 1)),
                 dict(r_range=(3, 1)), dict(radius=100000), dict(intensity_radius=-1),
                 dict(coords_units='bad'), dict(real_mask=np.ones((2, 3), dtype=int)),
                 dict(working_memory_mb=1e-6), dict(chunk_size=0), dict(progress=1),
                 dict(overwrite=True), dict(destination=self.folder / 'peaks.4denoise')]
        with self.open(self.saved) as source:
            tracked = TrackedArray(source._dataset)
            source._dataset = tracked
            for args in cases:
                options = dict(self.options, progress=False, **{})
                options.update(args)
                with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                    source.get_peaks(**options)
            self.assertEqual(tracked.reads, [])
            with self.assertRaisesRegex(TypeError, 'unexpected keyword'):
                source.get_peaks(**self.options, msak=np.ones((2, 3)))
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            source.get_peaks(**self.options)

    def test_polar_and_cepstral_rejected(self):
        for data in (self.data.to_polar(output_shape=(16, 64), progress=False), self.data.get_ewpc()):
            data.save(self.saved, overwrite=True)
            with self.open(self.saved) as source, self.assertRaisesRegex(ValueError, 'Cartesian'):
                source.get_peaks(**self.options)

    def test_malformed_peak_files_rejected(self):
        with self.open(self.saved) as source:
            source.get_peaks(**self.options, destination=self.output, progress=False)
        for name, value in (('format_version', '99'), ('scan_shape', (0, 3))):
            with h5py.File(self.output, 'r+') as file:
                old = file.attrs[name]
                file.attrs[name] = value
            with self.assertRaises(ValueError):
                fd.PeakScanResult.load(self.output)
            with h5py.File(self.output, 'r+') as file:
                file.attrs[name] = old
        with h5py.File(self.output, 'r+') as file:
            file['starts'][0, 0] = 100000
        with self.assertRaisesRegex(ValueError, 'starts/counts'):
            fd.PeakScanResult.load(self.output)


if __name__ == '__main__':
    unittest.main()
