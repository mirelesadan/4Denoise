"""Portable numerical and storage contracts for concentrated Bragg contrast."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f'Test imported the wrong fourdenoise.py: {fd.__file__}')


def reference(pattern, annulus, center, sigma, top_pixels, denominator_floor):
    """Literal notebook formula, independent of package helpers/mask generation."""
    yy, xx = np.indices(pattern.shape)
    radius = np.hypot(xx - center[1], yy - center[0])
    mask = (radius >= annulus[0]) & (radius < annulus[1])
    residual = np.maximum(pattern - ndimage.gaussian_filter(pattern, sigma=sigma), 0)
    top = np.partition(residual[mask], -top_pixels)[-top_pixels:]
    denominator = max(float(np.sum(pattern[mask], dtype=np.float64)), denominator_floor)
    return np.float32(np.sum(top, dtype=np.float64) / denominator)


class BraggnessTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(annulus=(1, 5), center=(3, 5), sigma=1.2,
                            top_pixels=7, denominator_floor=1.0)
        self.pattern = np.random.default_rng(19).uniform(0, 40, (7, 11)).astype(np.float32)

    def test_internal_routine_matches_notebook_float32_and_float64(self):
        yy, xx = np.indices(self.pattern.shape)
        mask = (np.hypot(yy - 3, xx - 5) >= 1) & (np.hypot(yy - 3, xx - 5) < 5)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                pattern = self.pattern.astype(dtype)
                value = fd._braggness_pattern(pattern, mask, 1.2, 7, 1.0)
                self.assertEqual(value, reference(pattern, **self.options))
                self.assertEqual(value.dtype, np.dtype('float32'))

    def test_single_pattern_stack_scan_shapes_and_independence(self):
        for shape in ((), (3,), (2, 3)):
            with self.subTest(shape=shape):
                original = np.random.default_rng(37).normal(20, 5, shape + (7, 11)).astype(np.float32)
                result = fd.HyperData(original).get_braggness(**self.options)
                self.assertIsInstance(result, fd.BraggnessResult)
                self.assertEqual(result.score.shape, shape)
                self.assertEqual(result.valid_mask.shape, shape)
                self.assertTrue(result.valid_mask.all())
                self.assertEqual(result.score.dtype, np.dtype('float32'))
                for index in np.ndindex(shape):
                    self.assertEqual(result.score[index], reference(original[index], **self.options))

    def test_reflect_filter_uses_full_pattern_not_annulus_or_scan_axes(self):
        patterns = np.zeros((2, 7, 11), dtype=np.float32)
        patterns[0, 0, 0] = 100
        patterns[0, 3, 5] = 80  # Inner-excluded peak affects the neighboring residual.
        patterns[0, 2, 5] = 12
        options = dict(self.options, annulus=(1, 4), sigma=2)
        with mock.patch.object(fd, 'gaussian_filter', wraps=ndimage.gaussian_filter) as filtering:
            result = fd.HyperData(patterns).get_braggness(**options)
        self.assertEqual(filtering.call_count, 2)
        for call in filtering.call_args_list:
            self.assertEqual(call.args[0].shape, (7, 11))
            self.assertEqual(call.kwargs, dict(sigma=2.0, mode='reflect', truncate=4.0))
        self.assertEqual(result.score[0], reference(patterns[0], **options))
        self.assertEqual(result.score[1], 0)

    def test_half_open_annulus_on_asymmetric_rectangular_detector(self):
        options = dict(annulus=(1, 2), center=(1, 2), sigma=1, top_pixels=8, denominator_floor=1)
        pattern = np.random.default_rng(11).uniform(0, 30, (3, 6)).astype(np.float32)
        result = fd.HyperData(pattern).get_braggness(**options)
        self.assertEqual(result.metadata['annulus_pixel_count'], 8)
        self.assertEqual(result.score, reference(pattern, **options))
        with self.assertRaisesRegex(ValueError, 'pixel count'):
            fd.HyperData(pattern).get_braggness(**dict(options, top_pixels=9))

    def test_subpixel_default_center_is_explicitly_different_from_64(self):
        pattern = np.random.default_rng(5).uniform(0, 10, (128, 128)).astype(np.float32)
        options = dict(annulus=(18, 50), sigma=2, top_pixels=50, denominator_floor=1)
        result = fd.HyperData(pattern).get_braggness(**options)
        self.assertEqual(result.metadata['center'], [63.5, 63.5])
        self.assertEqual(result.metadata['center_source'], 'image_center')
        self.assertEqual(result.score, reference(pattern, center=(63.5, 63.5), **options))
        explicit = fd.HyperData(pattern).get_braggness(center=(64, 64), **options)
        self.assertEqual(explicit.score, reference(pattern, center=(64, 64), **options))
        self.assertNotEqual(result.score, explicit.score)
        self.assertEqual(explicit.metadata['center_source'], 'explicit')

    def test_metadata_center_priority_and_explicit_override(self):
        metadata = dict(shape=(7, 11), center_px=(2.5, 4.25),
                        mean_fit_center_px=(3.1, 5.2),
                        center_calibrated=(0, 0), conv_factor=0.2,
                        initial_mean_fit_center_px=(1, 1), target_center_px=(3, 5))
        data = fd.HyperData(self.pattern, center_beam_metadata=metadata)
        before = deepcopy(data.center_beam_metadata)
        options = {key: value for key, value in self.options.items() if key != 'center'}
        result = data.get_braggness(**options)
        self.assertEqual(result.metadata['center'], [2.5, 4.25])
        self.assertEqual(result.metadata['center_source'], 'center_beam_metadata.center_px')
        self.assertEqual(result.score, reference(self.pattern, center=(2.5, 4.25), **options))
        explicit = data.get_braggness(**self.options)
        self.assertEqual(explicit.metadata['center_source'], 'explicit')
        self.assertEqual(explicit.score, reference(self.pattern, **self.options))
        self.assertEqual(data.center_beam_metadata, before)

    def test_mean_fit_metadata_center_fallback(self):
        data = fd.HyperData(self.pattern, center_beam_metadata={
            'shape': (7, 11), 'center_px': None, 'mean_fit_center_px': (2.75, 4.8),
        })
        options = {key: value for key, value in self.options.items() if key != 'center'}
        result = data.get_braggness(**options)
        self.assertEqual(result.metadata['center'], [2.75, 4.8])
        self.assertEqual(result.metadata['center_source'], 'center_beam_metadata.mean_fit_center_px')
        self.assertEqual(result.score, reference(self.pattern, center=(2.75, 4.8), **options))

    def test_calibrated_metadata_center_converts_to_array_pixels(self):
        center = (2.75, 6.125)
        stored_factor = 0.2
        calibrated = fd._center_to_calibrated(center, self.pattern.shape, stored_factor)
        options = {key: value for key, value in self.options.items() if key != 'center'}
        for use_stored_factor in (True, False):
            with self.subTest(use_stored_factor=use_stored_factor):
                metadata = dict(shape=(7, 11), center_calibrated=calibrated)
                if use_stored_factor:
                    metadata['conv_factor'] = stored_factor
                data = fd.HyperData(self.pattern, reciprocal_units='inverse_angstrom',
                                    reciprocal_conv_factor=0.5 if use_stored_factor else stored_factor,
                                    center_beam_metadata=metadata)
                result = data.get_braggness(**options)
                np.testing.assert_allclose(result.metadata['center'], center)
                self.assertEqual(result.metadata['center_source'], 'center_beam_metadata.center_calibrated')
                self.assertEqual(result.score, reference(self.pattern, center=center, **options))
                json.dumps(result.metadata, allow_nan=False)

    def test_actual_alignment_center_is_used_on_returned_object(self):
        yy, xx = np.indices((7, 11))
        patterns = np.stack([
            0.2 + 100 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 2)
            for cy, cx in ((2.7, 4.8), (3.25, 5.15))
        ])[None].astype(np.float32)
        data = fd.HyperData(patterns)
        with redirect_stdout(io.StringIO()):
            aligned = data.alignment(r_center=2, center=(3, 5), iterations=1)
        options = {key: value for key, value in self.options.items() if key != 'center'}
        result = aligned.get_braggness(**options)
        self.assertEqual(result.metadata['center_source'], 'center_beam_metadata.center_px')
        center = aligned.center_beam_metadata['center_px']
        np.testing.assert_array_equal(result.metadata['center'], center)
        for index in np.ndindex(aligned.scan_shape):
            self.assertEqual(result.score[index], reference(aligned.array[index], center=center, **options))
        self.assertIsNone(data.center_beam_metadata)

    def test_flipped_metadata_center_tracks_current_array(self):
        center = (2.25, 4.5)
        data = fd.HyperData(self.pattern, center_beam_metadata={
            'shape': (7, 11), 'center_px': center,
        }, flip_axis=0)
        options = {key: value for key, value in self.options.items() if key != 'center'}
        result = data.get_braggness(**options)
        flipped_center = (6 - center[0], center[1])
        self.assertEqual(result.metadata['center'], list(flipped_center))
        self.assertEqual(result.score, reference(data.array, center=flipped_center, **options))

    def test_stale_or_malformed_center_metadata_raises_unless_explicit(self):
        cases = [
            {'shape': (8, 11), 'center_px': (3, 5)},
            {'shape': 7, 'center_px': (3, 5)},
            {'center_px': (np.nan, 5)}, {'center_px': (3,)},
            {'mean_fit_center_px': (np.inf, 5)},
            {'center_calibrated': (0, 0)}, ['not a metadata dictionary'],
        ]
        options = {key: value for key, value in self.options.items() if key != 'center'}
        for metadata in cases:
            with self.subTest(metadata=metadata):
                data = fd.HyperData(self.pattern, center_beam_metadata=metadata)
                with self.assertRaises(ValueError):
                    data.get_braggness(**options)
                result = data.get_braggness(**self.options)
                self.assertEqual(result.metadata['center_source'], 'explicit')
                self.assertEqual(result.score, reference(self.pattern, **self.options))

    def test_metadata_without_center_uses_image_center(self):
        data = fd.HyperData(self.pattern, center_beam_metadata={
            'shape': (7, 11), 'radius_px': 1, 'initial_mean_fit_center_px': (1, 1),
        })
        options = {key: value for key, value in self.options.items() if key != 'center'}
        result = data.get_braggness(**options)
        self.assertEqual(result.metadata['center'], [3.0, 5.0])
        self.assertEqual(result.metadata['center_source'], 'image_center')

    def test_zero_and_constant_intensity(self):
        for value in (0, 100):
            for dtype in (np.uint16, np.float32):
                with self.subTest(value=value, dtype=dtype):
                    result = fd.HyperData(np.full((2, 7, 11), value, dtype=dtype)).get_braggness(**self.options)
                    np.testing.assert_array_equal(result.score, 0)
                    self.assertTrue(result.valid_mask.all())

    def test_dtype_policy_and_integer_subtraction_promotion(self):
        for dtype in (np.uint8, np.uint16, np.int32, np.int64, np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                pattern = self.pattern.astype(dtype)
                work_dtype = np.float32 if np.dtype(dtype).kind == 'f' and np.dtype(dtype).itemsize <= 4 else np.float64
                result = fd.HyperData(pattern).get_braggness(**self.options)
                self.assertEqual(result.score, reference(pattern.astype(work_dtype), **self.options))
                self.assertEqual(result.metadata['calculation_dtype'], np.dtype(work_dtype).name)
                self.assertEqual(result.score.dtype, np.dtype('float32'))

    def test_nonfinite_raise_reports_scan_index_and_checks_outside_annulus(self):
        for shape, index in (((), ()), ((2,), (1,)), ((2, 3), (1, 2))):
            for bad_value in (np.nan, np.inf, -np.inf):
                with self.subTest(shape=shape, bad_value=bad_value):
                    array = np.broadcast_to(self.pattern, shape + (7, 11)).copy()
                    array[index + (0, 0)] = bad_value  # Outside the annulus.
                    with self.assertRaises(ValueError) as raised:
                        fd.HyperData(array).get_braggness(**self.options)
                    self.assertIn(f'scan index {index}', str(raised.exception))

    def test_omit_and_scan_mask_do_not_repair_or_evaluate_excluded_patterns(self):
        array = np.broadcast_to(self.pattern, (2, 2, 7, 11)).copy()
        array[0, 1] = np.nan
        array[1, 0, 0, 0] = np.inf
        before = array.copy()
        mask = np.array([[True, False], [True, False]])
        result = fd.HyperData(array).get_braggness(**self.options, scan_mask=mask, nonfinite='omit')
        np.testing.assert_array_equal(result.valid_mask, [[True, False], [False, False]])
        self.assertEqual(int(np.isnan(result.score).sum()), 3)
        self.assertEqual(result.metadata['skipped_patterns'], 2)
        self.assertEqual(result.metadata['nonfinite_patterns'], 1)
        self.assertEqual(result.metadata['evaluated_patterns'], 1)
        np.testing.assert_array_equal(array, before)
        # Even geometric fill/refinement must not reintroduce omitted pixels.
        refined = ndimage.binary_fill_holes(np.ones((2, 2), dtype=bool)) & result.valid_mask
        np.testing.assert_array_equal(refined, result.valid_mask)
        fd.HyperData(array).get_braggness(**self.options, scan_mask=np.array([[True, False], [False, False]]))

    def test_scalar_scan_mask_for_single_pattern(self):
        pattern = np.full((7, 11), np.nan)
        result = fd.HyperData(pattern).get_braggness(**self.options, scan_mask=False)
        self.assertEqual(result.score.shape, ())
        self.assertTrue(np.isnan(result.score))
        self.assertFalse(result.valid_mask)
        self.assertEqual(result.metadata['skipped_patterns'], 1)

    def test_numerical_overflow_raise_and_omit(self):
        pattern = np.zeros((7, 11), dtype=np.float64)
        pattern[3, 4] = 1e40
        pattern[3, 6] = -1e40
        with self.assertRaisesRegex(ValueError, r'failed at scan index \(\)'):
            fd.HyperData(pattern).get_braggness(**self.options)
        result = fd.HyperData(pattern).get_braggness(**self.options, nonfinite='omit')
        self.assertFalse(result.valid_mask)
        self.assertTrue(np.isnan(result.score))
        self.assertEqual(result.metadata['numerical_failures'], 1)

    def test_result_scan_calibration_and_json_metadata(self):
        data = fd.HyperData(np.broadcast_to(self.pattern, (2, 3, 7, 11)),
                            real_units='nm', real_conv_factor=(-0.5, 0.75), real_origin=(10, 20))
        result = data.get_braggness(**self.options)
        image = result.as_real_space()
        self.assertIsInstance(image, fd.RealSpace)
        self.assertIs(image.array, result.score)
        self.assertEqual(image.units, 'nm')
        self.assertEqual(image.conv_factor, (-0.5, 0.75))
        self.assertEqual(image.origin, (10, 20))
        self.assertEqual(image.quantity, 'Braggness')
        self.assertIsNone(image.value_units)
        json.dumps(result.metadata, allow_nan=False)
        self.assertEqual(result.metadata['total_patterns'], 6)
        self.assertEqual(result.metadata['evaluated_patterns'], 6)
        self.assertNotIn('array', result.metadata)
        self.assertIn('(2, 3)', repr(result))
        for array in (self.pattern, self.pattern[None]):
            with self.assertRaisesRegex(ValueError, '2D scan map'):
                fd.HyperData(array).get_braggness(**self.options).as_real_space()

    def test_ordinary_trimmed_flipped_views_are_not_mutated(self):
        source = np.random.default_rng(9).uniform(0, 30, (2, 3, 9, 11)).astype(np.float32)
        before = source.copy()
        for view in (source[..., :7, :], source[..., :7, :][..., ::-1, :], source[..., :7, :][..., :, ::-1]):
            with self.subTest(strides=view.strides):
                self.assertFalse(view.flags.c_contiguous)
                data = fd.HyperData(view)
                data.center_beam_metadata = {'center_px': (3, 5), 'radius_px': 1}
                data.unfold_metadata = {'sentinel': [1, 2]}
                metadata = deepcopy((data.center_beam_metadata, data.unfold_metadata, data.acquisition_metadata))
                result = data.get_braggness(**self.options)
                self.assertTrue(np.shares_memory(data.array, source))
                self.assertEqual(data.shape, view.shape)
                self.assertEqual((data.center_beam_metadata, data.unfold_metadata, data.acquisition_metadata), metadata)
                for index in np.ndindex(data.scan_shape):
                    self.assertEqual(result.score[index], reference(view[index], **self.options))
                np.testing.assert_array_equal(source, before)

    def test_readonly_memmap_trimmed_and_flipped_preserves_backing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'patterns.raw'
            stored = np.memmap(path, dtype='<f4', mode='w+', shape=(2, 3, 9, 11))
            stored[:] = np.random.default_rng(7).uniform(0, 20, stored.shape)
            stored.flush()
            stored._mmap.close()
            before = path.read_bytes()
            source = np.memmap(path, dtype='<f4', mode='r', shape=(2, 3, 9, 11))
            try:
                for view in (source[..., :7, :], source[..., :7, :][..., ::-1, :]):
                    data = fd.HyperData(view)
                    self.assertFalse(data.array.flags.writeable)
                    self.assertTrue(np.shares_memory(data.array, source))
                    result = data.get_braggness(**self.options)
                    for index in np.ndindex(data.scan_shape):
                        self.assertEqual(result.score[index], reference(view[index], **self.options))
                self.assertEqual(path.read_bytes(), before)
            finally:
                del data, view
                source._mmap.close()

    def test_invalid_parameters(self):
        cases = [
            ('annulus', (-1, 4)), ('annulus', (4, 4)), ('annulus', (5, 4)),
            ('annulus', (1, np.inf)), ('annulus', (1,)), ('annulus', (1, 2, 3)),
            ('center', (np.nan, 3)), ('center', (1,)), ('center', ('a', 'b')),
            ('sigma', 0), ('sigma', -1), ('sigma', np.inf), ('sigma', np.nan), ('sigma', (1, 2)),
            ('denominator_floor', 0), ('denominator_floor', -2), ('denominator_floor', np.inf),
            ('top_pixels', 0), ('top_pixels', -1), ('top_pixels', 1.5), ('top_pixels', True),
            ('top_pixels', 10000), ('nonfinite', 'repair'), ('nonfinite', None), ('progress', 1),
            ('scan_mask', np.ones((2,))), ('scan_mask', np.array([True])),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex((ValueError, TypeError), name):
                    fd.HyperData(self.pattern).get_braggness(**dict(self.options, **{name: value}))

    def test_reject_polar_complex_object_boolean_and_5d(self):
        with self.assertRaisesRegex(ValueError, 'Cartesian'):
            fd.HyperData(self.pattern, polar_metadata={}).get_braggness(**self.options)
        for dtype in (complex, object, bool, str):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(TypeError, 'real numeric'):
                    fd.HyperData(self.pattern.astype(dtype)).get_braggness(**self.options)
        with self.assertRaisesRegex(ValueError, '2D, 3D, or 4D'):
            fd.HyperData(np.zeros((1, 1, 1, 7, 11))).get_braggness(**self.options)

    def test_empty_scan_and_empty_detector_region(self):
        result = fd.HyperData(np.empty((0, 7, 11), dtype=np.float32)).get_braggness(**self.options)
        self.assertEqual(result.score.shape, (0,))
        self.assertEqual(result.metadata['total_patterns'], 0)
        with self.assertRaisesRegex(ValueError, 'pixel count'):
            fd.HyperData(self.pattern).get_braggness(**dict(self.options, annulus=(30, 40)))


if __name__ == '__main__':
    unittest.main()
