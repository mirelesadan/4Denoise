"""Numerical, selection, metadata and bounded-allocation DP regressions."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_reductions as reductions

if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':
    raise ImportError(f'Imported the wrong module: {fd.__file__}')


def budget_for(source, operation, patterns=3):
    plan = reductions._diffraction_reduction_plan(source.shape, source.dtype, operation)
    return (plan['fixed_bytes'] + patterns * plan['pattern_bytes'] + 0.25) / 1024 ** 2


class RecordingSource:
    """Forbid full materialization, recording only basic bounded source reads."""

    def __init__(self, array, limit):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.limit = limit
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('The complete source must not be converted to an array.')

    def __getitem__(self, slices):
        if not isinstance(slices, tuple) or any(not isinstance(s, slice) for s in slices):
            raise AssertionError('Source access must use basic rectangular slices.')
        block = self.array[slices]
        size = int(np.prod(block.shape[:-2]))
        if size > self.limit:
            raise AssertionError(f'Read {size} patterns, exceeding the {self.limit} pattern limit.')
        self.reads.append(slices)
        return block


class DiffractionReductionTests(unittest.TestCase):
    def test_all_operations_dtypes_and_uneven_masks_match_numpy(self):
        rng = np.random.default_rng(21)
        base = rng.integers(-40, 50, size=(5, 7, 4, 6))
        mask = rng.random((5, 7)) > 0.4
        for dtype in (np.bool_, np.int8, np.uint16, np.int64, np.float16,
                      np.float32, np.float64, np.complex64, np.complex128):
            source = np.abs(base) if np.dtype(dtype).kind == 'u' else base
            source = source.astype(dtype)
            if np.dtype(dtype).kind == 'c':
                source += 1j * rng.integers(-10, 10, size=source.shape)
            data = fd.HyperData(source)
            for operation in ('mean', 'std', 'min', 'max'):
                for selection in (None, mask):
                    with self.subTest(dtype=dtype, operation=operation, masked=selection is not None):
                        selected = source if selection is None else source[selection]
                        expected = getattr(np, operation)(selected, axis=tuple(range(selected.ndim - 2)))
                        result = data.get_dp(
                            operation=operation, mask=selection,
                            working_memory_mb=budget_for(source, operation),
                        ).array
                        self.assertEqual(result.dtype, expected.dtype)
                        precision = result.real.dtype if result.dtype.kind in 'fc' else np.dtype('float64')
                        tolerance = max(1e-12, 16 * np.finfo(precision).eps)
                        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)

    def test_unequal_block_sizes_are_weighted_by_pattern_count(self):
        source = np.array([0, 1, 2, 3, 4, 5, 100], dtype=np.uint16)[:, None, None]
        for operation in ('mean', 'std'):
            result = fd.HyperData(source).get_dp(
                operation=operation, working_memory_mb=budget_for(source, operation),
            )
            np.testing.assert_allclose(result.array, getattr(np, operation)(source, axis=0))

    def test_variance_is_stable_with_large_offset_and_small_spread(self):
        source = (1e12 + np.arange(37, dtype=float))[:, None, None] * np.ones((1, 3, 4))
        result = fd.HyperData(source).get_dp('std', working_memory_mb=budget_for(source, 'std'))
        np.testing.assert_allclose(result.array, np.std(np.arange(37)), rtol=2e-6)

    def test_constant_and_single_pattern_std_are_zero(self):
        for shape in ((1, 3, 4), (1, 1, 3, 4), (5, 7, 3, 4)):
            source = np.full(shape, 60000, dtype=np.uint16)
            result = fd.HyperData(source).get_dp('std', working_memory_mb=budget_for(source, 'std'))
            np.testing.assert_array_equal(result.array, 0)

    def test_integer_extrema_and_accumulation_do_not_overflow(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64):
            bounds = np.iinfo(dtype)
            source = np.array([bounds.min, bounds.max] * 5, dtype=dtype)[:, None, None]
            for operation in ('mean', 'std', 'min', 'max'):
                with self.subTest(dtype=dtype, operation=operation):
                    actual = fd.HyperData(source).get_dp(
                        operation=operation, working_memory_mb=budget_for(source, operation),
                    ).array
                    expected = getattr(np, operation)(source, axis=0)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12)
                    self.assertEqual(actual.dtype, expected.dtype)

    def test_nan_inf_propagation_and_masked_exclusions(self):
        source = np.arange(9 * 2 * 3, dtype=float).reshape(3, 3, 2, 3)
        source[0, 0, 0, 0] = np.nan
        source[1, 1, 0, 1] = np.inf
        source[2, 1, 0, 2] = -np.inf
        source[0, 1, 1, 0] = np.inf
        source[2, 2, 1, 0] = -np.inf
        mask = np.ones((3, 3), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[2, 1] = False
        for selection in (None, mask):
            for operation in ('mean', 'std', 'min', 'max'):
                with self.subTest(operation=operation, masked=selection is not None), np.errstate(all='ignore'):
                    selected = source if selection is None else source[selection]
                    expected = getattr(np, operation)(selected, axis=tuple(range(selected.ndim - 2)))
                    actual = fd.HyperData(source).get_dp(
                        mask=selection, operation=operation,
                        working_memory_mb=budget_for(source, operation),
                    ).array
                    np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_noncontiguous_and_negative_stride_sources(self):
        array = np.arange(11 * 13 * 5 * 7, dtype=float).reshape(11, 13, 5, 7)
        source = array.transpose(1, 0, 3, 2)[::-1, ::2, ::-1, ::2]
        self.assertFalse(source.flags.c_contiguous)
        mask = np.indices(source.shape[:2]).sum(axis=0) % 3 != 0
        original = array.copy()
        for operation in ('mean', 'std', 'min', 'max'):
            actual = fd.HyperData(source).get_dp(
                mask=mask, operation=operation, working_memory_mb=budget_for(source, operation),
            ).array
            np.testing.assert_allclose(actual, getattr(np, operation)(source[mask], axis=0), rtol=1e-12)
        np.testing.assert_array_equal(array, original)

    def test_exact_median_keeps_full_selection_semantics(self):
        source = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        data = fd.HyperData(source)
        mask = np.indices((5, 7)).sum(axis=0) % 2 == 0
        for selection in (None, mask):
            selected = source if selection is None else source[selection]
            expected = np.median(selected, axis=tuple(range(selected.ndim - 2)))
            np.testing.assert_array_equal(data.get_dp(mask=selection, operation='median').array, expected)
        with self.assertRaisesRegex(ValueError, 'unsupported for median'):
            data.get_dp('median', working_memory_mb=256)


class DiffractionMemoryTests(unittest.TestCase):
    def test_all_source_reads_obey_the_budgeted_block_limit(self):
        array = np.arange(7 * 11 * 3 * 4, dtype=float).reshape(7, 11, 3, 4)
        mask = np.indices((7, 11)).sum(axis=0) % 3 == 0
        for operation in ('mean', 'std', 'min', 'max'):
            for selection in (None, mask):
                with self.subTest(operation=operation, masked=selection is not None):
                    source = RecordingSource(array, 3)
                    memory = budget_for(array, operation)
                    plan = reductions._diffraction_reduction_plan(array.shape, array.dtype, operation, memory)
                    self.assertEqual(plan['max_patterns'], 3)
                    self.assertLessEqual(plan['estimated_peak_bytes'], plan['budget_bytes'])
                    actual = reductions._reduce_diffraction(source, operation, mask=selection, working_memory_mb=memory)
                    self.assertGreater(len(source.reads), 1)
                    selected = array if selection is None else array[selection]
                    expected = getattr(np, operation)(selected, axis=tuple(range(selected.ndim - 2)))
                    np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_empty_mask_blocks_are_skipped_before_source_reads(self):
        array = np.arange(6 * 8 * 3 * 4).reshape(6, 8, 3, 4)
        mask = np.zeros((6, 8), dtype=bool)
        mask[-1, -1] = True
        source = RecordingSource(array, 3)
        result = reductions._reduce_diffraction(source, mask=mask, working_memory_mb=budget_for(array, 'mean'))
        self.assertEqual(len(source.reads), 1)
        np.testing.assert_array_equal(result, array[-1, -1])

    def test_iterator_slices_share_storage_for_unmasked_numpy_data(self):
        array = np.arange(9 * 8 * 3 * 4).reshape(9, 8, 3, 4).transpose(1, 0, 3, 2)
        counts = []
        for block, count in reductions._iter_diffraction_blocks(array, 5):
            self.assertTrue(np.shares_memory(block, array))
            self.assertLessEqual(count, 5)
            counts.append(count)
        self.assertEqual(sum(counts), np.prod(array.shape[:2]))

    def test_tiny_budget_errors_report_the_minimum(self):
        data = fd.HyperData(np.ones((2, 3, 128, 128)))
        for operation in ('mean', 'std', 'min', 'max'):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'need at least.*one pattern'):
                data.get_dp(operation=operation, working_memory_mb=0.001)

    def test_plan_budget_and_block_limits_for_large_detector_dtypes(self):
        for dtype in (np.uint16, np.float32, np.float64, np.complex128):
            for operation in ('mean', 'std', 'min', 'max'):
                plan = reductions._diffraction_reduction_plan((256, 256, 512, 512), dtype, operation, 256)
                self.assertGreater(plan['max_patterns'], 0)
                self.assertLessEqual(plan['estimated_peak_bytes'], 256 * 1024 ** 2)

    def test_auto_budget_is_fixed_not_proportional_to_scan_size(self):
        for shape in ((100, 100, 64, 64), (10000, 10000, 64, 64)):
            plan = reductions._diffraction_reduction_plan(shape, np.float32, 'std')
            self.assertEqual(plan['budget_bytes'], 256 * 1024 ** 2)
            self.assertLessEqual(plan['estimated_peak_bytes'], plan['budget_bytes'])

    def test_helper_reduces_real_hdf5_blocks_without_eager_loading(self):
        array = np.arange(5 * 7 * 3 * 4, dtype=np.uint16).reshape(5, 7, 3, 4)
        mask = np.indices((5, 7)).sum(axis=0) % 2 == 0
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / 'source.h5', 'w') as handle:
                source = handle.create_dataset('data', data=array, chunks=(1, 2, 3, 4))
                for operation in ('mean', 'std', 'min', 'max'):
                    result = reductions._reduce_diffraction(
                        source, operation, mask=mask, working_memory_mb=budget_for(array, operation),
                    )
                    np.testing.assert_allclose(result, getattr(np, operation)(array[mask], axis=0), rtol=1e-12)


class GetDPSelectionTests(unittest.TestCase):
    def setUp(self):
        self.array = np.arange(5 * 7 * 4 * 6, dtype=np.uint16).reshape(5, 7, 4, 6)
        self.data = fd.HyperData(
            self.array, real_units='nm', real_conv_factor=(2, 3), real_origin=(10, 20),
            reciprocal_units='mrad', reciprocal_conv_factor=0.2,
        )

    def test_pixel_regions_and_single_axis_selections(self):
        for y, x, expected in (
            ((1, 5), (2, 6), self.array[1:5, 2:6]),
            (2, None, self.array[2:3]),
            (None, 3, self.array[:, 3:4]),
        ):
            for operation in ('mean', 'std', 'min', 'max', 'median'):
                with self.subTest(y=y, x=x, operation=operation):
                    actual = self.data.get_dp(y, x, operation=operation).array
                    np.testing.assert_allclose(actual, getattr(np, operation)(expected, axis=(0, 1)))

    def test_calibrated_regions_respect_spacing_and_origin(self):
        selected = self.data.get_dp(
            y=(12, 20), x=(26, 38), selection_units='calibrated',
            working_memory_mb=budget_for(self.array, 'mean'),
        )
        np.testing.assert_array_equal(selected.array, np.mean(self.array[1:5, 2:6], axis=(0, 1)))

    def test_mask_takes_precedence_and_accepts_boolean_lists(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[4, 6] = True
        selected = self.data.get_dp(y=(0, 1), x=(0, 1), mask=mask.tolist())
        np.testing.assert_array_equal(selected.array, self.array[4, 6])

    def test_direct_index_does_not_allocate_reduction_buffers(self):
        with patch.object(reductions, '_reduce_diffraction', side_effect=AssertionError('Unexpected aggregation')):
            dp = self.data.get_dp(2, 3, working_memory_mb=0.000001)
            np.testing.assert_array_equal(dp.array, self.array[2, 3])
            self.assertTrue(np.shares_memory(dp.array, self.array))
            stack = fd.HyperData(self.array.reshape(35, 4, 6))
            np.testing.assert_array_equal(stack.get_dp(2).array, self.array.reshape(35, 4, 6)[2])
            np.testing.assert_array_equal(stack.get_dp(x=-1).array, self.array[-1, -1])

    def test_mask_errors_are_helpful_for_all_operations(self):
        for operation in ('mean', 'std', 'min', 'max', 'median', 'random'):
            for mask, error in (
                (np.ones((5, 7), dtype=int), TypeError),
                (np.ones((3, 4), dtype=bool), ValueError),
                (np.zeros((5, 7), dtype=bool), ValueError),
            ):
                with self.subTest(operation=operation, mask_shape=mask.shape), self.assertRaises(error):
                    self.data.get_dp(mask=mask, operation=operation)
        with self.assertRaisesRegex(ValueError, '4D'):
            fd.HyperData(self.array.reshape(35, 4, 6)).get_dp(mask=np.ones(35, dtype=bool))

    def test_invalid_memory_and_operations(self):
        for budget in (0, -1, True, np.nan, np.inf, '64', [64], 1e-20, 10**1000):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'working_memory_mb'):
                self.data.get_dp(working_memory_mb=budget)
        for operation in ('flat_mean', 'typo', ['mean']):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'operation must'):
                self.data.get_dp(operation=operation)
        for shape in ((2, 3), (0, 3, 4)):
            with self.assertRaisesRegex(ValueError, 'nonempty 3D'):
                fd.HyperData(np.ones(shape)).get_dp()

    def test_random_selection_stays_within_masks_and_regions(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[3, 5] = True
        with redirect_stdout(io.StringIO()):
            np.testing.assert_array_equal(self.data.get_dp(mask=mask, operation='random').array, self.array[3, 5])
            np.testing.assert_array_equal(self.data.get_dp(y=(3, 4), x=(5, 6), operation='random').array, self.array[3, 5])
        with self.assertRaisesRegex(ValueError, 'unsupported for random'):
            self.data.get_dp('random', working_memory_mb=64)

    def test_metadata_and_input_are_preserved(self):
        beam = fd._center_beam_metadata_from_pixels(
            1, (1.5, 2.5), (4, 6), units='mrad', conv_factor=0.2, source='alignment',
        )
        self.data.center_beam_metadata = beam
        before = self.array.copy()
        mask = np.indices((5, 7)).sum(axis=0) % 2 == 0
        for operation in ('mean', 'std', 'min', 'max', 'median'):
            dp = self.data.get_dp(mask=mask, operation=operation)
            self.assertIsInstance(dp, fd.ReciprocalSpace)
            self.assertEqual((dp.units, dp.conv_factor), ('mrad', 0.2))
            self.assertEqual(dp.center_beam_metadata, beam)
            dp.center_beam_metadata['center_px'] = (0, 0)
            self.assertEqual(self.data.center_beam_metadata, beam)
            self.assertFalse(np.shares_memory(dp.array, self.array))
        np.testing.assert_array_equal(self.array, before)

    def test_polar_and_cepstral_geometry_are_preserved(self):
        for data in (
            self.data.to_polar(output_shape=(4, 16), progress=False),
            self.data.get_ewpc(),
        ):
            expected = deepcopy(data.polar_metadata or data.cepstral_metadata)
            for operation in ('mean', 'std', 'min', 'max', 'median'):
                dp = data.get_dp(operation=operation)
                self.assertEqual(dp.is_polar, data.is_polar)
                self.assertEqual(dp.is_cepstral, data.is_cepstral)
                self.assertEqual(dp.polar_metadata or dp.cepstral_metadata, expected)


if __name__ == '__main__':
    unittest.main()
