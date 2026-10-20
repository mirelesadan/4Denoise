"""Regression tests for normalization arithmetic and explicit DP arguments."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Tests imported the wrong fourdenoise.py: {fd.__file__}")


class NormalizationTests(unittest.TestCase):
    def test_standardize_has_zero_mean_and_unit_variance(self):
        for shape in ((3, 4), (2, 3, 4), (2, 3, 3, 4)):
            source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data = fd.HyperData(source)
            for method in ('global', 'local'):
                with self.subTest(shape=shape, method=method):
                    result = data.standardize(method=method).array
                    axes = None if method == 'global' else (-2, -1)
                    np.testing.assert_allclose(result.mean(axis=axes), 0, atol=1e-6)
                    np.testing.assert_allclose(result.std(axis=axes), 1, atol=1e-6)
                    self.assertEqual(result.dtype, np.float32)

    def test_signed_integer_normalization_cannot_overflow(self):
        source = np.array([-128, -64, 0, 127], dtype=np.int8).reshape(2, 2)
        expected = (source.astype(np.float64) + 128) / 255
        for method in ('global', 'local'):
            with self.subTest(method=method):
                actual = fd.HyperData(source).normalize(method=method).array
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                self.assertGreaterEqual(actual.min(), 0)
                self.assertLessEqual(actual.max(), 1)

    def test_integer_boundaries_promote_before_arithmetic(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64):
            limits = np.iinfo(dtype)
            source = np.array([[limits.min, limits.max]], dtype=dtype)
            data = fd.HyperData(source)
            for method in ('global', 'local'):
                with self.subTest(dtype=dtype, method=method):
                    normalized = data.normalize(method=method).array
                    standardized = data.standardize(method=method).array
                    np.testing.assert_allclose(normalized, [[0, 1]])
                    np.testing.assert_allclose(standardized, [[-1, 1]])
                    expected_dtype = np.result_type(source.dtype, np.float32)
                    self.assertEqual(normalized.dtype, expected_dtype)
                    self.assertEqual(standardized.dtype, expected_dtype)

    def test_local_normalization_scales_each_pattern(self):
        source = np.array([[[10, 20], [30, 40]], [[-100, 0], [100, 200]]])
        expected = np.array([[[0, 1 / 3], [2 / 3, 1]]] * 2)
        np.testing.assert_allclose(
            fd.HyperData(source).normalize(method='local').array, expected,
        )

    def test_constant_arrays_and_patterns_become_zeros(self):
        for dtype in (np.int16, np.float32, np.float64, np.bool_):
            source = np.full((2, 2, 3, 4), 1, dtype=dtype)
            for operation in ('normalize', 'standardize'):
                for method in ('global', 'local'):
                    with self.subTest(dtype=dtype, operation=operation, method=method):
                        with np.errstate(all='raise'):
                            result = getattr(fd.HyperData(source), operation)(method=method)
                        np.testing.assert_array_equal(result.array, np.zeros(source.shape))
                        self.assertTrue(np.isfinite(result.array).all())

    def test_constant_patterns_do_not_affect_other_local_patterns(self):
        source = np.array([[[5, 5], [5, 5]], [[0, 1], [2, 3]]], dtype=np.float32)
        data = fd.HyperData(source)
        for operation in ('normalize', 'standardize'):
            with self.subTest(operation=operation):
                with np.errstate(all='raise'):
                    result = getattr(data, operation)(method='local').array
                np.testing.assert_array_equal(result[0], np.zeros((2, 2)))
                self.assertGreater(np.ptp(result[1]), 0)

    def test_small_float32_variance_is_not_treated_as_constant(self):
        source = np.arange(4, dtype=np.float32).reshape(2, 2) * 1e-30
        result = fd.HyperData(source).standardize().array
        np.testing.assert_allclose(result.std(), 1, atol=1e-6)

    def test_defaults_are_unchanged(self):
        data = fd.HyperData(np.arange(24, dtype=float).reshape(2, 3, 4))
        np.testing.assert_array_equal(
            data.standardize().array, data.standardize(method='local').array,
        )
        np.testing.assert_array_equal(
            data.normalize().array, data.normalize(method='global').array,
        )

    def test_invalid_methods_and_data_raise_helpful_errors(self):
        for operation in ('normalize', 'standardize'):
            with self.subTest(operation=operation, invalid='method'):
                with self.assertRaisesRegex(ValueError, 'global.*local'):
                    getattr(fd.HyperData(np.ones((2, 2))), operation)(method='typo')
            for method in ('global', 'local'):
                for value in (np.nan, np.inf, -np.inf):
                    source = np.array([[1, value], [2, 3]])
                    with self.subTest(operation=operation, method=method, value=value):
                        with self.assertRaisesRegex(ValueError, 'finite input'):
                            getattr(fd.HyperData(source), operation)(method=method)
                for source in (np.empty((0, 2)), np.ones((2, 2), dtype=complex),
                               np.array([['a', 'b']])):
                    with self.subTest(operation=operation, method=method, dtype=source.dtype):
                        with self.assertRaisesRegex(ValueError, 'nonempty.*real-valued'):
                            getattr(fd.HyperData(source), operation)(method=method)

    def test_input_and_metadata_are_preserved_through_unfold(self):
        source = np.arange(2 * 3 * 3 * 4, dtype=np.int16).reshape(2, 3, 3, 4)
        before = source.copy()
        data = fd.HyperData(
            source, real_units='nm', real_conv_factor=0.5,
            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.2,
            acquisition_metadata={'source': 'regression'},
        )
        unfolded = data.unfold(domain='real', method='serpentine')
        for operation in ('normalize', 'standardize'):
            for method in ('global', 'local'):
                with self.subTest(operation=operation, method=method):
                    result = getattr(unfolded, operation)(method=method)
                    self.assertIsNotNone(result.unfold_metadata)
                    np.testing.assert_allclose(
                        result.unfold(undo=True).array,
                        getattr(data, operation)(method=method).array,
                    )
                    self.assertEqual(result.real_units, data.real_units)
                    self.assertEqual(result.real_conv_factor, data.real_conv_factor)
                    self.assertEqual(result.real_origin, data.real_origin)
                    self.assertEqual(result.reciprocal_units, data.reciprocal_units)
                    self.assertEqual(result.reciprocal_conv_factor, data.reciprocal_conv_factor)
                    self.assertEqual(result.acquisition_metadata, data.acquisition_metadata)
                    self.assertFalse(np.shares_memory(result.array, unfolded.array))
        np.testing.assert_array_equal(source, before)


class GetDPArgumentTests(unittest.TestCase):
    def setUp(self):
        self.source = np.arange(4, dtype=float).reshape(2, 2, 1, 1) * np.ones((8, 8))
        self.data = fd.HyperData(self.source)
        self.mask = np.array([[True, False], [False, False]])

    def test_unknown_keywords_are_rejected_for_every_operation(self):
        for operation in ('mean', 'median', 'max', 'min', 'std', 'random'):
            for keyword in ('real_mask', 'msak', 'percentil'):
                with self.subTest(operation=operation, keyword=keyword):
                    with self.assertRaisesRegex(TypeError, keyword):
                        self.data.get_dp(operation=operation, **{keyword: self.mask})

    def test_valid_mask_is_applied(self):
        selected = self.data.get_dp(mask=self.mask, operation='mean').array
        np.testing.assert_array_equal(selected, self.source[0, 0])
        self.assertFalse(np.array_equal(selected, self.data.get_dp().array))

    def test_removed_operation_is_rejected(self):
        for call in (
            lambda: self.data.get_dp('flat_mean'),
            lambda: self.data.get_dp(operation='flat_mean'),
        ):
            with self.assertRaisesRegex(ValueError, 'operation must be one of'):
                call()

    def test_removed_options_are_rejected(self):
        for operation in ('mean', 'std', 'median', 'random'):
            for keyword in ('percentile', 'show_mask'):
                with self.subTest(operation=operation, keyword=keyword):
                    with self.assertRaisesRegex(TypeError, keyword):
                        self.data.get_dp(operation=operation, **{keyword: 10})


if __name__ == '__main__':
    unittest.main()
