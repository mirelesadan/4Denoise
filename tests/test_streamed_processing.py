"""Per-pattern HDF5 transforms: numerical parity, bounded reads and publication."""

from contextlib import contextmanager
from math import prod
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import Mock, patch
import weakref

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_streaming as streaming


class TrackedDataset:
    """Observe basic slices and reject any retention of the previous input."""

    def __init__(self, dataset, outputs=None):
        self.dataset = dataset
        self.reads = []
        self.previous_input = None
        self.outputs = outputs if outputs is not None else []

    def __array__(self, *args, **kwargs):
        raise AssertionError('The complete source was converted.')

    def __getitem__(self, selection):
        if not all(isinstance(item, slice) for item in selection):
            raise AssertionError('Only basic slices are allowed.')
        if self.previous_input is not None and self.previous_input() is not None:
            raise AssertionError('The previous input buffer survived until the next read.')
        if any(reference() is not None for reference in self.outputs):
            raise AssertionError('The previous transform output survived until the next read.')
        result = self.dataset[selection]
        self.reads.append((selection, result.nbytes))
        self.previous_input = weakref.ref(result)
        return result


class StreamedProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'source.4denoise'
        self.destination = self.folder / 'result.h5'
        self.values = np.arange(5 * 7 * 6 * 8, dtype=np.float32).reshape(5, 7, 6, 8)
        self.data = fd.HyperData(
            self.values, real_units='nm', real_conv_factor=(2, -3),
            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.25,
        )
        self.data.set_rq_calibration(rotation_deg=17, mirror_axis='x')
        self.data.acquisition_metadata = {'experiment': {'name': 'streamed'}, 'scan_shape': (5, 7)}
        self.data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            1.5, (2.5, 3.5), self.data.pattern_shape, units='mrad', conv_factor=0.25,
        )
        self.data.save(self.source)

    def stream(self, transform, *, output_pattern_shape=None, **kwargs):
        options = {
            'reader_factory': fd.HyperData.open_hdf5,
            'output_pattern_shape': output_pattern_shape or self.data.pattern_shape,
            'working_memory_mb': 0.04, 'progress': False,
        }
        options.update(kwargs)
        return streaming._stream_transform(self.source, self.destination, transform=transform, **options)

    def assert_metadata_equal(self, actual, expected):
        for name in streaming._METADATA_KEYS:
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)
        actual_rq = actual.rq_calibration.to_dict() if actual.rq_calibration is not None else None
        expected_rq = expected.rq_calibration.to_dict() if expected.rq_calibration is not None else None
        self.assertEqual(actual_rq, expected_rq)
        self.assertIsNone(actual.unfold_metadata)

    def tracked_factory(self, readers, datasets, outputs=None):
        @contextmanager
        def factory(path, hdf5_dataset=None):
            with fd.HyperData.open_hdf5(path, hdf5_dataset=hdf5_dataset) as reader:
                readers.append(reader)
                tracked = TrackedDataset(reader._dataset, outputs)
                datasets.append(tracked)
                reader._dataset = tracked
                yield reader
        return factory

    def assert_no_temporary(self):
        self.assertEqual(list(self.folder.glob('.result.h5.*.tmp')), [])

    def test_remove_background_matches_memory_and_global_metadata(self):
        background = np.arange(48, dtype=np.float64).reshape(6, 8) / 3
        expected = self.data.remove_bg(background, bg_frac=0.7, a_min=None, a_max=900)
        seen_shapes, origins = [], []

        def transform(block):
            seen_shapes.append(block.scan_shape)
            origins.append(block.real_origin)
            return block.remove_bg(background, bg_frac=0.7, a_min=None, a_max=900)

        result = self.stream(transform)
        self.assertEqual(result, str(self.destination))
        self.assertIsInstance(result, str)
        self.assertGreater(len(seen_shapes), 2)
        self.assertEqual(seen_shapes[0], (1, 1))
        self.assertEqual(sum(prod(shape) for shape in seen_shapes), 35)
        self.assertNotEqual(origins[-1], self.data.real_origin)
        actual = fd.HyperData(result)
        np.testing.assert_allclose(actual.array, expected.array)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assert_metadata_equal(actual, expected)
        self.assertEqual(actual.real_origin, (10, 20))
        with fd.HyperData.open_hdf5(result) as reader:
            np.testing.assert_array_equal(reader.get_dp(4, 6).array, expected.array[4, 6])

    def test_cartesian_canvas_variants_orders_and_cropped_sector_match_memory(self):
        for cropped in (False, True):
            polar = self.data.to_polar(output_shape=(5, 16), progress=False)
            if cropped:
                polar = polar.crop(kylim=(1, 5), kxlim=(3, 11))
            polar.save(self.source, overwrite=True)
            for canvas in (None, 'original', (7, 9)):
                pattern_shape = (
                    (2 * polar.pattern_shape[0],) * 2 if canvas is None
                    else self.data.pattern_shape if canvas == 'original' else canvas
                )
                for order in (0, 1, 3):
                    with self.subTest(cropped=cropped, canvas=canvas, order=order):
                        expected = polar.to_cartesian(
                            output_shape=canvas, order=order, fill_value=-3, progress=False,
                        )
                        result = self.stream(
                            lambda block: block.to_cartesian(
                                output_shape=canvas, order=order, fill_value=-3, progress=False,
                            ), output_pattern_shape=pattern_shape,
                            working_memory_mb=0.08, overwrite=True,
                        )
                        actual = fd.HyperData(result)
                        np.testing.assert_allclose(actual.array, expected.array)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assert_metadata_equal(actual, expected)
                        self.assertFalse(actual.is_polar)

    def test_3d_generic_dataset_selection_and_suffix_default(self):
        values = self.values.reshape(35, 6, 8).astype(np.uint16)
        source = self.folder / 'generic.h5'
        with h5py.File(source, 'w') as file:
            file.create_dataset('scan/images', data=values)
            file.create_dataset('other', data=np.ones((2, 3, 4)))
        readers, datasets = [], []
        result = streaming._stream_transform(
            source, self.folder / 'stack', hdf5_dataset='/scan/images',
            reader_factory=self.tracked_factory(readers, datasets),
            transform=lambda block: block.remove_bg(np.full((6, 8), 1.5), a_min=None),
            output_pattern_shape=(6, 8), working_memory_mb=0.04, progress=False,
            compression=None,
        )
        self.assertEqual(result, str(self.folder / 'stack.4denoise'))
        actual = fd.HyperData(result)
        expected = fd.HyperData(values).remove_bg(np.full((6, 8), 1.5), a_min=None)
        np.testing.assert_array_equal(actual.array, expected.array)
        self.assert_metadata_equal(actual, expected)
        self.assertGreater(len(datasets[0].reads), 2)
        self.assertTrue(readers[0]._closed)
        with h5py.File(result, 'r') as file:
            self.assertIsNone(file['array'].compression)

    def test_transformed_first_metadata_wins_and_unfold_undo_is_cleared(self):
        def transform(block):
            result = block.remove_bg(np.zeros((6, 8)))
            result.acquisition_metadata = {'returned_origin': block.real_origin}
            result.center_beam_metadata = None
            result.set_rq_calibration(rotation_deg=31)
            result.unfold_metadata = {'not_a_valid_partial_scan_undo': True}
            return result

        actual = fd.HyperData(self.stream(transform))
        self.assertEqual(actual.acquisition_metadata, {'returned_origin': (10, 20)})
        self.assertEqual(actual.rq_calibration.rotation_deg, 31)
        self.assertIsNone(actual.center_beam_metadata)
        self.assertIsNone(actual.unfold_metadata)

    def test_polar_and_cepstral_metadata_survive_background_transform(self):
        for data in (
            self.data.to_polar(output_shape=(4, 12), progress=False),
            self.data.get_ewpc(),
        ):
            with self.subTest(polar=data.is_polar, cepstral=data.is_cepstral):
                data.save(self.source, overwrite=True)
                background = np.zeros(data.pattern_shape)
                expected = data.remove_bg(background)
                result = self.stream(
                    lambda block: block.remove_bg(background),
                    output_pattern_shape=data.pattern_shape, overwrite=True,
                )
                actual = fd.HyperData(result)
                np.testing.assert_array_equal(actual.array, expected.array)
                self.assert_metadata_equal(actual, expected)

    def test_file_schema_matches_normal_hyperdata_save(self):
        expected = self.data.remove_bg(np.ones((6, 8)))
        reference = self.folder / 'reference.h5'
        expected.save(reference)
        self.stream(lambda block: block.remove_bg(np.ones((6, 8))))
        with h5py.File(reference, 'r') as reference_file, h5py.File(self.destination, 'r') as file:
            self.assertEqual(dict(file.attrs), dict(reference_file.attrs))
            self.assertEqual(set(file['metadata']), set(reference_file['metadata']))
            self.assertEqual(file['metadata'].attrs['kind'], 'dict')
            self.assertEqual(file['array'].shape, self.data.shape)
            self.assertEqual(file['array'].chunks, (1, 1, 6, 8))
            self.assertEqual(file['array'].compression, 'gzip')
            self.assertEqual(file['array'].compression_opts, 4)

    def test_probe_and_all_blocks_release_buffers_before_next_read(self):
        readers, datasets, outputs = [], [], []

        def transform(block):
            result = block.remove_bg(np.zeros((6, 8)))
            outputs.append(weakref.ref(result.array))
            return result

        self.stream(transform, reader_factory=self.tracked_factory(readers, datasets, outputs))
        self.assertGreater(len(datasets[0].reads), 2)
        self.assertIsNone(datasets[0].previous_input())
        self.assertTrue(all(reference() is None for reference in outputs))
        self.assertTrue(readers[0]._closed)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            readers[0].get_dp()
        coverage = np.zeros(self.data.scan_shape, dtype=int)
        for slices, read_bytes in datasets[0].reads:
            coverage[slices[:-2]] += 1
            self.assertLess(read_bytes, self.values.nbytes)
        np.testing.assert_array_equal(coverage, 1)

    def test_identity_transform_aliases_are_released(self):
        readers, datasets = [], []
        result = self.stream(lambda block: block, reader_factory=self.tracked_factory(readers, datasets))
        np.testing.assert_array_equal(fd.HyperData(result).array, self.values)
        self.assertIsNone(datasets[0].previous_input())

    def test_budget_plan_has_fixed_workspace_and_actual_block_buffers(self):
        plan = streaming._stream_transform_plan(
            self.data.shape, self.data.dtype, (7, 9), 0.05, output_dtype=np.float64,
        )
        self.assertGreaterEqual(plan['fixed_bytes'], 64 * 48 + 128 * 63)
        self.assertEqual(plan['source_workspace_bytes'], 64 * 48)
        self.assertEqual(plan['output_workspace_bytes'], 128 * 63)
        self.assertEqual(plan['angular_padding_bytes'], 64 * 12 * 6)
        self.assertEqual(plan['fixed_bytes'], 64 * 48 + 128 * 63 + 64 * 12 * 6)
        self.assertGreaterEqual(plan['pattern_bytes'], 48 * 4 + 63 * 8)
        self.assertLessEqual(plan['estimated_peak_bytes'], plan['budget_bytes'])
        self.assertLessEqual(prod(plan['chunk_shape']), plan['max_patterns'])
        default = streaming._stream_transform_plan(self.data.shape, self.data.dtype, (6, 8))
        self.assertEqual(default['budget_bytes'], 256 * 1024 ** 2)

    def test_padding_scales_with_radial_rows_and_standard_polar_fits_default_budget(self):
        narrow = streaming._stream_transform_plan((2, 256, 1), np.float32, (512, 512))
        standard = streaming._stream_transform_plan((2, 256, 1608), np.float32, (512, 512))
        self.assertEqual(narrow['angular_padding_bytes'], standard['angular_padding_bytes'])
        self.assertEqual(standard['angular_padding_bytes'], 64 * 12 * 256)
        self.assertEqual(standard['source_workspace_bytes'], 64 * 256 * 1608)
        self.assertLessEqual(standard['estimated_peak_bytes'], standard['budget_bytes'])
        self.assertGreaterEqual(standard['max_patterns'], 1)

    def test_public_cartesian_wrapper_canvas_variants_match_memory(self):
        polar = self.data.to_polar(output_shape=(5, 16), progress=False)
        polar.save(self.source, overwrite=True)
        for canvas in (None, 'original', (7, 9)):
            for order in (0, 1):
                with self.subTest(canvas=canvas, order=order):
                    expected = polar.to_cartesian(
                        output_shape=canvas, order=order, fill_value=-3, progress=False,
                    )
                    result = fd.HyperData.to_cartesian_hdf5(
                        self.source, self.destination, output_shape=canvas, order=order,
                        fill_value=-3, working_memory_mb=0.04, progress=False, overwrite=True,
                    )
                    self.assertIsInstance(result, str)
                    self.assertEqual(result, str(self.destination))
                    actual = fd.HyperData(result)
                    np.testing.assert_array_equal(actual.array, expected.array)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assert_metadata_equal(actual, expected)

    def test_public_background_wrapper_accepts_array_and_reciprocalspace(self):
        background = np.arange(48, dtype=np.float64).reshape(6, 8) / 3
        for input_background in (background, fd.ReciprocalSpace(background)):
            expected = self.data.remove_bg(input_background, bg_frac=0.4, a_min=None, a_max=600)
            result = fd.HyperData.remove_bg_hdf5(
                self.source, self.destination, input_background, bg_frac=0.4,
                a_min=None, a_max=600, working_memory_mb=0.04, progress=False,
                compression=None, overwrite=True,
            )
            self.assertIsInstance(result, str)
            actual = fd.HyperData(result)
            np.testing.assert_array_equal(actual.array, expected.array)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assert_metadata_equal(actual, expected)
            with h5py.File(result, 'r') as file:
                self.assertIsNone(file['array'].compression)

    def test_public_background_wrapper_selects_generic_3d_dataset(self):
        values = self.values.reshape(35, 6, 8).astype(np.uint16)
        source = self.folder / 'generic.h5'
        with h5py.File(source, 'w') as file:
            file.create_dataset('scan/images', data=values)
            file.create_dataset('other', data=np.ones((2, 3, 4)))
        result = fd.HyperData.remove_bg_hdf5(
            source, self.folder / 'stack', np.full((6, 8), 1.5),
            hdf5_dataset='/scan/images', a_min=None, working_memory_mb=0.04,
            progress=False,
        )
        self.assertEqual(result, str(self.folder / 'stack.4denoise'))
        actual = fd.HyperData(result)
        expected = fd.HyperData(values).remove_bg(np.full((6, 8), 1.5), a_min=None)
        np.testing.assert_array_equal(actual.array, expected.array)
        self.assert_metadata_equal(actual, expected)

    def test_public_wrappers_reject_tiny_budget_before_any_transform(self):
        with patch.object(fd.HyperData, 'remove_bg') as transform:
            with self.assertRaisesRegex(ValueError, 'too small'):
                fd.HyperData.remove_bg_hdf5(
                    self.source, self.destination, np.zeros((6, 8)),
                    working_memory_mb=0.001, progress=False,
                )
            transform.assert_not_called()
        polar = self.data.to_polar(output_shape=(5, 16), progress=False)
        polar.save(self.source, overwrite=True)
        with patch.object(fd.HyperData, 'to_cartesian') as transform:
            with self.assertRaisesRegex(ValueError, 'too small'):
                fd.HyperData.to_cartesian_hdf5(
                    self.source, self.destination, output_shape=(100000, 100000),
                    working_memory_mb=1, progress=False,
                )
            transform.assert_not_called()
        self.assertFalse(self.destination.exists())
        self.assert_no_temporary()

    def test_public_wrappers_validate_input_and_preserve_destination_on_failure(self):
        with self.assertRaisesRegex(ValueError, 'polar metadata'):
            fd.HyperData.to_cartesian_hdf5(self.source, self.destination, progress=False)
        with self.assertRaisesRegex(ValueError, 'background must have shape'):
            fd.HyperData.remove_bg_hdf5(
                self.source, self.destination, np.ones(self.data.shape), progress=False,
            )
        self.destination.write_bytes(b'original destination')
        with patch.object(streaming, '_write_block', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                fd.HyperData.remove_bg_hdf5(
                    self.source, self.destination, np.zeros((6, 8)),
                    progress=False, overwrite=True, working_memory_mb=0.04,
                )
        self.assertEqual(self.destination.read_bytes(), b'original destination')
        self.assert_no_temporary()

    def test_insufficient_budget_rejected_before_probe_read_or_temporary_file(self):
        readers, datasets = [], []
        transform = Mock(side_effect=AssertionError('Probe must not run.'))
        with patch.object(streaming.tempfile, 'mkstemp') as temporary:
            for pattern_shape, budget in (((6, 8), 0.001), ((100000, 100000), 1)):
                with self.subTest(pattern_shape=pattern_shape), self.assertRaisesRegex(ValueError, 'too small'):
                    self.stream(
                        transform, output_pattern_shape=pattern_shape, working_memory_mb=budget,
                        reader_factory=self.tracked_factory(readers, datasets),
                    )
            temporary.assert_not_called()
        transform.assert_not_called()
        self.assertTrue(all(not dataset.reads for dataset in datasets))
        self.assertTrue(all(reader._closed for reader in readers))
        self.assertFalse(self.destination.exists())

    def test_invalid_budget_and_pattern_shape_rejected_before_opening_reader(self):
        reader = Mock(side_effect=AssertionError('Reader should not be opened.'))
        transform = Mock()
        for budget in (True, 0, -1, float('nan'), float('inf'), '1', 1e-10):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.stream(transform, reader_factory=reader, working_memory_mb=budget)
        for shape in ((True, 8), (6.5, 8), (0, 8), (6,), (6, 8, 1), 8):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'output_pattern_shape'):
                self.stream(transform, reader_factory=reader, output_pattern_shape=shape)
        reader.assert_not_called()
        transform.assert_not_called()

    def test_malformed_outputs_never_publish_a_destination(self):
        def object_output(block):
            result = block.copy()
            result.array = np.full(block.shape, 'bad', dtype=object)
            return result

        transforms = (
            (lambda block: block.array, TypeError, 'HyperData'),
            (lambda block: fd.HyperData(np.ones((2, 1, 6, 8))), ValueError, 'preserve scan shape'),
            (lambda block: fd.HyperData(np.ones((1, 1, 5, 8))), ValueError, 'return shape'),
            (object_output, TypeError, 'numeric'),
        )
        for transform, exception, message in transforms:
            readers, datasets = [], []
            with self.subTest(message=message), self.assertRaisesRegex(exception, message):
                self.stream(transform, reader_factory=self.tracked_factory(readers, datasets))
            self.assertTrue(readers[0]._closed)
            self.assertFalse(self.destination.exists())
            self.assert_no_temporary()

    def test_dtype_change_after_probe_is_rejected_without_silent_casting(self):
        calls = 0

        def transform(block):
            nonlocal calls
            calls += 1
            return block._spawn(block.array.astype(np.float32 if calls == 1 else np.float64))

        with self.assertRaisesRegex(ValueError, 'dtype changed'):
            self.stream(transform)
        self.assertEqual(calls, 2)
        self.assertFalse(self.destination.exists())
        self.assert_no_temporary()

    def test_atomic_transform_failure_preserves_existing_or_absent_destination(self):
        for existing in (False, True):
            calls = 0
            readers, datasets = [], []
            if existing:
                self.destination.write_bytes(b'original destination')

            def transform(block):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('injected transform failure')
                return block.remove_bg(np.zeros((6, 8)))

            with self.subTest(existing=existing), self.assertRaisesRegex(RuntimeError, 'injected'):
                self.stream(transform, overwrite=existing,
                            reader_factory=self.tracked_factory(readers, datasets))
            self.assertTrue(readers[0]._closed)
            self.assertEqual(calls, 2)
            if existing:
                self.assertEqual(self.destination.read_bytes(), b'original destination')
            else:
                self.assertFalse(self.destination.exists())
            self.assert_no_temporary()

    def test_atomic_serializer_write_and_publish_failures_clean_temporary_files(self):
        self.destination.write_bytes(b'original destination')
        failures = (
            ('_write_hdf5_value', 'metadata serialization failed'),
            ('_write_block', 'block write failed'),
        )
        for target, message in failures:
            with self.subTest(target=target), patch.object(
                streaming, target, side_effect=RuntimeError(message),
            ), self.assertRaisesRegex(RuntimeError, message):
                self.stream(lambda block: block, overwrite=True)
            self.assertEqual(self.destination.read_bytes(), b'original destination')
            self.assert_no_temporary()
        with patch.object(streaming.os, 'replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                self.stream(lambda block: block, overwrite=True)
        self.assertEqual(self.destination.read_bytes(), b'original destination')
        self.assert_no_temporary()

    def test_reader_closes_before_atomic_publication(self):
        readers, datasets = [], []
        link = os.link

        def checked_link(source, destination):
            self.assertTrue(readers[0]._closed)
            with h5py.File(source, 'r+') as file:
                np.testing.assert_array_equal(file['array'][...], self.values)
            link(source, destination)

        with patch.object(streaming.os, 'link', side_effect=checked_link):
            self.stream(lambda block: block, reader_factory=self.tracked_factory(readers, datasets))
        self.assert_no_temporary()

    def test_no_overwrite_publication_race_preserves_competing_destination(self):
        link = os.link

        def competing_link(source, destination):
            Path(destination).write_bytes(b'competing destination')
            link(source, destination)

        with patch.object(streaming.os, 'link', side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self.stream(lambda block: block)
        self.assertEqual(self.destination.read_bytes(), b'competing destination')
        self.assert_no_temporary()

    def test_source_identity_and_hardlinks_are_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'different files'):
            streaming._stream_transform(
                self.source, self.source, reader_factory=fd.HyperData.open_hdf5,
                transform=lambda block: block, output_pattern_shape=(6, 8), overwrite=True,
            )
        os.link(self.source, self.destination)
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite), self.assertRaisesRegex(ValueError, 'different files'):
                self.stream(lambda block: block, overwrite=overwrite)
        np.testing.assert_array_equal(fd.HyperData(self.source).array, self.values)

    def test_file_options_require_explicit_overwrite_and_support_nonatomic_mode(self):
        self.destination.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            self.stream(lambda block: block)
        self.assertEqual(self.destination.read_bytes(), b'original')
        for atomic in (False, True):
            result = self.stream(lambda block: block, overwrite=True, atomic=atomic,
                                 compression='lzf', compression_opts=None)
            np.testing.assert_array_equal(fd.HyperData(result).array, self.values)
            self.assert_no_temporary()
        for options in ({'overwrite': 1}, {'atomic': 'yes'}, {'progress': None}, {'compression': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.stream(lambda block: block, **options)
        with self.assertRaisesRegex(ValueError, 'file path'):
            streaming._stream_transform(
                self.source, self.folder / 'bad.txt', reader_factory=fd.HyperData.open_hdf5,
                transform=lambda block: block, output_pattern_shape=(6, 8),
            )

    def test_single_pattern_and_narrow_scan_shapes_have_no_missing_positions(self):
        for scan_shape in ((1,), (1, 1), (1, 9), (9, 1), (4, 5)):
            values = np.arange(prod(scan_shape) * 48, dtype=np.int16).reshape(scan_shape + (6, 8))
            fd.HyperData(values).save(self.source, overwrite=True)
            calls = []

            def transform(block):
                calls.append(block.scan_shape)
                return block

            result = self.stream(transform, overwrite=True)
            np.testing.assert_array_equal(fd.HyperData(result).array, values)
            self.assertEqual(sum(prod(shape) for shape in calls), prod(scan_shape))

    def test_nonempty_numeric_source_is_required_before_transform(self):
        transform = Mock()
        for shape, dtype in (((0, 6, 8), np.float32), ((3, 6, 8), 'S1')):
            with self.subTest(shape=shape, dtype=dtype):
                with h5py.File(self.source, 'w') as file:
                    file.attrs['fourdenoise_format'] = streaming._HYPERDATA_HDF5_FORMAT
                    file.attrs['format_version'] = streaming._HYPERDATA_HDF5_VERSION
                    file.create_dataset('array', shape=shape, dtype=dtype)
                with self.assertRaises((TypeError, ValueError)):
                    self.stream(transform)
                self.assertFalse(self.destination.exists())
        transform.assert_not_called()

    def test_cartesian_peak_arrays_and_read_sizes_are_budgeted_not_scan_sized(self):
        source_data = fd.HyperData(np.ones((16, 17, 32, 48), dtype=np.float32))
        polar = source_data.to_polar(output_shape=(16, 32), progress=False)
        polar.save(self.source, overwrite=True)
        readers, datasets, outputs = [], [], []

        def transform(block):
            result = block.to_cartesian(output_shape=(32, 48), progress=False)
            outputs.append(weakref.ref(result.array))
            return result

        plan = streaming._stream_transform_plan(
            polar.shape, polar.dtype, (32, 48), 0.75, output_dtype=np.float32,
        )
        tracemalloc.start()
        try:
            result = self.stream(
                transform, output_pattern_shape=(32, 48), working_memory_mb=0.75,
                reader_factory=self.tracked_factory(readers, datasets, outputs),
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, plan['budget_bytes'] + 128 * 1024)
        self.assertLess(max(size for _, size in datasets[0].reads), polar.array.nbytes)
        for selection, _ in datasets[0].reads:
            count = prod(item.stop - item.start for item in selection[:-2])
            self.assertLessEqual(plan['fixed_bytes'] + count * plan['pattern_bytes'], plan['budget_bytes'])
        self.assertTrue(readers[0]._closed)
        np.testing.assert_array_equal(
            fd.HyperData(result).array,
            polar.to_cartesian(output_shape=(32, 48), progress=False).array,
        )

    def test_one_outer_progress_bar_and_no_runtime_main_import(self):
        bar = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=bar)
        context.__exit__ = Mock(return_value=False)
        with patch.object(streaming, 'tqdm', return_value=context) as progress:
            self.stream(lambda block: block)
        progress.assert_called_once_with(total=35, desc='Diffraction patterns', disable=True)
        self.assertEqual(sum(call.args[0] for call in bar.update.call_args_list), 35)
        check = subprocess.run(
            [sys.executable, '-B', '-c',
             "import sys; import fourdenoise_streaming; assert 'fourdenoise' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == '__main__':
    unittest.main()
