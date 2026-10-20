"""Exercise the actual file-based tutorial cells without full-source loads."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np

import fourdenoise as fd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'tutorials/07_large_data_processing.ipynb'


class LargeDataTutorialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='4d-tutorial-07-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.path = self.root / 'input.4denoise'
        self.plain_path = self.root / 'input.npy'
        self.centers = np.array([[8, 16], [12, 23], [20, 23], [24, 16], [20, 9], [12, 9]])
        yy, xx = np.ogrid[:32, :32]
        pattern = np.full((32, 32), 3.0)
        for cy, cx in self.centers:
            pattern += 80 * np.exp(-((yy - cy)**2 + (xx - cx)**2) / 2)
        self.values = np.broadcast_to(pattern.astype(np.uint16), (2, 3, 32, 32)).copy()
        data = fd.HyperData(self.values, real_units='nm', real_conv_factor=(4, 6),
                            real_origin=(20, 30), reciprocal_units='mrad', reciprocal_conv_factor=0.05,
                            center_beam_metadata={'center_px': [15.4, 16.1], 'radius_px': 3, 'shape': (32, 32)})
        data.save(self.path)
        np.save(self.plain_path, self.values)
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        figures = set(plt.get_fignums())
        self.addCleanup(lambda: [plt.close(number) for number in set(plt.get_fignums()) - figures])

    def run_cell(self, namespace, cell_id, **overrides):
        tree = ast.parse(self.cells[cell_id])
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in overrides for target in node.targets)
        )]
        namespace.update(overrides)
        with redirect_stdout(StringIO()), patch.object(plt, 'show'):
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)

    def select(self, **overrides):
        namespace = dict(Path=Path, np=np, HyperData=fd.HyperData, RealSpace=fd.RealSpace,
                         h5py=h5py, plt=plt, perf_counter=perf_counter, shutil=shutil,
                         output_dir=self.output, timings={}, written_paths={})
        configuration = dict(microscope_path=self.path, analysis_annulus_px=(4, 12))
        configuration.update(overrides)
        self.run_cell(namespace, 't07-input-selection', **configuration)
        return namespace

    def prepare(self, **overrides):
        namespace = self.select(**overrides)
        self.run_cell(namespace, 't07-source-inspection')
        self.run_cell(namespace, 't07-analysis-settings')
        return namespace

    def test_file_input_does_not_invent_calibration_or_enable_expensive_stages(self):
        namespace = self.prepare(microscope_path=self.plain_path)
        self.assertFalse(namespace['is_synthetic'])
        self.assertFalse(namespace['run_peak_detection'])
        self.assertFalse(namespace['run_transforms'])
        metadata = namespace['source_info']['metadata']
        self.assertIsNone(metadata.get('real_units'))
        self.assertIsNone(metadata.get('reciprocal_units'))
        self.assertIsNone(metadata.get('center_beam_metadata'))
        self.assertIsNone(metadata.get('rq_calibration'))
        self.assertEqual(list(self.output.iterdir()), [])

    def test_native_metadata_and_beam_center_are_retained(self):
        namespace = self.prepare()
        metadata = namespace['source_info']['metadata']
        self.assertEqual(metadata['real_units'], 'nm')
        np.testing.assert_allclose(metadata['real_conv_factor'], (4, 6))
        np.testing.assert_allclose(namespace['transform_center_px'], (15.4, 16.1))
        self.assertEqual(namespace['center_source'], 'current beam center_px metadata')
        self.assertTrue(namespace['source'].closed)

    def test_explicit_generic_file_calibration_does_not_change_input_file(self):
        before = self.plain_path.read_bytes()
        namespace = self.prepare(microscope_path=self.plain_path, input_origin='simulation',
                                  microscope_calibration={'real_units': 'nm', 'real_conv_factor': (7, 9),
                                                          'reciprocal_units': 'mrad', 'reciprocal_conv_factor': 0.2})
        self.assertEqual(namespace['source_info']['metadata']['real_units'], 'nm')
        self.assertEqual(namespace['source_info']['metadata']['reciprocal_conv_factor'], 0.2)
        self.assertEqual(self.plain_path.read_bytes(), before)

    def test_in_memory_inputs_conflicts_and_invalid_budgets_are_rejected(self):
        invalid = (
            {'input_data': self.values}, {'input_origin': 'guessed'}, {'microscope_backend': 'eager'},
            {'working_memory_mb': 0}, {'working_memory_mb': np.nan}, {'working_memory_mb': True},
            {'working_memory_mb': '16'},
            {'max_scan_map_mb': -1}, {'run_transforms': 'yes'},
            {'microscope_options': {'backend': 'native'}},
            {'microscope_options': {'real_units': 'nm'}, 'microscope_calibration': {'real_units': 'nm'}},
        )
        for configuration in invalid:
            with self.subTest(fields=tuple(configuration)), self.assertRaises(ValueError):
                self.select(**configuration)
        with self.assertRaises(FileNotFoundError):
            self.select(microscope_path=self.root / 'missing.npy')

    def test_external_analysis_settings_must_be_explicit_and_valid(self):
        invalid = (
            {'analysis_annulus_px': None}, {'analysis_annulus_px': (12, 4)},
            {'transform_center_px': (np.nan, 3)}, {'transform_center_px': (-1, 3)},
            {'scan_mask_input': np.ones((2, 3), dtype=int)}, {'scan_mask_input': np.ones((3, 2), dtype=bool)},
            {'run_peak_detection': True},
            {'run_peak_detection': True, 'peak_options_input': {'radius': 1, 'min_distance': 3}},
            {'run_peak_detection': True, 'scan_mask_input': np.zeros((2, 3), dtype=bool)},
            {'run_peak_detection': True, 'peak_options_input': {'radius': 1, 'min_distance': 3, 'threshold_abs': None}},
            {'run_peak_detection': True, 'peak_options_input': {'radius': 1, 'min_distance': 3, 'threshold_abs': 1, 'destination': 'other.h5'}},
        )
        for configuration in invalid:
            with self.subTest(fields=tuple(configuration)), self.assertRaises(ValueError):
                self.prepare(**configuration)

    def test_scan_map_guard_and_dimension_contract_run_before_pixel_operations(self):
        namespace = self.select(max_scan_map_mb=0.000001)
        with patch.object(fd.DiffractionSource, 'get_dp', side_effect=AssertionError('No pattern reads')):
            with self.assertRaises(MemoryError):
                self.run_cell(namespace, 't07-source-inspection')
        self.assertTrue(namespace['source'].closed)
        for values in (np.ones((3, 32, 32)), np.ones((2, 3, 32, 32), dtype=complex)):
            np.save(self.plain_path, values)
            with self.subTest(shape=values.shape, dtype=values.dtype):
                namespace = self.select(microscope_path=self.plain_path)
                with self.assertRaises(ValueError):
                    self.run_cell(namespace, 't07-source-inspection')

    def test_reductions_and_custom_chunks_never_load_a_full_source(self):
        before = self.path.read_bytes()
        namespace = self.prepare(chunk_shape=(1, 2))
        with patch.object(fd.DiffractionSource, 'load', side_effect=AssertionError('No full load')):
            self.run_cell(namespace, 't07-source-views')
            self.run_cell(namespace, 't07-chunk-reductions')
        expected = np.einsum('...ij,ij->...', self.values, namespace['detector_mask'], dtype=np.float64)
        np.testing.assert_allclose(namespace['virtual'].array, expected)
        np.testing.assert_array_equal(namespace['visited'], 1)
        self.assertLess(namespace['largest_block_bytes'], self.values.nbytes)
        self.assertEqual(self.path.read_bytes(), before)

    def test_incremental_peaks_use_selected_numeric_slices_not_full_collection_load(self):
        mask = np.ones((2, 3), dtype=bool)
        mask[0, 0] = False
        namespace = self.prepare(run_peak_detection=True, scan_mask_input=mask,
                                  peak_options_input={'radius': 1.5, 'min_distance': 3,
                                                      'threshold_abs': 40, 'r_range': (4, 12),
                                                      'intensity_radius': 2})
        with patch.object(fd.DiffractionSource, 'load', side_effect=AssertionError('No full diffraction load')):
            with patch.object(fd.PeakScanResult, 'load', side_effect=AssertionError('No full peak load')):
                self.run_cell(namespace, 't07-source-views')
                self.run_cell(namespace, 't07-peak-file')
                self.run_cell(namespace, 't07-selective-peaks')
        np.testing.assert_array_equal(namespace['processed_mask'], mask)
        self.assertEqual(namespace['peak_counts'][0, 0], 0)
        self.assertGreater(len(namespace['selected_coords']), 0)
        self.assertEqual(namespace['selected_intensities'].shape, (len(namespace['selected_coords']),))
        with h5py.File(namespace['peak_path'], 'r') as file:
            self.assertEqual(file['coords'].shape[1], 2)

    def test_disk_transforms_match_single_pattern_calls_and_update_scan_bin_origin(self):
        namespace = self.prepare(run_transforms=True)
        with patch.object(fd.DiffractionSource, 'load', side_effect=AssertionError('No full load')):
            self.run_cell(namespace, 't07-source-views')
            self.run_cell(namespace, 't07-output-plan')
            self.run_cell(namespace, 't07-disk-resize')
            self.run_cell(namespace, 't07-disk-polar')
        self.assertEqual(set(namespace['written_paths']), {'detector_resized', 'scan_resized', 'polar', 'cartesian'})
        with fd.HyperData.open(namespace['scan_resized_path']) as source:
            np.testing.assert_allclose(source.real_origin, (22, 36))
            self.assertEqual(source.scan_shape, (1, 1))
        self.assertTrue(namespace['source'].closed)


if __name__ == '__main__':
    unittest.main()
