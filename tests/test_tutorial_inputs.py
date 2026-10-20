"""Exercise the actual notebook input cells without starting Jupyter kernels."""

import ast
import hashlib
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from time import perf_counter
import unittest

import numpy as np

from fourdenoise import HyperData
from scripts.execute_tutorials import TUTORIALS


ROOT = Path(__file__).resolve().parents[1]
RESULT_NAMES = {
    '01_loading_and_calibration.ipynb': 'selected_data',
    '02_visualization_and_virtual_detectors.ipynb': 'selected_data',
    '03_alignment_and_coordinate_transforms.ipynb': 'inspection_data',
    '04_denoising_and_unfolding.ipynb': 'data',
    '05_peak_analysis_and_braggness.ipynb': 'data',
    '06_strain_mapping_and_rq_calibration.ipynb': 'data',
}


def execute_input_cell(name, **overrides):
    """Override configuration assignments, then execute the unchanged branch logic."""
    notebook = json.loads((ROOT / 'tutorials' / name).read_text(encoding='utf-8'))
    cells = [cell for cell in notebook['cells']
             if 'microscope-input' in cell['metadata'].get('tags', [])]
    if len(cells) != 1:
        raise AssertionError('Expected one configurable input cell.')
    tree = ast.parse(''.join(cells[0]['source']))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in overrides for target in node.targets)
    )]
    fixture = HyperData(np.full((2, 3, 8, 8), 17, dtype=np.float32),
                       real_units='nm', real_conv_factor=(9, 11),
                       reciprocal_units='mrad', reciprocal_conv_factor=0.75)
    namespace = {
        'HyperData': HyperData, 'np': np, 'Path': Path,
        'data': fixture, 'restored': fixture, 'synthetic_data': fixture,
        'clean_array': np.full(fixture.shape, 17, dtype=np.float32),
        'annulus_px': (1, 3), **overrides,
    }
    with redirect_stdout(StringIO()):
        exec(compile(tree, str(ROOT / 'tutorials' / name), 'exec'), namespace)
    return namespace, namespace[RESULT_NAMES[name]]


class TutorialInputTests(unittest.TestCase):
    def test_all_registered_lessons_have_a_checked_input_contract(self):
        # Tutorial 7 deliberately uses a file-only contract, checked separately.
        self.assertEqual(set(TUTORIALS), set(RESULT_NAMES) | {'07_large_data_processing.ipynb'})

    def test_in_memory_arrays_do_not_inherit_synthetic_calibration(self):
        array = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
        for name in RESULT_NAMES:
            for origin in ('experimental', 'simulation'):
                with self.subTest(notebook=name, origin=origin):
                    namespace, selected = execute_input_cell(name, input_data=array, input_origin=origin)
                    np.testing.assert_array_equal(selected.array, array)
                    self.assertFalse(np.shares_memory(selected.array, array))
                    self.assertIsNone(selected.real_units)
                    self.assertIsNone(selected.reciprocal_units)
                    self.assertIsNone(selected.center_beam_metadata)
                    self.assertIsNone(selected.rq_calibration)
                    if name.startswith('02'):
                        self.assertIsNone(namespace['selected_annulus_px'])
                    if name.startswith('04'):
                        self.assertIsNone(namespace['reference_array'])
                        self.assertFalse(namespace['run_tensorly_demo'])
                    if name.startswith('05'):
                        self.assertIsNone(namespace['reference_peaks'])
                        self.assertIsNone(namespace['analysis_annulus_px'])
                        self.assertIsNone(namespace['alignment_radius_px'])
                        self.assertFalse(namespace['run_alignment_example'])
                    if name.startswith('06'):
                        self.assertIsNone(namespace['measured_peaks_input'])
                        self.assertIsNone(namespace['reference_peaks_input'])
                        self.assertIsNone(namespace['intensity_input'])
                        self.assertIsNone(namespace['verified_rq_angle_deg'])
                        self.assertFalse(namespace['rq_verified'])

    def test_hyperdata_inputs_keep_their_own_metadata_and_own_their_pixels(self):
        source = HyperData(np.ones((2, 3, 8, 8), dtype=np.float32),
                           real_units='angstrom', real_conv_factor=(4, 6), real_origin=(20, 30),
                           reciprocal_units='inverse_angstrom', reciprocal_conv_factor=0.02,
                           acquisition_metadata={'source': 'external-simulation'})
        for name in RESULT_NAMES:
            with self.subTest(notebook=name):
                _, selected = execute_input_cell(name, input_data=source, input_origin='simulation')
                self.assertIsNot(selected, source)
                self.assertFalse(np.shares_memory(selected.array, source.array))
                self.assertEqual(selected.real_units, source.real_units)
                np.testing.assert_allclose(selected.real_conv_factor, source.real_conv_factor)
                np.testing.assert_allclose(selected.real_origin, source.real_origin)
                self.assertEqual(selected.reciprocal_conv_factor, source.reciprocal_conv_factor)
                self.assertEqual(selected.acquisition_metadata, source.acquisition_metadata)

    def test_native_files_load_without_invented_calibration(self):
        array = np.full((2, 3, 8, 8), 5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'simulation.npy'
            np.save(path, array)
            for name in RESULT_NAMES:
                with self.subTest(notebook=name):
                    _, selected = execute_input_cell(name, microscope_path=path, input_origin='simulation')
                    np.testing.assert_array_equal(selected.array, array)
                    self.assertIsNone(selected.real_units)
                    self.assertIsNone(selected.reciprocal_units)

    def test_calibration_is_only_applied_when_explicitly_supplied_to_bare_arrays(self):
        array = np.ones((2, 3, 8, 8), dtype=np.float32)
        calibration = {'real_units': 'nm', 'real_conv_factor': (1, 2),
                       'reciprocal_units': 'mrad', 'reciprocal_conv_factor': 0.1}
        for name in RESULT_NAMES:
            with self.subTest(notebook=name):
                _, selected = execute_input_cell(name, input_data=array, microscope_calibration=calibration)
                self.assertEqual(selected.real_units, 'nm')
                np.testing.assert_allclose(selected.real_conv_factor, (1, 2))
                self.assertEqual(selected.reciprocal_conv_factor, 0.1)

    def test_conflicting_inputs_overrides_and_unknown_origins_are_rejected(self):
        source = HyperData(np.ones((2, 3, 8, 8), dtype=np.float32))
        configurations = (
            {'input_data': source, 'microscope_path': 'unused.npy'},
            {'input_data': source, 'microscope_options': {'hdf5_dataset': '/data'}},
            {'input_data': source, 'microscope_calibration': {'real_units': 'nm'}},
            {'input_data': source, 'input_origin': 'guessed'},
        )
        for name in RESULT_NAMES:
            for configuration in configurations:
                with self.subTest(notebook=name, configuration=tuple(configuration)):
                    with self.assertRaises(ValueError):
                        execute_input_cell(name, **configuration)

    def test_denoising_reference_is_explicit_for_external_data(self):
        name = '04_denoising_and_unfolding.ipynb'
        array = np.full((2, 3, 8, 8), 5, dtype=np.float32)
        reference = np.full_like(array, 4)
        namespace, _ = execute_input_cell(name, input_data=array, reference_input=reference)
        self.assertIs(namespace['reference_array'], reference)
        self.assertFalse(namespace['is_synthetic'])
        for invalid in (np.ones((2, 3)), np.full_like(array, np.nan), array.astype(complex)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    execute_input_cell(name, input_data=array, reference_input=invalid)

    @staticmethod
    def _execute_strain_cell(namespace, cell_id):
        name = '06_strain_mapping_and_rq_calibration.ipynb'
        notebook = json.loads((ROOT / 'tutorials' / name).read_text(encoding='utf-8'))
        cell = next(cell for cell in notebook['cells'] if cell['id'] == cell_id)
        namespace.update(hashlib=hashlib, perf_counter=perf_counter)
        with redirect_stdout(StringIO()):
            exec(compile(''.join(cell['source']), str(ROOT / 'tutorials' / name), 'exec'), namespace)

    def _strain_fixture(self, **overrides):
        reference = np.array([[1.0, 1.0], [1.0, 6.0], [6.0, 1.0], [6.0, 6.0]])
        source = HyperData(np.ones((2, 3, 8, 8), dtype=np.float32),
                           real_units='nm', real_conv_factor=(4, 6), real_origin=(20, 30),
                           reciprocal_units='mrad', reciprocal_conv_factor=0.05,
                           center_beam_metadata={'center_px': [3.5, 3.5], 'shape': (8, 8)})
        values = dict(input_data=source, measured_peaks_input=np.broadcast_to(reference, (2, 3, 4, 2)),
                      reference_peaks_input=reference)
        values.update(overrides)
        namespace, selected = execute_input_cell('06_strain_mapping_and_rq_calibration.ipynb', **values)
        return namespace, selected

    def test_strain_lesson_requires_external_reference_and_measured_peaks(self):
        namespace, _ = execute_input_cell('06_strain_mapping_and_rq_calibration.ipynb',
                                          input_data=np.ones((2, 3, 8, 8)))
        with self.assertRaisesRegex(ValueError, 'measured_peaks_input and reference_peaks_input'):
            self._execute_strain_cell(namespace, 't06-analysis-settings')

    def test_strain_lesson_rejects_non_cartesian_or_oversized_tutorial_arrays(self):
        name = '06_strain_mapping_and_rq_calibration.ipynb'
        for array in (np.ones((2, 8, 8)), np.ones((2, 3, 8, 8), dtype=complex),
                      np.ones((2, 3, 8, 8), dtype=bool), np.ones((1, 3, 8, 8))):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                with self.assertRaises(ValueError):
                    execute_input_cell(name, input_data=array)
        with self.assertRaises(MemoryError):
            execute_input_cell(name, input_data=np.ones((2, 3, 8, 8)), max_tutorial_input_mb=0)

    def test_strain_lesson_global_and_local_references_use_supplied_beam_metadata(self):
        for local_reference in (False, True):
            with self.subTest(local_reference=local_reference):
                namespace, selected = self._strain_fixture()
                if local_reference:
                    namespace['reference_peaks_input'] = namespace['measured_peaks_input']
                self._execute_strain_cell(namespace, 't06-analysis-settings')
                self._execute_strain_cell(namespace, 't06-detector-fit')
                result = namespace['detector_result']
                np.testing.assert_allclose(result.exx, 0, atol=1e-12)
                np.testing.assert_allclose(result.peak_origin, [3.5, 3.5])
                self.assertEqual(result.peak_origin_source, 'center_beam_metadata.center_px')
                self.assertTrue(result.valid_mask.all())
                self.assertTrue(namespace['scan_mask'].all())
                self.assertEqual(result.real_origin, (20, 30))
                np.testing.assert_array_equal(selected.array, 1)

    def test_strain_lesson_real_output_requires_explicit_verification(self):
        namespace, _ = self._strain_fixture()
        namespace['data'].set_rq_calibration(23.0, 'x')
        self._execute_strain_cell(namespace, 't06-analysis-settings')
        self._execute_strain_cell(namespace, 't06-rq-settings')
        self.assertFalse(namespace['use_real_frame'])
        self.assertEqual(namespace['data'].rq_calibration.rotation_deg, 23)
        namespace['rq_verified'] = True
        self._execute_strain_cell(namespace, 't06-rq-settings')
        self.assertTrue(namespace['use_real_frame'])
        namespace['data'].rq_calibration = None
        with self.assertRaisesRegex(ValueError, 'requires an existing or explicitly supplied'):
            self._execute_strain_cell(namespace, 't06-rq-settings')

    def test_strain_lesson_explicit_rq_values_and_real_fit_keep_pixels_unchanged(self):
        namespace, selected = self._strain_fixture(rq_verified=True, verified_rq_angle_deg=41,
                                                   verified_rq_mirror_axis='y')
        before = selected.array.copy()
        self._execute_strain_cell(namespace, 't06-analysis-settings')
        self._execute_strain_cell(namespace, 't06-detector-fit')
        self._execute_strain_cell(namespace, 't06-rq-settings')
        self._execute_strain_cell(namespace, 't06-real-fit')
        self.assertEqual(namespace['result'].output_frame, 'real')
        self.assertEqual(namespace['result'].rq_calibration.mirror_axis, 'y')
        self.assertEqual(namespace['result'].rq_calibration.rotation_deg, 41)
        np.testing.assert_allclose(namespace['result'].exx, 0, atol=1e-12)
        np.testing.assert_array_equal(selected.array, before)

    def test_strain_lesson_rejects_bad_peak_weight_mask_and_calibration_inputs(self):
        invalid = (
            {'measured_peaks_input': np.ones((2, 3, 4, 3))},
            {'reference_peaks_input': np.full((4, 2), np.nan)},
            {'intensity_input': np.ones((2, 3, 3))},
            {'intensity_input': -np.ones((2, 3, 4))},
            {'scan_mask_input': np.zeros((2, 3), dtype=bool)},
            {'scan_mask_input': np.ones((2, 3), dtype=int)},
            {'peak_units_input': 'xy'},
            {'verified_rq_angle_deg': 45, 'rq_verified': False},
            {'verified_rq_mirror_axis': 'x'},
        )
        for configuration in invalid:
            with self.subTest(fields=tuple(configuration)):
                namespace, _ = self._strain_fixture(**configuration)
                with self.assertRaises(ValueError):
                    self._execute_strain_cell(namespace, 't06-analysis-settings')
        namespace, selected = self._strain_fixture()
        selected.center_beam_metadata = None
        with self.assertRaisesRegex(ValueError, 'strain_center_input or verified center_beam_metadata'):
            self._execute_strain_cell(namespace, 't06-analysis-settings')
        namespace, selected = self._strain_fixture(peak_units_input='calibrated')
        selected.reciprocal_units = None
        with self.assertRaisesRegex(ValueError, 'verified reciprocal unit/scale'):
            self._execute_strain_cell(namespace, 't06-analysis-settings')

    def test_strain_lesson_missing_peaks_remain_invalid_and_weights_follow_measurements(self):
        namespace, _ = self._strain_fixture(intensity_input=np.ones((2, 3, 4)))
        namespace['measured_peaks_input'] = [[points.copy() for points in row]
                                             for row in namespace['measured_peaks_input']]
        namespace['intensity_input'] = [[values.copy() for values in row]
                                        for row in namespace['intensity_input']]
        namespace['measured_peaks_input'][0][1] = np.empty((0, 2))
        namespace['intensity_input'][0][1] = np.empty(0)
        namespace['measured_peaks_input'][1][1] = np.vstack((namespace['measured_peaks_input'][1][1], [30, 30]))
        namespace['intensity_input'][1][1] = np.r_[np.ones(4), 1000]
        self._execute_strain_cell(namespace, 't06-analysis-settings')
        self._execute_strain_cell(namespace, 't06-detector-fit')
        result = namespace['detector_result']
        self.assertFalse(result.valid_mask[0, 1])
        self.assertTrue(np.isnan(result.exx[0, 1]))
        self.assertEqual(result.match_counts[1, 1], 4)
        np.testing.assert_allclose(result.exx[result.valid_mask], 0, atol=1e-12)

    def test_denoising_input_guards_and_explicit_tensorly_opt_in(self):
        name = '04_denoising_and_unfolding.ipynb'
        for array in (np.ones((3, 8, 8)), np.full((2, 3, 8, 8), np.nan),
                      np.ones((2, 3, 8, 8), dtype=complex), np.ones((1, 3, 8, 8)),
                      np.ones((2, 3, 8, 8), dtype=bool)):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                with self.assertRaises(ValueError):
                    execute_input_cell(name, input_data=array)
        array = np.ones((2, 3, 8, 8), dtype=np.float32)
        with self.assertRaises(MemoryError):
            execute_input_cell(name, input_data=array, max_tutorial_input_mb=0)
        namespace, _ = execute_input_cell(name, input_data=array, run_tensorly_demo=True)
        self.assertTrue(namespace['run_tensorly_demo'])

    def test_peak_lesson_requires_an_inspected_annulus_for_external_inputs(self):
        name = '05_peak_analysis_and_braggness.ipynb'
        array = np.full((2, 3, 8, 8), 5, dtype=np.float32)
        namespace, _ = execute_input_cell(name, input_data=array)
        with self.assertRaisesRegex(ValueError, 'explicitly set analysis_annulus_px'):
            self._execute_peak_settings(name, namespace)

    def test_peak_lesson_distinguishes_skipped_nonfinite_from_evaluated_patterns(self):
        name = '05_peak_analysis_and_braggness.ipynb'
        array = np.full((2, 3, 8, 8), 5, dtype=np.float32)
        array[0, 1] = np.nan
        mask = np.ones((2, 3), dtype=bool)
        mask[0, 1] = False
        namespace, _ = execute_input_cell(name, input_data=array, analysis_annulus_px=(1, 3),
                                        scan_mask_input=mask)
        self._execute_peak_settings(name, namespace)
        np.testing.assert_array_equal(namespace['scan_mask'], mask)
        self.assertIsNone(namespace['reference_peaks'])
        namespace, _ = execute_input_cell(name, input_data=array, analysis_annulus_px=(1, 3))
        with self.assertRaisesRegex(ValueError, 'finite nonnegative evaluated patterns'):
            self._execute_peak_settings(name, namespace)

    def test_peak_lesson_rejects_bad_masks_references_and_negative_intensities(self):
        name = '05_peak_analysis_and_braggness.ipynb'
        array = np.full((2, 3, 8, 8), 5, dtype=np.float32)
        invalid = (
            {'scan_mask_input': np.zeros((2, 3), dtype=bool)},
            {'scan_mask_input': np.ones((2, 3), dtype=int)},
            {'scan_mask_input': np.ones((1, 3), dtype=bool)},
            {'reference_peaks_input': [[np.ones((2, 3))] * 3] * 2},
            {'reference_peaks_input': [[np.full((1, 2), np.nan)] * 3] * 2},
        )
        for overrides in invalid:
            with self.subTest(fields=tuple(overrides)):
                namespace, _ = execute_input_cell(name, input_data=array, analysis_annulus_px=(1, 3), **overrides)
                with self.assertRaises(ValueError):
                    self._execute_peak_settings(name, namespace)
        namespace, _ = execute_input_cell(name, input_data=-array, analysis_annulus_px=(1, 3))
        with self.assertRaisesRegex(ValueError, 'finite nonnegative evaluated patterns'):
            self._execute_peak_settings(name, namespace)

    def test_peak_lesson_preserves_explicit_ragged_reference_coordinates(self):
        name = '05_peak_analysis_and_braggness.ipynb'
        references = [[np.empty((0, 2)), np.array([[2.0, 3.0]]), np.array([[2.0, 3.0], [4.0, 5.0]])],
                      [np.array([[3.0, 4.0]]), np.empty((0, 2)), np.empty((0, 2))]]
        namespace, _ = execute_input_cell(name, input_data=np.ones((2, 3, 8, 8)),
                                        reference_peaks_input=references, analysis_annulus_px=(1, 3))
        self._execute_peak_settings(name, namespace)
        self.assertEqual([[len(a) for a in row] for row in namespace['reference_peaks']], [[0, 1, 2], [1, 0, 0]])
        self.assertFalse(namespace['run_alignment_example'])

    def test_peak_lesson_braggness_cell_uses_the_supplied_beam_center(self):
        name = '05_peak_analysis_and_braggness.ipynb'
        source = HyperData(np.ones((2, 3, 8, 8), dtype=np.float32),
                           real_units='nm', real_conv_factor=(4, 6),
                           reciprocal_units='mrad', reciprocal_conv_factor=0.125,
                           center_beam_metadata={'center_px': (3.2, 4.1),
                                                 'radius_px': 1.25, 'shape': (8, 8)})
        namespace, selected = execute_input_cell(name, input_data=source,
                                               analysis_annulus_px=(1, 3), bragg_top_pixels=5)
        self.assertEqual(selected.center_beam_metadata, source.center_beam_metadata)
        self.assertIsNot(selected.center_beam_metadata, source.center_beam_metadata)
        self._execute_peak_settings(name, namespace)
        notebook = json.loads((ROOT / 'tutorials' / name).read_text(encoding='utf-8'))
        cell = next(cell for cell in notebook['cells'] if cell['id'] == 't05-braggness')
        with redirect_stdout(StringIO()):
            exec(compile(''.join(cell['source']), str(ROOT / 'tutorials' / name), 'exec'), namespace)
        result = namespace['braggness']
        self.assertEqual(result.metadata['center_source'], 'center_beam_metadata.center_px')
        np.testing.assert_allclose(result.metadata['center'], (3.2, 4.1))

    def _execute_peak_settings(self, name, namespace):
        notebook = json.loads((ROOT / 'tutorials' / name).read_text(encoding='utf-8'))
        cell = next(cell for cell in notebook['cells'] if cell['id'] == 't05-analysis-settings')
        with redirect_stdout(StringIO()):
            exec(compile(''.join(cell['source']), str(ROOT / 'tutorials' / name), 'exec'), namespace)


if __name__ == '__main__':
    unittest.main()
