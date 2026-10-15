"""Regression tests for structured HyperData strain results."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class StrainResultTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([
            [-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0],
            [2.0, 1.0], [3.0, 2.0],
        ])
        self.data = fd.HyperData(
            np.zeros((2, 2, 8, 8)),
            real_units='nm', real_conv_factor=(0.5, 0.75),
            real_origin=(10.0, 20.0),
        )
        self.measured = np.broadcast_to(
            self.reference, (2, 2) + self.reference.shape,
        ).copy()

    def test_result_names_quality_and_calibrated_maps(self):
        result = self.data.get_strains(
            centers=self.measured, ref_centers=self.reference,
            center=(0, 0), match_peaks='ordered',
        )

        self.assertIsInstance(result, fd.StrainResult)
        self.assertEqual(result.exx.shape, (2, 2))
        np.testing.assert_allclose(result.exx, 0, atol=1e-12)
        np.testing.assert_allclose(result.eyy, 0, atol=1e-12)
        np.testing.assert_allclose(result.exy, 0, atol=1e-12)
        np.testing.assert_allclose(result.erot, 0, atol=1e-12)
        np.testing.assert_allclose(result.fit_rmse, 0, atol=1e-12)
        np.testing.assert_allclose(result.relative_fit_rmse, 0, atol=1e-12)
        np.testing.assert_array_equal(result.match_counts, 5)
        self.assertTrue(result.valid_mask.all())
        self.assertEqual(result.peak_origin, (0.0, 0.0))
        self.assertEqual(result.peak_origin_source, 'user')
        self.assertEqual(result.real_units, 'nm')
        self.assertEqual(result.real_conv_factor, (0.5, 0.75))
        self.assertEqual(result.real_origin, (10.0, 20.0))

        image = result.as_real_space('exx')
        self.assertIsInstance(image, fd.RealSpace)
        self.assertIs(image.array, result.exx)
        self.assertEqual(image.units, 'nm')
        self.assertEqual(image.conv_factor, (0.5, 0.75))
        self.assertEqual(image.origin, (10.0, 20.0))
        self.assertEqual(result.as_real_space('erot').value_units, 'rad')
        self.assertEqual(result.as_real_space('match_counts').value_units, 'peaks')

        exx, eyy, exy, erot = result
        self.assertIs(exx, result.exx)
        self.assertIs(eyy, result.eyy)
        self.assertIs(exy, result.exy)
        self.assertIs(erot, result.erot)
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], result.exx)

    def test_normal_components_follow_yx_peaks_without_transposing_the_scan(self):
        measured = self.measured.copy()
        measured[0, 1, :, 1] *= 1.02  # Reciprocal x expansion at one scan pixel.
        measured[1, 0, :, 0] *= 1.03  # Reciprocal y expansion at another pixel.
        result = self.data.get_strains(
            centers=measured, ref_centers=self.reference, center=(0, 0),
            match_peaks='ordered',
        )
        np.testing.assert_allclose(result.exx, [[0, -0.02], [0, 0]], atol=1e-12)
        np.testing.assert_allclose(result.eyy, [[0, 0], [-0.03, 0]], atol=1e-12)
        np.testing.assert_allclose(result.exy, 0, atol=1e-12)
        np.testing.assert_allclose(result.erot, 0, atol=1e-12)
        np.testing.assert_array_equal(result.as_real_space('exx').array, result.exx)
        self.assertEqual(result.basis_angle_deg, 0)

    def test_basis_rotation_matches_tensor_rotation_and_preserves_the_fit(self):
        transform = np.array([[1.01, 0.015], [0.015, 0.98]])
        measured = self.measured @ transform.T
        options = dict(
            centers=measured, ref_centers=self.reference, center=(0, 0),
            match_peaks='ordered', return_transform=True,
        )
        original = self.data.get_strains(**options)
        for angle in (30, 90, 135):
            with self.subTest(angle=angle):
                rotated = self.data.get_strains(**options, ang=angle)
                expected = fd.rotate_strain(
                    original.exx, original.eyy, original.exy, np.radians(angle),
                )
                for name, values in zip(('exx', 'eyy', 'exy'), expected):
                    np.testing.assert_allclose(getattr(rotated, name), values, atol=1e-12)
                np.testing.assert_allclose(rotated.erot, original.erot, atol=1e-12)
                np.testing.assert_array_equal(rotated.fit_rmse, original.fit_rmse)
                np.testing.assert_array_equal(rotated.valid_mask, original.valid_mask)
                np.testing.assert_array_equal(
                    rotated.diagnostics['transforms'], original.diagnostics['transforms'],
                )
                self.assertEqual(rotated.basis_angle_deg, angle)
                if angle == 90:
                    np.testing.assert_allclose(rotated.exx, original.eyy, atol=1e-12)
                    np.testing.assert_allclose(rotated.eyy, original.exx, atol=1e-12)
                    np.testing.assert_allclose(rotated.exy, -original.exy, atol=1e-12)

    def test_detect_refine_and_fit_preserves_yx_on_rectangular_patterns(self):
        origin = np.array([20, 30])
        reference = np.array([[10, 15], [10, 45], [30, 15], [30, 45]])
        positions = np.broadcast_to(reference, (2, 3, 4, 2)).copy()
        positions[0, 2, :, 1] = [14, 46, 14, 46]  # Only kx expands.
        positions[1, 0, :, 0] = [9, 9, 31, 31]  # Only ky expands.
        yy, xx = np.ogrid[:41, :61]
        patterns = np.zeros((2, 3, 41, 61))
        for scan_index in np.ndindex(2, 3):
            for (y, x), amplitude in zip(positions[scan_index], (20, 50, 40, 30)):
                disk = (yy - y) ** 2 + (xx - x) ** 2 <= 1
                patterns[scan_index][disk] = amplitude
        data = fd.HyperData(patterns)
        detected = data.get_dp(0, 0).get_peaks(
            radius=1, min_distance=5, threshold_abs=1,
        )
        self.assertEqual(set(map(tuple, detected)), set(map(tuple, reference)))
        refined = data.get_centers(r=3, ref_coords=detected, method='CoM')
        order = [np.flatnonzero(np.all(reference == point, axis=1))[0]
                 for point in detected]
        np.testing.assert_allclose(refined, positions[:, :, order, :], atol=1e-12)

        expected_x = np.zeros((2, 3))
        expected_y = np.zeros((2, 3))
        expected_x[0, 2] = 1 - 16 / 15
        expected_y[1, 0] = 1 - 11 / 10
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                source = fd.HyperData(patterns.reshape(6, 41, 61)) if stacked else data
                centers = refined.reshape(6, 4, 2) if stacked else refined
                result = source.get_strains(
                    centers=centers, ref_centers=detected, center=origin,
                )
                self.assertTrue(result.valid_mask.all())
                self.assertEqual(result.exx.shape, (6,) if stacked else (2, 3))
                np.testing.assert_allclose(result.exx.reshape(2, 3), expected_x, atol=1e-12)
                np.testing.assert_allclose(result.eyy.reshape(2, 3), expected_y, atol=1e-12)
                np.testing.assert_allclose(result.exy, 0, atol=1e-12)
                np.testing.assert_allclose(result.erot, 0, atol=1e-12)
                if not stacked:
                    np.testing.assert_array_equal(result.as_real_space('exx').array, result.exx)

    def test_final_weighted_residual_matches_reported_error(self):
        measured = self.measured.copy()
        measured[0, 0, 0] += [0.35, -0.2]
        weights = np.broadcast_to(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (2, 2, 5),
        )
        result = self.data.get_strains(
            centers=measured, ref_centers=self.reference,
            intensity_array=weights, center=(0, 0),
            match_peaks='ordered', fit_translation=True,
            return_transform=True,
        )

        self.assertEqual(len(result), 5)
        self.assertIs(result[4], result.diagnostics)
        exx, eyy, exy, erot, diagnostics = result
        self.assertIs(exx, result.exx)
        self.assertIs(eyy, result.eyy)
        self.assertIs(exy, result.exy)
        self.assertIs(erot, result.erot)
        transform = diagnostics['transforms'][0, 0]
        translation = diagnostics['translations'][0, 0]
        residual = measured[0, 0] - (
            self.reference @ transform.T + translation
        )
        expected_rmse = np.sqrt(np.average(
            np.sum(residual ** 2, axis=1), weights=weights[0, 0],
        ))
        reference_rms = np.sqrt(np.average(
            np.sum(self.reference ** 2, axis=1), weights=weights[0, 0],
        ))
        self.assertAlmostEqual(result.fit_rmse[0, 0], expected_rmse)
        self.assertAlmostEqual(
            result.relative_fit_rmse[0, 0], expected_rmse / reference_rms,
        )
        self.assertGreater(result.fit_rmse[0, 0], 0)
        self.assertIs(diagnostics['fit_rmse'], result.fit_rmse)

    def test_auto_matching_is_invariant_to_equal_count_peak_order(self):
        permutation = [2, 0, 4, 1, 3]
        shuffled = self.measured[:, :, permutation, :]
        automatic = self.data.get_strains(
            centers=shuffled, ref_centers=self.reference, center=(0, 0),
        )
        for component in ('exx', 'eyy', 'exy', 'erot', 'fit_rmse'):
            np.testing.assert_allclose(
                getattr(automatic, component), 0, atol=1e-12,
            )
        np.testing.assert_array_equal(automatic.match_counts, 5)

        ordered = self.data.get_strains(
            centers=shuffled, ref_centers=self.reference, center=(0, 0),
            match_peaks='ordered',
        )
        self.assertGreater(float(ordered.fit_rmse[0, 0]), 0.1)

    def test_auto_matching_keeps_intensity_weights_with_measured_peaks(self):
        measured = self.measured.copy()
        measured[:, :, 0] += [0.35, -0.2]
        weights = np.broadcast_to(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (2, 2, 5),
        )
        permutation = [2, 0, 4, 1, 3]
        automatic = self.data.get_strains(
            centers=measured[:, :, permutation, :],
            ref_centers=self.reference,
            intensity_array=weights[:, :, permutation],
            center=(0, 0), fit_translation=True,
        )
        ordered = self.data.get_strains(
            centers=measured, ref_centers=self.reference,
            intensity_array=weights, center=(0, 0),
            fit_translation=True, match_peaks='ordered',
        )
        for component in ('exx', 'eyy', 'exy', 'erot', 'fit_rmse'):
            np.testing.assert_allclose(
                getattr(automatic, component),
                getattr(ordered, component), atol=1e-12,
            )

    def test_masked_locations_have_no_fit_quality(self):
        mask = np.array([[True, False], [True, True]])
        result = self.data.get_strains(
            centers=self.measured, ref_centers=self.reference,
            center=(0, 0), real_mask=mask,
        )

        np.testing.assert_array_equal(result.valid_mask, mask)
        self.assertTrue(np.isnan(result.fit_rmse[0, 1]))
        self.assertTrue(np.isnan(result.relative_fit_rmse[0, 1]))
        self.assertEqual(result.match_counts[0, 1], 0)

    def test_unrelated_scan_geometry_does_not_claim_calibration(self):
        result = self.data.get_strains(
            centers=self.measured[:1], ref_centers=self.reference,
            center=(0, 0),
        )

        self.assertIsNone(result.real_units)
        self.assertIsNone(result.real_conv_factor)
        self.assertEqual(result.real_origin, (0.0, 0.0))
        self.assertIsNone(result.as_real_space('exx').units)

    def test_stack_result_rejects_realspace_conversion(self):
        stack = fd.HyperData(np.zeros((3, 8, 8)))
        measured = np.broadcast_to(
            self.reference, (3,) + self.reference.shape,
        )
        result = stack.get_strains(
            centers=measured, ref_centers=self.reference, center=(0, 0),
        )

        self.assertEqual(result.exx.shape, (3,))
        self.assertIsNone(result.real_units)
        with self.assertRaisesRegex(ValueError, 'requires a 2D scan map'):
            result.as_real_space('exx')
        with self.assertRaisesRegex(ValueError, 'Unknown strain component'):
            result.as_real_space('something_else')

    def test_mismatched_intensity_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'peak count'):
            self.data.get_strains(
                centers=self.measured, ref_centers=self.reference,
                intensity_array=np.ones((2, 2, 4)), center=(0, 0),
            )

    def test_collinear_or_ill_conditioned_peaks_have_no_valid_fit(self):
        for reference in (
            np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [2.0, 1e-11], [3.0, 0.0]]),
        ):
            with self.subTest(reference=reference):
                measured = np.broadcast_to(reference, (2, 2) + reference.shape)
                result = self.data.get_strains(
                    centers=measured, ref_centers=reference,
                    center=(0, 0), match_peaks='ordered',
                )
                self.assertFalse(result.valid_mask.any())
                self.assertTrue(np.isnan(result.fit_rmse).all())

    def test_translation_fit_needs_two_dimensional_peak_geometry(self):
        reference = np.array([[0.0, 0.0], [1.0, 1.0]])
        measured = np.broadcast_to(reference, (2, 2, 2, 2))
        result = self.data.get_strains(
            centers=measured, ref_centers=reference,
            center=(0, 0), fit_translation=True,
        )
        self.assertFalse(result.valid_mask.any())

    def test_calibrated_peak_units_convert_stored_pixel_center(self):
        data = fd.HyperData(
            np.zeros((2, 2, 8, 8)),
            reciprocal_units='Å^-1', reciprocal_conv_factor=2.0,
            center_beam_metadata={'mean_fit_center_px': (4.0, 3.0)},
        )
        measured = np.broadcast_to(
            self.reference, (2, 2) + self.reference.shape,
        )
        result = data.get_strains(
            centers=measured, ref_centers=self.reference,
            peak_units='calibrated', match_peaks='ordered',
            return_transform=True,
        )
        self.assertEqual(result.peak_origin, (-1.0, -1.0))
        self.assertEqual(result.peak_units, 'Å^-1')
        self.assertEqual(result.as_real_space('fit_rmse').value_units, 'Å^-1')
        self.assertEqual(result.diagnostics['peak_units'], 'calibrated')
        self.assertTrue(result.valid_mask.all())

        with self.assertRaisesRegex(ValueError, 'requires reciprocal_units'):
            self.data.get_strains(
                centers=self.measured, ref_centers=self.reference,
                peak_units='calibrated',
            )


class StrainFrameTests(unittest.TestCase):
    def setUp(self):
        self.data = fd.HyperData(
            np.zeros((2, 3, 24, 32)),
            real_units='nm', real_conv_factor=(0.5, 0.75),
            real_origin=(10.0, 20.0),
            reciprocal_units='nm^-1', reciprocal_conv_factor=0.2,
        )
        self.reference_xy = np.array([
            [-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0],
            [2.0, 1.0], [3.0, 2.0],
        ])
        yy, xx = np.indices((2, 3))
        self.exx = 0.01 + 0.003 * yy
        self.eyy = -0.02 + 0.004 * xx
        self.exy = 0.007 + 0.001 * (yy - xx)
        self.erot = np.radians(3.0 + 0.5 * yy - xx)
        self.stretch = np.empty((2, 3, 2, 2))
        self.stretch[..., 0, 0] = 1 - self.exx
        self.stretch[..., 1, 1] = 1 - self.eyy
        self.stretch[..., 0, 1] = -self.exy
        self.stretch[..., 1, 0] = -self.exy
        self.rotation = np.empty_like(self.stretch)
        self.rotation[..., 0, 0] = np.cos(self.erot)
        self.rotation[..., 0, 1] = -np.sin(self.erot)
        self.rotation[..., 1, 0] = np.sin(self.erot)
        self.rotation[..., 1, 1] = np.cos(self.erot)

    def peak_options(self, calibration, peak_units='pixels', transform=None):
        self.data.rq_calibration = calibration
        if transform is None:
            transform = self.rotation @ self.stretch
        detector_transform = calibration.matrix @ transform @ calibration.matrix.T
        measured_xy = self.reference_xy @ detector_transform.swapaxes(-1, -2)
        if peak_units == 'pixels':
            center = np.array([8.25, 13.5])
            scale = self.data.reciprocal_conv_factor
            reference = center + self.reference_xy[:, ::-1] * [-1, 1] / scale
            measured = center + measured_xy[..., ::-1] * [-1, 1] / scale
        else:
            center = np.zeros(2)
            reference = self.reference_xy[:, ::-1].copy()
            measured = measured_xy[..., ::-1].copy()
        return dict(
            centers=measured, ref_centers=reference, center=center,
            peak_units=peak_units, match_peaks='ordered',
        )

    def test_known_real_tensor_and_rotation_in_every_detector_orientation(self):
        for angle in (0.0, 37.5, 90.0, 271.25):
            for mirror in (None, 'x', 'y'):
                for peak_units in ('pixels', 'calibrated'):
                    with self.subTest(angle=angle, mirror=mirror, units=peak_units):
                        calibration = fd.RQCalibration(angle, mirror)
                        options = self.peak_options(calibration, peak_units)
                        original_centers = options['centers'].copy()
                        original_reference = options['ref_centers'].copy()
                        result = self.data.get_strains(**options, output_frame='real')
                        for name in ('exx', 'eyy', 'exy', 'erot'):
                            np.testing.assert_allclose(
                                getattr(result, name), getattr(self, name), atol=1e-12,
                            )
                        self.assertTrue(result.valid_mask.all())
                        np.testing.assert_allclose(result.fit_rmse, 0, atol=1e-12)
                        self.assertEqual(result.output_frame, 'real')
                        self.assertEqual(result.rq_calibration, calibration)
                        self.assertEqual(result.basis_angle_deg, 0)
                        np.testing.assert_array_equal(options['centers'], original_centers)
                        np.testing.assert_array_equal(options['ref_centers'], original_reference)
                        np.testing.assert_array_equal(self.data.array, 0)

    def test_default_and_manual_detector_outputs_ignore_rq_metadata(self):
        options = self.peak_options(fd.RQCalibration(0))
        self.data.rq_calibration = None
        original = self.data.get_strains(**options)
        manually_rotated = self.data.get_strains(**options, ang=37.5)
        self.data.set_rq_calibration(112.0, 'y')
        default = self.data.get_strains(**options)
        explicit = self.data.get_strains(**options, output_frame='reciprocal')
        manual = self.data.get_strains(**options, ang=37.5)
        for result, expected in ((default, original), (explicit, original),
                                 (manual, manually_rotated)):
            for name in ('exx', 'eyy', 'exy', 'erot', 'fit_rmse'):
                np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))
            self.assertEqual(result.output_frame, 'reciprocal')
            self.assertIsNone(result.rq_calibration)

    def test_pixel_real_frame_matches_manual_rotation_without_reflection(self):
        options = self.peak_options(fd.RQCalibration(37.5))
        real = self.data.get_strains(**options, output_frame='real')
        manual = self.data.get_strains(**options, ang=37.5)
        for name in ('exx', 'eyy', 'exy', 'erot'):
            np.testing.assert_allclose(getattr(real, name), getattr(manual, name), atol=1e-12)

    def test_reporting_conversion_preserves_mask_quality_and_raw_fit_diagnostics(self):
        calibration = fd.RQCalibration(37.5, 'x')
        options = self.peak_options(calibration)
        mask = np.array([[True, False, True], [True, True, True]])
        options.update(real_mask=mask, return_transform=True)
        detector = self.data.get_strains(**options)
        real = self.data.get_strains(**options, output_frame='real')
        for name in ('fit_rmse', 'relative_fit_rmse', 'match_counts',
                     'initial_match_counts', 'outlier_counts', 'valid_mask'):
            np.testing.assert_array_equal(getattr(real, name), getattr(detector, name))
        for name in ('transforms', 'translations', 'center'):
            np.testing.assert_array_equal(real.diagnostics[name], detector.diagnostics[name])
        for name in ('exx', 'eyy', 'exy', 'erot'):
            np.testing.assert_allclose(getattr(real, name)[mask], getattr(self, name)[mask], atol=1e-12)
            self.assertTrue(np.isnan(getattr(real, name)[~mask]).all())
        image = real.as_real_space('exx')
        np.testing.assert_array_equal(image.array, real.exx)
        self.assertEqual(image.conv_factor, (0.5, 0.75))
        self.assertEqual(image.origin, (10.0, 20.0))
        self.assertEqual(real.diagnostics['output_frame'], 'real')
        self.assertEqual(real.diagnostics['rq_calibration'], calibration.to_dict())
        np.testing.assert_array_equal(
            real.diagnostics['output_basis_matrix'], real.output_basis_matrix,
        )
        self.assertFalse(np.shares_memory(
            real.diagnostics['output_basis_matrix'], real.output_basis_matrix,
        ))
        self.data.set_rq_calibration(11)
        self.assertEqual(real.rq_calibration, calibration)

    def test_saved_calibration_is_used_for_stack_output(self):
        calibration = fd.RQCalibration(112, 'y')
        options = self.peak_options(calibration, 'calibrated')
        stack = self.data.unfold(domain='real')
        options['centers'] = options['centers'].reshape(6, 5, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibrated.4denoise'
            stack.save(path)
            restored = fd.HyperData(path)
            result = restored.get_strains(**options, output_frame='real')
        self.assertEqual(result.exx.shape, (6,))
        self.assertEqual(result.rq_calibration, calibration)
        for name in ('exx', 'eyy', 'exy', 'erot'):
            np.testing.assert_allclose(
                getattr(result, name), getattr(self, name).reshape(6), atol=1e-12,
            )

    def test_ewpc_inverse_uses_the_same_real_reporting_frame(self):
        for peak_units in ('pixels', 'calibrated'):
            with self.subTest(units=peak_units):
                options = self.peak_options(
                    fd.RQCalibration(52, 'x'), peak_units,
                    transform=np.linalg.inv(self.stretch),
                )
                result = self.data.get_strains(**options, ewpc=True, output_frame='real')
                for name in ('exx', 'eyy', 'exy'):
                    np.testing.assert_allclose(getattr(result, name), getattr(self, name), atol=1e-12)
                np.testing.assert_allclose(result.erot, 0, atol=1e-12)

    def test_output_frame_requires_calibration_and_rejects_two_basis_corrections(self):
        self.data.acquisition_metadata = {'scan_rotation_deg': 87.276}
        with self.assertRaisesRegex(ValueError, 'requires stored rq_calibration'):
            self.data.get_strains(output_frame='real')
        self.data.set_rq_calibration(30)
        with self.assertRaisesRegex(ValueError, 'leave ang=0'):
            self.data.get_strains(output_frame='real', ang=30)
        for invalid in ('auto', 'scan', None, 1, ['real']):
            with self.subTest(frame=invalid):
                with self.assertRaisesRegex(ValueError, 'output_frame'):
                    self.data.get_strains(output_frame=invalid)
        for invalid in (np.nan, np.inf, True, None, [0]):
            with self.subTest(angle=invalid):
                with self.assertRaisesRegex(ValueError, 'ang must be a finite scalar'):
                    self.data.get_strains(ang=invalid)


if __name__ == '__main__':
    unittest.main()
