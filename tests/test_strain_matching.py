"""Unequal-count, local-candidate, initial-transform and diagnostic contracts."""
import unittest
import numpy as np
import fourdenoise as fd
from fourdenoise_strain import _initial_transform


class StrainMatchingTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([[-2., -1.], [-1., 2.], [1., -2.], [2., 1.], [3., 3.]])
        self.data = fd.HyperData(np.zeros((1, 1, 64, 64)), reciprocal_units='mrad', reciprocal_conv_factor=0.1)

    def fit(self, measured, **kwargs):
        return self.data.get_strains(centers=[[measured]], ref_centers=self.reference,
                                    center=(0, 0), return_transform=True, **kwargs)

    def test_extra_peaks_do_not_move_pairing_origin(self):
        measured = np.vstack((self.reference[[2, 4, 0, 3, 1]], [[100, 100], [-80, 70]]))
        result = self.fit(measured)
        self.assertEqual(result.match_counts[0, 0], 5)
        np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], np.eye(2), atol=1e-12)
        info = result.diagnostics['matching_information'][0, 0]
        np.testing.assert_array_equal(info['unmatched_measured_indices'], [5, 6])
        self.assertEqual(result.fit_status[0, 0], 'success')

    def test_missing_peaks_and_extras_are_left_unmatched(self):
        measured = np.vstack((self.reference[:4], [[100, 100]]))
        result = self.fit(measured)
        self.assertEqual(result.match_counts[0, 0], 4)
        info = result.diagnostics['matching_information'][0, 0]
        np.testing.assert_array_equal(info['unmatched_reference_indices'], [4])
        np.testing.assert_array_equal(info['unmatched_measured_indices'], [4])
        np.testing.assert_allclose(result.exx, 0, atol=1e-12)

    def test_no_candidates_has_an_explicit_failure_status(self):
        result = self.fit(self.reference + 100, match_radius=0.5)
        self.assertFalse(result.valid_mask.any())
        self.assertEqual(result.fit_status[0, 0], 'no_candidates')
        self.assertTrue(np.isnan(result.exx).all())

    def test_initial_rotation_guides_matching_but_keeps_total_transform(self):
        for units in ('pixels', 'calibrated'):
            matrix = _initial_transform(120, None, units)
            result = self.fit(self.reference @ matrix.T, initial_rotation=120, peak_units=units, match_radius=0.1)
            self.assertEqual(result.match_counts[0, 0], 5)
            np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], matrix, atol=1e-12)
            np.testing.assert_allclose(result.exx, 0, atol=1e-12)

    def test_initial_matrix_can_include_scale_and_shear(self):
        matrix = np.array([[0.0, -1.03], [0.98, 0.02]])
        result = self.fit(self.reference @ matrix.T, initial_transform=matrix, match_radius=0.1)
        np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], matrix, atol=1e-12)

    def test_translation_consensus_ignores_distant_extra_peaks(self):
        shift = np.array([5., -7.])
        measured = np.vstack((self.reference + shift, [[100, 100], [-100, 100]]))
        result = self.fit(measured, fit_translation=True, match_radius=0.1)
        self.assertEqual(result.match_counts[0, 0], 5)
        np.testing.assert_allclose(result.diagnostics['translations'][0, 0], shift, atol=1e-12)

    def test_pair_confidence_and_optional_ambiguity_rejection(self):
        measured = np.vstack((self.reference[1:], self.reference[0] + [0.05, 0], self.reference[0] - [0.05, 0]))
        result = self.fit(measured, match_radius=0.5, ambiguity_ratio=0.8)
        info = result.diagnostics['matching_information'][0, 0]
        self.assertEqual(info['ambiguous_pair_count'], 1)
        self.assertEqual(result.match_counts[0, 0], 4)
        self.assertEqual(len(np.unique(info['matched_measured_indices'])), 4)

    def test_original_indices_survive_nonfinite_and_intensity_filtering(self):
        measured = np.vstack(([np.nan, np.nan], self.reference, [100, 100]))
        weights = [[[0, 2, 3, 4, 5, 6, 1]]]
        result = self.fit(measured, intensity_array=weights, intensity_percentile=10, intensity_clip='lower')
        info = result.diagnostics['matching_information'][0, 0]
        self.assertTrue(np.all(info['matched_measured_indices'] >= 1))
        np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], np.eye(2), atol=1e-12)

    def test_scan_shaped_rotation_guesses(self):
        data = fd.HyperData(np.zeros((2, 2, 16, 16)))
        angles = np.array([[0, 100], [-80, 50]])
        measured = np.array([[self.reference @ _initial_transform(a, None, 'pixels').T for a in row] for row in angles])
        result = data.get_strains(centers=measured, ref_centers=self.reference, center=(0, 0),
                                  initial_rotation=angles, match_radius=0.1)
        np.testing.assert_array_equal(result.match_counts, 5)

    def test_ordered_correspondence_keeps_original_rows_after_filtering(self):
        weights = [[[1, 3, 2, 6, 4]]]
        result = self.fit(self.reference, intensity_array=weights, intensity_percentile=30,
                          intensity_clip='lower', match_peaks='ordered')
        np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], np.eye(2), atol=1e-12)
        info = result.diagnostics['matching_information'][0, 0]
        np.testing.assert_array_equal(info['matched_reference_indices'], info['matched_measured_indices'])
        np.testing.assert_array_equal(info['intensity_filtered_indices'], [0, 2])

    def test_matching_parameter_validation(self):
        for kwargs in ({'match_radius': -1}, {'ambiguity_ratio': 1},
                       {'initial_rotation': 10, 'initial_transform': np.eye(2)},
                       {'initial_translation': [1, 2]}, {'initial_rotation': [1, 2]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fit(self.reference, **kwargs)

    def test_collapsed_transform_is_reported_not_as_full_compression(self):
        result = self.fit(np.zeros_like(self.reference), match_peaks='ordered')
        self.assertFalse(result.valid_mask.any())
        self.assertEqual(result.fit_status[0, 0], 'singular_transform')


if __name__ == '__main__':
    unittest.main()
