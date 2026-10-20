"""Shared operation-specific geometry and canonical beam-center contracts."""
import unittest
import numpy as np
import fourdenoise as fd
import fourdenoise_geometry as geometry
from fourdenoise_rq import _rq_center


class GeometryOperationTests(unittest.TestCase):
    def test_crop_resize_flip_chain_preserves_current_coordinates(self):
        beam = fd._center_beam_metadata_from_pixels(3, (7.5, 7.5), (16, 16), units='mrad', conv_factor=0.2)
        data = fd.HyperData(np.ones((3, 4, 16, 16)), real_units='nm', real_conv_factor=0.5,
                            real_origin=(10, 20), reciprocal_units='mrad', reciprocal_conv_factor=0.2,
                            center_beam_metadata=beam, rq_calibration=fd.RQCalibration(25))
        cropped = data.crop(ylim=(1, 3), xlim=(1, 4), kylim=(4, 12), kxlim=(4, 12),
                            real_limit_units='pixels', reciprocal_limit_units='pixels')
        resized = cropped.resize((4, 4), 'reciprocal')
        flipped = fd.HyperData(resized.array, **resized._coordinate_metadata(),
                               rq_calibration=resized.rq_calibration, flip_axis=(0, 2))
        self.assertEqual(flipped.center_beam_metadata['shape'], (4, 4))
        np.testing.assert_allclose(flipped.center_beam_metadata['center_px'], (1.5, 1.5))
        self.assertEqual(flipped.center_beam_metadata['radius_px'], 1.5)
        np.testing.assert_allclose(flipped.real_origin, (11, 20.5))
        np.testing.assert_allclose(flipped.real_conv_factor, (-0.5, 0.5))
        np.testing.assert_allclose(flipped.rq_calibration.matrix @ flipped.rq_calibration.matrix.T, np.eye(2), atol=1e-12)
        self.assertEqual(data.center_beam_metadata['shape'], (16, 16))

    def test_hyperdata_and_pattern_crop_use_the_same_geometry(self):
        beam = fd._center_beam_metadata_from_pixels(2, (7.5, 7.5), (16, 16))
        data = fd.HyperData(np.ones((1, 1, 16, 16)), reciprocal_units='mrad', reciprocal_conv_factor=0.2, center_beam_metadata=beam)
        for bounds in (((4, 12), (4, 12)), ((4.2, 12.2), (4, 12))):
            hyper = data.crop(kylim=bounds[0], kxlim=bounds[1], kshape=(4, 4), reciprocal_limit_units='pixels')
            pattern = data.get_dp(0, 0).crop(kylim=bounds[0], kxlim=bounds[1], kshape=(4, 4))
            self.assertEqual(hyper.center_beam_metadata, pattern.center_beam_metadata)
            self.assertEqual(hyper.reciprocal_conv_factor, pattern.conv_factor)

    def test_strain_uses_current_beam_center_not_historical_fit_location(self):
        center = np.array([7.5, 7.5])
        beam = fd._center_beam_metadata_from_pixels(2, center, (16, 16), mean_fit_center_px=(4, 5))
        data = fd.HyperData(np.zeros((1, 1, 16, 16)), center_beam_metadata=beam)
        reference = np.array([[-3, -2], [-2, 3], [2, -3], [3, 2]]) + center
        matrix = np.array([[1.01, 0.01], [0, 0.99]])
        measured = (reference - center) @ matrix.T + center
        result = data.get_strains(centers=[[measured]], ref_centers=reference, return_transform=True)
        self.assertEqual(result.peak_origin, tuple(center))
        self.assertEqual(result.peak_origin_source, 'center_beam_metadata.center_px')
        np.testing.assert_allclose(result.diagnostics['transforms'][0, 0], matrix, atol=1e-12)
        self.assertEqual(_rq_center((16, 16), beam), geometry._beam_center_pixels((16, 16), beam))

    def test_geometry_dispatch_rejects_unsupported_and_shape_changing_values(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            geometry._transform_geometry({}, (8, 8), (8, 8), 'invented')
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            geometry._transform_geometry({}, (8, 8), (4, 4), 'values')

    def test_image_center_rotation_maps_stored_beam_center_and_radius(self):
        beam = fd._center_beam_metadata_from_pixels(1, (2, 4), (7, 9))
        data = fd.HyperData(np.zeros((1, 7, 9)), center_beam_metadata=beam)
        result = data.rotate_dps(90, order=0)
        self.assertEqual(result.pattern_shape, (9, 7))
        np.testing.assert_allclose(result.center_beam_metadata['center_px'], (4, 2), atol=1e-12)
        self.assertEqual(result.center_beam_metadata['radius_px'], 1)

    def test_value_metadata_is_independent_and_beam_validation_is_shared(self):
        beam = fd._center_beam_metadata_from_pixels(2, (3.5, 3.5), (8, 8))
        original = {'center_beam_metadata': beam}
        copied = geometry._transform_geometry(original, (8, 8), (8, 8), 'values')
        copied['center_beam_metadata']['radius_px'] = 99
        self.assertEqual(beam['radius_px'], 2)
        for bad in ({'shape': 5}, {'shape': (4, 4)}, {'center_px': [np.nan, 0]}):
            with self.assertRaises(ValueError):
                geometry._beam_center_pixels((8, 8), bad)


if __name__ == '__main__':
    unittest.main()
