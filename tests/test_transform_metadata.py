"""Coordinate and intensity fidelity for polar, elliptical and EWPC transforms."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':
    raise ImportError(f'Imported the wrong fourdenoise.py: {fd.__file__}')


def calibrated_data(shape=(2, 3, 16, 16)):
    return fd.HyperData(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 10,
        real_units='nm', real_conv_factor=(2, 3), real_origin=(5, 7),
        reciprocal_units='mrad', reciprocal_conv_factor=0.2,
        acquisition_metadata={'experiment': {'name': 'metadata regression'}},
        rq_calibration=fd.RQCalibration(17),
    )


class PolarGeometryTests(unittest.TestCase):
    def setUp(self):
        self.data = calibrated_data()
        self.polar = self.data.to_polar(output_shape=(8, 32), progress=False)
        self.original_metadata = deepcopy(self.polar.polar_metadata)

    def tearDown(self):
        plt.close('all')

    def test_area_resize_updates_samples_coverage_and_rotation(self):
        resized = self.polar.resize((4, 16), domain='reciprocal')
        meta = resized.polar_metadata
        self.assertEqual(meta['output_shape'], (4, 16))
        self.assertEqual(meta['theta_range'], (0, 360))
        self.assertEqual(meta['theta_step'], 22.5)
        self.assertEqual(meta['theta_sample_start'], 5.625)
        np.testing.assert_allclose(meta['radius_range_pixels'], (7.5 / 14, 7.5 * 13 / 14))
        np.testing.assert_allclose(meta['radius_sample_step_pixels'], 15 / 7)
        self.assertEqual(meta['radius_support_range_pixels'], (0, 7.5))
        np.testing.assert_array_equal(
            resized.rotate_dps(22.5).array, np.roll(resized.array, -1, axis=-1),
        )
        self.assertEqual(self.polar.polar_metadata, self.original_metadata)

    def test_nearest_resize_keeps_radial_endpoints_and_dtype(self):
        polar = self.data.to_polar(output_shape=(8, 32), order=0, progress=False)
        for shape in ((4, 16), (16, 64)):
            with self.subTest(shape=shape):
                result = polar.resize(shape, domain='reciprocal', method='nearest')
                self.assertEqual(result.dtype, self.data.dtype)
                self.assertEqual(result.polar_metadata['radius_range_pixels'], (0, 7.5))
                self.assertEqual(result.polar_metadata['radius_sample_step_pixels'], 7.5 / (shape[0] - 1))
                self.assertEqual(result.polar_metadata['theta_step'], 360 / shape[1])
                result.rotate_dps(360 / shape[1])

    def test_linear_upsampling_wraps_theta_but_not_radius(self):
        original = np.broadcast_to(np.arange(8, dtype=np.uint16), (4, 8)).copy()
        polar = fd.HyperData(original[None], polar_metadata={
            'output_shape': (4, 8), 'radius_range_pixels': (0, 3),
            'theta_range': (0, 360), 'theta_step': 45,
        })
        resized = polar.resize((7, 16), domain='reciprocal', method='linear')
        np.testing.assert_allclose(resized.array[0, :, -1], 3.5)
        np.testing.assert_allclose(resized.array[0, :, 0], 0)
        self.assertEqual(resized.dtype.kind, 'f')
        radial = fd.HyperData(
            np.broadcast_to(np.arange(4)[:, None], (4, 8))[None],
            polar_metadata=polar.polar_metadata,
        ).resize((7, 16), domain='reciprocal', method='linear')
        np.testing.assert_allclose(radial.array[0, :, 0], np.linspace(0, 3, 7))

    def test_crop_records_actual_annulus_and_sector(self):
        result = self.polar.crop(kylim=(2, 6), kxlim=(4, 20))
        meta = result.polar_metadata
        self.assertEqual(meta['output_shape'], (4, 16))
        np.testing.assert_allclose(meta['radius_range_pixels'], (15 / 7, 37.5 / 7))
        np.testing.assert_allclose(meta['radius_support_range_pixels'], (11.25 / 7, 41.25 / 7))
        np.testing.assert_allclose(meta['radius_display_range'], (3 / 7, 7.5 / 7))
        self.assertEqual(meta['theta_range'], (45, 225))
        self.assertEqual(meta['theta_sample_start'], 45)
        self.assertEqual(meta['theta_step'], 11.25)
        with self.assertRaisesRegex(ValueError, '360'):
            result.rotate_dps(10)
        self.assertEqual(self.polar.polar_metadata, self.original_metadata)

    def test_hyper_and_reciprocal_crop_agree(self):
        hyper = self.polar.crop(kylim=(1.2, 6.2), kxlim=(3.2, 27.2))
        dp = self.polar.get_dp(0, 0).crop(kylim=(1.2, 6.2), kxlim=(3.2, 27.2))
        np.testing.assert_allclose(hyper.array[0, 0], dp.array)
        self.assertEqual(hyper.polar_metadata, dp.polar_metadata)
        self.assertAlmostEqual(dp.polar_metadata['theta_sample_start'], 36)

    def test_scan_only_changes_preserve_polar_geometry(self):
        for output in (
            self.polar.crop(ylim=(0, 1), real_limit_units='pixels'),
            self.polar.resize((1, 2), domain='real'), self.polar.copy(), self.polar.clip(),
        ):
            self.assertEqual(output.polar_metadata, self.original_metadata)

    def test_virtual_detector_uses_resized_sample_origins(self):
        resized = self.polar.resize((4, 16), domain='reciprocal')
        resized.array = np.ones(resized.shape)
        # The first angular sample is 5.625 degrees, not zero.
        virtual = resized.virtual_image(
            annulus=(0, 10), theta_range=(4, 7), detector_units='pixels',
            show=False, plot_mask=False,
        )
        np.testing.assert_array_equal(virtual.array, 4)
        radial = self.polar.crop(kylim=(2, 6))
        radial.array = np.ones(radial.shape)
        virtual = radial.virtual_image(
            annulus=(0.4, 0.9), detector_units='calibrated', show=False, plot_mask=False,
        )
        np.testing.assert_array_equal(virtual.array, 3 * 32)

    def test_cartesian_cropped_annulus_and_sector_remain_missing(self):
        self.polar.array = np.full(self.polar.shape, 100.0)
        cropped = self.polar.crop(kylim=(2, 8), kxlim=(0, 8))
        cartesian = cropped.to_cartesian(output_shape='original', fill_value=-3, progress=False)
        image = cartesian.array[0, 0]
        self.assertEqual(image[7, 7], -3)  # Removed inner disk.
        self.assertEqual(image[8, 11], 100)  # Within the retained 0..90 degree sector.
        self.assertEqual(image[8, 4], -3)  # Removed angular sector.
        self.assertEqual(image[0, 0], -3)  # Outside the retained circle.
        self.assertFalse(cartesian.is_polar)
        self.assertAlmostEqual(cartesian.reciprocal_conv_factor, 0.2)

    def test_cartesian_resize_uses_actual_first_angular_sample(self):
        self.polar.array = np.broadcast_to(
            np.arange(32, dtype=float), self.polar.shape,
        ).copy()
        resized = self.polar.resize((4, 16), domain='reciprocal')
        cartesian = resized.to_cartesian(output_shape=(17, 17), center=(8, 8), progress=False)
        # Area-averaged samples are .5 at 5.625 degrees and 2.5 at 28.125.
        # At theta=45 degrees interpolation therefore gives exactly 4.
        self.assertAlmostEqual(cartesian.array[0, 0, 11, 11], 4)
        self.assertAlmostEqual(cartesian.reciprocal_conv_factor, 1.5 / 8.5)

    def test_cartesian_sector_without_zero_angle_does_not_wrap(self):
        self.polar.array = np.full(self.polar.shape, 100.0)
        cropped = self.polar.crop(kxlim=(8, 16))  # 90..180 degrees.
        output = cropped.to_cartesian(output_shape=(17, 17), center=(8, 8), progress=False)
        self.assertEqual(output.array[0, 0, 11, 5], 100)
        self.assertEqual(output.array[0, 0, 5, 11], 0)

    def test_metadata_mismatch_is_rejected(self):
        for field, value in (('output_shape', (99, 32)), ('theta_step', 1), ('radius_sample_step_pixels', 12)):
            with self.subTest(field=field):
                bad = self.polar.copy()
                bad.polar_metadata[field] = value
                with self.assertRaisesRegex(ValueError, 'metadata|inconsistent'):
                    bad.resize((4, 16), domain='reciprocal')
                with self.assertRaises(ValueError):
                    bad.to_cartesian(progress=False)

    def test_unsupported_polar_crop_has_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, 'calibrated|angular'):
            self.polar.crop(kylim=(0, 1), reciprocal_limit_units='calibrated')
        with self.assertRaisesRegex(ValueError, 'radial grid'):
            self.polar.crop(kshape=(16, 32))

    def test_single_radial_sample_preserves_support(self):
        resized = self.polar.resize((1, 16), domain='reciprocal', method='linear')
        self.assertEqual(resized.polar_metadata['radius_range_pixels'], (0, 0))
        self.assertEqual(resized.polar_metadata['radius_support_range_pixels'], (0, 7.5))
        output = resized.to_cartesian(output_shape='original', progress=False)
        self.assertEqual(output.pattern_shape, (16, 16))

    def test_updated_polar_metadata_survives_save_load_and_chunks(self):
        resized = self.polar.resize((4, 16), domain='reciprocal').crop(kylim=(1, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'polar.4denoise'
            resized.save(path)
            loaded = fd.HyperData(path)
            self.assertEqual(loaded.polar_metadata, resized.polar_metadata)
            with fd.HyperData.open_hdf5(path) as reader:
                self.assertEqual(reader.get_dp(0, 0).polar_metadata, resized.polar_metadata)
                _, block = next(reader.iter_chunks(1))
                self.assertEqual(block.polar_metadata, resized.polar_metadata)
                block.rotate_dps(22.5)


class EllipticalMetadataTests(unittest.TestCase):
    def correct(self, data, params=(0.31, 4.5, 2.75), **kwargs):
        with (
            patch.object(fd.HyperData, '_extract_ellipse', return_value=params),
            patch.object(fd, 'tqdm', new=lambda iterable, **kwargs: iterable),
            redirect_stdout(io.StringIO()),
        ):
            return data.fix_elliptical_distortions(r=1, R=5, **kwargs)

    def test_integer_warp_preserves_fractional_intensities(self):
        data = calibrated_data()
        original = data.array.copy()
        corrected = self.correct(data)
        expected = self.correct(data._spawn(data.array.astype(np.float32)))
        self.assertEqual(corrected.dtype.kind, 'f')
        np.testing.assert_array_equal(corrected.array, expected.array)
        self.assertTrue(np.any(np.abs(corrected.array - np.rint(corrected.array)) > 1e-3))
        np.testing.assert_array_equal(data.array, original)

    def test_warp_preserves_scan_geometry_but_clears_detector_calibration(self):
        data = calibrated_data()
        data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            2, (7.5, 7.5), data.pattern_shape, units='mrad', conv_factor=0.2, source='alignment',
        )
        result = self.correct(data)
        self.assertEqual(result.real_units, data.real_units)
        self.assertEqual(result.real_conv_factor, data.real_conv_factor)
        self.assertEqual(result.real_origin, data.real_origin)
        self.assertEqual(result.acquisition_metadata, data.acquisition_metadata)
        self.assertIsNone(result.reciprocal_units)
        self.assertIsNone(result.reciprocal_conv_factor)
        self.assertIsNone(result.center_beam_metadata)
        self.assertIsNone(result.rq_calibration)
        result.acquisition_metadata['experiment']['name'] = 'changed'
        self.assertNotEqual(result.acquisition_metadata, data.acquisition_metadata)

    def test_identity_preserves_detector_calibration_and_beam(self):
        data = calibrated_data()
        data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            2, (7.5, 7.5), data.pattern_shape, units='mrad', conv_factor=0.2, source='alignment',
        )
        result = self.correct(data, params=(0.31, 4, 4))
        np.testing.assert_array_equal(result.array, data.array)
        self.assertEqual(result.reciprocal_units, data.reciprocal_units)
        self.assertEqual(result.reciprocal_conv_factor, data.reciprocal_conv_factor)
        self.assertEqual(result.center_beam_metadata, data.center_beam_metadata)
        self.assertEqual(result.rq_calibration, data.rq_calibration)

    def test_three_dimensional_stacks_are_supported(self):
        data = calibrated_data((3, 16, 20))
        output = self.correct(data)
        self.assertEqual(output.shape, data.shape)
        self.assertEqual(output.dtype.kind, 'f')

    def test_fitting_and_warping_share_stored_subpixel_center(self):
        data = calibrated_data()
        center = (6.25, 8.75)
        data.center_beam_metadata = fd._center_beam_metadata_from_pixels(
            2, center, data.pattern_shape, source='alignment',
        )
        with (
            patch.object(fd.HyperData, '_extract_ellipse', return_value=(0, 4, 3)) as fit,
            patch.object(fd.HyperData, '_apply_affine_transformation', wraps=data._apply_affine_transformation) as warp,
            patch.object(fd, 'tqdm', new=lambda iterable, **kwargs: iterable),
            redirect_stdout(io.StringIO()),
        ):
            data.fix_elliptical_distortions(r=1, R=5)
        np.testing.assert_allclose(fit.call_args_list[0].args[1:3], center)
        self.assertEqual(warp.call_args.kwargs['center'], center)

    def test_no_fix_returns_independent_copy_without_warp(self):
        data = calibrated_data()
        with patch.object(fd.HyperData, '_apply_affine_transformation') as warp:
            result = self.correct(data, return_fix=False)
        warp.assert_not_called()
        self.assertFalse(np.shares_memory(result.array, data.array))
        self.assertEqual(result.rq_calibration, data.rq_calibration)

    def test_integer_fitting_weights_do_not_overflow(self):
        data = calibrated_data()
        costs = []
        def optimizer(cost, guess, **kwargs):
            costs.append(cost(np.array([0.2, 4, 3])))
            self.assertGreater(guess[1], 0)
            self.assertIn('bounds', kwargs)
            return SimpleNamespace(success=True, x=np.array([0.2, 4, 3]))
        image = np.full((16, 16), 60000, dtype=np.uint16)
        with patch.object(fd, 'minimize', side_effect=optimizer):
            data._extract_ellipse(image, 7.5, 7.5, 1, 5)
            data._extract_ellipse(image.astype(float), 7.5, 7.5, 1, 5)
        self.assertTrue(np.all(np.isfinite(costs)))
        self.assertEqual(costs[0], costs[1])
        self.assertGreater(costs[0], 0)

    def test_failed_fit_and_empty_annulus_are_explicit(self):
        data = calibrated_data()
        with patch.object(fd, 'minimize', return_value=SimpleNamespace(
            success=False, x=np.array([0, 2, 3]), message='not converged',
        )):
            with self.assertRaisesRegex(RuntimeError, 'not converged'):
                data._extract_ellipse(np.ones((16, 16)), 7.5, 7.5, 1, 5)
        with self.assertRaisesRegex(ValueError, 'positive intensity'):
            data._extract_ellipse(np.zeros((16, 16)), 7.5, 7.5, 1, 5)

    def test_actual_fit_recovers_rotated_ring_and_corrects_it(self):
        y, x = np.indices((65, 65), dtype=float)
        x -= 32
        y -= 32
        for angle in (0.2, np.pi / 4, 1.2):
            with self.subTest(angle=angle):
                radius = np.sqrt(
                    ((x * np.cos(angle) + y * np.sin(angle)) / 15) ** 2
                    + ((y * np.cos(angle) - x * np.sin(angle)) / 10) ** 2
                )
                image = (60000 * np.exp(-((radius - 1) / 0.03) ** 2)).astype(np.uint16)
                data = fd.HyperData(image[None])
                fitted_angle, major, minor = data._extract_ellipse(image, 32, 32, 8, 18)
                self.assertAlmostEqual(major, 15, delta=0.2)
                self.assertAlmostEqual(minor, 10, delta=0.2)
                self.assertGreater(np.cos(2 * (fitted_angle - angle)), 0.999)
                with (
                    redirect_stdout(io.StringIO()),
                    patch.object(fd, 'tqdm', new=lambda iterable, **kwargs: iterable),
                ):
                    corrected = data.fix_elliptical_distortions(r=8, R=18)
                _, corrected_a, corrected_b = data._extract_ellipse(corrected.array[0], 32, 32, 8, 18)
                self.assertAlmostEqual(corrected_a / corrected_b, 1, delta=0.03)

    def test_invalid_correction_contracts(self):
        data = calibrated_data()
        for kwargs in ({'r': -1, 'R': 5}, {'r': 5, 'R': 4}, {'interp_method': 'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                data.fix_elliptical_distortions(**kwargs)
        for bad in (
            fd.HyperData(np.ones((16, 16))),
            data.to_polar(progress=False), data.get_ewpc(),
        ):
            with self.assertRaisesRegex(ValueError, 'Cartesian'):
                bad.fix_elliptical_distortions()


class CepstralGeometryTests(unittest.TestCase):
    def setUp(self):
        self.data = calibrated_data((2, 3, 8, 10))
        self.ewpc = self.data.get_ewpc()

    def tearDown(self):
        plt.close('all')

    def test_fft_formula_and_input_preservation(self):
        original = self.data.array.copy()
        expected = np.abs(fftshift(fft2(np.log(original.astype(np.float32)), axes=(-2, -1)), axes=(-2, -1)))
        np.testing.assert_array_equal(self.ewpc.array, expected)
        np.testing.assert_array_equal(self.data.array, original)
        floored = self.data.get_ewpc(a_min=50)
        expected = np.abs(fftshift(fft2(np.log(np.maximum(original.astype(np.float32), 50)), axes=(-2, -1)), axes=(-2, -1)))
        np.testing.assert_array_equal(floored.array, expected)

    def test_ewpc_coordinate_identity_and_scan_calibration(self):
        self.assertTrue(self.ewpc.is_cepstral)
        self.assertFalse(self.ewpc.is_polar)
        self.assertEqual(self.ewpc.cepstral_metadata['axis_order'], ('cy', 'cx'))
        self.assertEqual(self.ewpc.cepstral_metadata['center_px'], (4, 5))
        self.assertEqual(self.ewpc.cepstral_metadata['source_reciprocal_units'], 'mrad')
        self.assertIsNone(self.ewpc.reciprocal_units)
        self.assertIsNone(self.ewpc.reciprocal_conv_factor)
        self.assertIsNone(self.ewpc.center_beam_metadata)
        self.assertIsNone(self.ewpc.rq_calibration)
        self.assertEqual(self.ewpc.real_units, self.data.real_units)
        self.assertEqual(self.ewpc.real_conv_factor, self.data.real_conv_factor)
        self.assertEqual(self.ewpc.real_origin, self.data.real_origin)
        self.assertEqual(self.ewpc.acquisition_metadata, self.data.acquisition_metadata)

    def test_show_labels_and_scatter_use_actual_fft_zero(self):
        for shape in ((8, 10), (7, 9)):
            with self.subTest(shape=shape):
                data = calibrated_data((1, 1) + shape).get_ewpc()
                center = tuple(size // 2 for size in shape)
                dp = data.get_dp(0, 0)
                with patch.object(plt, 'show'):
                    dp.show(logScale=False, coords=[center])
                ax = plt.gcf().axes[0]
                self.assertIn('c_x', ax.get_xlabel())
                self.assertIn('c_y', ax.get_ylabel())
                self.assertEqual(ax.get_title(), 'EWPC')
                np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 0]])
                extent = ax.images[0].get_extent()
                self.assertAlmostEqual(extent[0] + center[1] + 0.5, 0)
                self.assertAlmostEqual(extent[3] - center[0] - 0.5, 0)
                plt.close('all')

    def test_value_only_processing_and_copy_keep_independent_identity(self):
        for output in (
            self.ewpc.copy(), self.ewpc.clip(), self.ewpc.normalize(),
            self.ewpc.denoise('median', domain='reciprocal', window_size=3),
        ):
            self.assertEqual(output.cepstral_metadata, self.ewpc.cepstral_metadata)
            output.cepstral_metadata['transform'] = 'changed'
            self.assertEqual(self.ewpc.cepstral_metadata['transform'], 'ewpc')
        self.assertTrue(self.ewpc.get_dp(0, 0).copy().is_cepstral)

    def test_crop_resize_and_flip_update_fft_zero(self):
        cropped = self.ewpc.crop(kylim=(1, 7), kxlim=(2, 10))
        self.assertEqual(cropped.cepstral_metadata['center_px'], (3, 3))
        self.assertEqual(cropped.cepstral_metadata['shape'], (6, 8))
        resized = self.ewpc.resize((4, 5), domain='reciprocal')
        np.testing.assert_allclose(resized.cepstral_metadata['center_px'], (1.75, 2.25))
        dp = self.ewpc.get_dp(0, 0).crop(kylim=(1, 7), kxlim=(2, 10))
        self.assertEqual(dp.cepstral_metadata, cropped.cepstral_metadata)
        flipped = fd.HyperData(self.ewpc.array, cepstral_metadata=self.ewpc.cepstral_metadata, flip_axis=(2, 3))
        self.assertEqual(flipped.cepstral_metadata['center_px'], (3, 4))
        self.assertEqual(self.ewpc.cepstral_metadata['center_px'], (4, 5))

    def test_incompatible_array_assignment_clears_cepstral_identity(self):
        data = self.ewpc.copy()
        data.array = np.ones((2, 3, 5, 7))
        self.assertFalse(data.is_cepstral)
        with self.assertRaisesRegex(ValueError, 'cannot|both'):
            fd.HyperData(self.ewpc.array, polar_metadata={}, cepstral_metadata=self.ewpc.cepstral_metadata)

    def test_invalid_log_floor_and_transform_domains(self):
        for floor in (0, -1, np.nan, np.inf, True, 'a', [1], 1j):
            with self.subTest(floor=floor), self.assertRaisesRegex(ValueError, 'a_min'):
                self.data.get_ewpc(a_min=floor)
        for data in (self.data.to_polar(progress=False), self.ewpc):
            with self.assertRaisesRegex(ValueError, 'Cartesian'):
                data.get_ewpc()
        for call in (lambda: self.ewpc.to_polar(progress=False), lambda: self.ewpc.rotate_dps(10)):
            with self.assertRaisesRegex(ValueError, 'cepstral|EWPC|Cartesian'):
                call()

    def test_save_load_chunks_preserve_cepstral_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ewpc.4denoise'
            self.ewpc.save(path)
            loaded = fd.HyperData(path)
            self.assertTrue(loaded.is_cepstral)
            self.assertEqual(loaded.cepstral_metadata, self.ewpc.cepstral_metadata)
            np.testing.assert_array_equal(loaded.array, self.ewpc.array)
            with fd.HyperData.open_hdf5(path) as reader:
                self.assertEqual(reader.get_dp(0, 0).cepstral_metadata, self.ewpc.cepstral_metadata)
                for _, chunk in reader.iter_chunks(1):
                    self.assertEqual(chunk.cepstral_metadata, self.ewpc.cepstral_metadata)
                    self.assertTrue(chunk.is_cepstral)


if __name__ == '__main__':
    unittest.main()
