"""EMPAD sidecar association, provenance, and HyperData metadata lifecycle."""

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
import fourdenoise_metadata as fm

if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':
    raise ImportError(f'Wrong fourdenoise module: {fd.__file__}')


def empad_xml(raw='scan.raw', shape=(2, 3), rotation='1.5232616538187775'):
    return f'''<root name="test acquisition">
      <timestamp timestamp="1664142581.207085" isoformat="2022-09-25T16:49:41.207085"/>
      <software_version>1.2.0</software_version><raw_file filename="{raw}"/>
      <scan_parameters mode="search"><scan_resolution_y>99</scan_resolution_y>
        <scan_resolution_x>99</scan_resolution_x></scan_parameters>
      <scan_parameters mode="acquire"><scan_resolution_y>{shape[0]}</scan_resolution_y>
        <scan_resolution_x>{shape[1]}</scan_resolution_x></scan_parameters>
      <exposure_time>1</exposure_time><post_exposure_time>0</post_exposure_time>
      <iom_measurements><scan_rotation>{rotation}</scan_rotation>
        <high_voltage>80000</high_voltage><nominal_camera_length>0.7225</nominal_camera_length>
        <full_scan_field_of_view><x>6.66e-7</x><y>6.67e-7</y><scale_factor>0.72</scale_factor>
        </full_scan_field_of_view><spot_size_index>8</spot_size_index></iom_measurements>
      <unrecognized_setting foo="bar">keep this value</unrecognized_setting>
    </root>'''


class AcquisitionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw_path = self.root / 'scan.raw'
        self.xml_path = self.root / 'acquisition.xml'
        self.values = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
        self.values.tofile(self.raw_path)
        self.xml_path.write_text(empad_xml(), encoding='utf-8')

    def load(self, **kwargs):
        return fd.HyperData(self.raw_path, raw_shape=self.values.shape, **kwargs)

    def test_auto_reads_empad_settings_without_changing_data_or_calibration(self):
        data = self.load()
        meta = data.acquisition_metadata
        np.testing.assert_array_equal(data.array, self.values)
        self.assertEqual(meta['scan_shape'], (2, 3))
        self.assertAlmostEqual(meta['scan_rotation_deg'], 87.2764638579338)
        self.assertEqual(meta['accelerating_voltage_V'], 80000)
        self.assertEqual(meta['nominal_camera_length_m'], .7225)
        self.assertEqual(meta['nominal_field_of_view_y_m'], 6.67e-7)
        self.assertEqual(meta['exposure_time'], 1)
        self.assertIsNone(meta['exposure_time_units'])
        self.assertEqual(meta['timestamp'], '2022-09-25T16:49:41.207085')
        self.assertEqual(meta['source_path'], str(self.xml_path.resolve()))
        self.assertEqual(meta['association'], 'raw_filename_match')
        self.assertEqual(len(meta['source_sha256']), 64)
        self.assertIsNone(data.rq_calibration)
        self.assertIsNone(data.real_conv_factor)
        self.assertIsNone(data.reciprocal_conv_factor)
        source = meta['field_sources']['scan_rotation_deg']
        self.assertEqual(source['original_value'], '1.5232616538187775')
        self.assertEqual(source['source_unit'], 'rad')
        self.assertEqual(source['unit_origin'], 'EMPAD schema')
        original_nodes = meta['raw_metadata']['children']
        extra = next(node for node in original_nodes if node['tag'] == 'unrecognized_setting')
        self.assertEqual(extra['attributes'], {'foo': 'bar'})
        self.assertEqual(extra['text'], 'keep this value')

    def test_explicit_units_are_converted_and_original_values_retained(self):
        xml = empad_xml(rotation='87.3').replace('<scan_rotation>', '<scan_rotation units="deg">')
        xml = xml.replace('<high_voltage>80000', '<high_voltage unit="kV">80')
        xml = xml.replace('<exposure_time>1', '<exposure_time units="ms">1')
        self.xml_path.write_text(xml, encoding='utf-8')
        meta = self.load(metadata_path=self.xml_path).acquisition_metadata
        self.assertEqual(meta['scan_rotation_deg'], 87.3)
        self.assertEqual(meta['accelerating_voltage_V'], 80000)
        self.assertEqual(meta['exposure_time_units'], 'ms')
        self.assertEqual(meta['field_sources']['accelerating_voltage_V']['original_value'], '80')
        self.assertEqual(meta['field_sources']['scan_rotation_deg']['unit_origin'], 'xml')

    def test_explicit_calibrations_are_not_overwritten(self):
        data = self.load(real_units='nm', real_conv_factor=(2, 3),
                         reciprocal_units='mrad', reciprocal_conv_factor=.4,
                         rq_calibration=fd.RQCalibration(rotation_deg=12))
        self.assertEqual(data.real_conv_factor, (2, 3))
        self.assertEqual(data.reciprocal_conv_factor, .4)
        self.assertEqual(data.rq_calibration.rotation_deg, 12)

    def test_missing_sidecar_and_disabled_discovery_are_empty(self):
        self.assertEqual(self.load(metadata_path=None).acquisition_metadata, {})
        self.xml_path.unlink()
        self.assertEqual(self.load().acquisition_metadata, {})

    def test_auto_ignores_nonmatching_and_unsupported_xml(self):
        self.xml_path.write_text(empad_xml(raw='different.raw'), encoding='utf-8')
        (self.root / 'settings.xml').write_text('<root><other>data</other></root>', encoding='utf-8')
        self.assertEqual(self.load().acquisition_metadata, {})
        with self.assertRaisesRegex(ValueError, 'references.*different.raw'):
            self.load(metadata_path=self.xml_path)
        with self.assertRaisesRegex(ValueError, 'not a supported acquisition XML'):
            self.load(metadata_path=self.root / 'settings.xml')

    def test_auto_does_not_attach_raw_metadata_to_numpy_exports(self):
        path = self.root / 'scan.npy'
        np.save(path, self.values)
        self.assertEqual(fd.HyperData(path).acquisition_metadata, {})
        attached = fd.HyperData(path, metadata_path=self.xml_path)
        self.assertEqual(attached.acquisition_metadata['association'], 'explicit_attachment')

    def test_multiple_matching_sidecars_require_explicit_selection_before_reading_data(self):
        (self.root / 'second.xml').write_text(empad_xml(), encoding='utf-8')
        with patch.object(fd, 'read_4D') as loader:
            with self.assertRaisesRegex(ValueError, 'Multiple acquisition sidecars') as error:
                self.load()
            loader.assert_not_called()
        self.assertIn('second.xml', str(error.exception))
        self.assertEqual(self.load(metadata_path=self.xml_path).acquisition_metadata['format'], 'empad_xml')

    def test_absolute_recorded_raw_path_can_match_relocated_basename(self):
        self.xml_path.write_text(empad_xml(raw='/acquisition/computer/scan.raw'), encoding='utf-8')
        self.assertEqual(self.load().acquisition_metadata['scan_shape'], (2, 3))

    def test_missing_optional_fields_remain_unknown(self):
        self.xml_path.write_text('<root><raw_file filename="scan.raw"/>'
                                 '<scan_parameters mode="acquire"/><iom_measurements/></root>', encoding='utf-8')
        metadata = self.load().acquisition_metadata
        self.assertNotIn('scan_rotation_deg', metadata)
        self.assertNotIn('scan_shape', metadata)
        self.assertNotIn('exposure_time', metadata)

    def test_dimension_mismatch_is_not_silently_accepted(self):
        self.xml_path.write_text(empad_xml(shape=(3, 2)), encoding='utf-8')
        for mode in ('auto', self.xml_path):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'scan shape.*does not match'):
                self.load(metadata_path=mode)
        stack = fd.HyperData(self.values.reshape(6, 8, 8), metadata_path=self.xml_path)
        self.assertEqual(stack.acquisition_metadata['scan_shape'], (3, 2))

    def test_malformed_xml_warns_in_auto_and_raises_if_explicit(self):
        self.xml_path.write_text('<root>', encoding='utf-8')
        with self.assertWarnsRegex(UserWarning, 'Skipping acquisition sidecar'):
            self.assertEqual(self.load().acquisition_metadata, {})
        with self.assertRaisesRegex(ValueError, 'Malformed acquisition XML'):
            self.load(metadata_path=self.xml_path)

    def test_dtd_entities_and_large_xml_are_rejected(self):
        for encoding in ('utf-8', 'utf-16'):
            xml = '<?xml version="1.0" encoding="' + encoding + '"?>'
            xml += '<!DOCTYPE root [<!ENTITY test "value">]><root>&test;</root>'
            self.xml_path.write_bytes(xml.encode(encoding))
            with self.subTest(encoding=encoding), self.assertRaisesRegex(ValueError, 'DTD'):
                self.load(metadata_path=self.xml_path)
        self.xml_path.write_bytes(b'x' * (fm._MAX_XML_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'size limit'):
            self.load(metadata_path=self.xml_path)

    def test_invalid_numbers_units_paths_and_parameter_types(self):
        for rotation in ('nan', 'inf', 'text'):
            self.xml_path.write_text(empad_xml(rotation=rotation), encoding='utf-8')
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                self.load(metadata_path=self.xml_path)
        self.xml_path.write_text(empad_xml().replace('<scan_rotation>', '<scan_rotation units="turns">'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unsupported units'):
            self.load(metadata_path=self.xml_path)
        with self.assertRaises(FileNotFoundError):
            self.load(metadata_path=self.root / 'missing.xml')
        with self.assertRaisesRegex(ValueError, 'Only EMPAD'):
            self.load(metadata_path='settings.json')
        with self.assertRaisesRegex(TypeError, 'metadata_path'):
            self.load(metadata_path=True)
        with self.assertRaisesRegex(TypeError, 'acquisition_metadata'):
            self.load(acquisition_metadata=[])
        with self.assertRaisesRegex(ValueError, 'not both'):
            self.load(acquisition_metadata={}, metadata_path=self.xml_path)

    def test_processing_preserves_original_record_with_independent_ownership(self):
        data = self.load()
        original = data.acquisition_metadata
        outputs = (
            data.copy(), data.clip(a_min=5), data.crop(ylim=(0, 1), xlim=(1, 2)),
            data.resize((1, 2), domain='real'), data.bin_data(domain='real'),
            data.denoise('median', domain='reciprocal', window_size=3),
            data.unfold(domain='real'), data.reshape(6, 8, 8),
            data.to_polar(output_shape=(4, 12), progress=False),
        )
        for transformed in outputs:
            with self.subTest(shape=transformed.shape):
                self.assertEqual(transformed.acquisition_metadata, original)
                self.assertEqual(transformed.acquisition_metadata['scan_shape'], (2, 3))
                transformed.acquisition_metadata['field_sources']['scan_shape']['original_value'][0] = 'changed'
                self.assertEqual(original['field_sources']['scan_shape']['original_value'][0], '2')
        data.array = np.zeros((4, 4))
        self.assertEqual(data.acquisition_metadata, original)

    def test_axis_flip_does_not_rewrite_the_recorded_scan_rotation(self):
        data = self.load(flip_axis=(0, 3))
        np.testing.assert_array_equal(data.array, self.values[::-1, :, :, ::-1])
        self.assertAlmostEqual(data.acquisition_metadata['scan_rotation_deg'], 87.2764638579338)

    def test_elliptical_correction_preserves_acquisition_record(self):
        data = self.load()
        with patch.object(fd.HyperData, '_extract_ellipse', return_value=(0, 2, 2)):
            corrected = data.fix_elliptical_distortions(r=1, R=3)
        self.assertEqual(corrected.acquisition_metadata, data.acquisition_metadata)

    def test_manual_dictionary_is_copied_and_can_replace_saved_metadata(self):
        record = {'experiment': {'name': 'my experiment'}}
        data = fd.HyperData(self.values, acquisition_metadata=record)
        record['experiment']['name'] = 'changed'
        self.assertEqual(data.acquisition_metadata['experiment']['name'], 'my experiment')
        path = self.root / 'test.4denoise'
        data.save(path)
        self.assertEqual(fd.HyperData(path, acquisition_metadata={}).acquisition_metadata, {})

    def test_save_reload_and_chunks_do_not_require_the_original_sidecar(self):
        data = self.load().crop(ylim=(0, 1), xlim=(1, 2))
        path = self.root / 'saved.4denoise'
        data.save(path)
        expected = data.acquisition_metadata
        self.xml_path.unlink()
        loaded = fd.HyperData(path, metadata_path=None)
        self.assertEqual(loaded.acquisition_metadata, expected)
        self.assertNotEqual(loaded.scan_shape, expected['scan_shape'])
        with fd.HyperData.open_hdf5(path) as source:
            for _, block in source.iter_chunks((1, 1)):
                self.assertEqual(block.acquisition_metadata, expected)
        target = self.root / 'polar.4denoise'
        fd.HyperData.to_polar_hdf5(path, target, chunk_shape=1,
                                 output_shape=(4, 8), progress=False)
        self.assertEqual(fd.HyperData(target).acquisition_metadata, expected)

    def test_old_saved_files_without_acquisition_metadata_still_load(self):
        path = self.root / 'old.4denoise'
        fd.HyperData(self.values).save(path)
        with h5py.File(path, 'r+') as file:
            del file['metadata/acquisition_metadata']
        self.assertEqual(fd.HyperData(path).acquisition_metadata, {})
        with fd.HyperData.open_hdf5(path) as source:
            _, block = next(source.iter_chunks(1))
            self.assertEqual(block.acquisition_metadata, {})

    def test_summary_distinguishes_acquisition_from_current_calibration(self):
        summary = self.load().resize((1, 2), domain='real').metadata_summary()
        self.assertIn('Current data shape: (1, 2, 8, 8)', summary)
        self.assertIn('Acquired scan shape (Ry, Rx): (2, 3)', summary)
        self.assertIn('87.276464 deg', summary)
        self.assertIn('units unspecified', summary)
        self.assertIn('not a verified R/Q', summary)
        self.assertIn('not available', fd.HyperData(self.values).metadata_summary())


if __name__ == '__main__':
    unittest.main()
