"""Private readers for acquisition records, separate from image calibration.

The first supported sidecar is the legacy EMPAD XML schema. Instrument
settings are recorded, never promoted to a detector/scan transform or pixel
calibration. Missing settings remain absent rather than receiving defaults.
"""

from copy import deepcopy
from hashlib import sha256
from math import degrees, isfinite, prod
import os
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET


_MAX_XML_BYTES = 2 * 1024 * 1024


class _NoDTDTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError('Acquisition XML must not contain a DTD or entity declarations.')


def _parse_xml(path):
    """Read a bounded XML document without DTD/entity expansion."""
    with path.open('rb') as file:
        payload = file.read(_MAX_XML_BYTES + 1)
    if len(payload) > _MAX_XML_BYTES:
        raise ValueError('Acquisition XML exceeds the 2 MiB size limit.')
    try:
        root = ET.fromstring(payload, parser=ET.XMLParser(target=_NoDTDTreeBuilder()))
    except ET.ParseError as exc:
        raise ValueError(f'Malformed acquisition XML: {exc}') from exc
    return root, sha256(payload).hexdigest()


def _xml_record(node, depth=0):
    """Retain original values and attributes, including unrecognized fields."""
    if depth > 64:
        raise ValueError('Acquisition XML exceeds the 64-level nesting limit.')
    return {
        'tag': node.tag,
        'attributes': dict(node.attrib),
        'text': (node.text or '').strip(),
        'children': [_xml_record(child, depth + 1) for child in node],
    }


def _number(value, field, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid numeric acquisition field {field!r}: {value!r}.') from exc
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f'Acquisition field {field!r} must be finite'
                         + (' and positive.' if positive else '.'))
    return result


def _read_empad_xml(root):
    """Normalize a recognized EMPAD XML document; return None for other schemas."""
    raw_node = root.find('raw_file')
    if (root.tag != 'root' or raw_node is None
            or not raw_node.get('filename')
            or root.find('iom_measurements') is None
            or root.find('scan_parameters') is None):
        return None

    meta = {
        'format': 'empad_xml',
        'reader_version': 1,
        'raw_file': raw_node.get('filename'),
        'field_sources': {},
        'raw_metadata': _xml_record(root),
    }
    sources = meta['field_sources']

    def text_field(key, value, location):
        if value is not None and str(value).strip():
            meta[key] = str(value).strip()
            sources[key] = {'xml_path': location, 'original_value': value}

    text_field('acquisition_name', root.get('name'), 'root/@name')
    text_field('software_version', root.findtext('software_version'), 'software_version')
    timestamp = root.find('timestamp')
    if timestamp is not None:
        text_field('timestamp', timestamp.get('isoformat'), 'timestamp/@isoformat')
        if timestamp.get('timestamp') is not None:
            meta['timestamp_unix_s'] = _number(timestamp.get('timestamp'), 'timestamp')
            sources['timestamp_unix_s'] = {
                'xml_path': 'timestamp/@timestamp',
                'original_value': timestamp.get('timestamp'), 'source_unit': 's',
            }

    scans = root.findall("scan_parameters[@mode='acquire']")
    if len(scans) > 1:
        raise ValueError('EMPAD XML has multiple acquisition scan_parameters records.')
    if scans:
        values = [scans[0].findtext(f'scan_resolution_{axis}') for axis in ('y', 'x')]
        if any(value is not None for value in values):
            dimensions = [_number(value, f'scan_resolution_{axis}', positive=True)
                          for axis, value in zip(('y', 'x'), values)]
            if any(value != int(value) for value in dimensions):
                raise ValueError('Acquisition scan dimensions must be positive integers.')
            meta['scan_shape'] = tuple(int(value) for value in dimensions)
            sources['scan_shape'] = {
                'xml_path': "scan_parameters[@mode='acquire']/scan_resolution_{y,x}",
                'original_value': values, 'coordinate_order': 'y,x',
            }

    def numeric_field(key, location, unit, conversions, *, positive=False, default_unit=None):
        node = root.find(location)
        if node is None or not (node.text or '').strip():
            return
        source_unit = node.get('units', node.get('unit', default_unit or unit))
        unit_key = source_unit.strip().lower()
        if unit_key not in conversions:
            raise ValueError(f'Unsupported units {source_unit!r} for {location}.')
        value = _number(node.text, location, positive=positive)
        factor = conversions[unit_key]
        converted = _number(value * factor, key, positive=positive)
        meta[key] = converted
        sources[key] = {
            'xml_path': location, 'original_value': node.text.strip(),
            'source_unit': source_unit, 'normalized_unit': unit,
            'conversion_factor': factor,
            'unit_origin': 'xml' if 'unit' in node.attrib or 'units' in node.attrib
                           else 'EMPAD schema',
        }

    # Legacy EMPAD stores this angle in radians, even without a units attribute.
    numeric_field('scan_rotation_deg', 'iom_measurements/scan_rotation', 'deg',
                  {'rad': degrees(1), 'radian': degrees(1), 'radians': degrees(1),
                   'deg': 1, 'degree': 1, 'degrees': 1}, default_unit='rad')
    numeric_field('accelerating_voltage_V', 'iom_measurements/high_voltage', 'V',
                  {'v': 1, 'kv': 1000}, positive=True)
    numeric_field('nominal_camera_length_m', 'iom_measurements/nominal_camera_length', 'm',
                  {'m': 1, 'mm': 1e-3}, positive=True)
    for axis in ('y', 'x'):
        numeric_field(f'nominal_field_of_view_{axis}_m',
                      f'iom_measurements/full_scan_field_of_view/{axis}', 'm',
                      {'m': 1, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9}, positive=True)
    numeric_field('field_of_view_scale_factor',
                  'iom_measurements/full_scan_field_of_view/scale_factor', '1',
                  {'1': 1}, positive=True)
    numeric_field('spot_size_index', 'iom_measurements/spot_size_index', '1', {'1': 1})
    for key in ('exposure_time', 'post_exposure_time'):
        node = root.find(key)
        if node is not None and (node.text or '').strip():
            # Unlabelled exposure units are deliberately not guessed.
            meta[key] = _number(node.text, key)
            meta[f'{key}_units'] = node.get('units', node.get('unit'))
            sources[key] = {'xml_path': key, 'original_value': node.text.strip(),
                            'source_unit': meta[f'{key}_units']}
    return meta


_XML_READERS = (_read_empad_xml,)


def _same_filename(recorded, path):
    name = recorded.replace('\\', '/').rsplit('/', 1)[-1]
    return os.path.normcase(name) == os.path.normcase(path.name)


def _read_sidecar(path, data_path, *, automatic):
    root, checksum = _parse_xml(path)
    if automatic:
        raw = root.find('raw_file')
        if raw is None or not _same_filename(raw.get('filename', ''), data_path):
            return None
    for reader in _XML_READERS:
        metadata = reader(root)
        if metadata is not None:
            break
    else:
        if automatic:
            return None
        raise ValueError(f"'{path}' is not a supported acquisition XML (expected EMPAD).")
    if (data_path is not None and data_path.suffix.lower() == '.raw'
            and not _same_filename(metadata['raw_file'], data_path)):
        raise ValueError(f"'{path}' references {metadata['raw_file']!r}, not '{data_path.name}'.")
    metadata.update({
        'source_path': str(path.resolve()), 'source_sha256': checksum,
        'data_path': str(data_path.resolve()) if data_path is not None else None,
        'association': 'raw_filename_match' if automatic else 'explicit_attachment',
    })
    return metadata


def _load_acquisition_metadata(data_path, metadata_path='auto'):
    """Discover same-directory EMPAD sidecars for raw files, or read an explicit one."""
    if metadata_path is None:
        return {}
    if not isinstance(metadata_path, (str, Path)):
        raise TypeError("metadata_path must be 'auto', a file path, or None.")
    if metadata_path != 'auto':
        path = Path(metadata_path).expanduser()
        if path.suffix.lower() != '.xml':
            raise ValueError('Only EMPAD .xml acquisition sidecars are currently supported.')
        return _read_sidecar(path, data_path, automatic=False)
    if data_path is None or data_path.suffix.lower() != '.raw' or not data_path.is_file():
        return {}
    candidates = []
    for path in sorted(data_path.parent.iterdir()):
        if not path.is_file() or path.suffix.lower() != '.xml':
            continue
        try:
            metadata = _read_sidecar(path, data_path, automatic=True)
        except (ValueError, OSError) as exc:
            warnings.warn(f"Skipping acquisition sidecar '{path}': {exc}",
                          UserWarning, stacklevel=3)
            continue
        if metadata is not None:
            candidates.append(metadata)
    if len(candidates) > 1:
        listing = '\n'.join(f"  {item['source_path']}" for item in candidates)
        raise ValueError('Multiple acquisition sidecars reference this raw file. '
                         'Select metadata_path explicitly, or use metadata_path=None.\n'
                         + listing)
    return candidates[0] if candidates else {}


def _validate_acquisition_shape(metadata, shape):
    """Validate freshly imported sidecars, not inherited historical records."""
    expected = metadata.get('scan_shape')
    if expected is None:
        return
    compatible = (
        (len(shape) == 4 and tuple(shape[:2]) == tuple(expected))
        or (len(shape) == 3 and shape[0] == prod(expected))
        or (len(shape) == 2 and prod(expected) == 1)
    )
    if not compatible:
        raise ValueError(f'Acquisition scan shape {expected} does not match data shape {shape}. '
                         'Check the selected XML and raw_shape; metadata_path=None disables sidecar loading.')


def _copy_acquisition_metadata(value):
    """Ensure independent ownership of the descriptive metadata dictionary."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError('acquisition_metadata must be a dictionary or None.')
    return deepcopy(value)


def _acquisition_summary(metadata):
    """Format normalized fields without dumping the original XML tree."""
    if not metadata:
        return ['Acquisition metadata: not available.']
    lines = ['Acquisition metadata (original acquisition, not current geometry):']
    for key, label, suffix in (
        ('format', 'Format', ''), ('source_path', 'Source', ''),
        ('association', 'Association', ''), ('acquisition_name', 'Acquisition', ''),
        ('timestamp', 'Timestamp', ''), ('software_version', 'Software', ''),
        ('scan_shape', 'Acquired scan shape (Ry, Rx)', ''),
        ('scan_rotation_deg', 'Recorded scan rotation', ' deg'),
        ('accelerating_voltage_V', 'Accelerating voltage', ' V'),
        ('nominal_camera_length_m', 'Nominal camera length', ' m'),
        ('nominal_field_of_view_y_m', 'Nominal field of view y', ' m'),
        ('nominal_field_of_view_x_m', 'Nominal field of view x', ' m'),
        ('field_of_view_scale_factor', 'Recorded FOV scale factor (not applied)', ''),
        ('spot_size_index', 'Spot-size index', ''),
    ):
        if key in metadata:
            value = metadata[key]
            formatted = f'{value:.8g}' if isinstance(value, float) else str(value)
            lines.append(f'  {label}: {formatted}{suffix}')
    for key in ('exposure_time', 'post_exposure_time'):
        if key in metadata:
            unit = metadata.get(f'{key}_units') or 'units unspecified'
            lines.append(f'  {key}: {metadata[key]} ({unit})')
    lines.append('Instrument settings are not a verified R/Q or pixel calibration; '
                 'no calibration or rotation is inferred.')
    return lines
