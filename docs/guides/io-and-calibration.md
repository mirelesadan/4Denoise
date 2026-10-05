# I/O and Calibration

Load an array or a supported file path with `HyperData(...)`. Generic file
loading preserves stored values by default: clipping below one and repairing
NaNs are explicit `clip_on_load` and `repair_nans` choices.

## Raw Detector Data

For a binary file with known layout, provide the full stored shape and dtype:

```python
from fourdenoise import HyperData

data = HyperData(
    "scan.raw", raw_shape=(Ry, Rx, Ky, Kx),
    raw_dtype=">u2", raw_order="C",
)
```

Here `>u2` means big-endian unsigned 16-bit data; use the dtype of your
actual detector file. With an explicit `raw_shape`, detector rows are not
trimmed unless you set `raw_trim_meta=True` and `raw_trim_dims=(Ky, Kx)`.
The [case-study scan](../case-studies/3d-strain-mapping.md) uses float32
data with two stored metadata rows. Rectangular scans need an explicit
`raw_shape`; otherwise the legacy float32 EMPAD reader infers a square scan.

## Acquisition Metadata

For `.raw` files, `HyperData` looks for **legacy EMPAD XML** files in the same
directory by default. A sidecar must explicitly name the raw file; its XML
filename need not match the raw filename. Other metadata formats are not yet
imported automatically.

```python
data = HyperData("scan_x256_y256.raw")  # metadata_path="auto" is the default
print(data.metadata_summary())

# Select a sidecar explicitly if several XML files describe the same raw file.
data = HyperData("scan_x256_y256.raw", metadata_path="acquisition.xml")
angle = data.acquisition_metadata.get("scan_rotation_deg")
voltage = data.acquisition_metadata.get("accelerating_voltage_V")
```

Multiple matching sidecars raise an error listing the candidates. Explicit
paths are resolved from the working directory. A malformed sidecar warns and
is skipped in auto mode; an explicitly selected malformed or unsupported XML
raises an error. Imported acquisition scan dimensions must match the loaded
scan, or the pattern count for a 3D stack. This metadata import does not change
the raw shape, dtype, storage order, or trimming rules described above.
Use `metadata_path=None` to disable external sidecar loading.

The dictionary stores the acquisition timestamp, software version, acquired
`scan_shape` in `(Ry, Rx)` order, scan rotation in degrees, accelerating voltage
in volts, nominal camera length and field of view in meters, and available
exposure information. Missing fields are absent, not zero. Unlabelled EMPAD
exposure units remain unknown. The recorded FOV scale factor is retained but
not applied to infer a scan-pixel size.

`source_path` and `source_sha256` identify the XML source. `field_sources`
records original values, source units, and conversions; `raw_metadata`
preserves the parsed XML tree, including fields the reader does not normalize.
Metadata can contain instrument names and original filesystem paths; review
it before sharing a saved dataset.

**Acquisition settings are not verified calibration.** In particular,
`scan_rotation_deg` does not populate `rq_calibration`, and nominal camera
length or field of view does not set pixel calibration. Explicit user
calibrations remain unchanged. An angle such as the case-study acquisition's
87.276-degree scan setting is not, by itself, a measured detector-to-scan
rotation.

Acquisition metadata describes the **original experiment**, so its recorded
scan shape stays unchanged after cropping, resizing, unfolding, or axis
flips. Derived `HyperData` objects have independently copied metadata;
their current dimensions are in `data.shape` and `data.scan_shape`.

NumPy and generic HDF5 exports do not automatically inherit nearby raw-file
sidecars. You can explicitly attach an XML to an unchanged exported array
(the scan-shape check still applies), or supply a descriptive dictionary:

```python
data = HyperData(array, acquisition_metadata={"experiment": "sample A"})
```

For already cropped/processed arrays, pass a copy of their source object's
`acquisition_metadata` rather than attaching an XML whose full-scan shape no
longer matches. A supplied dictionary takes precedence over saved acquisition
metadata and cannot be combined with an explicit `metadata_path`.

`.save()` retains this record. Reloading a saved HyperData file, including
with `metadata_path=None`, restores embedded acquisition metadata without
requiring the original XML. HDF5 chunks and streamed polar transforms also
preserve it. Older saved files without this field load with an empty dictionary.

## HDF5 and Large Files

`HyperData(path)` loads a selected dataset eagerly. When a generic HDF5
file has multiple datasets, pass `hdf5_dataset="/entry/data"`. For 3D/4D
HDF5 files too large to load at once, read bounded scan blocks:

```python
with HyperData.open_hdf5("experiment.h5") as source:
    pattern = source.get_dp(0, 0)  # 4D scan position
    for scan_slices, block in source.iter_chunks((16, 16)):
        print(scan_slices, block.shape)
```

Each block is an in-memory `HyperData`; operations on one block do not
automatically process the rest of the source. A 3D stack uses one pattern
index and a scalar chunk size.

## Save and Reopen

```python
data.save("processed.4denoise")
reopened = HyperData("processed.4denoise")
```

The save format retains calibration and supported object metadata. Saving
is atomic by default: on overwrite, the old and new files coexist briefly,
so ensure there is enough free disk space. Use `atomic=False` only if that
extra space is unavailable and a partial destination on failure is
acceptable. `requirements.lock.txt` contains machine-specific `file:///`
paths; use `environment.yml` or the editable install on another computer.
