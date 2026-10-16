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

## Disk-Backed Sources and Large Files

`HyperData(path)` loads a selected dataset eagerly. Use `HyperData.open(path)`
instead when a nonempty 3D stack or 4D scan should stay on disk. It returns a
read-only `DiffractionSource` inside a `with` block. Supported sources are
memory-mapped `.npy` arrays, generic `.h5`/`.hdf5`/`.hdf` files, and saved
`.4denoise` files. Other microscopy formats can use the optional RosettaSciIO
backend described below. `HyperData(path)` remains the native eager loader;
use `open(...).load()` when importing a RosettaSciIO file into RAM.

When generic HDF5 has multiple datasets, pass `hdf5_dataset="/entry/data"`.
Saved HyperData files automatically use `/array`. NumPy files contain one
array and do not accept an HDF5 dataset selector. Object arrays, `.npz`
archives, empty arrays and dimensions other than 3D/4D are rejected.

```python
with HyperData.open("experiment.h5") as source:
    print(source.shape, source.dtype, source.nbytes)  # no detector pixels read
    print(source.metadata)  # independent descriptive/calibration record
    pattern = source.get_dp(0, 0)  # 4D scan position
    mean_pattern = source.get_dp(operation="mean", working_memory_mb=64)
    regional_std = source.get_dp(
        y=(10, 30), x=(20, 50), operation="std", working_memory_mb=64,
    )
    for scan_slices, block in source.iter_chunks((16, 16), working_memory_mb=64):
        print(scan_slices, block.shape)
```

`HyperData.open_hdf5()` uses this same source interface and remains available
for HDF5-specific workflows. Both entry points close their owned resource
when leaving the `with` block, including on exceptions. `source.close()` is
also safe to call explicitly and repeatedly. Subsequent reads raise an error;
already returned patterns and blocks remain usable because they own their
data. Their modifications cannot write back to the original file.

Each block is an in-memory `HyperData`; operations on one block do not
automatically process the rest of the source. A 3D stack uses one pattern
index and a scalar or one-tuple chunk size; it does not invent a 2D scan grid.
For 4D chunks, `real_origin` accounts for the chunk's position in the original
scan, including signed or unequal axis spacing. `chunk_shape` is a maximum:
the working-memory budget can reduce it. Do not collect all chunks into a
list for a dataset too large to fit in RAM.

The reader's `get_dp()` also supports Boolean scan masks, calibrated scan
limits, and mean/std/min/max reductions using the same bounded reducer as
in-memory `HyperData`. Returned patterns retain saved detector calibration
and geometry and remain usable after the reader closes. Exact median is
explicitly rejected on this bounded reader; load data into `HyperData` when
an exact, potentially memory-intensive median is required.

Generic HDF5 and NumPy files have no assumed calibration. Set it explicitly
when opening, without modifying the file:

```python
with HyperData.open(
    "experiment.npy",
    real_units="nm", real_conv_factor=(2.0, 3.0), real_origin=(10.0, 20.0),
    reciprocal_units="mrad", reciprocal_conv_factor=0.5,
) as source:
    calibrated_pattern = source.get_dp(14.0, 29.0, selection_units="calibrated")
```

Specify both units and conversion factor for a domain. Saved HyperData files
restore their calibration and acquisition, R/Q, beam, polar and cepstral
records automatically; explicit calibration pairs override stored pairs only
in returned objects. Original acquisition metadata remains descriptive.
Cartesian beam metadata retains its pixel center/radius and updates its
calibrated labels when detector calibration is overridden. Reciprocal
calibration overrides are rejected for saved polar/cepstral data, because
those transformed axes have their own coordinate records; omit the overrides
to preserve the saved coordinate system.

### Streamed Virtual Images

Integrate a detector over a complete 4D scan while keeping the diffraction
patterns on disk. This works for native NumPy/HDF5 sources and the optional
RosettaSciIO backend:

```python
with HyperData.open("processed.4denoise") as source:
    image = source.virtual_image(
        annulus=(10, 35), detector_units="pixels",
        working_memory_mb=64, show=False,
    )

# The returned RealSpace owns its scan-sized array and keeps scan calibration.
image.show(axis_units="auto")
```

The inputs and display options match `HyperData.virtual_image()`. Choose one
of an annulus, a disk (or union of disks) using `radius` and `centers`, or a
Boolean/signed/fractional detector `mask`. `detector_units="auto"` uses saved
reciprocal calibration when available; specify `"pixels"` for pixel radii and
positions. `axis_units` chooses real-space image units; `units` chooses the
optional mean-detector preview units. Polar data accepts radial annuli and
wrapped angular `theta_range`, but not Cartesian disks.

The detector is built once, then basic scan blocks are integrated in a single
pass. Both eager and streamed methods use at least float64 products/sums,
without a full-size weighted tensor. `plot_mask=True` or
`return_detector=True` accumulates the mean diffraction pattern in that same
pass; otherwise no mean is computed. The latter returns
`(image, masked_mean_detector)`; polar detectors retain their coordinate record.
Overlapping disks count each pixel once, weights are not normalized, and
NaN/Inf propagate like ordinary NumPy multiplication/sums.
Complex amplitudes are rejected; supply real-valued intensities.

`working_memory_mb` bounds numerical input/work buffers (default 256 MiB).
Rosetta sources also reserve one declared reader chunk. The compact returned
scan image, detector construction, metadata, caches and visualization are
outside that budget. An insufficient budget raises before pixel reads. A
3D stack is rejected: folding it into a 2D scan requires an explicit scan shape.

### Explicit Full Loading

```python
with HyperData.open("processed.4denoise") as source:
    # Deliberately allocate the complete numerical array, capped here at 4 GiB.
    data = source.load(max_memory_mb=4096, working_memory_mb=64)

# Independent HyperData remains usable after source closure.
denoised = data.denoise("median", domain="reciprocal", window_size=3)
```

`load()` preserves dtype and saved metadata, including unfolding undo data.
Opening a source does **not** read large undo payloads; they are deferred until
full loading. Partial chunks do not carry those payloads and cannot undo a
full-source traversal. `source.has_unfold_metadata` reports their presence.

`max_memory_mb` caps the numerical output allocation and fails before pixel
reads if too small. None allows full loading without a size cap.
`working_memory_mb` separately bounds temporary numerical read/copy buffers
or reduction workspaces (256 MiB by default). Metadata, preserved undo values,
OS/HDF5 caches, allocator overhead and objects retained by the caller are
excluded; these options are not a process-RAM limit. NumPy memory mapping
also does not prevent the OS from caching pages while data is processed.

Disk-backed **loading** is not lazy execution of every analysis method. A
source has no public `array` and rejects implicit NumPy conversion; use
`get_dp`, `virtual_image`, `iter_chunks`, or explicit `load`. Global tensor decompositions still
require a full in-memory input, not independent chunk-by-chunk fitting.

## RosettaSciIO Microscopy Formats

Install the optional backend into the environment used by your notebook:

```bash
python -m pip install -e ".[rosetta]"
```

RosettaSciIO readers provide data, axes and metadata through a
[common reader interface](https://hyperspy.org/rosettasciio/api/index.html).
4Denoise currently enables lazy-capable plugins for Digital Micrograph,
EMD/DE5, HyperSpy, EMPAD, FEI, MRC, Blockfile, Ripple, Quantum Detector,
TIFF, TVIPS and NumPy. This is a curated initial subset, not a promise to
read every RosettaSciIO format, every vendor variant, or 2D images. Reader
errors are not hidden and unsupported lazy readers have no eager fallback.
Actual EMD, HyperSpy and Ripple fixtures are exercised by the optional tests;
other vendor acquisitions still need validation against your files.

```python
with HyperData.open("scan.dm4") as source:
    print(source.shape, source.dtype, source.reader_chunk_nbytes)
    print(source.reader_metadata["axes"])
    mean_pattern = source.get_dp(operation="mean", working_memory_mb=64)
    # Explicitly load everything only when RAM permits it.
    data = source.load(max_memory_mb=4096, working_memory_mb=64)
```

`backend="auto"` uses native readers for `.npy` and native/generic HDF5
extensions, and RosettaSciIO otherwise. To interpret vendor-specific HDF5
metadata, select the Rosetta backend explicitly rather than treating the file
as an arbitrary HDF5 array:

```python
with HyperData.open("vendor.hdf5", backend="rosetta", reader="hspy") as source:
    pattern = source.get_dp(0, 0)
```

`reader` accepts a registered plugin name, alias, or module suffix, such as
`"emd"` or `"digitalmicrograph"`. Ambiguous extensions report candidate
readers instead of selecting one silently. `backend="native"` forbids
Rosetta dispatch. Existing `open_hdf5()` remains a native-only entry point.

### Dataset and Axis Selection

Rosetta may return several datasets, including thumbnails. Only nonempty
numeric 3D/4D arrays are candidates. One suitable array is selected
automatically; multiple candidates raise an error listing every returned
dataset's original index, title, shape, dimensionality and dtype. Pass
`dataset=1` or `dataset="unique title"` to choose. Indices refer to the full
reader result list, including thumbnails, not a renumbered candidate list.
`hdf5_dataset` is for native HDF5; Rosetta plugin selectors such as
`dataset_path` belong in `reader_kwargs` when supported.

Axis dictionaries are interpreted using `index_in_array`, not their list
order. Automatic mapping requires explicit `Ry/Rx/Ky/Kx` labels, or clear
navigation/signal roles plus y/x names. Ambiguous axes raise an error showing
their labels and navigation flags. Confirm the raw storage layout before
specifying `axis_order`; do not assume row-major storage proves a physical
y/x assignment.

```python
# Only for a confirmed original layout (Rx, Ry, Kx, Ky):
with HyperData.open("scan.emd", axis_order=(1, 0, 3, 2)) as source:
    data = source.load(max_memory_mb=4096)
```

The permutation lists original array indices in target
`(Ry, Rx, Ky, Kx)` or `(N, Ky, Kx)` order. Reordering happens on selected
blocks; it never requires transposing or loading the complete source.
Source `get_dp()` selections default to pixel indices; request
`selection_units="calibrated"` for physical scan coordinates. After `load()`,
the resulting HyperData follows its existing `selection_units="auto"` default,
which uses imported calibration when present. Use `selection_units="pixels"`
explicitly when selecting scan indices on that loaded object.

### Calibration and Metadata

Uniform, compatible spatial scan units and signed axis spacings/offsets are
imported into `real_units`, `real_conv_factor` and `real_origin`. Common length
units can be converted between the two scan axes. Isotropic reciprocal-space
pixel-size magnitudes in inverse-length or angular units are imported as one
positive `reciprocal_conv_factor`. Unsupported units, nonlinear axes and
unequal detector pixel scales warn and remain descriptive; supply explicit
calibration overrides if appropriate.

Detector offsets and spacing signs are preserved in the original axis record,
but are **not** assumed to locate the central beam or define an R/Q transform.
Diffraction display coordinates retain 4Denoise's centered, y-up Cartesian
convention. Use alignment or explicit beam metadata to establish a beam
center, and independent R/Q calibration to establish the scan/detector frame.

Original axes, normalized reader metadata and vendor metadata are archived in
`acquisition_metadata`, along with reader/version, original dataset index,
source shape, applied permutation and calibration notes. The archive preserves
slash-containing vendor keys, numeric arrays, binary strings and timestamps
without pickle, and survives full loading and `.save()`/reload.

```python
from fourdenoise import unpack_reader_metadata

reader_record = unpack_reader_metadata(data.acquisition_metadata)
original_axes = reader_record["axes"]
vendor_metadata = reader_record["original_metadata"]
```

`source.reader_metadata` provides the same decoded defensive copy. Metadata
decoding works for saved files even when the optional RosettaSciIO/Dask
packages are not installed. Original acquisition records are descriptive;
cropping or resizing does not rewrite the vendor's original axes.

### Lazy Reader Memory

Rosetta readers are always called with `lazy=True`. Selected Dask slices are
computed synchronously, not with unconstrained parallel readers. Nevertheless,
a small selection can read an entire upstream chunk. Inspect
`source.reader_chunk_nbytes`: its largest declared chunk is reserved before
4Denoise budgets managed copy/reduction arrays. A budget too small for that
chunk fails before selection computation. Prefer smaller plugin chunks when
the reader supports them:

```python
with HyperData.open(
    "scan.dm4", reader_kwargs={"chunks": (1, 1, 128, 128)},
) as source:
    mean_pattern = source.get_dp(operation="mean", working_memory_mb=64)
```

`reader_kwargs` is plugin-specific; 4Denoise controls `filename` and `lazy`
and rejects attempts to override them or request writable file/memory-map
modes. Automatic Dask chunking uses a 16 MiB
target during reader construction, but plugins may have fixed or stored
chunking. That target is not an allocation guarantee. Task graphs, decompression
and other library workspaces, metadata and OS caches remain outside the budget.
The source closes discoverable owned file handles, blocks all later reads, and
keeps returned patterns and chunks independent of those resources.

## Memory-Budgeted Resizing

`HyperData.resize()` and `RealSpace.resize()` use one tiled numerical engine:

```python
resized = data.resize(
    (64, 64), domain="real", method="area", working_memory_mb=64,
)
```

`area` averages overlapping pixel/scan bins, including fractional weights for
non-integer reductions. It preserves constant intensity, not total summed
counts. `linear` and `nearest` also support upsampling. Bins and interpolation
halos are defined in global coordinates, so changing the block size does not
introduce seams or change which neighboring diffraction patterns contribute.

Floating/complex outputs retain their dtype; averaging and linear interpolation
of integer/Boolean data produce float64. Accumulators use at least float64
precision. Nearest sampling and completely unchanged resizes retain the exact
source dtype/values, including large integers. Results own independent storage.

The default `working_memory_mb=None` uses a 256 MiB **temporary-array** budget.
Resident source and complete returned output arrays, allocator overhead, and
HDF5 caches are excluded. A budget cannot make a multi-GB output fit in RAM;
use the disk-output path instead. Too-small budgets raise an informative error
before allocating the complete output.

```python
saved_path = HyperData.resize_hdf5(
    "experiment.h5", "resized.4denoise", (64, 64), domain="real",
    method="area", working_memory_mb=64, progress=True,
)
with HyperData.open_hdf5(saved_path) as resized_source:
    mean_pattern = resized_source.get_dp(working_memory_mb=64)
```

The source and output stay on disk; output tiles are written directly to HDF5.
For a generic input with several arrays, specify `hdf5_dataset="/entry/data"`.
Saved HyperData inputs retain their calibration and acquisition/R-Q records;
generic inputs are not assumed to have calibration. Real-space spacing and
origin, beam/cepstral geometry, and polar radial/angular sampling receive the
same updates as in-memory resizing. Polar interpolation remains periodic for
full-circle data. Unfolding undo metadata is not attached to resized outputs.

Writes are atomic by default: failure leaves an existing destination intact
and removes the temporary file. Replacement requires `overwrite=True`; source
and destination must be different files. The return value is a path, not a
fully loaded `HyperData`. `HyperData(saved_path)` deliberately loads the whole
result into RAM. Atomic overwrites need space for both old and new outputs.

## Streamed Cartesian and Background Processing

Saved polar data can be converted without holding the full Cartesian output:

```python
cartesian_path = HyperData.to_cartesian_hdf5(
    "polar.4denoise", "cartesian.4denoise",
    output_shape="original", working_memory_mb=256,
)
background_path = HyperData.remove_bg_hdf5(
    "cartesian.4denoise", "background-corrected.4denoise",
    background=background_dp, bg_frac=0.8, a_min=None,
    working_memory_mb=64,
)
```

`to_cartesian_hdf5()` requires saved polar metadata and uses the same radius,
angle, interpolation, support-fill and output-calibration rules as
`to_cartesian()`. None chooses `(2*n_r, 2*n_r)`; `"original"` uses its recorded
Cartesian canvas. `remove_bg_hdf5()` accepts a fixed 2D background matching the
diffraction axes, not a scan-shaped background stack. Estimate that background
explicitly first; inferred global/residual backgrounds are not recomputed on
independent blocks.

Both methods preserve global scan origin, acquisition and R/Q records, use
one optional outer progress bar, and return paths rather than full arrays.
The default is atomic publication with explicit `overwrite=True` for existing
destinations. An insufficient one-pattern workspace budget fails before the
probe transform. Coordinate/interpolation workspaces, conversion and input/
output blocks are budgeted; caller data/background, metadata, allocator
overhead and HDF5 caches are excluded. The estimate is conservative, so the
required minimum may exceed measured allocations. Use `open_hdf5()` for
bounded access to the saved result.

Only independent per-pattern transforms are streamed this way. Splitting
global tensor decompositions into independent chunks changes their mathematical
meaning and is deliberately not provided as an automatic denoising shortcut.

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
