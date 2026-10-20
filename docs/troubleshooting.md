# Troubleshooting

## A Different Package Version Imports

The active working directory does not choose the Python environment. In the
terminal and notebook, compare `sys.executable` and inspect the imported
module path:

```python
import sys
import fourdenoise
print(sys.executable)
print(fourdenoise.__file__)
```

Activate the intended environment and run `python -m pip install -e .`
from this repository. Select the matching Jupyter kernel, then restart it
to reload edited source code.

## A Raw File Has the Wrong Shape

Binary `.raw` data does not describe its own dimensions or byte order.
Pass `raw_shape`, `raw_dtype`, and `raw_order` explicitly when known. For
the study's EMPAD scan, use the dimensions shown in the
[case study](case-studies/3d-strain-mapping.md). Generic file loading does
not clip values below one or repair NaNs unless requested.

## An HDF5 File Has Multiple Datasets

Pass `hdf5_dataset="/path/to/array"` to select the intended numeric data.
For a multi-gigabyte file, use `HyperData.open_hdf5(...)` and
the format-independent `HyperData.open(...)` rather than an eager
`HyperData(path)` load. Use source reductions, virtual imaging, chunk
iteration, or supported disk-output transforms. See
[I/O and calibration](guides/io-and-calibration.md).

## A Microscopy Format Needs RosettaSciIO

Install `python -m pip install -e ".[rosetta]"` in the active interpreter.
Use `HyperData.open(path)`, not the native eager constructor, for an optional
Rosetta reader; call `source.load(max_memory_mb=...)` only if full loading is
intended. The adapter requires a supported lazy reader and does not fall back
to an eager load. A plugin may need additional format-specific dependencies.

For vendor-specific HDF5 metadata, choose `backend="rosetta"` and a confirmed
`reader`, such as `"hspy"` for a HyperSpy file. Automatic HDF5 dispatch is
native and does not interpret arbitrary vendor metadata. Check the
[reader support boundary](guides/io-and-calibration.md#reader-support-and-validation-boundary)
before assuming a vendor acquisition has been validated.

## Dataset or Axis Selection Is Ambiguous

Native HDF5 uses `hdf5_dataset="/entry/data"`. Rosetta uses `dataset` with
the original result-list index or a unique title; thumbnails are not
renumbered away. Error messages list the alternatives.

An unlabeled array does not establish physical y/x order. Confirm the stored
layout before giving `axis_order`, whose entries list original axes in target
`(Ry, Rx, Ky, Kx)` or `(N, Ky, Kx)` order. Do not guess a permutation just to
silence a warning. See [dataset and axis selection](guides/io-and-calibration.md#dataset-and-axis-selection).

## Calibration Is Missing or Selections Use Unexpected Units

Generic NumPy/HDF5 arrays do not acquire calibration merely from their file
format. Rosetta imports only compatible supported axis calibration; unsupported
units, nonlinear axes, and unequal detector scales remain descriptive with
warnings. Inspect `source.metadata` and `source.reader_metadata`. Supply
verified unit/scale pairs explicitly if needed; do not substitute guessed
beam coordinates or R/Q rotation from vendor field names.

Both source and in-memory `get_dp()` selections default to
`selection_units="pixels"`, even after loading a calibrated source.
Use `"calibrated"` for physical scan coordinates or `"auto"` to choose
calibration when available.
`detector_units` chooses virtual-detector input units; `axis_units` chooses
display units. For streamed peaks, detection/integration settings remain pixels
and `coords_units` selects only the returned coordinate frame. Calibrated peak
coordinates are y-up about the detector midpoint, not shifted to the search center.

## A Source Is Closed or Has No Array Attribute

Perform reads inside `with HyperData.open(path) as source:`. Source methods
borrow the open handle and do not close it early; leaving the context closes
it even on an exception. Reopen the source for later reads. Returned images,
patterns, chunks, and peak results own their data and remain usable.
No public `.array` or implicit NumPy conversion is provided; use a selection,
`iter_chunks`, or explicit full `load()` instead.

## A Peak File Will Not Load as HyperData

Incremental `peaks.h5` files contain peak tables, not diffraction tensors.
Use `PeakScanResult.load(path, max_memory_mb=...)` to materialize the collection,
or h5py `/starts` and `/counts` plus corresponding table slices for a selection.
The [peak guide](guides/peaks-and-strain.md#incremental-peak-files) describes
the schema. Zero counts can mean either no detections or an excluded location;
inspect `processed_mask`. Symmetry-generated rows have NaN measured scores
and intensities by design, not fabricated observations.

## An Unfolded Result Changes Shape

Full-shape traversals retain all scan positions. Hilbert, Morton, Peano,
and block-meander methods may crop to a compatible centered domain or
resize first. Check the emitted warning and `unfold_metadata`. With
`preserve_excess=True`, excluded values are retained for exact undo but
were not included in denoising. With `preserve_excess=False`, undo returns
only the cropped geometry. See [denoising and unfolding](guides/denoising-and-unfolding.md).

## A Large Workflow Exhausts Memory

The mini-data DEMO is the routine smoke test. The experimental scan is
multi-gigabyte and eager loading requires substantially more RAM than the
raw file size once intermediates are allocated. Use bounded HDF5 scan
chunks where a method supports them, avoid keeping multiple full outputs,
and review export cells before running the experimental notebook. Saving
intermediates uses disk space and does not free live arrays from RAM. Not
every `HyperData` method is lazy.

`working_memory_mb` is a temporary-buffer estimate, not a process-RAM cap.
An in-memory peak collection or returned scan image is outside that budget.
Use `get_peaks(destination=...)` for large peak collections and source-to-file
transforms for large diffraction outputs. Global TensorLy fits still require
in-memory data; independent per-chunk fitting would be a different model.

Rosetta may need an entire upstream chunk for one requested pattern. Inspect
`source.reader_chunk_nbytes`; lowering the scan block size cannot shrink a
fixed reader chunk. Increase the budget or use supported plugin-specific
chunk options. Decompression, task graphs, metadata, and caches add memory
beyond declared numerical buffers. A too-small budget fails rather than
silently loading the complete array.

## A Saved Output Already Exists

Streamed writes refuse replacement by default. Choose a new destination or
explicitly pass `overwrite=True` after reviewing the existing output. Default
atomic writes publish each completed file and preserve the prior destination
on processing failure, but need temporary disk space. `atomic=False` trades
that protection for lower disk requirements and can leave partial output.
Never reuse the source path as the destination.

## The Website Does Not Show Local Documentation Changes

Build and inspect the local guide first. The public site follows validated
`main` commits; editing files, committing on another branch, or opening a
pull request does not publish it. The Pages deployment waits for its required
checks. Failed validation leaves the previous site in place. See
[building this guide](getting-started/install.md#build-this-guide).
