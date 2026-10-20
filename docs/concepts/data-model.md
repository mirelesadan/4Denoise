# Data Model

`HyperData` treats the final two axes as an image or diffraction pattern.
For a 4D-STEM scan, the convention is `(Ry, Rx, Ky, Kx)`: two real-space
scan coordinates followed by two reciprocal-space detector coordinates.

| Input | `scan_shape` | `pattern_shape` | `real_shape` |
| --- | --- | --- | --- |
| 2D image | `()` | `(Ky, Kx)` | `None` |
| 3D stack | `(N,)` | `(Ky, Kx)` | `None` |
| 4D scan | `(Ry, Rx)` | `(Ky, Kx)` | `(Ry, Rx)` |

A 3D stack does not retain a 2D scan layout unless you provide it when
reshaping or undoing a row-major unfold. `real_units` and
`real_conv_factor` describe the scan axes; `reciprocal_units` and
`reciprocal_conv_factor` describe the detector axes. Conversion factors
are units per pixel. `real_conv_factor` can also be `(y, x)` for unequal
scan-axis steps, and `real_origin` locates scan pixel `(0, 0)`.

Scan axes and detector axes are not necessarily physically aligned. A strain
map's `(Ry, Rx)` layout locates the measurement, while `exx` and `eyy` describe
directions in the fitted tensor basis. The default
`get_strains(output_frame="reciprocal", ang=0)` reports detector kx and ky.
Use `output_frame="real"` to report Cartesian real-space components using
stored `rq_calibration`, including its rotation and optional reflection.
The result records its output frame and the calibration used. Converting
the result to `RealSpace` displays those components at the original scan
positions without applying another tensor rotation. A recorded acquisition
scan rotation does not automatically establish R/Q calibration.
See [peaks and strain](../guides/peaks-and-strain.md) for this distinction.

## Choose the Object for the Job

| Object | What it contains | Typical return path |
| --- | --- | --- |
| `HyperData` | A complete in-memory diffraction array and its metadata | `HyperData(path)`, `source.load()`, or a returned chunk |
| `DiffractionSource` | Read-only file-backed access, not a public `.array` | `HyperData.open(path)` inside a `with` block |
| `ReciprocalSpace` | One in-memory diffraction pattern or reduction | `data.get_dp()` or `source.get_dp()` |
| `RealSpace` | One in-memory 2D scan image with scan calibration | A virtual image or a strain component view |
| `PeakScanResult` | Ragged per-pattern coordinates, scores, provenance, and optional intensities | `source.get_peaks()` without a destination, or `PeakScanResult.load(path)` |

`PeakScanResult` is not itself a `RealSpace` image: each scan position can
contain a different number of peaks. Its entries are NumPy **object arrays**
of shape `(Ry, Rx)` or `(N,)`; `coords[index]` is an `(n_peaks, 2)` array.
`observed_peak_counts` is a scan-shaped count array, while `processed_mask`
distinguishes excluded positions from evaluated patterns with no peaks.
Scores measure template response; optional intensities are raw integrated
counts. Neither peak count nor template response is a crystalline fraction.
See [peak collections](../guides/peaks-and-strain.md#peak-extraction-from-disk-backed-sources).

## Current Geometry and Original Acquisition Records

Active geometry determines how present pixels are interpreted: scan spacing
and origin, reciprocal sampling, the stored beam center, polar/cepstral axes,
and verified R/Q calibration. Crop, resize, flips, and diffraction rotation
update these together or invalidate records that no longer describe the
result. A current stored beam center takes precedence over historical
mean-fit coordinates; stale shapes are rejected rather than used silently.

`acquisition_metadata` instead describes the original experiment. Its
recorded shape, vendor axes, and instrument settings do not become the
geometry of a later crop. Compatible RosettaSciIO axis calibration can be
imported, but detector offsets are not assumed to identify the beam, and a
vendor rotation field does not automatically establish an R/Q relationship.
Explicit calibration overrides apply to returned objects, not the source file.
See [metadata interpretation](../guides/io-and-calibration.md#active-calibration-versus-acquisition-records).

Value-only methods generally return a new object rather than altering the
input. Geometric operations can change calibration or invalidate metadata
that no longer describes the result. Inspect the method's return value and
metadata before chaining a shape-changing operation.

## In-Memory Objects and Disk-Backed Sources

`HyperData(path)` loads pixels into RAM. `HyperData.open(path)` instead owns
a read-only `DiffractionSource` for a 3D/4D file within a `with` block. Its
shape, dtype and descriptive/calibration metadata are available without
reading the detector array. `get_dp()` returns a pattern or bounded reduction;
`iter_chunks()` returns independent, calibrated in-memory HyperData blocks.
Only `load()` deliberately materializes the complete array, including saved
unfolding undo payloads. Closing the source invalidates future reads but not
objects already returned.

An output is not necessarily small merely because the input is disk-backed:

| Operation on a source | Return value | Retained numerical output |
| --- | --- | --- |
| `get_dp()` | `ReciprocalSpace` | One pattern/reduction |
| `virtual_image()` | `RealSpace` | One complete 2D scan image; 4D scans only |
| `get_peaks()` | `PeakScanResult` | The entire ragged peak collection |
| `get_peaks(destination=...)` | Saved path | Incrementally written peak tables |
| `to_polar`, `to_cartesian`, `resize`, `remove_bg` with `destination` | Saved path | Diffraction output written to disk |
| `load()` | `HyperData` | The entire diffraction array |

The source-to-file background operation requires a fixed 2D background;
global background estimation is not silently repeated on separate chunks.
`working_memory_mb` limits estimated temporary numerical buffers, **not total
process RAM**. Retained results, metadata, caches, allocator overhead, and
some reader internals are outside that estimate. Rosetta sources additionally
reserve a declared reader chunk. `load(max_memory_mb=...)` separately caps
the full numerical array; `PeakScanResult.load(max_memory_mb=...)` caps its
numeric tables and object-pointer arrays, not all Python object overhead.

This separation prevents implicit full-array conversions from looking like
lazy analysis. Global tensor decompositions are still in-memory operations.
See [I/O and calibration](../guides/io-and-calibration.md) for formats,
memory-budget semantics and explicit full loading.

## Saved Diffraction Data and Peak Files

`data.save("processed.4denoise")` and source-to-file transforms save pixels
plus supported metadata. Reopen with `HyperData.open(...)` for bounded access
or `HyperData(...)` to load all pixels. A peak file from
`source.get_peaks(destination="peaks.h5")` has a separate
`4denoise.peak_scan` schema; use `PeakScanResult.load`, not `HyperData`.
Its numeric tables also allow selected h5py slices without loading all peaks.
No pickle is used for these saved metadata/peak records.

File-output operations are atomic by default and replacement is explicit.
Atomicity applies to each output file, not a transaction across several
workflow steps. Reserve disk space for temporary output and, when replacing,
the existing destination. See the [streaming tutorial](../getting-started/streaming.md).

## Implementation Boundaries

Continue to import the public objects from `fourdenoise`; the implementation
is split into supporting modules without changing notebook imports.
`HyperData` owns calibration, unfolding orchestration, and result metadata.
The private `fourdenoise_denoising` module separates numerical algorithms
from method lookup, validation, and dimensional routing. It receives an
object factory when it needs to construct a `HyperData`, rather than importing
the main module.

`RealSpace.show()` and `ReciprocalSpace.show()` keep their public signatures
and delegate rendering to `fourdenoise_views`. Plotting does not own or replace
the underlying data. These supporting modules are implementation details,
not alternative user-facing APIs.
