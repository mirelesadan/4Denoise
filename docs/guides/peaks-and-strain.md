# Peaks and Strain Results

For an independently executable synthetic example, see
[Tutorial 6: Strain mapping and R/Q calibration](../tutorials/strain-mapping-and-rq-calibration.md).
It checks coordinate ordering and known real-frame components without using
an arbitrary 90-degree basis correction, and distinguishes residuals from
confidence and reciprocal-stretch strain from finite direct-lattice strain.

[Tutorial 5](../tutorials/peak-analysis-and-braggness.md) provides a runnable
synthetic and replaceable-input lesson on independent detection, ragged
centers/intensities, threshold sensitivity, and Braggness. It keeps detection
scores, intensity windows, and sharp-contrast scores distinct.

The [DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb) shows
one verified path from a reference diffraction pattern through peak
detection, center refinement, intensity integration, and a relative strain
map. Peak-selection thresholds and reference regions are data-dependent;
the [experimental case study](../case-studies/3d-strain-mapping.md) uses
different choices.

For a shared reference, detect spots once on an averaged diffraction pattern
and pass those `(y, x)` pixel positions to `get_centers`. Alternatively,
`HyperData.get_peaks(...)` detects each pattern independently and returns
ragged peak lists when counts vary. Keep the peak coordinate order and units
consistent when supplying `ref_centers` or `intensity_array` to a strain fit.

```python
reference_dp = data.get_dp(operation="mean")
reference_peaks = reference_dp.get_peaks(
    radius=3, min_distance=7, threshold_rel=0.1, r_range=(10, 61),
)
measured = data.get_centers(r=5, ref_coords=reference_peaks)
```

These thresholds illustrate the included mini dataset, not general detector
defaults. Inspect an overlay before fitting strain and adjust the peak set
for your material and calibration.

The [streaming tutorial](../getting-started/streaming.md) runs the independent
detector on the included mini dataset and writes a numeric peak file. This is
separate from reference-based center refinement: no shared reference or
fixed peak count is assumed.

## Peak Extraction From Disk-Backed Sources

`DiffractionSource.get_peaks()` uses the same detector as in-memory peak
finding, with bounded input blocks and one reused detector/template:

```python
with HyperData.open("experiment.h5") as source:
    peaks = source.get_peaks(
        radius=3, min_distance=7, threshold_rel=0.1, r_range=(10, 61),
        intensity_radius=3, coords_units="pixels",
        working_memory_mb=64, progress=True,
    )

counts = peaks.observed_peak_counts        # (Ry, Rx), excluding repaired peaks
positions = peaks.coords[0, 0]             # (n_peaks, 2), ordered (y, x)
intensities = peaks.intensities[0, 0]       # raw circular-window sums
```

The result is a `PeakScanResult`, not a diffraction tensor or RealSpace image.
Its ragged **object arrays** share the original scan shape: `(Ry, Rx)` for 4D
or `(N,)` for a 3D stack. `coords`, `scores`, `synthetic_mask`, `orbit_ids` and
optional `intensities` align row-by-row in each pattern. `processed_mask`
distinguishes masked-out positions from evaluated patterns containing no
peaks. Use `real_mask` for a Boolean 4D scan selection; fully masked read
blocks are skipped, while unselected patterns inside mixed blocks are never
evaluated. Returned arrays/metadata survive source closure independently.

Detection radii, separation, radial ranges, centers and integration radii
remain **pixels**. `coords_units` affects only output: `"pixels"` is the
default; `"auto"` uses reciprocal calibration when available; `"calibrated"`
requires it. Calibrated coordinates are y-up/x-right relative to the geometric
detector midpoint, consistent with diffraction display. The search center
uses current beam metadata (or an explicit pixel override), but does not
change that coordinate origin. The result records both settings and source
scan/detector calibration. Pixel results can be passed as reference positions
for subsequent center refinement; streamed detection itself does not refine
coordinates, align images, normalize intensities or estimate background.

`scores` are template-correlation responses, **not integrated intensities**.
`intensity_radius` requests raw sums in circular windows, accumulated in
float64 and clipped to the detector edges. Supply background-subtracted
patterns beforehand when that is scientifically appropriate. Counts above
float64's exact-integer range can lose precision. Symmetry repair adds
coordinates, not observations: generated peaks have True `synthetic_mask`
and NaN scores/intensities. `peak_counts` includes repaired rows;
`observed_peak_counts` excludes them. Neither count is a crystalline fraction.

### Incremental Peak Files

For large peak collections, write directly to a distinct peak file:

```python
from fourdenoise import PeakScanResult

with HyperData.open("experiment.dm4") as source:
    peak_path = source.get_peaks(
        radius=3, min_distance=7, intensity_radius=3,
        destination="peaks.h5", working_memory_mb=64,
    )

# Explicitly materialize the peak collection, not the diffraction dataset.
peaks = PeakScanResult.load(peak_path, max_memory_mb=256)
```

Peak files use append-only numeric `/coords`, `/scores`, `/synthetic_mask`,
`/orbit_ids` and optional `/intensities` tables, plus scan-shaped `/starts`,
`/counts` and `/processed_mask`. They do not use pickle or variable-length
object datasets. For a selected scan index, use h5py to read its start/count
and only the corresponding numeric table slices. Metadata includes coordinate
convention, detection/integration settings, resolved search center and the
source calibration/acquisition record. These files are not HyperData saves:
use `PeakScanResult.load`, not `HyperData(path)`.

File mode returns a path and never allocates the full ragged collection. The
source remains open and unmodified. Atomic publication is default: failures
remove the temporary sibling and preserve an existing destination. Replacement
requires `overwrite=True`; `atomic=False` explicitly permits partial output.
Use `.h5`/`.hdf5`/`.hdf`; no suffix appends `.peaks.h5`.

The temporary `working_memory_mb` budget defaults to 256 MiB and reserves one
Rosetta reader chunk plus conservative FFT/template/candidate/symmetry
workspace and two raw read/copy buffers. Too-small budgets fail before pixel
reads. `chunk_size` further limits patterns per read; 4D reads stay within a
scan row. Metadata, existing caches, allocator overhead and retained results
are outside the estimate, which is not a process-RAM limit. Symmetry can need
larger budgets. Without file mode, result memory grows with scan size and peak
count. `PeakScanResult.load` explicitly materializes the full peak collection;
its cap excludes metadata and Python/NumPy object-header overhead.

To read one entry without materializing the complete peak collection:

```python
import h5py

scan_index = (0, 0)  # use (i,) for a 3D stack
with h5py.File(peak_path, "r") as file:
    processed = bool(file["processed_mask"][scan_index])
    start = int(file["starts"][scan_index])
    count = int(file["counts"][scan_index])
    positions = file["coords"][start:start + count]
    synthetic = file["synthetic_mask"][start:start + count]
    measured_positions = positions[~synthetic]
```

The slices own their returned values after the file closes. This is a
low-level read of a known peak schema, not a dedicated lazy peak-file object.
Use `PeakScanResult.load` for schema validation and whole-collection access.
For a `PeakScanResult` from memory, inspect `processed_mask` before interpreting
a zero in `observed_peak_counts`. Repair-generated rows must not be counted
as measured peaks or passed as measured intensity observations.

## Strain Results

`HyperData.get_strains(centers=measured, ref_centers=reference)` returns a
`StrainResult` with named maps and fit diagnostics:

```python
result = data.get_strains(centers=measured, ref_centers=reference)
result.as_real_space("exx").show()
result.as_real_space("relative_fit_rmse").show()
valid = result.valid_mask
```

`exx`, `eyy`, and `exy` are dimensionless. `erot` is in radians.

**Tensor directions are separate from map positions.** Peak coordinates have
order `(ky, kx)`. The default `output_frame="reciprocal"`, at `ang=0`, assigns
`exx` to detector kx and `eyy` to detector ky. A `(Ry, Rx)` map's location in
the scan does not make those tensor directions scan-aligned.

To report strain along real-space directions, establish the real-to-detector
orientation and store it on the dataset. You can use `compare_rq(...).apply()`
after adjusting the interactive comparison, or set a known calibration:

```python
data.set_rq_calibration(rotation_deg=verified_angle, mirror_axis=None)
result = data.get_strains(
    centers=measured,
    ref_centers=reference,
    output_frame="real",
)
result.as_real_space("exx").show()
print(result.output_frame)
print(result.rq_calibration)
```

`verified_angle` must describe the counterclockwise real-to-detector rotation
in Cartesian image axes (x right, y up). Set `mirror_axis="x"` or `"y"` when
the real frame is reflected about that axis before rotation. Recorded
acquisition scan rotation alone does not establish this relationship.

`output_frame="real"` reads the stored calibration and uses its full rotation
and reflection matrix. `exx` and `eyy` then follow real-space x and y, and
`exy` is the Cartesian off-diagonal component of `I - U`, where `U` is the
polar stretch. `erot` is positive counterclockwise. Pixel peaks have downward
positive rows, while calibrated peaks have upward positive ky; both are
converted to the same Cartesian reporting frame. Reflections can reverse the
rotation sign relative to detector-frame output. Map locations, match counts,
and fit errors are retained. Real output requires stored `rq_calibration`.

For a manual detector-basis correction, use
`get_strains(..., ang=angle_degrees, output_frame="reciprocal")`. For example,
`ang=90` exchanges the normal components and reverses the shear sign without
moving map pixels. A nonzero `ang` cannot be combined with `output_frame="real"`.
With pixel peaks and a rotation-only calibration, real output is equivalent
to the corresponding manual angle; using metadata also handles reflections
and the different handedness of calibrated peak coordinates.

`result.output_basis_matrix` maps centered input `(ky, kx)` vectors to the
reporting coordinates. `result.rq_calibration` records the calibration used
for real output; `result.basis_angle_deg` records only the manual `ang` and
is zero for real output. Fitted transforms and translations in `diagnostics`
remain in the supplied peak-coordinate frame. `as_real_space()` displays
the already-selected components without applying another basis conversion.

`fit_rmse` measures residual peak-position error in the supplied peak
coordinate units; `relative_fit_rmse` divides that error by the RMS
reference-peak radius. Neither number is a confidence probability. Check
`match_counts`, `outlier_counts`, and `valid_mask` as well, especially
when only a few peaks are fitted. A 2D map inherits scan calibration from
the source 4D data when its shape matches that scan.

Existing four-value unpacking and numeric indexing still work. With
`return_transform=True`, transform and peak-rejection diagnostics are
available in `result.diagnostics` and as `result[4]`.

## Local Peak Pairing and Failure Diagnostics

The default matcher uses one-to-one candidates within a neighborhood of each
reference peak, not centroids of differently populated peak sets. Extra,
missing, or distant peaks can remain unmatched. `match_radius` uses the same
units as the supplied peaks; None selects 45% of the median nearest-reference
peak spacing. For a large misorientation, supply an initial guess:

```python
result = data.get_strains(
    centers=measured, ref_centers=reference, center=beam_center,
    initial_rotation=rotation_angles, match_radius=0.15,
    peak_units="calibrated", return_transform=True,
)
```

Angles are Cartesian counterclockwise degrees, either scalar or scan-shaped.
`initial_transform` instead accepts global/scan-shaped (2, 2) matrices in
the input (ky, kx) order. Guesses guide matching only: the full fitted transform
and its rotation are retained, not just a residual rotation. Stored current
beam centers take precedence over historical pre-alignment mean-fit centers.

`fit_translation=True` enables translation consensus that does not assume
equal peak counts. An explicit `initial_translation` may be supplied in peak
units. `ambiguity_ratio` optionally rejects close competing alternatives;
smaller values are stricter. The reported pair-confidence margin is geometric,
not a calibrated probability. Intensity weights remain attached to original
measured-peak indices through cleaning, filtering and assignment.

Inspect `result.fit_status` for masked locations, insufficient peaks/matches,
no candidates, ambiguity rejection, ill-conditioned fits, or success.
`condition_numbers` reports fitted design conditioning when available.
With `return_transform=True`, `diagnostics['matching_information'][index]`
contains original matched/unmatched indices, candidate counts, pair distances,
confidence margins and the initial transform/translation actually used.
Failed fits remain NaN; they are not silently converted to zero strain.

Use `match_peaks="ordered"` only for known correspondences. Explicit
`"hungarian"` allows unrestricted assignment when no gate is specified; it
can force unrelated peaks into pairs and is not the recommended default.
