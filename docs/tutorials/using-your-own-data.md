# Using Your Own Data

Every completed tutorial has one marked input configuration cell. All run
without external files by default, and all can later accept experimental or
simulation input without copying the illustrative synthetic calibration.

## Choose One Input

The in-memory recipes in this section apply to Tutorials 1-6. Tutorial 7 is
intentionally file-based: keep `input_data=None` and select a file with
`microscope_path`, reader/dataset options, and only verified calibration.
If your data is already a HyperData, save it deliberately first, then reopen
that saved file. The lesson never creates an eager copy and calls it lazy.

In that cell, supply exactly one of `input_data` and `microscope_path`:

```python
input_data = my_dataset                 # Existing HyperData, with its own metadata.
input_origin = 'experimental'           # Or 'simulation'.
microscope_path = None
microscope_options = {}
microscope_calibration = {}
```

An existing object is copied, so later processing does not overwrite its
pixels. Configure its geometry before passing it and leave loader/calibration
overrides empty. This is an eager copy, not a shared or lazy view.

For a numerical array from a simulation or an acquisition reader:

```python
input_data = simulated_array            # Verified numerical (Ry, Rx, Ky, Kx) data.
input_origin = 'simulation'
microscope_path = None
microscope_options = {}
microscope_calibration = {}             # Supply only verified unit/scale pairs.
```

Convert simulator-specific objects to numerical arrays first. Confirm axis
order, handedness, beam position, and units; a shape or field name cannot
prove these. A bare array receives no calibration unless you supply it.
The `input_origin` label is a user declaration, not verified acquisition
provenance, a simulation reader, or an R/Q calibration.

For a file instead, set `input_data=None`, provide `microscope_path`, and
select native or optional Rosetta loading. Native options can include an
HDF5 dataset path, MAT variable, or known raw shape/dtype/order. Rosetta
may need reader/dataset/axis-order choices. The historical variable name
`microscope_path` can also name a saved simulation file. Saved native
HyperData metadata is retained; source data is not overwritten.

## What Changes in Each Lesson

| Lesson | External-data entry point |
| --- | --- |
| 1: Loading/calibration | A separate selected-input branch after known-truth format/calibration exercises. |
| 2: Visualization | Selected-input preview and an optional, explicitly chosen experimental/simulation annulus. |
| 3: Alignment/transforms | Selected-input preview and a separate opt-in alignment using inspected center/radius/search parameters. |
| 4: Denoising/unfolding | Selection before the common processing workflow; all subsequent denoising uses the selected input. |
| 5: Peaks/Braggness | Selection before the common analysis, an explicitly inspected detector annulus, optional reference coordinates, and a separate opt-in alignment example. |
| 6: Strain/RQ | Selection before common fitting, explicit measured/reference peaks, a verified center, optional measured intensities, and opt-in verified real-frame reporting. |
| 7: Large-data processing | A file-only source, an inspected pixel annulus, and explicit opt-ins/settings for peak extraction and full disk-output transforms. |

Tutorials 1-3 intentionally retain their small synthetic construction,
analytical coordinates, and truth assertions. They do not apply those
assertions or assumed radii to your acquisition. Reuse their API calls after
selecting and inspecting your input; do not simply rename the original
synthetic array and run every known-truth assertion as if it were experimental.
The experimental DEMO and research notebooks remain separate.

## References and Scientific Validation

Tutorial 4 uses `reference_input=None` by default. For external data this
means no clean-reference RMSE/NRMSE or reference-error plot. You may supply a
clean numerical/HyperData reference only after checking matching registration,
grid, orientation, calibration, and intensity units. Matching shapes alone
do not establish these conditions. No synthetic reference is silently used
for a microscope file or your own simulation.

Review filter widths, intensity-dependent weights, rank, noise assumptions,
and the chosen detector for each input. A smoother image, more iterations,
or lower reconstruction error against noisy data does not prove improved
peak positions, intensities, or strain. Verify those downstream quantities.

Tutorial 5 accepts `reference_peaks_input` as nested `[Ry][Rx]` arrays of
absolute detector `(y, x)` pixel coordinates, including empty `(0, 2)` entries.
It skips known-position accuracy checks for external data unless you supply
that reference. Choose `analysis_annulus_px` explicitly for an external input;
beam metadata is preferred when the center argument is omitted. Keep detector
pixel parameters separate from calibrated display axes. Evaluated patterns
must be finite and nonnegative for this lesson; the explicit scan mask can
exclude unusable positions without declaring them zero-peak or non-crystalline.
Any alignment example is opt-in for external data and uses a separate copy.

Tutorial 6 requires `measured_peaks_input` in nested `[Ry][Rx]` layout and
`reference_peaks_input` as a global `(n_ref, 2)` array or the same nested layout.
Each entry uses finite `(n_peaks, 2)` coordinates; empty entries are `(0, 2)`.
Use `peak_units_input='pixels'` for absolute downward-positive `(ky, kx)` pixel
coordinates, or `'calibrated'` for upward-positive ky in the dataset's verified
reciprocal units. A matched shape does not establish a shared reference lattice,
phase, registration, or handedness. Tutorial 5's refined coordinates can be
measured input, but are not automatically an unstrained reference.

Optional `intensity_input` follows measured peaks exactly, not reference order.
Supply `strain_center_input` if the input lacks current beam metadata. Review
the matching gate in peak-coordinate units and provide a justified coarse
rotation guess if needed. Local strain fitting is not full orientation matching.
No synthetic peaks, beam center, or R/Q calibration are inherited by external
inputs. Set `rq_verified=True` only after checking the input's stored calibration
or providing `verified_rq_angle_deg` and its mirror setting. Otherwise detector
components are retained, even if unverified orientation metadata exists.

R/Q changes component directions, not scan-map positions. Preserve the result's
frame and strain definition when exporting maps: real-frame components use
`I - U` of the reciprocal transform, and rotation is in radians. Tutorial 6's
synthetic truth and pixel/calibrated-equivalence assertions are skipped for
external input; fit residuals alone do not validate its physical strain.

Tutorial 7 requires `analysis_annulus_px` explicitly for an external file.
It never inherits fixture calibration, a beam center, or an artificial peak
mask. `run_peak_detection` and `run_transforms` default to False for external
inputs. To enable extraction, also supply `peak_options_input` with reviewed
`radius`, `min_distance`, and threshold settings in pixels. The selected scan
mask applies only to peak extraction; reductions and transforms cover the full
scan. Template scores and optional raw intensity sums have different meanings.

Native files and supported lazy Rosetta readers remain on disk. Multiple HDF5
datasets need a selected path; ambiguous Rosetta results/axes need explicit
choices. Read single patterns or selected peak-table rows instead of calling
a full loader. The lesson does not explicitly materialize a full-source object
or peak collection. A small file can still fit in a budgeted temporary read
block. Compact scan maps and figures are retained in RAM, so inspect their
size separately from the temporary-buffer budget.

Before enabling transforms, review their detector/polar output shapes, center,
interpolation, disk-size estimate, and free space. Several complete transformed
copies can be expensive. Outputs are atomic per file, not an all-or-nothing
pipeline, and their default temporary destinations are removed on cleanup.
Choose persistent paths deliberately if you want to retain results. Passing
the synthetic CI path is not validation of original vendor data or a large
process-RAM benchmark.

## Memory and Reproducibility

Native loading and in-memory input copying are eager. Optional Rosetta
branches cap full numerical loads, but these budgets do not cap total process
RAM. Tutorial 4 also rejects oversized selected arrays before denoising;
that guard cannot prevent an earlier eager file load. It overwrites the full
denoised result between methods and retains only small 2D summaries. Global
decomposition still needs full-tensor memory and cannot be streamed by
independently processing arbitrary chunks without changing the method.

For multi-GB acquisitions, start with [read-only sources](../guides/io-and-calibration.md)
and the [streaming example](../getting-started/streaming.md), or choose an
explicit representative input on your machine. No tutorial automatically
extracts a scientifically representative ROI for you.

Default CI checks the reproducible synthetic paths. Branch tests exercise
input selection with generated arrays/files and supplied metadata; they do
not validate your original microscope acquisition or the scientific meaning
of a simulation. Record the original file, calibration source, package version,
processing settings, and any ROI before interpreting results.
