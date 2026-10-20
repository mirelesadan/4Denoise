# Stream the Included Dataset

For a broader independently runnable notebook covering chunk iteration,
memory/disk planning, selected peak-file reads, and streamed resizing, see
[Tutorial 7: Large-data processing](../tutorials/large-data-processing.md).
That lesson uses synthetic hexagonal patterns; the example below uses the
complete included measured mini dataset.

This executable tutorial uses the **complete included experimental mini
dataset**, `mini_dataset_binned.npy`, with shape `(15, 13, 128, 128)`.
It needs only the core installation, not RosettaSciIO, simulation tools, or
an external acquisition. Run it from the repository root:

```bash
python scripts/streaming_example.py
```

The example makes a mean diffraction pattern and annular virtual image,
extracts independent peaks into a numeric HDF5 peak file, and writes polar
and Cartesian diffraction outputs. It then reloads the **peak collection**
and checks the saved shapes. It never calls full-source `load()` or modifies
the input, DEMO, or research notebooks. Outputs use a temporary directory
and are removed after a successful run or an exception.

To keep the three output files for inspection, choose an output directory:

```bash
python scripts/streaming_example.py --output docs/_build/streaming-example
```

Existing output filenames are not overwritten. Use a new directory on the
next run, or review and move your earlier outputs. Each file is published
atomically; the entire multi-step example is not a single transaction.

## What Runs

The code below is included directly from the executable script, so the guide
and tested example cannot drift into separate copies. Paths are supplied by
the command above; the same function can be imported in a notebook.
All detector radii, separation, and output peak coordinates here are pixels.
The mini-data thresholds are illustrative, not universal defaults.

```{literalinclude} ../../scripts/streaming_example.py
:language: python
:pyobject: run_example
```

The caller receives owned `RealSpace`/`ReciprocalSpace` views and a ragged
`PeakScanResult` after all source contexts have closed. The script runs
headlessly; in a notebook you can call the function and display its results:

```python
from scripts.streaming_example import run_example

example = run_example("mini_dataset_binned.npy", "docs/_build/my-streaming-run")
example["image"].show(axis_units="pixels", title="Annular virtual image")
example["mean_dp"].show(axis_units="pixels", title="Mean diffraction pattern")
counts = example["peaks"].observed_peak_counts  # (15, 13)
```

This imports the repository's example script; it is not an installed package
entry point. Start the notebook at the repository root, as with the quickstart.

## Interpret the Outputs

| Output | Meaning and loading |
| --- | --- |
| Mean DP | One independent `ReciprocalSpace` array, not a link to an open file |
| Virtual image | One complete `(Ry, Rx)` `RealSpace` array, with annular intensity sums |
| `peaks.h5` | Ragged coordinates, correlation scores, provenance, and raw circular-window intensity sums; use `PeakScanResult.load` or selected h5py table slices |
| `polar.4denoise` | Diffraction pixels on saved radial/angular axes; reopen with `HyperData.open` |
| `cartesian.4denoise` | Interpolated Cartesian diffraction pixels on the recorded original canvas; reopen with `HyperData.open` |

An original canvas shape does **not** mean original values are recovered.
Polar conversion resamples a circular region, and Cartesian conversion
interpolates it back; it cannot restore excluded corners or information lost
during interpolation. The example does not assert numerical identity after
this round trip.

Peak coordinates are `(y, x)` pixel positions. `scores` are detector-template
responses, not integrated intensities. `intensity_radius=3` requests raw
circular sums; no background subtraction, center refinement, or strain fit
is implied. All positions are processed, with symmetry repair disabled.
`observed_peak_counts` counts detections, **not crystalline fraction**.

## Memory and File Lifetime

Diffraction input and transformed output tensors stay on disk. The returned
mean pattern, scan image, and explicitly loaded peak collection stay in RAM.
For a very large scan, do not load the whole peak file merely because its
diffraction data was processed in blocks; use the
[selective peak-file recipe](../guides/peaks-and-strain.md#incremental-peak-files).

`working_memory_mb=64` budgets estimated temporary numerical buffers, not
total process memory. Caches, metadata, retained results, allocator overhead,
and reader internals can add memory. `PeakScanResult.load(max_memory_mb=64)`
separately caps its numeric tables and object-pointer arrays. Default budgets
are 256 MiB when omitted. Large external patterns may need larger budgets.
Global tensor denoising is not automatically streamed.

## Move to an Original Acquisition

Install `.[rosetta]` only when needed. For microscope formats use
`HyperData.open(path, ...)`, with a confirmed reader, dataset, and axis order
where automatic interpretation is ambiguous. The example's detector thresholds
and canvas choices must be reconsidered for the new file; `--data` is not a
universal acquisition recipe. Native HDF5 also needs an explicit dataset path
when several arrays are present; adapt the open call accordingly.

Compatible calibration and archived vendor metadata are distinct. A scan
rotation field is not automatically R/Q calibration, and a detector offset
is not automatically the beam center. Current evidence includes actual
reader-generated EMD/HyperSpy/Ripple fixtures and mini-data workflows, not
independent validation of every original vendor acquisition. Follow the
[reader validation boundary](../guides/io-and-calibration.md#reader-support-and-validation-boundary)
before trusting new file-specific geometry.
