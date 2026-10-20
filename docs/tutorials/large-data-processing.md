# Large-Data Processing

**[Read the complete tutorial online, with code and figures](notebooks/07_large_data_processing.ipynb).**

```{toctree}
:hidden:

notebooks/07_large_data_processing
```

Tutorial 7 keeps diffraction pixels on disk using `HyperData.open`. It builds
a small synthetic hexagonal HDF5 fixture one pattern at a time, then produces
compact views, incremental peak tables, and transformed files without a
full-source load. The fixture is laptop-sized: this validates the processing
path, not a multi-GB process-RAM benchmark or original microscope acquisition.

{download}`Download the notebook <../../tutorials/07_large_data_processing.ipynb>`.

## What You Will Learn

- Inspect shape, dtype, numerical size, calibration, and reader chunk information.
- Read owned patterns and compute bounded mean/std reductions and virtual images.
- Iterate independent blocks without collecting them, preserving global scan origins.
- Write peaks incrementally and read selected numeric table slices with h5py.
- Resize scan/detector axes and write polar/Cartesian transforms directly to disk.
- Check global bin-center calibration and streamed/single-pattern equivalence.
- Plan disk use, retain settings, close resources, and protect existing outputs.

The native default needs no external data, simulator, or optional microscopy
reader. It constructs a `(12, 16, 64, 64)` uint16 dataset with HDF5 storage
chunks and processes it through separate read-only source contexts. Its seven
figures inspect compact results, selected peaks, polar axes, and lossy
reconstruction. All output paths are owned temporary files.

## Memory Is More Than a Buffer Budget

Diffraction input and full transform outputs stay on disk. Mean/std patterns,
scan maps, selected peaks, and figures remain in RAM by design. The custom
loop checks coverage and agrees with source reductions, without retaining
blocks. No full `DiffractionSource.load` or `PeakScanResult.load` is used.

`working_memory_mb` budgets estimated numerical work buffers, not total
process RAM. Reader/HDF5/OS caches, metadata, custom work, allocators, plotting,
and retained results add memory. A separate map-size guard limits each complete
float64 scan map, not the sum of all retained arrays. Reader/storage chunks
and requested processing blocks are separate concepts.

A small fixture can fit into one automatically budgeted temporary block.
Bounded processing does not require every read to be a tiny fraction of a
small input; it avoids an explicit persistent eager full-source object and
splits larger inputs according to the same operation-specific planner.

Global tensor denoising is not made equivalent by independently processing
arbitrary blocks. Neighborhood methods need correctly designed halos, while
global decompositions require a separate algorithm or an explicit full load.

## Replace the Input With a File

Unlike the earlier lessons, Tutorial 7 intentionally accepts a **file**, not
an in-memory array or HyperData. Save already loaded data deliberately first,
or use an original lazy-readable acquisition. Configure `microscope_path`,
backend, dataset/axis selection, and only verified calibration. A generic
HDF5/NumPy file does not inherit the fixture's units or beam center.

External files require an inspected pixel annulus. Peak extraction and disk
transforms default to off; enable them only after checking parameters and
output-size estimates. Multiple acquisition-sized output files can exceed
available disk space even when processing buffers are small. Atomic output
is per-file, not a transaction covering the entire workflow.

For optional Rosetta readers, follow the
[reader validation boundary](../guides/io-and-calibration.md#reader-support-and-validation-boundary).
No supported reader silently falls back to an eager load. Preserved acquisition
settings are not verified beam/RQ calibration. See
[using your own data](using-your-own-data.md) for the file-based input contract.

## Interpret the Checks

Scan resizing updates spacing and shifts pixel-center origins to the new bin
centers; it does not pin the first output center to the old one. Area resizing
averages rather than sums intensities. Polar reconstruction restores an
original canvas shape, not original intensities or discarded corners.

Peak counts are not crystalline fraction, correlation scores are not integrated
intensities, and an excluded peak-extraction position is not an evaluated
zero-peak pattern. The processed mask preserves that distinction. Other
operations still cover the full selected scan.

## Run the Notebook

```bash
python scripts/execute_tutorials.py --notebook 07_large_data_processing.ipynb
```

Run from the repository root. A fresh kernel saves a separate report in
`docs/_build/tutorials/`; source notebooks remain clean. The companion
[experimental mini-data streaming example](../getting-started/streaming.md)
uses the included measured data rather than this synthetic fixture. This
completes the seven-notebook foundation.
