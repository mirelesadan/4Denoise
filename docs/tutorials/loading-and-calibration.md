# Tutorial 1: Loading and Calibration

**[Read the complete tutorial online, with code and figures](notebooks/01_loading_and_calibration.ipynb).**

```{toctree}
:hidden:

notebooks/01_loading_and_calibration
```

{download}`Download the notebook <../../tutorials/01_loading_and_calibration.ipynb>`.
In a repository checkout, open `tutorials/01_loading_and_calibration.ipynb`
with Jupyter instead. Use a fresh kernel in the environment where 4Denoise
is installed.

## What You Will Learn

- Interpret `(Ry, Rx, Ky, Kx)` and distinguish a 3D stack from a known 2D scan.
- Load NumPy, generic HDF5, MATLAB and raw data, selecting datasets explicitly.
- Define units per pixel, unequal scan steps, and the scan origin.
- Separate physical scan selections from pixel indices and display units.
- Save supported metadata with the pixels and reopen the result.
- Choose eager `HyperData` loading or a read-only `DiffractionSource`.

## Data and Requirements

The notebook creates a deterministic `(8, 10, 64, 64)` uint16 fixture with
diffraction-like spots and Poisson noise. It uses **invented** nm/mrad scales
and acquisition settings. This is a teaching illustration, not multislice
simulation, an experimental dataset, or verification of a microscope.
No included mini dataset, simulation extra, or RosettaSciIO is required.
Core packages plus an IPython/Jupyter kernel are sufficient.

All generated format fixtures and saved outputs live in an owned temporary
directory; the final cell removes only those files. The notebook contains
assertions for exact native-format values/dtypes, coordinate selection,
metadata round trips, and source/result lifetime. It also shows explicitly
requested load-time clipping without altering the stored input.

## Introduce Experimental or Simulation Data Later

The marked input cell defaults to a synthetic copy. Replace its path and
reader options only when ready, and leave its calibration overrides empty
unless you have verified values. It does not copy the illustrative scales
or acquisition settings onto a real file. Earlier file-format exercises
continue to export only their small synthetic fixture.

The native branch loads eagerly; do not point it at a file too large for
RAM. The optional Rosetta branch explicitly caps the numerical full load at
512 MiB. For larger acquisitions, use the read-only source workflow instead.
Reader/dataset/axis-order ambiguity requires confirmation, not guessing.
The notebook's default CI run does not exercise your future microscope file.

The same cell accepts an existing `HyperData` or numerical array through
`input_data`, labelled with `input_origin='experimental'` or `'simulation'`.
An existing object is copied with its own metadata; an array receives only
explicit calibration. Do not set a file and an in-memory input together.
See [using your own data](using-your-own-data.md) for the common contract and
the distinction between synthetic teaching exercises and input branches.

Next, use [Tutorial 2: Visualization and virtual detectors](visualization-and-virtual-detectors.md)
for synthetic hexagonal scans, detector masks, weighted sums, and scale bars.

See [I/O and calibration](../guides/io-and-calibration.md) for supported
readers, original acquisition records, and the distinction between recorded
scan rotation and verified R/Q calibration. Continue to the
[data model](../concepts/data-model.md) or
[visualization guide](../guides/visualization.md) after this lesson.
