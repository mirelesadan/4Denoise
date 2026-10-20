# Tutorial 2: Visualization and Virtual Detectors

**[Read the complete tutorial online, with code and figures](notebooks/02_visualization_and_virtual_detectors.ipynb).**

```{toctree}
:hidden:

notebooks/02_visualization_and_virtual_detectors
```

{download}`Download the notebook <../../tutorials/02_visualization_and_virtual_detectors.ipynb>`.
In a checkout, open `tutorials/02_visualization_and_virtual_detectors.ipynb`
with Jupyter. It runs independently of Tutorial 1 and external microscope data.

## What You Will Learn

- Display diffraction patterns with known-coordinate overlays, grids, and
  scale bars, separating pixel selections from calibrated display units.
- Survey a scan through `visualize(grid_shape=..., padding=...)`, where tiles
  are area-averaged patterns and the outer axes describe scan position.
- Form annular, displaced circular, multiple-disk, and arbitrary detectors.
- Supply Boolean, soft fractional, or signed masks and interpret their sums.
- Compose comparable subplots or an axes-free image with a calibrated bar.
- Preview circular polar coordinates without transforming the stored data.
- Apply the same virtual detector to a saved, read-only source.

## Data and Expected Contrast

A deterministic `(16, 20, 80, 80)` uint16 dataset contains 320 synthetic
patterns. The left and right halves use regular hexagons with an assigned
30-degree rotation between them in detector-array coordinates. Direct-beam
intensity, diffuse signal, smooth spot-amplitude variation, an opposing-spot
imbalance, and Poisson noise provide visible teaching targets. The notebook
checks the hexagons' equal radii and side lengths.

The nm/mrad calibration and intensities are invented. This is not multislice
simulation, an indexed orientation map, or evidence for phase, strain, or
crystalline fraction. A weak selected-spot image may simply mean the spot
falls outside that particular detector. The whole-scan mean combines both
hexagons, so its additional spot locations do not establish another phase.

## Quantitative Checks

Fresh-kernel execution checks direct NumPy detector sums, equivalent pixel
and calibrated inputs, disk unions without double-counting, signed-mask
differences, unchanged original counts, and disk/in-memory agreement. Numeric
weights are not automatically normalized. The source's midpoint defines the
Cartesian annulus origin; an experimental beam center must be verified.

All display transforms affect figures only. Detector preview tint marks
nonzero support, not numeric weight magnitude/sign. The notebook plots soft
and signed weight images explicitly and uses a zero-centered diverging map
for signed results. Polar display does not set polar metadata or replace
the coordinate transforms demonstrated in
[Tutorial 3](alignment-and-coordinate-transforms.md).

Cartesian diffraction display axes use centered offsets and y-up directions
even in pixel mode; scatter overlays still accept absolute `(row, column)`
array coordinates. Calibrated disk centers instead use centered y-up
reciprocal coordinates. The notebook checks the conversion explicitly.

## Run or Replace the Input

Use core dependencies plus a Jupyter/IPython kernel, or install `.[docs-test]`
and run from the repository root:

```bash
python scripts/execute_tutorials.py --notebook 02_visualization_and_virtual_detectors.ipynb
```

The source stays unexecuted; the report is saved separately under
`docs/_build/tutorials/`. Owned temporary files are removed by the notebook's
final cell. No research notebook or acquisition file is modified.

The marked microscope-input cell defaults to synthetic data and inherits no
invented calibration when a real path is supplied. Native loading is eager;
the optional Rosetta branch explicitly caps a full numerical load at 512 MiB.
For large acquisitions, use a read-only source instead. Choose a detector
after inspecting the data: no experimental annulus or beam center is guessed.
The default synthetic CI run does not validate that future acquisition.

For an already loaded experiment or converted simulation, set `input_data`
to a `HyperData` or numerical array instead of a file path. The former keeps
its own metadata; the latter gets only explicit calibration. External inputs
do not inherit the synthetic annulus or analytical spot coordinates. See
[using your own data](using-your-own-data.md).

See the [visualization guide](../guides/visualization.md) for API context and
[I/O and calibration](../guides/io-and-calibration.md) for source memory and
reader-validation boundaries. Revisit [Tutorial 1](loading-and-calibration.md)
for file-format and calibration fundamentals.
