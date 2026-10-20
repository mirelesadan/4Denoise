# Strain Mapping and R/Q Calibration

**[Read the complete tutorial online, with code and figures](notebooks/06_strain_mapping_and_rq_calibration.ipynb).**

```{toctree}
:hidden:

notebooks/06_strain_mapping_and_rq_calibration
```

Tutorial 6 fits subpixel peak sets in small synthetic hexagonal diffraction
patterns. It separates scan-map positions from the directions of strain
components and uses a known 37-degree real-to-detector orientation with a
reflection to verify real-frame reporting. It does not infer an acquisition
calibration from a hexagon or use an arbitrary 90-degree component swap.

{download}`Download the notebook <../../tutorials/06_strain_mapping_and_rq_calibration.ipynb>`.

## What You Will Learn

- Supply global or per-pattern references and variable-length measured peaks.
- Fit missing/extra peaks with neighborhood-gated assignment and optional
  measured-intensity weights, while retaining failures as invalid values.
- Reuse alignment beam-center metadata without confusing it with R/Q orientation.
- Compare detector components with components along verified real directions.
- Inspect match counts, fit statuses, conditioning, residuals, and equal-aspect
  transformed-reference overlays.
- Check exact noiseless tensor recovery and pixel/calibrated-coordinate
  equivalence, then export maps with their reporting-frame metadata.

The default run is independent of other notebooks and requires no external
data, simulator, or interactive plotting backend. It creates an 8 by 12 scan
of 64 by 64 patterns, known deformations, localization noise, missing peaks,
unrelated spots, one masked position, and one underdetermined peak input.
All generated files live in an owned temporary directory and are cleaned up.

## Scientific Conventions

Data axes are `(Ry, Rx, Ky, Kx)`, and peak columns are `(ky, kx)`. Pixel rows
increase downward; calibrated ky increases upward. R/Q orientation uses
Cartesian `(x, y)`, x right and y up, with reflection before the
counterclockwise real-to-detector rotation. Map pixels are not transposed or
rotated when the reporting frame changes.

Real-frame strain is `I - U`, where `U` is the right polar stretch of the
fitted reciprocal reference-to-measured transform in that frame. Components
are dimensionless; rotation is in radians. This reciprocal-stretch convention
is not an exact finite direct-lattice strain measure. Fit RMSE is a residual
in peak-coordinate units, not a confidence probability or phase identity.

## Replace the Synthetic Input

Read [using your own data](using-your-own-data.md), then supply the marked
cell with a file, array, or existing `HyperData`, measured peaks, and a justified
reference. Global `(n_ref, 2)` and local nested `[Ry][Rx]` references are
supported. Peak collections must use the selected input's geometry and stated
units. Optional intensity vectors must follow the measured peak order.

External data receives no synthetic peaks, beam center, artificial scan mask,
or R/Q orientation. The lesson requires either current beam metadata or an
explicit peak origin. Real-frame output is opt-in through `rq_verified=True`
and an existing or explicitly supplied calibration. Otherwise the common
workflow reports detector components and skips synthetic truth checks.
Vendor scan-rotation metadata alone does not verify R/Q orientation.

An optional, unexecuted interactive recipe explains `compare_rq` with an
independently justified real/shadow image pair. Default CI does not open a GUI
or validate an original microscope acquisition. Large rotations and symmetry
ambiguity need orientation matching or a justified initial guess before this
local fitting workflow.

## Run and Continue

```bash
python scripts/execute_tutorials.py --notebook 06_strain_mapping_and_rq_calibration.ipynb
```

Run from the repository root. The runner starts a fresh kernel and saves the
executed report separately in `docs/_build/tutorials/`, leaving the downloadable
source clean. See [peaks and strain](../guides/peaks-and-strain.md) for API
details. Continue with [Tutorial 7: Large-data processing](large-data-processing.md).
