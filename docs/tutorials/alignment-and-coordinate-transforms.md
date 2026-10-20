# Tutorial 3: Alignment and Coordinate Transforms

**[Read the complete tutorial online, with code and figures](notebooks/03_alignment_and_coordinate_transforms.ipynb).**

```{toctree}
:hidden:

notebooks/03_alignment_and_coordinate_transforms
```

{download}`Download the notebook <../../tutorials/03_alignment_and_coordinate_transforms.ipynb>`.
In a checkout, open `tutorials/03_alignment_and_coordinate_transforms.ipynb`
with Jupyter. It runs independently of earlier tutorials or microscope data.

## What You Will Learn

- Align a central disk using a local negative-trench template search, fitting
  its radius on a representative mean pattern before matching each pattern.
- Check center spread against known synthetic motion and a separate centroid
  diagnostic, rather than relying only on the template estimator.
- Interpret current-canvas beam metadata in pixels and calibrated units,
  then reuse it for direct-beam blocking.
- Distinguish Cartesian image rotation from periodic polar-column shifts.
- Choose radial/angular sampling and preserve the intended physical scales.
- Form polar ring/sector images and reconstruct onto original or new canvases.
- Save polar geometry and aligned beam metadata alongside the samples.

## Data and Expected Results

An `(8, 10, 80, 80)` float32 array contains synthetic softened central disks
of radius 6 pixels and regular hexagons of Gaussian spots at radius 22 pixels.
A known, smooth subpixel translation moves each complete pattern, with
reproducible Poisson noise. The invented calibration is 2 and 3 nm per scan
row/column and 0.25 mrad per detector pixel. This is not a physical scattering
simulation, strain map, orientation measurement, or verified R/Q calibration.

Alignment should reduce the independent center spread substantially while
retaining scan calibration. Disk alignment crops to a common detector canvas;
it need not preserve detector-edge data. The radius fitted on the drift-blurred
mean need not equal the constructed radius, and is not presented as a physical
semiangle measurement. The notebook explains COM, Gaussian, disk, and
negative-trench choices without claiming one fits every acquisition.

## Geometry and Reversibility

Alignment inputs and the explicit `to_polar(center=...)` use absolute
detector `(row, column)` pixels. The notebook passes the stored alignment
center explicitly because `to_polar()` defaults to the geometric midpoint.
Cartesian `rotate_dps()` rotates about that geometric midpoint and expands
the canvas; it is not the off-center-aware R/Q overlay tool. No R/Q
calibration is inferred or assigned by this tutorial.

The polar grid uses 48 radial and 180 angular samples, covering 30 original
detector pixels (7.5 mrad) and 0-360 degrees. Metadata distinguishes radial
sample-center spacing from display-bin width. Stored theta increases
clockwise, so positive counterclockwise rotation shifts columns left.
Checks establish exact integer rolls, 360-degree identity, calibrated/pixel
detector equivalence, and consistent inverse-output pixel scales.

Polar transformation and reconstruction use interpolation and exclude
unsampled support. A default inverse canvas is `(2*n_r, 2*n_r)`;
`output_shape='original'` means the canvas passed into `to_polar()`, here
the aligned canvas. Neither option restores the original detector corners
or guarantees identical intensities. Beam metadata cleared by conversion is
not recreated by the inverse. Polar virtual-image sums are sums of polar
samples, not automatically Jacobian-corrected Cartesian integrals.

## Run or Replace the Input

Install core dependencies and a Jupyter/IPython kernel, or use `.[docs-test]`
and run from the repository root:

```bash
python scripts/execute_tutorials.py --notebook 03_alignment_and_coordinate_transforms.ipynb
```

The runner saves an executed report under `docs/_build/tutorials/`, without
overwriting the clean source. Temporary files are removed by the final cell.
No package calculations, research notebooks, or acquisition files are edited.

The marked microscope-input cell uses synthetic data by default. Real native
loading is eager; the optional Rosetta full-load branch has an explicit
512-MiB budget. Real alignment is a separate opt-in and requires inspected
center/radius/search choices in pixels. It receives no invented calibration
or beam parameters. Routine synthetic execution does not validate this future
microscope branch. For large files use read-only sources and disk-output
transforms rather than loading the whole acquisition for this small lesson.

The input cell also accepts an existing `HyperData` or numerical simulation/
experimental array. These branches preserve supplied metadata, not synthetic
centers or radii. Alignment remains an explicit opt-in with inspected pixel
parameters. See [using your own data](using-your-own-data.md).

Continue with [Tutorial 4](denoising-and-unfolding.md) for denoising a selected
input using full-shape and cropped traversals.

Revisit [Tutorial 2](visualization-and-virtual-detectors.md) for detector and
display conventions. See the [data model](../concepts/data-model.md),
[visualization guide](../guides/visualization.md), and
[streaming tutorial](../getting-started/streaming.md) for related workflows.
