# Tutorial 4: Denoising and Unfolding

**[Read the complete tutorial online, with code and figures](notebooks/04_denoising_and_unfolding.ipynb).**

```{toctree}
:hidden:

notebooks/04_denoising_and_unfolding
```

{download}`Download the notebook <../../tutorials/04_denoising_and_unfolding.ipynb>`.
In a checkout, open `tutorials/04_denoising_and_unfolding.ipynb` with Jupyter.
It needs core dependencies plus an IPython kernel, not a microscope file.

## What You Will Learn

- Select a synthetic fixture, file, numerical array, or existing `HyperData`
  before the common processing workflow.
- Browse all implemented traversal methods in a compact three-column gallery
  near the top of the notebook, with configurable maximum preview grid size.
- Distinguish per-diffraction median filtering, full-array windows, and
  real/reciprocal unfolding into different 3D representations.
- Visualize traversal paths and understand the neighborhood relations that
  row-major, serpentine, and Hilbert orderings do or do not preserve.
- Apply median, total variation, and diffusion with explicit intensity-scale
  parameters and automatic refolding.
- Display cropped-curve processing coverage and verify excluded positions
  return unchanged when preserved in metadata.
- Time each denoise call and compare copied diffraction/virtual-image
  summaries without retaining every complete reconstruction.
- Use errors from one short CP/PARAFAC run for convergence plotting, with
  separate opt-ins for rank sweeps and optional BM4D.

## Data and Interpretation

The default `(10, 12, 48, 48)` float32 fixture contains two deliberately
rotated hexagonal regions with Gaussian spots, a central beam, and Poisson
noise. The noiseless intensity model is available as a reference. Calibration
and intensities are invented; this low-complexity model can favor particular
factorizations and does not establish a best denoiser for microscopy.

Real unfolding produces `(Ry*Rx, Ky, Kx)`; reciprocal unfolding produces
`(Ky*Kx, Ry, Rx)`. A `(3, 1, 1)` filter follows the first traversal axis,
whose meaning therefore changes between domains. A row-major unfolding of
both domains is a 2D matrix. A 3D traversal does not preserve every neighbor
relationship of either original 2D coordinate grid.

For the default scan, center-cropped Hilbert processing visits rows `[1:9]`
and columns `[2:10]`, or 64 of 120 patterns. The other 56 are saved in metadata
and returned unchanged, not denoised. The notebook checks this distinction
and reports coverage separately from whole-array clean-reference error.

CP convergence errors describe the fit to noisy input, not error against
ground truth. The default rank-6, 20-iteration run is a short API example,
not proof of convergence or an optimal rank. Rank sweeps require one fit per
rank. TV and diffusion wrappers do not expose iteration-error histories.
Volume-filter parameters are sample/intensity scales, not physically
equivalent distances across nm and mrad axes.

## Replace the Input

The marked configuration cell accepts a file or an in-memory array/object.
Existing metadata is retained; no synthetic units, beam center, orientation,
or reference is assigned to an external input. Without an explicitly supplied
registered clean reference, reference-error metrics and their plot are skipped.
The short TensorLy example defaults off for external data. All illustrative
filter settings still need reviewing for that input.

Native file loading and array copies are eager. A 64-MiB input-size guard
prevents inadvertently processing a huge tensor but does not cap total RAM
or an earlier eager load. The optional Rosetta full load has an explicit
output budget. Use a chosen representative input or the source workflow for
large acquisitions. See [using your own data](using-your-own-data.md).

## Run It

From the repository root with the `docs-test` extra installed:

```bash
python scripts/execute_tutorials.py --notebook 04_denoising_and_unfolding.ipynb
```

The clean source is not overwritten. The executed report is saved separately
under `docs/_build/tutorials/`. Rank sweeps and BM4D remain disabled in the
default run; BM4D requires the `bm` extra and a suitable Gaussian noise model
and noise estimate. No package is installed by the notebook itself.

The lesson checks exact unprocessed folding, metadata, included/excess
coverage, a 2D/4D median equivalence, native saving, and unchanged input.
These checks are not validation of a future original microscope acquisition.
See the [denoising guide](../guides/denoising-and-unfolding.md) and
[Tutorial 3](alignment-and-coordinate-transforms.md) for related workflows.

Continue with [Tutorial 5](peak-analysis-and-braggness.md) for independent
peak detection, subpixel refinement, intensity integration, and sharp-contrast
maps. Denoising is not automatically applied there: changing intensities can
change peak and Braggness measurements and needs separate validation.
