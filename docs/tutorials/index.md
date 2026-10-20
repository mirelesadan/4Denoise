# Tutorial Notebooks

The notebook series teaches individual parts of 4Denoise using small,
reproducible **synthetic examples**. No external microscope data is needed.
Each completed tutorial has a clearly marked section for introducing your
own data later, without treating the synthetic calibration as experimental.

These lessons complement the [experimental mini-data quickstart](../getting-started/quickstart.md)
and [streaming example](../getting-started/streaming.md). They are not
physical scattering simulations or substitutes for acquisition validation.

## Available

Each overview links to the **complete online lesson**, including notebook
explanations, code, saved figures, and a clean notebook download. The online
lessons display validated fresh-kernel results; your browser runs no Python.
Synthetic teaching data is labeled throughout and is not experimental
validation. Keep the included experimental DEMO and the synthetic lessons
distinct when interpreting results.

- [Tutorial 1: Loading and calibration](loading-and-calibration.md): array
  conventions, native formats, units, metadata, saves, and read-only sources.
- [Tutorial 2: Visualization and virtual detectors](visualization-and-virtual-detectors.md):
  synthetic hexagonal patterns, scan montages, masks, weighted detector
  sums, calibrated axes, scale bars, and disk-backed imaging.
- [Tutorial 3: Alignment and coordinate transforms](alignment-and-coordinate-transforms.md):
  known subpixel motion, disk fitting, beam metadata, direct-beam blocking,
  Cartesian/polar rotation, sampling, and lossy reconstruction.
- [Tutorial 4: Denoising and unfolding](denoising-and-unfolding.md):
  selectable inputs, 2D/3D filter axes, traversal coverage, timed comparisons,
  and a short TensorLy convergence run.
- [Tutorial 5: Peak analysis and Braggness](peak-analysis-and-braggness.md):
  variable peak counts, subpixel centers, intensity windows, threshold
  sensitivity, validity masks, and detection-independent sharp contrast.
- [Tutorial 6: Strain mapping and R/Q calibration](strain-mapping-and-rq-calibration.md):
  missing/extra peaks, fit diagnostics, verified reporting directions,
  intensity weights, and exact coordinate checks.
- [Tutorial 7: Large-data processing](large-data-processing.md):
  read-only sources, bounded reductions, independent blocks, incremental peak
  files, direct-to-disk transforms, and resource/disk planning.

Read [using your own data](using-your-own-data.md) before replacing a fixture
with an experimental file, simulation array, or existing `HyperData` object.

The seven-notebook foundation is complete. Tutorial 7 deliberately uses a
file-based input contract; the earlier lessons also accept in-memory input.

## Run a Tutorial

Use the [installation guide](../getting-started/install.md) to select the
correct Python environment. Download the notebook below or open its copy in
the repository's `tutorials/` directory, restart the kernel, and run all cells.
For automated execution from the repository root:

```bash
python -m pip install -e ".[docs-test]"
python scripts/execute_tutorials.py
```

Only registered tutorials run; research notebooks are never discovered.
Executed reports are saved separately in `docs/_build/tutorials/`.

```{toctree}
:maxdepth: 1

loading-and-calibration
visualization-and-virtual-detectors
alignment-and-coordinate-transforms
denoising-and-unfolding
peak-analysis-and-braggness
strain-mapping-and-rq-calibration
large-data-processing
using-your-own-data
```
