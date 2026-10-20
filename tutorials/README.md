# 4Denoise Tutorial Notebooks

These self-contained lessons use reproducible **synthetic teaching data**.
They do not require an original microscope acquisition and do not claim to
validate a microscope calibration or reproduce electron-scattering physics.
The experimental DEMO and research notebooks remain separate.

## Start Here

1. Install 4Denoise in your intended environment and select its Jupyter kernel.
2. Open [Tutorial 1: Loading and calibration](01_loading_and_calibration.ipynb).
3. Restart the kernel and run all cells from top to bottom.
4. Read the clearly marked input section before substituting a file, array, or HyperData object.

Tutorials 1-7 use core dependencies plus a Jupyter/IPython kernel. Their examples
work when Jupyter starts in either the repository root or this directory:
there are no hard-coded research paths or external input datasets. Generated
files live in a temporary directory that the final cell cleans up.

The repository installation options are described in the
[installation guide](../docs/getting-started/install.md).
Each lesson also has a complete online version in the documentation, with code
and figures from the validated fresh-kernel run. Downloadable sources stay
output-free; rendered files are generated, not committed notebook copies.
For a lightweight automated run, install the `docs-test` extra and run:

```bash
python -m pip install -e ".[docs-test]"
python scripts/execute_tutorials.py
```

Run those commands from the repository root. Each registered notebook gets a
fresh kernel using that Python interpreter. Executed reports, including
partial output on failure, go to ignored `docs/_build/tutorials/`; the source
notebooks are never overwritten. Passing CI validates the synthetic workflow,
not the optional microscope-input branch against an original acquisition.

## Series

| Number | Topic | Status |
| --- | --- | --- |
| 1 | [Loading and calibration](01_loading_and_calibration.ipynb) | Available; synthetic, executable, no external data |
| 2 | [Visualization and virtual detectors](02_visualization_and_virtual_detectors.ipynb) | Available; synthetic hexagonal regions and quantitative detector checks |
| 3 | [Alignment and coordinate transforms](03_alignment_and_coordinate_transforms.ipynb) | Available; drifting hexagonal patterns, beam metadata, polar rotation, and reconstruction checks |
| 4 | [Denoising and unfolding](04_denoising_and_unfolding.ipynb) | Available; selectable inputs, filter axes, traversal coverage, timings, and TensorLy convergence |
| 5 | [Peak analysis and Braggness](05_peak_analysis_and_braggness.ipynb) | Available; ragged detections, refinement, intensity integration, validity, and sharp-contrast maps |
| 6 | [Strain mapping and R/Q calibration](06_strain_mapping_and_rq_calibration.ipynb) | Available; missing/extra peaks, fit diagnostics, verified real-frame components, and coordinate checks |
| 7 | [Large-data processing](07_large_data_processing.ipynb) | Available; file-based inputs, bounded reductions, incremental peaks, and direct-to-disk transforms |

Only completed notebooks are linked. New lessons should retain an explicit
data-origin statement, a replaceable input section, units/shape explanations,
expected results, and checks suitable for fresh-kernel execution.

To execute only the visualization tutorial:

```bash
python scripts/execute_tutorials.py --notebook 02_visualization_and_virtual_detectors.ipynb
```

Tutorial 2 is independent of Tutorial 1's temporary files. It constructs a
regular hexagonal spot pattern in two deliberately rotated teaching regions.
It covers diffraction overlays, a scan montage, annuli, displaced/combined
disks, Boolean/weighted masks, units, calibrated scale bars, and an equivalent
disk-backed virtual image. Original microscope data remains optional.

Tutorial 3 independently constructs a translated central disk and regular
hexagon. It checks alignment against known motion and an independent centroid,
reuses beam metadata, demonstrates Cartesian/polar rotations, and quantifies
lossy Cartesian reconstruction. It distinguishes exact integer angular rolls
from interpolated transforms and includes an opt-in microscope-alignment cell.

Tutorial 4 selects its input before the common denoising workflow. It compares
per-pattern and unfolded-volume filtering, displays paths and processing
coverage, preserves small diagnostics instead of every 4D output, and includes
a short TensorLy convergence example. Rank sweeps and BM4D are explicit opt-ins.

Tutorial 5 constructs strong/weak spot and diffuse regions. It distinguishes
detection scores from integrated intensities, demonstrates local-background
correction, checks localization and the Braggness formula, and reuses alignment
beam metadata on a separate aligned copy. It exports small numeric ragged
tables without pickle, not a phase map or a crystalline volume fraction.

Tutorial 6 fits synthetic subpixel hexagonal peaks with known reciprocal
stretch and rotation, then checks reporting under a known R/Q rotation and
reflection. It reuses beam metadata, compares weighting, and exports maps
with frame information. Its exact checks do not infer a real calibration.
External inputs require their own measured/reference peaks; verified real-frame
reporting is explicit, while unverified inputs retain detector components.

Tutorial 7 constructs its HDF5 fixture one pattern at a time, then uses
read-only sources throughout. It checks chunk coverage/origins, selected
peak-file reads, streamed/single-pattern transforms, output protection, and
resource lifetime. It demonstrates a bounded execution path, not a large-data
RAM benchmark. External peak extraction and transforms are explicit opt-ins.

The first six input cells accept a file, a numerical array, or an existing
`HyperData`, including simulation output. Synthetic calibration and reference
values are never used as defaults for an external input. Tutorials 1-3 keep
their earlier known-truth exercises synthetic and provide a separate selected-
input branch; Tutorials 4-6 process the selected input throughout. See
[using your own data](../docs/tutorials/using-your-own-data.md) before migrating
the workflows to an acquisition or simulation.

Tutorial 7 is intentionally file-based. Use a verified lazy-readable
acquisition/simulation file, or save an existing HyperData explicitly first.
It never copies an in-memory input and calls that a lazy workflow. No full
diffraction or peak collection is loaded in its default path. This completes
the seven-notebook foundation; all source notebooks remain output-free.
