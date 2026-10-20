# 4Denoise Guide

4Denoise provides loading, visualization, denoising, peak analysis, and strain
mapping for 4D-STEM data. Start with [installation](getting-started/install.md)
and the [quickstart](getting-started/quickstart.md), which uses the included
mini dataset. Continue with the [streaming tutorial](getting-started/streaming.md)
to keep diffraction pixels on disk, or the full
[DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb)
for the in-memory peak-refinement and relative-strain workflow.

For self-contained lessons without microscope data, start with the
[tutorial notebook series](tutorials/index.md). Seven notebooks cover
loading/calibration, visualization/virtual detectors, alignment/coordinate
transforms, denoising/unfolding, peak analysis/Braggness, strain/RQ calibration,
and disk-backed large-data processing using reproducible
synthetic data, including regular hexagonal diffraction patterns. Each also accepts a chosen
experimental or simulation input, with a file-based contract for the last lesson;
see [using your own data](tutorials/using-your-own-data.md).
The lessons can be read directly online with code and validated figures, or
downloaded as clean notebooks. Synthetic results verify the teaching workflow,
not an original acquisition or an experimental physical interpretation.

`HyperData(path)` loads into RAM; `HyperData.open(path)` provides a read-only
source with bounded processing and selected disk-output operations. Neither
an installed reader nor a preserved vendor field implies that an original
microscope acquisition or its physical calibration has been validated.

## Concepts and Tasks

- [Data model](concepts/data-model.md): in-memory objects, sources, peak collections, axes, and calibration.
- [I/O and calibration](guides/io-and-calibration.md): native and optional RosettaSciIO readers, metadata, streamed processing, and saving.
- [Visualization](guides/visualization.md): diffraction patterns, virtual detectors, and calibrated axes.
- [Denoising and unfolding](guides/denoising-and-unfolding.md): what happens to excluded data during a traversal.
- [Peaks and strain](guides/peaks-and-strain.md): ragged peak collections, incremental peak files, refinement, and strain diagnostics.
- [Braggness](guides/braggness.md): concentrated sharp diffraction contrast and validity masks.

The [API reference](reference/hyperdata.rst) lists selected public entry points
and draws their signatures and descriptions from the source docstrings.

## Research Workflow

The [3D strain-mapping case study](case-studies/3d-strain-mapping.md) links the
full experimental and simulation notebooks and describes the external data,
compute requirements, and MATLAB GUI. It is not a routine installation test.

This guide follows the repository's `main` branch. GitHub Actions validates
the documentation, mini-data DEMO, and registered tutorials before publishing updates to GitHub
Pages. You can also [build the guide locally](getting-started/install.md#build-this-guide).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/install
getting-started/quickstart
getting-started/streaming
concepts/data-model
```

```{toctree}
:maxdepth: 2
:caption: Tutorial notebooks

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Tasks

guides/io-and-calibration
guides/visualization
guides/denoising-and-unfolding
guides/peaks-and-strain
guides/braggness
```

```{toctree}
:maxdepth: 2
:caption: Reference and research

reference/hyperdata
reference/views-and-results
reference/utilities
case-studies/3d-strain-mapping
troubleshooting
```
