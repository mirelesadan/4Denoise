# 4Denoise

4Denoise is a research-oriented Python toolkit for loading, visualizing,
denoising, and analyzing four-dimensional scanning transmission electron
microscopy (4D-STEM) data. Its main `HyperData` object represents a scan as
`(Ry, Rx, Ky, Kx)`: two scan axes followed by two diffraction axes.

The included mini dataset and DEMO notebook are the fastest way to try the
package. The full 3D strain-mapping study uses separate, much larger data and
optional simulation and MATLAB tools.

## Install

The supplied Conda environment uses Python 3.11. From a terminal:

```bash
git clone https://github.com/mirelesadan/4Denoise.git
cd 4Denoise
conda env create -f environment.yml
conda activate 4denoise-main
python -m pip install -e .
python -c "from fourdenoise import HyperData; print(HyperData.__name__)"
```

For a pip-only setup, use `python -m pip install -e ".[notebook]"` in a
Python 3.10+ environment. Microscopy readers beyond native NumPy/HDF5 are
optional via `python -m pip install -e ".[rosetta]"`. The study's simulation
workflow additionally needs
`python -m pip install -e ".[simulation]"`; BM3D/BM4D support is optional via
`.[bm]`. An editable install uses the current local source when you edit it.

## Quick Example

Run this in Jupyter from the repository root, where
`mini_dataset_binned.npy` is included:

```python
from fourdenoise import HyperData

data = HyperData("mini_dataset_binned.npy")
print(data.shape)  # (15, 13, 128, 128)

image = data.virtual_image(
    annulus=(37, 50), detector_units="pixels", show=False,
)
image.show(title="Annular virtual image")
```

For a fuller verified example, run
[the DEMO notebook](DEMO_exp_4dstem_ripple_processing.ipynb) from a fresh
Jupyter kernel with `jupyter lab`.

## Larger Datasets

`HyperData(path)` loads the complete dataset into RAM. Use a read-only source
when diffraction pixels should stay on disk:

```python
with HyperData.open("mini_dataset_binned.npy") as source:
    image = source.virtual_image(
        annulus=(37, 50), detector_units="pixels",
        working_memory_mb=64, show=False,
    )

image.show()  # The compact RealSpace result survives source closure.
```

Sources support bounded reductions, virtual images, independent peak
detection, and selected transforms written directly to disk. They do not make
every analysis method lazy; global tensor denoising remains in memory.
See the [executable streaming tutorial](docs/getting-started/streaming.md)
and [I/O guide](docs/guides/io-and-calibration.md) for supported backends,
metadata, memory-budget limits, and the separate `PeakScanResult` file format.

## Explore

- [Documentation website](https://mirelesadan.github.io/4Denoise/): installation, task guides, a curated API reference, and the case study.
- [Tutorial notebooks](tutorials/README.md): seven lessons covering loading/calibration, visualization, alignment/transforms, denoising/unfolding, peaks/Braggness, strain/RQ calibration, and disk-backed processing, with synthetic patterns and replaceable experimental/simulation inputs.
- [3D strain-mapping case study](https://mirelesadan.github.io/4Denoise/case-studies/3d-strain-mapping.html): external data, experimental notebook, simulation, and MATLAB GUI.
- [Experimental processing notebook](exp_MoS2_MoSe2_processing.ipynb): the full, compute-intensive analysis.
- [Simulation notebook](generateSimulatedRipple_4Ddata.ipynb): the companion ripple simulation workflow.

The large experimental and simulated datasets are not bundled with this
repository; they are available from the
[Zenodo data record](https://zenodo.org/records/17246822). The case study
explains which stages have been verified. The documentation guide can be built
locally with `python -m pip install -e ".[docs,docs-test]"`, then
`python scripts/execute_tutorials.py` and
`python -m sphinx -b html -W --keep-going docs docs/_build/html`. Tutorial pages
render the validated code and figures without executing kernels during the
HTML build; clean notebooks remain downloadable. Its
[source pages](docs/index.md) are maintained in this repository. GitHub Pages
publishes the validated `main` branch automatically; PyPI publication is not
required for either the website or the editable install above.

To run the local unit tests: `python -m unittest discover -s tests`.

Pull requests also check the documentation build, guide links, and a
fresh-kernel runs of the mini-data DEMO and registered synthetic tutorials. See the
[documentation maintenance notes](docs/README.md#automated-checks-and-maintenance)
for local commands and downloadable CI reports. The large experimental
workflow is not executed by these checks.
