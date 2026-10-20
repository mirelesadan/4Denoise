# Install 4Denoise

The documented Conda environment uses Python 3.11. An editable install is
recommended while the package is being developed; publishing to PyPI is not
required. Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate 4denoise-main
python -m pip install -e .
python -c "from fourdenoise import HyperData; print(HyperData.__name__)"
```

If the environment already exists, use
`conda env update -f environment.yml --prune` instead of creating it again.
The Python distribution is named `fourdenoise-main`, while the import is
`fourdenoise`.

## Optional Tools

In a pip-only Python 3.10+ environment, the package metadata provides
optional dependencies:

```bash
python -m pip install -e ".[notebook]"    # Jupyter and notebook analysis tools
python -m pip install -e ".[simulation]"  # abTEM/ASE simulation tools
python -m pip install -e ".[bm]"          # BM3D and BM4D denoisers
python -m pip install -e ".[rosetta]"     # RosettaSciIO microscopy readers
```

The Conda environment already includes Jupyter and most notebook tools, but
the complete simulation extra is separate. Install only the extras your
workflow needs. You can check the active interpreter inside Jupyter with:

```python
import sys
print(sys.executable)
```

If this path is not from the intended environment, select the correct
Jupyter kernel and restart it before importing `fourdenoise`.

After pulling changes that add modules or dependencies, rerun the editable
installation command with your required extras in that same environment,
then restart the notebook kernel. Merely changing the working directory
does not update the imported package or already-created object classes.

The `rosetta` extra currently supports RosettaSciIO 0.14.x and installs Dask
for lazy reader arrays. It is optional: native NumPy/HDF5 I/O and the core
package do not import RosettaSciIO at startup. Some plugin families require
additional reader-specific dependencies; see the
[RosettaSciIO format documentation](https://hyperspy.org/rosettasciio/supported_formats/index.html).
The [I/O guide](../guides/io-and-calibration.md#rosettasciio-microscopy-formats)
explains dataset selection, axis ordering and the limits of lazy reading.
The [streaming tutorial](streaming.md) uses only core dependencies and the
included NumPy mini dataset; RosettaSciIO is not required to run it.

## Build This Guide

Install the documentation and notebook-test extras, execute the seven small
synthetic tutorials once, then build HTML locally:

```bash
python -m pip install -e ".[docs,docs-test]"
python scripts/execute_tutorials.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. Sphinx uses MyST-NB to render
the validated reports into static pages with code and figures. It never runs
notebook kernels itself. Missing, failed, changed, or stale tutorial reports
stop an HTML build with instructions to rerun the tutorials. Changing package
code or execution infrastructure invalidates the reports; ordinary guide edits
do not. Link checking uses source-only copies and needs no execution reports.

The build imports the core
`fourdenoise` module to read docstrings; it does not run the large
experimental notebook. The public site is rebuilt from the validated `main`
branch, so it describes the development version rather than a versioned
PyPI release.

Check the executable streaming example separately from rendering the pages:

```bash
python scripts/streaming_example.py
```

It uses temporary outputs by default and does not modify the mini dataset or
research notebooks. The [streaming tutorial](streaming.md) explains its
outputs and validation boundaries.

Continue with the [mini-data quickstart](quickstart.md).
Alternatively, [Tutorial 1: Loading and calibration](../tutorials/loading-and-calibration.md)
creates its own synthetic data. Registered tutorial notebooks can be checked
in fresh kernels with `python scripts/execute_tutorials.py` after installing
`.[docs-test]`; source notebooks are never overwritten.
