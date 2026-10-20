"""Execute only the explicitly curated tutorial notebooks in fresh kernels.

Run from the repository root with ``python scripts/execute_tutorials.py``.
Reports go to docs/_build/tutorials; source notebooks are never overwritten.
No glob discovers research notebooks, and no microscope dataset is required.
"""

import argparse
from pathlib import Path

if __package__:
    from .execute_demo import execute_notebook
else:
    from execute_demo import execute_notebook


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = (
    '01_loading_and_calibration.ipynb',
    '02_visualization_and_virtual_detectors.ipynb',
    '03_alignment_and_coordinate_transforms.ipynb',
    '04_denoising_and_unfolding.ipynb',
    '05_peak_analysis_and_braggness.ipynb',
    '06_strain_mapping_and_rq_calibration.ipynb',
    '07_large_data_processing.ipynb',
)


def execute_tutorials(root=ROOT, *, notebooks=None, timeout=120):
    """Validate the selection, then execute each registered tutorial separately."""
    root = Path(root).resolve()
    selected = TUTORIALS if notebooks is None else tuple(notebooks)
    if not selected or any(name not in TUTORIALS for name in selected):
        raise ValueError('Select only registered tutorial notebook names.')
    if len(set(selected)) != len(selected):
        raise ValueError('Tutorial notebook names must not be repeated.')
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError('timeout must be a positive integer number of seconds.')
    for name in selected:
        if not (root / 'tutorials' / name).is_file():
            raise FileNotFoundError(f'Missing registered tutorial: {name}')
    return [
        execute_notebook(
            root / 'tutorials' / name, root=root / 'tutorials',
            output=root / 'docs' / '_build' / 'tutorials' / name, timeout=timeout,
        )
        for name in selected
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', action='append', choices=TUTORIALS,
                        help='Registered notebook to run; repeat to select several.')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Maximum seconds per code cell (default: 120).')
    args = parser.parse_args()
    execute_tutorials(notebooks=args.notebook, timeout=args.timeout)


if __name__ == '__main__':
    main()
