"""Small, dependency-free provenance checks for executed notebook reports."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform


REPORT_KEY = '4denoise_execution'
REPORT_VERSION = 1


def source_digest(path):
    """Hash source text consistently across LF/CRLF checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def execution_context(root):
    """Fingerprint numerical modules, packaging, and execution infrastructure.

    Documentation prose and tests do not invalidate numerical results. Paths
    are repository-relative, so separate CI jobs/checkouts can share reports.
    This detects accidental stale artifacts, not maliciously forged reports.
    """
    root = Path(root).resolve()
    if root.name == 'tutorials':
        root = root.parent
    paths = set(root.glob('fourdenoise*.py'))
    for name in ('abTEM_helpers.py', 'rippleModel.py', 'kin_diff_sim.py',
                 'pyproject.toml', 'scripts/execute_demo.py',
                 'scripts/execute_tutorials.py', 'scripts/notebook_reports.py'):
        path = root / name
        if path.is_file():
            paths.add(path)
    files = {path.relative_to(root).as_posix(): source_digest(path)
             for path in sorted(paths)}
    payload = json.dumps(files, sort_keys=True).encode('utf-8')
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'files': files}


def output_digest(notebook):
    """Fingerprint code execution counts and saved outputs, excluding timing."""
    payload = [(cell.get('id'), cell.get('execution_count'), cell.get('outputs', []))
               for cell in notebook['cells'] if cell['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def new_report_provenance(source, root):
    """Create a failed-by-default execution record before starting a kernel."""
    packages = {}
    for name in ('numpy', 'scipy', 'matplotlib', 'scikit-image', 'scikit-learn',
                 'tensorly', 'h5py', 'nbclient', 'ipykernel'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'not installed'
    return dict(version=REPORT_VERSION, status='failed', source=Path(source).name,
                source_sha256=source_digest(source), context=execution_context(root),
                python=platform.python_version(), packages=packages)


def complete_report(notebook, source, root):
    """Mark a report complete only if execution inputs did not change mid-run."""
    record = notebook['metadata'][REPORT_KEY]
    if (record['source_sha256'] != source_digest(source)
            or record['context'] != execution_context(root)):
        raise RuntimeError('Notebook execution inputs changed during the run; rerun before publishing.')
    record['outputs_sha256'] = output_digest(notebook)
    record['status'] = 'complete'


def validate_report_provenance(notebook, source, root):
    """Reject failed, changed, or stale reports before rendering their outputs."""
    record = notebook.get('metadata', {}).get(REPORT_KEY, {})
    if record.get('version') != REPORT_VERSION or record.get('status') != 'complete':
        raise ValueError('Missing successful execution provenance; rerun scripts/execute_tutorials.py.')
    if record.get('source') != Path(source).name or record.get('source_sha256') != source_digest(source):
        raise ValueError('Executed report does not match its current notebook source; rerun the tutorial.')
    if record.get('context') != execution_context(root):
        raise ValueError('Executed report has stale package/executor inputs; rerun scripts/execute_tutorials.py.')
    if record.get('outputs_sha256') != output_digest(notebook):
        raise ValueError('Saved notebook outputs changed after validation; rerun the tutorial.')


def source_signature(notebook):
    """Compare original cell identities/content/metadata, ignoring run timing."""
    signatures = []
    for cell in notebook['cells']:
        metadata = {key: value for key, value in cell.get('metadata', {}).items()
                    if key != 'execution'}
        signatures.append((cell.get('id'), cell['cell_type'], cell['source'], metadata))
    return signatures
