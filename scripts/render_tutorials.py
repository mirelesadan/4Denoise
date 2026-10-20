"""Prepare only curated tutorials for static MyST-NB rendering; never execute."""

from copy import deepcopy
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from scripts.execute_demo import clear_execution_state, validate_execution
from scripts.execute_tutorials import TUTORIALS
from scripts.notebook_reports import source_signature, validate_report_provenance


OVERVIEW_PAGES = (
    'loading-and-calibration', 'visualization-and-virtual-detectors',
    'alignment-and-coordinate-transforms', 'denoising-and-unfolding',
    'peak-analysis-and-braggness', 'strain-mapping-and-rq-calibration',
    'large-data-processing',
)


def _rewrite_links(text, root, destination):
    """Resolve notebook-relative Markdown links in the generated site location."""
    def replace(match):
        url = match.group(1)
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        target = (root / 'tutorials' / unquote(parts.path)).resolve()
        if target.parent == root / 'tutorials' and target.name in TUTORIALS:
            rewritten = target.name
        elif target.is_relative_to(root / 'docs'):
            rewritten = Path(os.path.relpath(target, destination.parent)).as_posix()
        else:
            return match.group(0)
        return '](' + urlunsplit(('', '', quote(rewritten, safe='/'), parts.query, parts.fragment)) + ')'
    return re.sub(r'\]\(([^\s)]+)\)', replace, text)


def _render_copy(notebook, name, overview, root, destination, executed):
    """Adjust only a generated copy, keeping downloadable sources untouched."""
    import nbformat

    notebook = deepcopy(notebook)
    notebook['metadata'].pop('widgets', None)
    notebook['metadata']['mystnb'] = {'execution_mode': 'off'}
    notebook['metadata']['4denoise_rendered'] = {'source': name, 'executed': executed}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            cell['source'] = _rewrite_links(cell['source'], root, destination)
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            data.pop('application/vnd.jupyter.widget-view+json', None)
            data.pop('application/javascript', None)
            if 'data' in output and not data:
                data['text/plain'] = 'Interactive output omitted; run the downloaded notebook locally.'
    status = ('Code and figures below come from a validated fresh-kernel run.' if executed else
              'Source-only rendering for link validation; no output figures are claimed.')
    banner = nbformat.v4.new_markdown_cell(
        ':::{note}\n'
        '**Synthetic teaching data, not experimental validation.** ' + status + '\n\n'
        f'{{download}}`Download the clean notebook <../../../tutorials/{name}>` '
        f'or read the [lesson overview](../{overview}.md).\n'
        ':::\n', id='rendered-tutorial-notice',
    )
    notebook['cells'].insert(1, banner)
    return notebook


def prepare_tutorials(root, *, require_executed=True):
    """Validate all reports before generating pages inside the docs tree.

    HTML/other publication builds require successful current reports. Linkcheck
    uses source-only copies, never cached figures and never a running kernel.
    All paths come from the explicit registry; research notebooks are ignored.
    """
    import nbformat

    root = Path(root).resolve()
    destination = root / 'docs' / 'tutorials' / 'notebooks'
    if destination.is_symlink() or not destination.resolve().is_relative_to(root / 'docs'):
        raise ValueError('Generated tutorial destination must stay inside the documentation tree.')
    if len(OVERVIEW_PAGES) != len(TUTORIALS):
        raise ValueError('Each registered tutorial needs one explicit overview mapping.')
    if destination.exists():
        unexpected = {path.name for path in destination.glob('*.ipynb')} - set(TUTORIALS)
        if unexpected:
            raise ValueError('Unexpected notebook in generated tutorial directory: ' + ', '.join(sorted(unexpected)))
    prepared = []
    for name, overview in zip(TUTORIALS, OVERVIEW_PAGES):
        source = root / 'tutorials' / name
        original = nbformat.read(source, as_version=4)
        nbformat.validate(original)
        if any(cell.get('outputs') or cell.get('execution_count') is not None
               for cell in original['cells'] if cell['cell_type'] == 'code'):
            raise ValueError(f'{name}: downloadable notebook sources must have clean execution state.')
        if original['metadata'].get('4denoise', {}).get('data_origin') != 'synthetic':
            raise ValueError(f'{name} must retain its explicit synthetic-data declaration.')
        if require_executed:
            report = root / 'docs' / '_build' / 'tutorials' / name
            if not report.is_file():
                raise ValueError(f'Missing executed tutorial {name}; run python scripts/execute_tutorials.py first.')
            notebook = nbformat.read(report, as_version=4)
            nbformat.validate(notebook)
            validate_report_provenance(notebook, source, root)
            validate_execution(notebook)
            if source_signature(notebook) != source_signature(original):
                raise ValueError(f'{name}: executed cells do not match the clean source notebook.')
            if notebook['metadata'].get('4denoise') != original['metadata'].get('4denoise'):
                raise ValueError(f'{name}: executed report changed the declared teaching-data metadata.')
        else:
            notebook = deepcopy(original)
            clear_execution_state(notebook)
        target = destination / name
        prepared.append((target, _render_copy(notebook, name, overview, root, target, require_executed)))
    destination.mkdir(parents=True, exist_ok=True)
    for target, notebook in prepared:
        content = nbformat.writes(notebook, version=4)
        if not target.exists() or target.read_text(encoding='utf-8') != content:
            target.write_text(content, encoding='utf-8')
    return [target for target, _ in prepared]
