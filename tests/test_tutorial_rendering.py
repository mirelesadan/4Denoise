"""Contracts for rendering validated notebooks without executing them in Sphinx."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts import notebook_reports as reports
from scripts import render_tutorials as render
from scripts.execute_tutorials import TUTORIALS


ROOT = Path(__file__).resolve().parents[1]
HAS_NBFORMAT = importlib.util.find_spec('nbformat') is not None


class NotebookReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'lesson.ipynb'
        self.source.write_text('source\n', encoding='utf-8')
        (self.root / 'fourdenoise.py').write_text('# kernel\n', encoding='utf-8')
        self.notebook = {'metadata': {}, 'cells': [dict(id='code', cell_type='code',
                         source='print(1)', execution_count=1, metadata={},
                         outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': '1\n'}])]}
        self.notebook['metadata'][reports.REPORT_KEY] = reports.new_report_provenance(self.source, self.root)

    def test_successful_report_records_current_context_and_output_digest(self):
        reports.complete_report(self.notebook, self.source, self.root)
        reports.validate_report_provenance(self.notebook, self.source, self.root)
        record = self.notebook['metadata'][reports.REPORT_KEY]
        self.assertEqual(record['status'], 'complete')
        self.assertIn('fourdenoise.py', record['context']['files'])
        self.assertIn('numpy', record['packages'])

    def test_incomplete_source_changed_code_changed_and_tampered_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'successful execution provenance'):
            reports.validate_report_provenance(self.notebook, self.source, self.root)
        reports.complete_report(self.notebook, self.source, self.root)
        self.notebook['cells'][0]['outputs'][0]['text'] = '2\n'
        with self.assertRaisesRegex(ValueError, 'outputs changed'):
            reports.validate_report_provenance(self.notebook, self.source, self.root)
        self.source.write_text('changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'current notebook source'):
            reports.validate_report_provenance(self.notebook, self.source, self.root)

    def test_code_changes_invalidate_reports_but_guide_edits_do_not(self):
        reports.complete_report(self.notebook, self.source, self.root)
        (self.root / 'guide.md').write_text('Updated prose', encoding='utf-8')
        reports.validate_report_provenance(self.notebook, self.source, self.root)
        (self.root / 'fourdenoise.py').write_text('# changed kernel\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'stale package/executor'):
            reports.validate_report_provenance(self.notebook, self.source, self.root)

    def test_mid_run_changes_cannot_be_marked_successful(self):
        self.source.write_text('changed mid-run', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'changed during the run'):
            reports.complete_report(self.notebook, self.source, self.root)
        self.assertEqual(self.notebook['metadata'][reports.REPORT_KEY]['status'], 'failed')

    def test_context_is_independent_of_line_endings_and_checkout_location(self):
        before = reports.execution_context(self.root)
        (self.root / 'fourdenoise.py').write_bytes(b'# kernel\r\n')
        self.assertEqual(reports.execution_context(self.root), before)
        other = self.root / 'another-checkout'
        other.mkdir()
        (other / 'fourdenoise.py').write_bytes(b'# kernel\n')
        self.assertEqual(reports.execution_context(other), before)

    def test_source_signature_ignores_only_execution_timing(self):
        original = deepcopy(self.notebook)
        self.notebook['cells'][0]['metadata']['execution'] = {'start': 'now'}
        self.assertEqual(reports.source_signature(original), reports.source_signature(self.notebook))
        self.notebook['cells'][0]['source'] = 'print(2)'
        self.assertNotEqual(reports.source_signature(original), reports.source_signature(self.notebook))


@unittest.skipUnless(HAS_NBFORMAT, 'Install the docs extra for notebook-rendering tests.')
class TutorialRenderingTests(unittest.TestCase):
    def setUp(self):
        import nbformat
        self.nbformat = nbformat
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / 'tutorials').mkdir()
        (self.root / 'docs/tutorials').mkdir(parents=True)
        (self.root / 'docs/_build/tutorials').mkdir(parents=True)
        (self.root / 'fourdenoise.py').write_text('# numerical module\n', encoding='utf-8')
        self.source_bytes = {}
        self.report_bytes = {}
        for index, name in enumerate(TUTORIALS):
            original = nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_markdown_cell('# Synthetic lesson\n\n[Guide](../docs/guides/io-and-calibration.md#native)\n'
                                            '[Next](02_visualization_and_virtual_detectors.ipynb)\n'
                                            '[External](https://example.org/page?q=1#part)'),
                nbformat.v4.new_code_cell('print(1)', id='code'),
            ], metadata={'4denoise': {'data_origin': 'synthetic', 'tutorial': index + 1},
                         'mystnb': {'execution_mode': 'force'}})
            source = self.root / 'tutorials' / name
            nbformat.write(original, source)
            executed = deepcopy(original)
            executed['cells'][1]['execution_count'] = 1
            executed['cells'][1]['outputs'] = [nbformat.v4.new_output('display_data', data={
                'image/png': 'aW1hZ2U=', 'text/plain': 'Static plot',
                'application/vnd.jupyter.widget-view+json': {'model_id': 'fixture'},
            }), nbformat.v4.new_output('stream', name='stderr', text='A visible diagnostic\n')]
            executed['metadata']['widgets'] = {'state': 'not needed by a static page'}
            executed['metadata'][reports.REPORT_KEY] = reports.new_report_provenance(source, self.root)
            reports.complete_report(executed, source, self.root)
            report = self.root / 'docs/_build/tutorials' / name
            nbformat.write(executed, report)
            self.source_bytes[name] = source.read_bytes()
            self.report_bytes[name] = report.read_bytes()

    def change_report(self, name, change, *, reseal=False):
        path = self.root / 'docs/_build/tutorials' / name
        notebook = self.nbformat.read(path, as_version=4)
        change(notebook)
        if reseal:
            reports.complete_report(notebook, self.root / 'tutorials' / name, self.root)
        self.nbformat.write(notebook, path)

    def test_static_rendering_preserves_code_figures_and_downloadable_sources(self):
        generated = render.prepare_tutorials(self.root)
        self.assertEqual(len(generated), len(TUTORIALS))
        for path in generated:
            notebook = self.nbformat.read(path, as_version=4)
            self.assertEqual(notebook['metadata']['mystnb']['execution_mode'], 'off')
            self.assertNotIn('widgets', notebook['metadata'])
            self.assertIn('Synthetic teaching data', notebook['cells'][1]['source'])
            code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
            self.assertEqual(code['source'], 'print(1)')
            self.assertEqual(code['outputs'][0]['data']['image/png'], 'aW1hZ2U=')
            self.assertNotIn('application/vnd.jupyter.widget-view+json', code['outputs'][0]['data'])
            self.assertEqual(code['outputs'][1]['text'], 'A visible diagnostic\n')
            self.assertEqual((self.root / 'tutorials' / path.name).read_bytes(), self.source_bytes[path.name])
            self.assertEqual((self.root / 'docs/_build/tutorials' / path.name).read_bytes(), self.report_bytes[path.name])

    def test_notebook_relative_links_are_rebased_but_external_links_are_unchanged(self):
        path = render.prepare_tutorials(self.root)[0]
        markdown = self.nbformat.read(path, as_version=4)['cells'][0]['source']
        self.assertIn('../../guides/io-and-calibration.md#native', markdown)
        self.assertIn('](02_visualization_and_virtual_detectors.ipynb)', markdown)
        self.assertIn('https://example.org/page?q=1#part', markdown)

    def test_missing_reports_fail_publication_without_generating_partial_pages(self):
        (self.root / 'docs/_build/tutorials' / TUTORIALS[-1]).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing executed tutorial'):
            render.prepare_tutorials(self.root)
        self.assertFalse((self.root / 'docs/tutorials/notebooks').exists())

    def test_source_only_linkcheck_ignores_reports_and_strips_all_cached_outputs(self):
        for path in (self.root / 'docs/_build/tutorials').iterdir():
            path.unlink()
        paths = render.prepare_tutorials(self.root, require_executed=False)
        for path in paths:
            notebook = self.nbformat.read(path, as_version=4)
            self.assertFalse(notebook['metadata']['4denoise_rendered']['executed'])
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])

    def test_changed_cells_are_rejected_even_if_output_digest_is_valid(self):
        self.change_report(TUTORIALS[0], lambda nb: nb['cells'][1].update(source='print(2)'))
        with self.assertRaisesRegex(ValueError, 'cells do not match'):
            render.prepare_tutorials(self.root)

    def test_error_outputs_and_skipped_cells_are_rejected_even_with_a_complete_stamp(self):
        name = TUTORIALS[0]
        self.change_report(name, lambda nb: nb['cells'][1].update(execution_count=None), reseal=True)
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            render.prepare_tutorials(self.root)
        (self.root / 'docs/_build/tutorials' / name).write_bytes(self.report_bytes[name])
        self.change_report(name, lambda nb: nb['cells'][1]['outputs'].append(
            self.nbformat.v4.new_output('error', ename='ValueError', evalue='bad', traceback=[])), reseal=True)
        with self.assertRaisesRegex(RuntimeError, 'contains an error'):
            render.prepare_tutorials(self.root)

    def test_incomplete_provenance_and_stale_numeric_modules_fail(self):
        name = TUTORIALS[0]
        self.change_report(name, lambda nb: nb['metadata'].pop(reports.REPORT_KEY))
        with self.assertRaisesRegex(ValueError, 'successful execution provenance'):
            render.prepare_tutorials(self.root)
        (self.root / 'docs/_build/tutorials' / name).write_bytes(self.report_bytes[name])
        (self.root / 'fourdenoise.py').write_text('# changed kernel', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'stale package/executor'):
            render.prepare_tutorials(self.root)

    def test_non_synthetic_declarations_and_dirty_downloads_fail(self):
        path = self.root / 'tutorials' / TUTORIALS[0]
        source = self.nbformat.read(path, as_version=4)
        source['metadata']['4denoise']['data_origin'] = 'experimental'
        self.nbformat.write(source, path)
        with self.assertRaisesRegex(ValueError, 'synthetic-data declaration'):
            render.prepare_tutorials(self.root, require_executed=False)
        path.write_bytes(self.source_bytes[path.name])
        source = self.nbformat.read(path, as_version=4)
        source['cells'][1]['execution_count'] = 1
        self.nbformat.write(source, path)
        with self.assertRaisesRegex(ValueError, 'clean execution state'):
            render.prepare_tutorials(self.root, require_executed=False)

    def test_unregistered_research_files_are_never_discovered(self):
        (self.root / 'tutorials/research.ipynb').write_text('not a notebook', encoding='utf-8')
        paths = render.prepare_tutorials(self.root)
        self.assertEqual({path.name for path in paths}, set(TUTORIALS))
        (paths[0].parent / 'research.ipynb').write_text('unexpected output', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unexpected notebook'):
            render.prepare_tutorials(self.root)

    def test_unchanged_preparation_does_not_rewrite_generated_sources(self):
        paths = render.prepare_tutorials(self.root)
        timestamps = {path: path.stat().st_mtime_ns for path in paths}
        render.prepare_tutorials(self.root)
        self.assertEqual({path: path.stat().st_mtime_ns for path in paths}, timestamps)


class TutorialSiteConfigurationTests(unittest.TestCase):
    def test_ci_reuses_current_run_reports_instead_of_reexecuting_in_html_build(self):
        workflow = (ROOT / '.github/workflows/docs.yml').read_text(encoding='utf-8')
        build = workflow.split('  docs-build:', 1)[1].split('  docs-links:', 1)[0]
        self.assertIn('needs: demo-notebook', build)
        self.assertIn('name: executed-tutorial-notebooks', build)
        self.assertIn('path: docs/_build/tutorials', build)
        self.assertNotIn('run: python scripts/execute_tutorials.py', build)
        self.assertIn('test_tutorial_rendering.py', build)

    def test_all_overviews_offer_online_lessons_and_keep_clean_downloads(self):
        for name, overview in zip(TUTORIALS, render.OVERVIEW_PAGES):
            text = (ROOT / 'docs/tutorials' / f'{overview}.md').read_text(encoding='utf-8')
            self.assertIn(f'(notebooks/{name})', text)
            self.assertIn('{download}', text)
            self.assertIn('notebooks/' + name.removesuffix('.ipynb'), text)
        config = (ROOT / 'docs/conf.py').read_text(encoding='utf-8')
        self.assertIn('nb_execution_mode = "off"', config)
        self.assertIn('/docs/tutorials/notebooks/', (ROOT / '.gitignore').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
