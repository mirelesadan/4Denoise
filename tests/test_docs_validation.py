"""Notebook validation contracts that do not require optional Jupyter packages."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from scripts import execute_tutorials as tutorials


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scripts.execute_demo', ROOT / 'scripts' / 'execute_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
conf_spec = importlib.util.spec_from_file_location('docs_conf', ROOT / 'docs' / 'conf.py')
docs_conf = importlib.util.module_from_spec(conf_spec)
conf_spec.loader.exec_module(docs_conf)


class DemoValidationTests(unittest.TestCase):
    def test_cached_outputs_and_execution_metadata_are_cleared(self):
        notebook = {'cells': [
            {'cell_type': 'markdown', 'source': 'Instructions', 'metadata': {}},
            {'cell_type': 'code', 'source': 'print(1)', 'execution_count': 5,
             'outputs': [{'output_type': 'stream', 'text': 'old output'}],
             'metadata': {'execution': {'old': True}, 'tags': ['example']}},
        ]}
        demo.clear_execution_state(notebook)
        self.assertEqual(notebook['cells'][1]['outputs'], [])
        self.assertIsNone(notebook['cells'][1]['execution_count'])
        self.assertEqual(notebook['cells'][1]['metadata'], {'tags': ['example']})
        self.assertNotIn('outputs', notebook['cells'][0])

    def test_skipped_nonempty_cells_are_rejected(self):
        cell = {'cell_type': 'code', 'source': 'print(1)', 'execution_count': None, 'outputs': []}
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            demo.validate_execution({'cells': [cell]})

    def test_error_outputs_are_rejected(self):
        cell = {'cell_type': 'code', 'source': '1 / 0', 'execution_count': 1,
                'outputs': [{'output_type': 'error', 'ename': 'ZeroDivisionError'}]}
        with self.assertRaisesRegex(RuntimeError, 'contains an error'):
            demo.validate_execution({'cells': [cell]})

    def test_valid_execution_ignores_markdown_and_empty_cells(self):
        notebook = {'cells': [
            {'cell_type': 'markdown', 'source': 'Instructions'},
            {'cell_type': 'code', 'source': '  ', 'execution_count': None},
            {'cell_type': 'code', 'source': 'print(1)', 'execution_count': 1, 'outputs': []},
        ]}
        self.assertEqual(demo.validate_execution(notebook), 1)
        with self.assertRaisesRegex(RuntimeError, 'no executable'):
            demo.validate_execution({'cells': notebook['cells'][:2]})

    def test_output_is_separate_from_source(self):
        output = demo.resolve_output(ROOT)
        self.assertEqual(output, ROOT / 'docs' / '_build' / 'demo' / demo.DEMO)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            demo.resolve_output(ROOT, ROOT / demo.DEMO)
        with self.assertRaisesRegex(ValueError, 'extension'):
            demo.resolve_output(ROOT, ROOT / 'report.txt')


class LinkValidationTests(unittest.TestCase):
    def test_repo_file_links_check_the_exact_file_on_the_content_host(self):
        for repository in ('4Denoise', '4denoise'):
            uri = f'https://github.com/mirelesadan/{repository}/blob/main/missing_file.py'
            self.assertEqual(
                docs_conf._github_file_check_uri(None, uri),
                'https://raw.githubusercontent.com/mirelesadan/4Denoise/main/missing_file.py',
            )

    def test_other_destinations_queries_and_fragments_are_unchanged(self):
        base = 'https://github.com/mirelesadan/4Denoise/blob/main/gui.m'
        for uri in (base + '#L10', base + '?plain=1',
                    'https://github.com/another/repo/blob/main/gui.m',
                    'https://zenodo.org/records/17246822',
                    'https://github.com/mirelesadan/4Denoise/pull/9'):
            self.assertIsNone(docs_conf._github_file_check_uri(None, uri))

    def test_validation_hooks_are_registered(self):
        app = Mock()
        docs_conf.setup(app)
        app.connect.assert_any_call('builder-inited', docs_conf._prepare_tutorial_pages)
        app.connect.assert_any_call('linkcheck-process-uri', docs_conf._github_file_check_uri)
        app.connect.assert_any_call('build-finished', docs_conf._require_verified_external_links)


class TutorialValidationTests(unittest.TestCase):
    def test_registered_notebook_has_clean_state_and_declared_synthetic_origin(self):
        for name in tutorials.TUTORIALS:
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / 'tutorials' / name).read_text(encoding='utf-8'))
                self.assertEqual(notebook['nbformat'], 4)
                self.assertEqual(notebook['metadata']['4denoise']['data_origin'], 'synthetic')
                self.assertFalse(notebook['metadata']['4denoise']['requires_external_data'])
                ids = [cell['id'] for cell in notebook['cells']]
                self.assertEqual(len(ids), len(set(ids)))
                code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
                self.assertGreater(len(code), 10)
                for cell in code:
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])
                parameter_cells = [cell for cell in code if 'microscope-input' in cell['metadata'].get('tags', [])]
                self.assertEqual(len(parameter_cells), 1)
                self.assertIn('microscope_path = None', ''.join(parameter_cells[0]['source']))
                self.assertIn('input_data = None', ''.join(parameter_cells[0]['source']))
                self.assertIn("input_origin = 'experimental'", ''.join(parameter_cells[0]['source']))

    def test_only_registered_tutorials_are_dispatched_to_separate_reports(self):
        outputs = [ROOT / 'docs' / '_build' / 'tutorials' / name for name in tutorials.TUTORIALS]
        with patch.object(tutorials, 'execute_notebook', side_effect=outputs) as execute:
            self.assertEqual(tutorials.execute_tutorials(ROOT, timeout=15), outputs)
        self.assertEqual(execute.call_args_list, [
            call(ROOT / 'tutorials' / name, root=ROOT / 'tutorials', output=output, timeout=15)
            for name, output in zip(tutorials.TUTORIALS, outputs)
        ])

    def test_explicit_selection_does_not_run_other_tutorials(self):
        selected = tutorials.TUTORIALS[-1]
        output = ROOT / 'docs' / '_build' / 'tutorials' / selected
        with patch.object(tutorials, 'execute_notebook', return_value=output) as execute:
            self.assertEqual(tutorials.execute_tutorials(ROOT, notebooks=[selected]), [output])
        execute.assert_called_once_with(
            ROOT / 'tutorials' / selected, root=ROOT / 'tutorials', output=output, timeout=120,
        )

    def test_research_paths_empty_and_duplicate_selections_are_rejected(self):
        for selected in ([], ['../' + demo.DEMO], [demo.DEMO],
                         [tutorials.TUTORIALS[0], tutorials.TUTORIALS[0]]):
            with self.subTest(selected=selected), patch.object(tutorials, 'execute_notebook') as execute:
                with self.assertRaises(ValueError):
                    tutorials.execute_tutorials(ROOT, notebooks=selected)
                execute.assert_not_called()

    def test_missing_tutorial_is_reported_before_any_kernel_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tutorials, 'execute_notebook') as execute:
                with self.assertRaisesRegex(FileNotFoundError, 'Missing registered'):
                    tutorials.execute_tutorials(directory)
                execute.assert_not_called()

    def test_shared_runner_rejects_source_overwrite_and_bad_paths_before_imports(self):
        source = ROOT / 'tutorials' / tutorials.TUTORIALS[0]
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            demo.execute_notebook(source, root=ROOT, output=source)
        with self.assertRaisesRegex(ValueError, 'extension'):
            demo.execute_notebook(source, root=ROOT, output=ROOT / 'report.txt')
        with self.assertRaisesRegex(FileNotFoundError, 'Notebook does not exist'):
            demo.execute_notebook(ROOT / 'absent.ipynb', root=ROOT, output=ROOT / 'report.ipynb')
        for timeout in (True, 0, -1, 1.5):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'timeout'):
                demo.execute_notebook(source, root=ROOT, output=ROOT / 'report.ipynb', timeout=timeout)

    def test_existing_demo_uses_the_shared_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / demo.DEMO).write_text('{}', encoding='utf-8')
            (root / 'mini_dataset_binned.npy').write_bytes(b'fixture')
            output = demo.resolve_output(root)
            with patch.object(demo, 'execute_notebook', return_value=output) as execute:
                self.assertEqual(demo.execute_demo(root, timeout=30), output)
            execute.assert_called_once_with(root / demo.DEMO, root=root, output=output, timeout=30)


if __name__ == '__main__':
    unittest.main()
