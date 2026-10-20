"""Standalone module boundaries and parent re-export integration."""

import ast
import builtins
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import symtable
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = {
    name: importlib.import_module(name)
    for name in ('fourdenoise_traversals', 'fourdenoise_plotting', 'fourdenoise_io')
}
traversals = MODULES['fourdenoise_traversals']
plotting = MODULES['fourdenoise_plotting']
io = MODULES['fourdenoise_io']
PLOTTING_HELPERS = (
    '_BLOCK_TRAVERSAL_METHODS', '_TRAVERSAL_INDEX_GENERATORS',
    '_block_size_for_method', '_get_traversal_indices',
    '_moore_indices', '_normalize_traversal_method',
)


def _node_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    return []


def _module_node_names(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    return [name for node in tree.body for name in _node_names(node)]


def _free_globals(table):
    # Python 3.12 inlines module-level comprehensions: their assigned loop
    # variables are both local and global, but are not free global references.
    names = {
        symbol.get_name() for symbol in table.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
        and not (symbol.is_local() and symbol.is_assigned())
    }
    for child in table.get_children():
        names.update(_free_globals(child))
    return names


class ModuleBoundaryTests(unittest.TestCase):
    def test_free_globals_distinguishes_comprehension_bindings_and_dependencies(self):
        source = (
            'mapping = {tuple(coord): idx for idx, coord in enumerate(points)}\n'
            'values = [transform(idx) for idx in values_in]\n'
            'nested = [[combine(idx, coord) for coord in row] for idx, row in rows]\n'
            'def unresolved():\n'
            '    return idx + missing\n'
        )
        self.assertEqual(
            _free_globals(symtable.symtable(source, '<comprehensions>', 'exec')),
            {'tuple', 'enumerate', 'points', 'transform', 'values_in',
             'combine', 'rows', 'idx', 'missing'},
        )

    def test_no_duplicate_implementations_and_direct_parent_reexports(self):
        parent = importlib.import_module('fourdenoise')
        tree = ast.parse((ROOT / 'fourdenoise.py').read_text(encoding='utf-8'))
        parent_definitions = {name for node in tree.body for name in _node_names(node)}
        seen = set()
        for name, module in MODULES.items():
            with self.subTest(module=name):
                names = _module_node_names(module)
                self.assertFalse(seen.intersection(names))
                self.assertFalse(parent_definitions.intersection(names))
                seen.update(names)
                for exported in names:
                    self.assertIs(getattr(parent, exported), getattr(module, exported))

    def test_free_globals_decorators_and_initializers_are_resolved(self):
        for name, module in MODULES.items():
            with self.subTest(module=name):
                source = Path(module.__file__).read_text(encoding='utf-8')
                unresolved = (
                    _free_globals(symtable.symtable(source, module.__file__, 'exec'))
                    - set(vars(module)) - set(vars(builtins))
                )
                self.assertEqual(unresolved, set())
                self.assertTrue(all(hasattr(module, exported)
                                    for exported in _module_node_names(module)))

    def test_fresh_imports_never_load_main(self):
        for name in MODULES:
            with self.subTest(module=name):
                script = (
                    'import importlib, sys; '
                    f'importlib.import_module({name!r}); '
                    'assert "fourdenoise" not in sys.modules; '
                    'assert not any(x in sys.modules for x in '
                    '("tensorly", "sklearn", "numba", "cv2"))'
                )
                if name == 'fourdenoise_traversals':
                    script += '; assert "matplotlib" not in sys.modules; assert "h5py" not in sys.modules'
                completed = subprocess.run(
                    [sys.executable, '-B', '-c', script], cwd=ROOT,
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_shared_dependency_objects_and_cached_function(self):
        self.assertIs(plotting._TRAVERSAL_INDEX_GENERATORS, traversals._TRAVERSAL_INDEX_GENERATORS)
        self.assertIs(io._unfold_payload_arrays, traversals._unfold_payload_arrays)
        for name in PLOTTING_HELPERS:
            self.assertIs(getattr(plotting, name), getattr(traversals, name))
        path = traversals._meander5_block_path('W', 'C')
        self.assertIs(path, traversals._meander5_block_path('W', 'C'))
        self.assertEqual(traversals._meander5_block_path.cache_parameters()['maxsize'], None)


class StandaloneTraversalTests(unittest.TestCase):
    def test_every_implemented_method_round_trips_both_stack_domains(self):
        original = np.arange(10 * 11 * 10 * 12, dtype=np.int16).reshape(10, 11, 10, 12)
        methods = [name for name in traversals._TRAVERSAL_INDEX_GENERATORS if name != 'moore']
        for domain in ('real', 'reciprocal'):
            for method in methods:
                with self.subTest(domain=domain, method=method):
                    unfolded, metadata = traversals._unfold_array(
                        original, domain=domain, method=method, return_metadata=True,
                    )
                    restored = traversals._unfold_array(unfolded, undo=True, metadata=metadata)
                    np.testing.assert_array_equal(restored, original)
                    self.assertEqual(restored.dtype, original.dtype)
                    payloads = tuple(traversals._unfold_payload_arrays(metadata))
                    self.assertEqual(metadata['preserved_values_nbytes'], sum(x.nbytes for x in payloads))
                    self.assertTrue(all(not x.flags.writeable for x in payloads))

    def test_both_domain_new_and_legacy_excess_encodings(self):
        original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        for method in ('row_major', 'morton'):
            with self.subTest(method=method):
                unfolded, metadata = traversals._unfold_array(
                    original, domain='both', method=method, return_metadata=True,
                )
                np.testing.assert_array_equal(
                    traversals._unfold_array(unfolded, undo=True, metadata=metadata), original,
                )
                if method == 'morton':
                    self.assertEqual(metadata['excess_values_encoding'], 'separated_domains')
                    legacy = deepcopy(metadata)
                    legacy['excess_values'] = original.copy()
                    legacy['excess_values_encoding'] = 'full_tensor'
                    np.testing.assert_array_equal(
                        traversals._unfold_array(unfolded, undo=True, metadata=legacy), original,
                    )

    def test_coordinate_aligned_and_metadata_less_row_major_undo(self):
        original = np.arange(4 ** 4).reshape((4,) * 4)
        for domain in ('real', 'reciprocal'):
            with self.subTest(domain=domain, method='coordinate_aligned'):
                unfolded, metadata = traversals._unfold_array(
                    original, domain=domain, method='coordinate_aligned', return_metadata=True,
                )
                np.testing.assert_array_equal(
                    traversals._unfold_array(unfolded, undo=True, metadata=metadata), original,
                )
        for domain in ('real', 'reciprocal', 'both'):
            with self.subTest(domain=domain, method='row_major'):
                unfolded = traversals._unfold_array(original, domain=domain)
                np.testing.assert_array_equal(
                    traversals._unfold_array(unfolded, domain=domain, undo=True), original,
                )

    def test_nonpreserving_crop_and_preserved_resize_undo(self):
        original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        unfolded, metadata = traversals._unfold_array(
            original, method='hilbert', preserve_excess=False, return_metadata=True,
        )
        np.testing.assert_array_equal(
            traversals._unfold_array(unfolded, undo=True, metadata=metadata), original[0:4, 1:5],
        )
        working = original[0:4, 1:5]
        unfolded, metadata = traversals._unfold_array(
            working, method='hilbert', curve_shape_strategy='resize',
            preserve_original=True, original_shape=original.shape,
            original_values=original, return_metadata=True,
        )
        np.testing.assert_array_equal(
            traversals._unfold_array(unfolded, undo=True, metadata=metadata), original,
        )

    def test_clone_can_share_only_read_only_payloads(self):
        original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        _, metadata = traversals._unfold_array(original, method='hilbert', return_metadata=True)
        shared = traversals._clone_unfold_metadata(metadata, share_payload=True)
        independent = traversals._clone_unfold_metadata(metadata)
        self.assertIsNot(shared, metadata)
        self.assertIs(shared['excess_values'], metadata['excess_values'])
        self.assertFalse(shared['excess_values'].flags.writeable)
        self.assertFalse(np.shares_memory(independent['excess_values'], metadata['excess_values']))
        self.assertFalse(np.shares_memory(shared['traversal_indices'], metadata['traversal_indices']))

    def test_registry_entry_mutation_reaches_plotting(self):
        with patch.dict(traversals._TRAVERSAL_INDEX_GENERATORS, hilbert=lambda shape: np.array([[0, 0]])):
            with self.assertRaisesRegex(ValueError, 'did not cover'):
                plotting.plot_traversals('hilbert', 8, show=False)
        plt.close('all')

    def test_plot_layout_aliases_and_failure_modes(self):
        fig, axes = plotting.plot_traversals('all', 10, show=False)
        try:
            self.assertEqual(axes.shape, (4, 3))
            titles = [ax.get_title() for ax in axes.flat if ax.get_visible()]
            self.assertEqual(len(titles), 10)
            self.assertIn('peano\nused 9x9 / max 10x10', titles)
            self.assertIn('hilbert\nused 8x8 / max 10x10', titles)
        finally:
            plt.close(fig)
        fig, axes = plotting.plot_traversals('z_order', 8, show=False)
        try:
            self.assertTrue(axes[0, 0].get_title().startswith('morton\n'))
        finally:
            plt.close(fig)
        with self.assertRaises(NotImplementedError):
            plotting.plot_traversals('moore', 8, show=False)
        with self.assertRaisesRegex(ValueError, 'not a 2D traversal'):
            plotting.plot_traversals('coordinate_aligned', 8, show=False)


class StandaloneHDF5Tests(unittest.TestCase):
    def test_recursive_serialization_and_compression(self):
        objects = np.empty(3, dtype=object)
        objects[:] = [{'label': 'inner'}, None, (3, 4)]
        metadata = {
            'none': None, 'mapping': {'value': 2}, 'list': [1, None, 'text'],
            'tuple': (True, 2.5), 'bytes': b'bytes', 'objects': objects,
            'strings': np.array([['caf\u00e9', '\u03b2'], ['plain', '\u4e2d\u6587']]),
            'array': np.arange(12).reshape(3, 4),
        }
        with h5py.File('module-boundaries.h5', 'w', driver='core', backing_store=False) as file:
            io._write_hdf5_value(file, 'metadata', metadata, compression='gzip', compression_opts=1)
            restored = io._read_hdf5_value(file, 'metadata')
            self.assertEqual(restored['none'], None)
            self.assertEqual(restored['mapping'], metadata['mapping'])
            self.assertEqual(restored['list'], metadata['list'])
            self.assertEqual(restored['tuple'], metadata['tuple'])
            self.assertEqual(restored['bytes'], 'bytes')
            np.testing.assert_array_equal(restored['objects'], objects)
            np.testing.assert_array_equal(restored['strings'], metadata['strings'])
            np.testing.assert_array_equal(restored['array'], metadata['array'])
            self.assertEqual(file['metadata/array'].compression, 'gzip')
            with self.assertRaisesRegex(ValueError, 'cannot contain'):
                io._write_hdf5_value(file, 'invalid-key', {'a/b': 1})
            with self.assertRaisesRegex(TypeError, 'unsupported type'):
                io._write_hdf5_value(file, 'invalid-value', object())

    def test_generic_dataset_selection_numeric_validation_and_internal_exclusion(self):
        with h5py.File('module-boundaries.h5', 'w', driver='core', backing_store=False) as file:
            data = file.create_dataset('data', data=np.arange(12).reshape(3, 4))
            file.create_dataset('scalar', data=2)
            file.create_dataset('text', data=np.array([[b'a', b'b']]))
            self.assertEqual(io._select_hdf5_dataset(file).name, data.name)
            with self.assertRaisesRegex(ValueError, 'at least 2'):
                io._select_hdf5_dataset(file, 'scalar')
            with self.assertRaisesRegex(TypeError, 'non-numeric'):
                io._select_hdf5_dataset(file, 'text')
            with self.assertRaisesRegex(KeyError, 'Available datasets'):
                io._select_hdf5_dataset(file, 'missing')
            file.create_dataset('#refs#/internal', data=np.ones((2, 2)))
            self.assertEqual(io._select_hdf5_dataset(file, exclude_internal=True).name, data.name)
            with self.assertRaisesRegex(ValueError, '2 numeric multidimensional datasets'):
                io._select_hdf5_dataset(file)
            self.assertEqual(io._select_hdf5_dataset(file, Path('data')).name, data.name)

    def test_saved_format_loading_and_legacy_payload_migration(self):
        original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        unfolded, metadata = traversals._unfold_array(
            original, domain='both', method='morton', return_metadata=True,
        )
        metadata.pop('preserved_values_nbytes')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'legacy.4denoise'
            with h5py.File(path, 'w') as file:
                file.attrs['fourdenoise_format'] = io._HYPERDATA_HDF5_FORMAT
                file.attrs['format_version'] = '1.0'
                file.create_dataset('array', data=unfolded)
                io._write_hdf5_value(file.create_group('metadata'), 'unfold_metadata', metadata)
            self.assertTrue(io._is_hyperdata_hdf5_file(path))
            loaded, saved = io._load_hyperdata_hdf5(path)
            self.assertEqual(saved['real_origin'], (0.0, 0.0))
            saved_metadata = saved['unfold_metadata']
            self.assertEqual(saved_metadata['preserved_values_nbytes'], original.nbytes - unfolded.nbytes)
            np.testing.assert_array_equal(
                traversals._unfold_array(loaded, undo=True, metadata=saved_metadata), original,
            )
            np.testing.assert_array_equal(io._read_hdf5_file(path, 'array'), unfolded)
            with h5py.File(path, 'r+') as file:
                file.attrs['format_version'] = '2.0'
            with self.assertRaisesRegex(ValueError, 'unsupported.*2.0'):
                io._load_hyperdata_hdf5(path)
            with h5py.File(path, 'r+') as file:
                del file.attrs['format_version']
            with self.assertRaisesRegex(ValueError, 'missing.*format_version'):
                io._load_hyperdata_hdf5(path)


class ParentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = importlib.import_module('fourdenoise')

    def test_direct_reexports_have_original_module_identity(self):
        for name, module in MODULES.items():
            for exported in _module_node_names(module):
                with self.subTest(module=name, exported=exported):
                    self.assertIs(getattr(self.parent, exported), getattr(module, exported))
        self.assertEqual(self.parent.plot_traversals.__module__, 'fourdenoise_plotting')
        self.assertIs(self.parent._TRAVERSAL_INDEX_GENERATORS, plotting._TRAVERSAL_INDEX_GENERATORS)

    def test_main_unfold_plotting_and_hdf5_class_orchestration(self):
        original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        data = self.parent.HyperData(original)
        with patch.object(self.parent, 'plot_traversals', wraps=plotting.plot_traversals) as plotter:
            unfolded = data.unfold(
                domain='both', method='morton', warn_on_crop=False,
                plot_traversal=True, plot_kwargs={'show': False},
            )
        self.assertEqual(plotter.call_count, 2)
        plt.close('all')
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, original)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'unfolded.4denoise'
            unfolded.save(path)
            loaded = self.parent.HyperData(path)
            np.testing.assert_array_equal(loaded.unfold(undo=True).array, original)
            path = Path(directory) / 'data.4denoise'
            data.save(path)
            with self.parent.HyperData.open_hdf5(path) as reader:
                np.testing.assert_array_equal(reader.get_dp(0, 0).array, original[0, 0])


if __name__ == '__main__':
    unittest.main()
