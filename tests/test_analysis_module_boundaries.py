"""Denoising and rendering extraction preserves public behavior and boundaries."""

import ast
import builtins
import inspect
from pathlib import Path
import subprocess
import sys
import symtable
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd
import fourdenoise_denoising as denoising
import fourdenoise_views as views


class AnalysisModuleBoundaryTests(unittest.TestCase):
    def test_extracted_dependencies_are_explicit(self):
        def free_globals(table):
            names = {
                symbol.get_name() for symbol in table.get_symbols()
                if symbol.is_referenced() and symbol.is_global()
                and not (symbol.is_local() and symbol.is_assigned())
            }
            for child in table.get_children():
                names.update(free_globals(child))
            return names

        for module in (denoising, views):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding='utf-8')
                unresolved = (
                    free_globals(symtable.symtable(source, module.__file__, 'exec'))
                    - set(vars(module)) - set(vars(builtins))
                )
                self.assertEqual(unresolved, set())

    def test_denoising_reexports_are_the_same_classes(self):
        self.assertIs(fd._DenoisingMethods, denoising._DenoisingMethods)
        self.assertIs(fd._DenoiseEngine, denoising._DenoiseEngine)
        self.assertIs(fd._DENOISING_METHOD_CONTRACTS, denoising._DENOISING_METHOD_CONTRACTS)
        definitions = {
            n.name for n in ast.parse((ROOT / 'fourdenoise.py').read_text(encoding='utf-8')).body
            if isinstance(n, ast.ClassDef)
        }
        self.assertNotIn('_DenoisingMethods', definitions)
        self.assertNotIn('_DenoiseEngine', definitions)

    def test_standalone_imports_do_not_import_public_facade(self):
        for name in ('fourdenoise_denoising', 'fourdenoise_views'):
            script = (
                f'import {name}; import sys; '
                'assert "fourdenoise" not in sys.modules; '
                'assert "sklearn" not in sys.modules'
            )
            if name == 'fourdenoise_views':
                script += '; assert "tensorly" not in sys.modules; assert "cv2" not in sys.modules'
            with self.subTest(module=name):
                result = subprocess.run(
                    [sys.executable, '-B', '-c', script], cwd=ROOT,
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_array_kernels_need_no_hyperdata_factory(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        engine = denoising._DenoiseEngine(data)
        result = engine.apply('median', window_size=(1, 3, 3))
        self.assertEqual(result.shape, data.shape)
        self.assertEqual(set(engine.available_methods), set(denoising._DENOISING_METHOD_CONTRACTS))
        with self.assertRaisesRegex(RuntimeError, 'HyperData factory'):
            engine.methods._unfold_with_hyperdata(np.ones((2, 3, 4, 5)), 'real')

    def test_factory_routes_unfolding_through_hyperdata(self):
        data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        methods = denoising._DenoisingMethods(hyperdata_type=fd.HyperData)
        array, metadata = methods._unfold_with_hyperdata(data, 'real', 'serpentine')
        self.assertEqual(array.shape, (6, 4, 5))
        np.testing.assert_array_equal(methods._refold_with_hyperdata(array, metadata), data)

    def test_reconstruction_validation_uses_injected_type(self):
        data = fd.HyperData(np.ones((3, 4, 5)))
        validate = denoising._DenoiseEngine.require_reconstruction
        self.assertIs(validate(data.array, data.shape, method='test'), data.array)
        self.assertIs(validate(data, data.shape, method='test', hyperdata_type=fd.HyperData),
                      data.array)
        with self.assertRaisesRegex(TypeError, 'expected an ndarray'):
            validate(data, data.shape, method='test')
        denoised = data.denoise('median', window_size=(1, 3, 3))
        np.testing.assert_array_equal(denoised.array, data.array)

    def test_factory_preserves_decomposition_object_payloads(self):
        methods = denoising._DenoisingMethods(hyperdata_type=fd.HyperData)
        data = np.random.default_rng(12).random((3, 4, 5)) + 0.1
        for method, array, kwargs in (
            ('parafac2', data, {'rank': 1, 'n_iter_max': 2, 'n_iter_parafac': 1, 'random_state': 0}),
            ('tensor_train', data, {'rank': 1}),
            ('tensor_train_matrix', data[:2, :2, :4].reshape(2, 2, 2, 2), {'rank': 1}),
        ):
            with self.subTest(method=method):
                result = getattr(methods, method)(array, return_decomposition=True, **kwargs)
                self.assertIsInstance(result[1], fd.HyperData)
                self.assertEqual(result[1].shape, array.shape)

    def test_show_signatures_and_forwarding_are_preserved(self):
        for cls, renderer in (
            (fd.RealSpace, '_show_real_space'),
            (fd.ReciprocalSpace, '_show_reciprocal_space'),
        ):
            with self.subTest(view=cls.__name__):
                self.assertEqual(inspect.signature(cls.show),
                                 inspect.signature(getattr(views, renderer)))
                view = cls(np.ones((8, 8)), units='nm', conv_factor=0.2)
                marker = object()
                with patch.object(fd, renderer, return_value=marker) as draw:
                    result = view.show(
                        axis_units='pixels', vmin=0, vmax=2, cmap='magma',
                        grid=True, scale_bar=2, scale_bar_color='red',
                    )
                self.assertIs(result, marker)
                self.assertIs(draw.call_args.args[0], view)
                self.assertEqual(draw.call_args.kwargs['axis_units'], 'pixels')
                self.assertEqual(draw.call_args.kwargs['scale_bar_color'], 'red')
                np.testing.assert_array_equal(view.array, np.ones((8, 8)))


if __name__ == '__main__':
    unittest.main()
