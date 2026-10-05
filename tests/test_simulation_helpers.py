"""Contracts for lightweight simulation helpers and their plotting wrapper."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_kin_diff_sim():
    skopt = types.ModuleType('skopt')
    skopt.gp_minimize = lambda *args, **kwargs: None
    space = types.ModuleType('skopt.space')
    space.Real = lambda *args, **kwargs: (args, kwargs)
    utils = types.ModuleType('skopt.utils')
    utils.use_named_args = lambda domain: lambda function: function
    with patch.dict(sys.modules, {
        'skopt': skopt,
        'skopt.space': space,
        'skopt.utils': utils,
    }):
        return _load_module('_kin_diff_sim_contract_test', SOURCES / 'kin_diff_sim.py')


class SimulationHelperTests(unittest.TestCase):
    def test_complete_ripple_mirrors_both_axes(self):
        module = _load_module('_ripple_contract_test', SOURCES / 'rippleModel.py')
        quarter = np.array([[1, 2], [3, 4]])
        expected = np.array([
            [4, 3, 3, 4],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [4, 3, 3, 4],
        ])
        np.testing.assert_array_equal(module.complete_ripple(quarter), expected)

    def test_optimize_and_plot_passes_domains_and_criterion(self):
        module = _load_kin_diff_sim()
        figures_before = set(plt.get_fignums())
        try:
            with patch.object(module, 'compute_intensities', return_value=np.array([1.0])) as simulate:
                with patch.object(module, 'find_optimal_params', return_value=np.array([0.5, 0.2])) as fit:
                    with patch.object(module.plt, 'show'):
                        module.optimize_and_plot([0.1], [0.2], ['mse'], sample=3)

            simulate.assert_called_once_with(0.1, 0.2, sample=3)
            fit.assert_called_once()
            self.assertIs(fit.call_args.args[1], module.domain1)
            self.assertIs(fit.call_args.args[2], module.domain2)
            self.assertEqual(fit.call_args.kwargs['criterion'], 'mse')
            self.assertEqual(fit.call_args.kwargs['sample'], 3)
        finally:
            for figure in set(plt.get_fignums()) - figures_before:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
