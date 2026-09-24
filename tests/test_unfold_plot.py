"""Tests for optional traversal visualization during HyperData unfolding."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class UnfoldPlotTests(unittest.TestCase):
    def setUp(self):
        self.original = np.arange(5 * 7 * 4 * 6).reshape(5, 7, 4, 6)
        self.data = fd.HyperData(self.original)
        self.addCleanup(plt.close, 'all')

    def test_real_center_crop_uses_data_grid_and_does_not_change_undo(self):
        with patch.object(fd, 'plot_traversals', wraps=fd.plot_traversals) as plotter:
            unfolded, metadata = self.data.unfold(
                method='hilbert', plot_traversal=True,
                plot_kwargs={'cmap': 'viridis', 'linewidth': 2, 'show': False},
                return_metadata=True,
            )

        self.assertEqual(unfolded.shape, (16, 4, 6))
        self.assertEqual(plotter.call_args.args, ('hilbert',))
        self.assertEqual(plotter.call_args.kwargs['grid_shape'], (5, 7))
        self.assertEqual(plotter.call_args.kwargs['cmap'], 'viridis')
        self.assertEqual(plotter.call_args.kwargs['linewidth'], 2)
        self.assertFalse(plotter.call_args.kwargs['show'])
        fig = plt.gcf()
        self.assertIn('used 4x4 / max 5x7', fig.axes[0].get_title())
        self.assertEqual(fig._suptitle.get_text(), 'Real-space traversal')
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)
        self.assertEqual(metadata['method'], 'hilbert')

    def test_reciprocal_uses_diffraction_grid(self):
        with patch.object(fd, 'plot_traversals', wraps=fd.plot_traversals) as plotter:
            unfolded = self.data.unfold(
                domain='reciprocal', method='serpentine',
                plot_traversal=True, plot_kwargs={'show': False},
            )

        self.assertEqual(unfolded.shape, (24, 5, 7))
        self.assertEqual(plotter.call_args.kwargs['grid_shape'], (4, 6))
        self.assertEqual(plt.gcf()._suptitle.get_text(), 'Reciprocal-space traversal')

    def test_both_domains_get_separate_figures_and_one_show_call(self):
        with patch.object(fd, 'plot_traversals', wraps=fd.plot_traversals) as plotter:
            with patch.object(fd.plt, 'show') as show:
                unfolded = self.data.unfold(
                    domain='both', method='row_major', plot_traversal=True,
                )

        self.assertEqual(unfolded.shape, (35, 24))
        self.assertEqual(plotter.call_count, 2)
        self.assertEqual(
            [call.kwargs['grid_shape'] for call in plotter.call_args_list],
            [(5, 7), (4, 6)],
        )
        self.assertEqual(
            [plt.figure(number)._suptitle.get_text() for number in plt.get_fignums()],
            ['Real-space traversal', 'Reciprocal-space traversal'],
        )
        show.assert_called_once_with()
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)

    def test_resize_plots_resized_grid_and_labels_it(self):
        with patch.object(fd, 'plot_traversals', wraps=fd.plot_traversals) as plotter:
            unfolded, metadata = self.data.unfold(
                method='hilbert', curve_shape_strategy='resize',
                resize_method='nearest', plot_traversal=True,
                plot_kwargs={'show': False}, return_metadata=True,
            )

        resized_grid = tuple(metadata['resized_shape'][:2])
        self.assertEqual(plotter.call_args.kwargs['grid_shape'], resized_grid)
        self.assertEqual(unfolded.shape[0], resized_grid[0] * resized_grid[1])
        self.assertIn('resized from 5x7 to', plt.gcf()._suptitle.get_text())

    def test_custom_grid_is_labeled_preview_and_alias_is_normalized(self):
        options = {'grid_shape': (4, 4), 'show': False}
        with patch.object(fd, 'plot_traversals', wraps=fd.plot_traversals) as plotter:
            unfolded = self.data.unfold(
                method='z_order', plot_traversal=True, plot_kwargs=options,
            )

        self.assertEqual(plotter.call_args.args, ('morton',))
        self.assertEqual(plotter.call_args.kwargs['grid_shape'], (4, 4))
        self.assertIn('preview (data grid 5x7)', plt.gcf()._suptitle.get_text())
        self.assertEqual(options, {'grid_shape': (4, 4), 'show': False})
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)

    def test_invalid_plotting_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only available when unfolding'):
            self.data.unfold(undo=True, plot_traversal=True)
        with self.assertRaisesRegex(ValueError, 'no 2D traversal'):
            self.data.unfold(method='coordinate_aligned', plot_traversal=True)
        with self.assertRaisesRegex(ValueError, 'cannot override'):
            self.data.unfold(plot_traversal=True, plot_kwargs={'method': 'spiral'})
        with self.assertRaisesRegex(TypeError, 'Unknown plot_kwargs'):
            self.data.unfold(plot_traversal=True, plot_kwargs={'not_an_option': 1})
        with self.assertRaisesRegex(ValueError, 'Set plot_traversal=True'):
            self.data.unfold(plot_kwargs={'show': False})
        with self.assertRaisesRegex(TypeError, 'must be a dictionary'):
            self.data.unfold(plot_traversal=True, plot_kwargs=['show'])
        with self.assertRaisesRegex(TypeError, 'must be a boolean'):
            self.data.unfold(plot_traversal='yes')
        self.assertEqual(plt.get_fignums(), [])

    def test_default_unfold_does_not_plot(self):
        with patch.object(fd, 'plot_traversals') as plotter:
            unfolded = self.data.unfold()
        plotter.assert_not_called()
        np.testing.assert_array_equal(unfolded.unfold(undo=True).array, self.original)


if __name__ == '__main__':
    unittest.main()
