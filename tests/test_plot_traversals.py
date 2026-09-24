"""Regression tests for traversal-path visualization."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class PlotTraversalsTests(unittest.TestCase):
    def test_single_method_uses_one_panel_and_colors_each_segment(self):
        fig, axes = fd.plot_traversals(
            'row_major', (4, 6), cmap='viridis', linewidth=2, show=False,
        )
        try:
            self.assertEqual(axes.shape, (1, 1))
            self.assertIn('used 4x6 / max 4x6', axes[0, 0].get_title())
            path = next(
                collection for collection in axes[0, 0].collections
                if isinstance(collection, LineCollection)
                and collection.get_array() is not None
            )
            self.assertEqual(len(path.get_segments()), 23)
            self.assertEqual(path.get_linewidths()[0], 2)
            np.testing.assert_allclose(
                path.get_array()[[0, -1]], [0, 1],
            )
            self.assertEqual(axes[0, 0].get_ylim(), (3.5, -0.5))
        finally:
            plt.close(fig)

    def test_all_methods_use_three_columns_and_show_crop_sizes(self):
        fig, axes = fd.plot_traversals('all', (10, 10), show=False)
        try:
            self.assertEqual(axes.shape, (4, 3))
            self.assertEqual(sum(ax.get_visible() for ax in axes.flat), 10)
            titles = {ax.get_title().split('\n')[0]: ax.get_title()
                      for ax in axes.flat if ax.get_visible()}
            self.assertEqual(len(titles), 10)
            self.assertIn('used 9x9 / max 10x10', titles['peano'])
            self.assertIn('used 8x8 / max 10x10', titles['hilbert'])
            self.assertIn('used 8x8 / max 10x10', titles['meander-4'])
            self.assertIn('used 10x10 / max 10x10', titles['meander-5'])
            self.assertNotIn('moore', titles)
            self.assertNotIn('z_order', titles)
        finally:
            plt.close(fig)

    def test_too_small_block_methods_are_labeled_in_all_view(self):
        fig, axes = fd.plot_traversals('all', (3, 3), show=False)
        try:
            titles = [ax.get_title() for ax in axes.flat if ax.get_visible()]
            self.assertIn('meander-4\nunavailable / max 3x3', titles)
            self.assertIn('meander-5\nunavailable / max 3x3', titles)
        finally:
            plt.close(fig)

    def test_alias_and_invalid_requests(self):
        fig, axes = fd.plot_traversals('z_order', 8, show=False)
        try:
            self.assertTrue(axes[0, 0].get_title().startswith('morton\n'))
        finally:
            plt.close(fig)
        with self.assertRaises(NotImplementedError):
            fd.plot_traversals('moore', 8, show=False)
        with self.assertRaisesRegex(ValueError, 'not a 2D traversal'):
            fd.plot_traversals('coordinate_aligned', 8, show=False)
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            fd.plot_traversals('row_major', (8, 4.5), show=False)
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            fd.plot_traversals('meander-5', 3, show=False)

    def test_incomplete_curve_is_rejected_before_plotting(self):
        original = fd._TRAVERSAL_INDEX_GENERATORS['hilbert']
        fd._TRAVERSAL_INDEX_GENERATORS['hilbert'] = lambda shape: np.array([[0, 0]])
        try:
            with self.assertRaisesRegex(ValueError, 'did not cover'):
                fd._get_traversal_indices((8, 8), 'hilbert')
        finally:
            fd._TRAVERSAL_INDEX_GENERATORS['hilbert'] = original


if __name__ == '__main__':
    unittest.main()
