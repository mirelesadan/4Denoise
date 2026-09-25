"""Regression tests for 4D scan clustering choices and detector weights."""

import unittest
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class GetClustersTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.data = rng.poisson(2, size=(8, 8, 6, 6)).astype(np.float32)
        self.data[:4, :, 2, 2] += 12
        self.data[4:, :, 2, 3] += 12
        self.original = self.data.copy()
        self.dataset = fd.HyperData(self.data)

    def tearDown(self):
        np.testing.assert_array_equal(self.data, self.original)

    def test_existing_pca_and_kmeans_call(self):
        labels, details = self.dataset.get_clusters(
            n_PCAcomponents=3, n_clusters=2, std_Threshold=0,
            intensity_transform='sqrt', return_diagnostics=True,
        )
        self.assertEqual(labels.shape, (8, 8))
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(details['feature_weighting'], 'hard')
        self.assertEqual(details['reduction_method'], 'pca')
        self.assertEqual(details['explained_variance_ratio'].shape, (3,))

    def test_soft_weighted_pca_and_gaussian_mixture(self):
        detector_mask = np.ones((6, 6), dtype=bool)
        detector_mask[0] = False
        labels, details = self.dataset.get_clusters(
            n_components=3, n_clusters=2, reduction_method='pca',
            clustering_method='gaussian-mixture', feature_weighting='soft',
            detector_mask=detector_mask, intensity_transform='sqrt',
            return_diagnostics=True,
        )
        self.assertEqual(labels.shape, (8, 8))
        self.assertEqual(len(np.unique(labels)), 2)
        weights = details['feature_weights']
        self.assertEqual(weights.shape, (6, 6))
        self.assertTrue(np.all(weights[0] == 0))
        self.assertTrue(np.all((weights[detector_mask] > 0) & (weights[detector_mask] < 1)))
        self.assertGreater(details['soft_weight_scale'], 0)
        self.assertEqual(details['model_confidence'].shape, (8, 8))
        self.assertTrue(np.all((details['model_confidence'] >= 0) &
                               (details['model_confidence'] <= 1)))

    def test_soft_weighted_nmf_and_kmeans(self):
        labels, details = self.dataset.get_clusters(
            n_components=3, n_clusters=2, reduction_method='nmf',
            feature_weighting='soft', intensity_transform='sqrt',
            fit_samples=0.75, return_diagnostics=True,
        )
        self.assertEqual(labels.shape, (8, 8))
        self.assertIsNone(details['explained_variance_ratio'])
        self.assertEqual(details['component_vectors'].shape,
                         (3, details['feature_count']))
        self.assertTrue(np.all(details['component_vectors'] >= 0))
        self.assertEqual(details['reduction_fit_patterns'], 48)

    def test_explicit_soft_scale_and_no_weighting(self):
        _, soft = self.dataset.get_clusters(
            n_components=2, n_clusters=2, feature_weighting='soft',
            soft_weight_scale=3, return_diagnostics=True,
        )
        _, unweighted = self.dataset.get_clusters(
            n_components=2, n_clusters=2, feature_weighting='none',
            return_diagnostics=True,
        )
        self.assertEqual(soft['soft_weight_scale'], 3)
        self.assertTrue(np.all(soft['feature_weights'] <= 1))
        np.testing.assert_array_equal(
            unweighted['feature_weights'][unweighted['detector_mask']], 1,
        )

    def test_hdbscan_can_choose_cluster_count(self):
        labels, details = self.dataset.get_clusters(
            n_components=3, n_clusters=None, clustering_method='hdbscan',
            hdbscan_min_cluster_size=4, intensity_transform='sqrt',
            return_diagnostics=True,
        )
        self.assertEqual(labels.shape, (8, 8))
        self.assertTrue(np.all(labels >= -1))
        self.assertEqual(details['noise_count'], np.count_nonzero(labels == -1))

    def test_invalid_combinations_are_explained(self):
        with self.assertRaisesRegex(ValueError, 'Specify only one'):
            self.dataset.get_clusters(3, 2, n_components=3)
        with self.assertRaisesRegex(ValueError, 'n_clusters must be None'):
            self.dataset.get_clusters(3, 2, clustering_method='hdbscan')
        with self.assertRaisesRegex(ValueError, 'plotScree is only defined for PCA'):
            self.dataset.get_clusters(3, 2, reduction_method='nmf', plotScree=True)


if __name__ == '__main__':
    unittest.main()
