import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.visualize import (
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_feature_importance
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        self.features = ['f1', 'f2', 'f3']
    
    def test_feature_distributions(self):
        """Test feature distribution plotting."""
        fig = plot_feature_distributions(self.df, self.features)
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close('all')
    
    def test_correlation_matrix(self):
        """Test correlation matrix plotting."""
        fig = plot_correlation_matrix(self.df[self.features])
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close('all')
    
    def test_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        y_true = np.random.binomial(1, 0.3, 100)
        y_prob = np.random.random(100)
        fig = plot_precision_recall_curve(y_true, y_prob)
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close('all')
    
    def test_class_distribution(self):
        """Test class distribution plotting."""
        fig = plot_class_distribution(self.df['target'])
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close('all')
    
    def test_feature_importance(self):
        """Test feature importance plotting."""
        feature_importance = np.random.random(len(self.features))
        fig = plot_feature_importance(feature_importance, self.features)
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close('all')

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

if __name__ == '__main__':
    unittest.main()