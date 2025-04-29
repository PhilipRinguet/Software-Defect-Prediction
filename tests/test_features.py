import unittest
import numpy as np
import pandas as pd
from src.features.build_features import (
    FeatureSelector,
    handle_class_imbalance,
    select_features
)

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create synthetic test data
        self.X = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100)
        })
        self.y = pd.Series(np.random.binomial(1, 0.2, 100))  # imbalanced classes
    
    def test_feature_selector(self):
        """Test FeatureSelector class."""
        selector = FeatureSelector(k=2)
        selector.fit(self.X, self.y)
        selected = selector.transform(self.X)
        
        self.assertEqual(selected.shape[1], 2)  # Check number of selected features
        self.assertIsInstance(selected, pd.DataFrame)
    
    def test_handle_class_imbalance(self):
        """Test class imbalance handling."""
        # Test ADASYN
        X_resampled, y_resampled = handle_class_imbalance(
            self.X, self.y, method='adasyn'
        )
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertAlmostEqual(counts[0], counts[1], delta=10)  # Allow small difference
        
        # Test SMOTE
        X_resampled, y_resampled = handle_class_imbalance(
            self.X, self.y, method='smote'
        )
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertAlmostEqual(counts[0], counts[1], delta=10)
    
    def test_select_features(self):
        """Test feature selection functionality."""
        X_train = self.X.copy()
        X_test = self.X.copy()
        y_train = self.y.copy()
        
        X_train_selected, X_test_selected, selected_features = select_features(
            X_train, y_train, X_test, k=2
        )
        
        self.assertEqual(X_train_selected.shape[1], 2)
        self.assertEqual(X_test_selected.shape[1], 2)
        self.assertEqual(len(selected_features), 2)

if __name__ == '__main__':
    unittest.main()