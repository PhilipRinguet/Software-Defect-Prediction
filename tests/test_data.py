import unittest
import pandas as pd
import numpy as np
from src.data.make_dataset import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.jm1_df, self.kc1_df = load_data()

    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        X_train, X_test, y_train, y_test, transformer = preprocess_data(
            self.jm1_df, self.kc1_df, test_size=0.3
        )
        
        # Check shapes
        total_samples = len(self.jm1_df) + len(self.kc1_df)
        expected_test_size = int(total_samples * 0.3)
        # Allow for ±1 sample difference due to stratification
        self.assertLessEqual(abs(len(X_test) - expected_test_size), 1)
        
        # Check that transformer works
        self.assertIsNotNone(transformer)
        transformed_data = transformer.transform(X_test)
        self.assertEqual(transformed_data.shape, X_test.shape)
        
        # Check target values
        self.assertTrue(all(y in [0, 1] for y in y_train))
        self.assertTrue(all(y in [0, 1] for y in y_test))

if __name__ == '__main__':
    unittest.main()