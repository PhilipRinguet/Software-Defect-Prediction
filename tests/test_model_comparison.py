import os
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from src.models.model_comparison import ModelComparison

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X, y = make_classification(
        n_samples=100, 
        n_features=20,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def model_comparison(tmp_path):
    """Create ModelComparison instance with temporary directory"""
    return ModelComparison(str(tmp_path))

@pytest.fixture
def dummy_models():
    """Create dummy models for testing"""
    return {
        'model1': DummyClassifier(strategy='stratified', random_state=42),
        'model2': DummyClassifier(strategy='most_frequent', random_state=42)
    }

def test_evaluate_model(model_comparison, sample_data):
    """Test model evaluation with single model"""
    X_train, X_test, y_train, y_test = sample_data
    model = DummyClassifier(strategy='stratified', random_state=42)
    model.fit(X_train, y_train)
    
    metrics = model_comparison.evaluate_model(model, X_test, y_test)
    
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ['Precision', 'Recall', 'F-beta Score', 'PR-AUC'])
    assert all(isinstance(value, float) for value in metrics.values())

def test_compare_models(model_comparison, sample_data, dummy_models):
    """Test comparison of multiple models"""
    X_train, X_test, y_train, y_test = sample_data
    
    # Train models
    for model in dummy_models.values():
        model.fit(X_train, y_train)
    
    results = model_comparison.compare_models(dummy_models, X_test, y_test)
    
    assert isinstance(results, pd.DataFrame)
    assert list(results.index) == list(dummy_models.keys())
    assert all(col in results.columns for col in ['Precision', 'Recall', 'F-beta Score', 'PR-AUC'])

def test_save_and_load_results(model_comparison):
    """Test saving and loading of results"""
    # Create sample results
    sample_results = pd.DataFrame({
        'Precision': [0.8, 0.7],
        'Recall': [0.75, 0.8],
        'F-beta Score': [0.77, 0.75],
        'PR-AUC': [0.82, 0.79]
    }, index=['model1', 'model2'])
    
    # Save results
    output_file = model_comparison.save_results(sample_results)
    assert os.path.exists(output_file)
    
    # Load results
    loaded_results = model_comparison.load_results()
    pd.testing.assert_frame_equal(sample_results, loaded_results)

def test_invalid_model_comparison(model_comparison, sample_data):
    """Test handling of invalid model in comparison"""
    X_train, X_test, y_train, y_test = sample_data
    
    # Create an untrained model
    untrained_model = {'bad_model': DummyClassifier()}
    
    with pytest.raises(Exception):
        model_comparison.compare_models(untrained_model, X_test, y_test)