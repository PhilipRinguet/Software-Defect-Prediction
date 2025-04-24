import os
import json
import pandas as pd
from src.models.model_comparison import compare_models
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pytest

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def sample_models():
    return {
        'dummy_model': DummyClassifier(strategy='most_frequent')
    }

def test_compare_models(sample_data, sample_models, tmp_path):
    X_train, X_test, y_train, y_test = sample_data
    models = sample_models

    # Train models
    for model in models.values():
        model.fit(X_train, y_train)

    # Run comparison
    output_path = tmp_path / "results"
    results = compare_models(models, X_test, y_test, output_path)

    # Check if results are saved
    assert (output_path / 'model_comparison_results.json').exists()
    assert (output_path / 'model_comparison_results.csv').exists()

    # Validate results format
    with open(output_path / 'model_comparison_results.json') as f:
        json_results = json.load(f)
    assert isinstance(json_results, list)
    assert all('model' in entry for entry in json_results)

    csv_results = pd.read_csv(output_path / 'model_comparison_results.csv')
    assert not csv_results.empty