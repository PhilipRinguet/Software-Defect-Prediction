import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score
import mlflow
from model_comparison import ModelComparison
from train_model import train_and_evaluate
from data.make_dataset import load_data, preprocess_data
from features.build_features import select_features, handle_class_imbalance

def run_full_comparison():
    """Run complete model comparison with all configurations."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    jm1_df, kc1_df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(jm1_df, kc1_df)
    
    # Feature engineering
    print("Performing feature selection...")
    X_train_selected, X_test_selected, _ = select_features(X_train, y_train, X_test)
    
    # Initialize ModelComparison
    model_comparison = ModelComparison(reports_dir)
    
    # Define model configurations to test
    model_configs = {
        'logistic': {'model_type': 'logistic'},
        'svm_balanced': {'model_type': 'svm', 'class_weight': 'balanced'},
        'svm_smote': {'model_type': 'svm', 'resampling': 'smote'},
        'svm_adasyn': {'model_type': 'svm', 'resampling': 'adasyn'},
        'balanced_rf': {'model_type': 'balanced_rf'},
        'xgboost': {'model_type': 'xgboost'}
    }
    
    # Train and evaluate all models
    results = {}
    for name, config in model_configs.items():
        print(f"\nTraining {name}...")
        
        # Handle resampling if specified
        if config.get('resampling'):
            X_train_res, y_train_res = handle_class_imbalance(
                X_train_selected, 
                y_train, 
                method=config['resampling']
            )
        else:
            X_train_res, y_train_res = X_train_selected, y_train
        
        # Train and evaluate model
        model, metrics = train_and_evaluate(
            X_train_res, X_test_selected,
            y_train_res, y_test,
            model_type=config['model_type']
        )
        
        results[name] = metrics
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame.from_dict(results, orient='index')
    model_comparison.save_results(results_df)
    
    print("\nModel comparison complete. Results saved to reports directory.")
    return results_df

if __name__ == "__main__":
    with mlflow.start_run(run_name="model_comparison"):
        results = run_full_comparison()
        print("\nFinal Results:")
        print(results.round(3))