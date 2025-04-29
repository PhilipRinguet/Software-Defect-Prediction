import mlflow
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
import joblib

from src.features.build_features import select_features, transform_features
from src.models.train_model import (
    create_svm_pipeline,
    create_brf_pipeline,
    create_xgb_pipeline,
    train_and_evaluate,
    train_with_optimization
)
from src.utils.config import load_config

def load_and_prepare_data(config):
    """Load and prepare the JM1 and KC1 datasets"""
    datasets = []
    for dataset in config['data']['datasets']:
        data_path = f"{config['data']['raw_data_path']}/{dataset}.arff"
        data, _ = arff.loadarff(data_path)
        df = pd.DataFrame(data)
        
        # Lowercase column names
        df.columns = df.columns.str.lower()
        
        # Rename target columns consistently
        if 'label' in df.columns:
            df.rename(columns={"label": "target"}, inplace=True)
        elif 'defective' in df.columns:
            df.rename(columns={"defective": "target"}, inplace=True)
        
        # Convert target to numeric
        df["target"] = df["target"].str.decode("utf-8")
        df["target"] = np.where(df["target"] == "Y", 1, 0)
        datasets.append(df)
    
    # Combine datasets
    combined_data = pd.concat(datasets, ignore_index=True)
    return combined_data

def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    
    # Set up MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load and prepare data
    data = load_and_prepare_data(config)
    
    # Split features and target
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        stratify=y, 
        random_state=config['data']['random_state']
    )
    
    # Feature selection
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)
    
    # Define models to train
    model_configs = {
        "SVM-RBF": {"type": "svm", "beta": config['training']['metrics']['beta']},
        "BRF": {"type": "rf", "beta": config['training']['metrics']['beta']},
        "XGBoost": {"type": "xgb", "beta": config['training']['metrics']['beta']}
    }
    
    results = {}
    for name, model_config in model_configs.items():
        print(f"\nOptimizing {name}...")
        pipeline, best_params, best_score = train_with_optimization(
            X_train_selected,
            y_train,
            model_type=model_config["type"],
            cv=config['training']['optimization']['cv_folds'],
            n_trials=config['training']['optimization']['n_trials'],
            timeout=config['training']['optimization']['timeout'],
            features=selected_features
        )
        
        print(f"Best validation score: {best_score:.3f}")
        print("Best parameters:", best_params)
        
        # Evaluate on test set
        _, metrics, threshold = train_and_evaluate(
            pipeline,
            X_train_selected,
            X_test_selected,
            y_train,
            y_test,
            name,
            model_config["beta"]
        )
        
        results[name] = {
            "metrics": metrics,
            "threshold": threshold,
            "best_params": best_params,
            "best_val_score": best_score
        }
        
        # Save best model (SVM-RBF)
        if name == "SVM-RBF":
            print("\nSaving best model...")
            joblib.dump(pipeline, "pipeline_svm_rbf.pkl")
    
    # Print final results
    print("\nModel Comparison Results:")
    print("-" * 50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Precision: {result['metrics']['precision']:.3f}")
        print(f"Recall: {result['metrics']['recall']:.3f}")
        print(f"F-beta Score: {result['metrics']['f_beta']:.3f}")
        print(f"PR-AUC: {result['metrics'].get('pr_auc', 'N/A')}")
        print(f"Optimal Threshold: {result['threshold']:.3f}")
        print(f"Best Validation Score: {result['best_val_score']:.3f}")
        print("Best Parameters:", result['best_params'])

if __name__ == "__main__":
    main()