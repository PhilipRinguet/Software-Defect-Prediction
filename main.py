import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import json

from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import select_features
from src.models.train_model import (
    create_svm_pipeline,
    create_brf_pipeline,
    create_xgb_pipeline,
    train_and_evaluate,
    train_with_optimization
)
from src.models.metrics import (
    calculate_basic_metrics,
    calculate_threshold_metrics,
    calculate_detailed_metrics
)
from src.utils.config import load_config
from src.visualization.visualize import (
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_class_distribution,
    plot_feature_importance,
    plot_pr_curves,
    plot_metrics_comparison
)
from src.visualization.plotting import plot_manager

# Define model configurations
model_configs = {
    'SVM-RBF': {
        'type': 'svm',
        'sampling': 'adasyn',  # or 'smote' or None
        'class_weight': 'balanced',
        'beta': 2.0
    },
    'BRF': {
        'type': 'rf',
        'sampling_strategy': 0.7,
        'beta': 2.0
    },
    'XGBoost': {
        'type': 'xgb',
        'eval_metric': 'logloss',
        'beta': 2.0
    }
}

def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    
    # Set up MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name="full_analysis") as parent_run:
            # Load and process data, feature selection
            print("Loading datasets...")
            jm1_df, kc1_df = load_data()
            
            print("Preprocessing data...")
            X_train, X_test, y_train, y_test, transformer = preprocess_data(
                jm1_df, 
                kc1_df, 
                test_size=config['data']['test_size'],
                random_state=config['data']['random_state']
            )
            
            # Log dataset info
            mlflow.log_param("n_samples_train", len(X_train))
            mlflow.log_param("n_samples_test", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_distribution_train", dict(pd.Series(y_train).value_counts()))
            mlflow.log_param("class_distribution_test", dict(pd.Series(y_test).value_counts()))
            
            # Feature selection
            print("Performing feature selection...")
            X_train_selected, X_test_selected, selected_features = select_features(
                X_train, y_train, X_test
            )
            
            # Train models and store results
            results = {}
            for name, model_config in model_configs.items():
                print(f"\nOptimizing {name}...")
                with mlflow.start_run(run_name=f"train_{name.lower()}", nested=True) as child_run:
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
                    pipeline.fit(X_train_selected, y_train)
                    y_pred = pipeline.predict(X_test_selected)
                    y_prob = pipeline.predict_proba(X_test_selected)[:, 1]
                    
                    # Calculate all metrics
                    basic_metrics = calculate_basic_metrics(y_test, y_pred, y_prob, beta=model_config['beta'])
                    threshold_metrics = calculate_threshold_metrics(y_test, y_prob, beta=model_config['beta'])
                    detailed_metrics = calculate_detailed_metrics(y_test, y_pred, y_prob)
                    
                    results[name] = {
                        "metrics": basic_metrics,
                        "threshold": threshold_metrics['optimal_threshold'],
                        "detailed": detailed_metrics,
                        "best_params": best_params,
                        "best_val_score": best_score
                    }
            
            # Generate visualizations
            plots = {}
            print("\nGenerating visualizations...")
            plots['class_distribution'] = plot_class_distribution(y_train, title="Training Set Class Distribution")
            plots['feature_distributions'] = plot_feature_distributions(
                pd.concat([X_train_selected, pd.Series(y_train, name='target')], axis=1),
                features=selected_features[:5],  # Show top 5 selected features
                target='target'
            )
            plots['correlation_matrix'] = plot_correlation_matrix(X_train_selected[selected_features])
            
            # Add PR curves for all models
            y_probs_dict = {name: r["detailed"]["probabilities"] for name, r in results.items()}
            plots['pr_curves'] = plot_pr_curves(y_test, y_probs_dict)
            
            # Add metrics comparison
            plots['metrics_comparison'] = plot_metrics_comparison({name: r["metrics"] for name, r in results.items()})
            
            # Save plots
            plot_manager.save_plots(plots, subdir='figures')
            
            # Compare model results
            print("\nModel Comparison Results:")
            print("-" * 50)
            for name, result in results.items():
                metrics = result['metrics']
                print(f"\n{name}:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F-beta Score: {metrics['f_beta']:.3f}")
                print(f"PR-AUC: {metrics.get('pr_auc', 'N/A')}")
                print(f"Optimal Threshold: {result['threshold']:.3f}")
                print(f"Best Validation Score: {result['best_val_score']:.3f}")
            
            # Save results
            os.makedirs("reports/model_comparison", exist_ok=True)
            with open("reports/model_comparison/results.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for name, result in results.items():
                    json_results[name] = {
                        "metrics": {k: float(v) for k, v in result["metrics"].items()},
                        "threshold": float(result["threshold"]),
                        "best_val_score": float(result["best_val_score"]),
                        "best_params": result["best_params"]
                    }
                json.dump(json_results, f, indent=4)
                
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()