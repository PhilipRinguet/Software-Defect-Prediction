import argparse
import mlflow
import logging
from pathlib import Path

from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import select_features, handle_class_imbalance
from src.models.train_model import optimize_hyperparameters, train_and_evaluate
from src.visualization.visualize import (
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_precision_recall_curve,
    plot_class_distribution
)

def setup_logging(log_path='logs'):
    """Set up logging configuration."""
    Path(log_path).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_path}/sdp.log'),
            logging.StreamHandler()
        ]
    )

def run_pipeline(args):
    """Run the complete software defect prediction pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting software defect prediction pipeline")
    
    # Start MLflow run
    with mlflow.start_run():
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        jm1_df, kc1_df = load_data(
            jm1_path=args.jm1_path,
            kc1_path=args.kc1_path
        )
        
        # Generate initial visualizations
        if args.visualize:
            logger.info("Generating initial visualizations")
            plot_class_distribution(jm1_df['target'], 
                                 save_path='reports/figures/jm1_class_dist.png')
            plot_class_distribution(kc1_df['target'],
                                 save_path='reports/figures/kc1_class_dist.png')
            plot_correlation_matrix(jm1_df.drop('target', axis=1),
                                save_path='reports/figures/correlation_matrix.png')
        
        # Preprocess data
        X_train, X_test, y_train, y_test, transformer = preprocess_data(
            jm1_df, kc1_df, test_size=args.test_size
        )
        
        # Feature engineering
        logger.info("Performing feature engineering")
        X_train_selected, X_test_selected, selected_features = select_features(
            X_train, y_train, X_test, k=args.n_features
        )
        
        # Handle class imbalance
        logger.info("Handling class imbalance")
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train_selected, y_train,
            method=args.resampling_method
        )
        
        # Optimize hyperparameters if requested
        if args.optimize:
            logger.info("Optimizing hyperparameters")
            best_params, best_score = optimize_hyperparameters(
                X_train_balanced, X_test_selected,
                y_train_balanced, y_test,
                model_type=args.model_type,
                n_trials=args.n_trials
            )
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best validation score: {best_score}")
        else:
            best_params = None
        
        # Train and evaluate model
        logger.info("Training and evaluating model")
        model, metrics = train_and_evaluate(
            X_train_balanced, X_test_selected,
            y_train_balanced, y_test,
            model_type=args.model_type,
            params=best_params
        )
        
        # Log metrics
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.3f}")
            mlflow.log_metric(metric_name, value)
        
        # Generate evaluation visualizations
        if args.visualize:
            logger.info("Generating evaluation visualizations")
            y_prob = model.predict_proba(X_test_selected)[:, 1]
            plot_precision_recall_curve(
                y_test, y_prob,
                save_path='reports/figures/precision_recall_curve.png'
            )

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Software Defect Prediction Pipeline'
    )
    
    parser.add_argument('--jm1-path', type=str, default='data/raw/jm1.arff',
                      help='Path to JM1 dataset')
    parser.add_argument('--kc1-path', type=str, default='data/raw/kc1.arff',
                      help='Path to KC1 dataset')
    parser.add_argument('--model-type', type=str, default='svm',
                      choices=['logistic', 'svm', 'balanced_rf', 'xgboost'],
                      help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.3,
                      help='Proportion of dataset to use for testing')
    parser.add_argument('--n-features', type=int, default=10,
                      help='Number of features to select')
    parser.add_argument('--resampling-method', type=str, default='adasyn',
                      choices=['smote', 'adasyn'],
                      help='Method for handling class imbalance')
    parser.add_argument('--optimize', action='store_true',
                      help='Whether to perform hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                      help='Number of optimization trials')
    parser.add_argument('--visualize', action='store_true',
                      help='Whether to generate visualizations')
    
    args = parser.parse_args()
    setup_logging()
    run_pipeline(args)

if __name__ == "__main__":
    main()