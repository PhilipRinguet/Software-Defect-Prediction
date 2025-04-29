"""Model training utilities."""
import os
import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    fbeta_score, 
    average_precision_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

from src.features.build_features import ColumnSelector, transform_features
from src.models.metrics import (
    calculate_basic_metrics,
    calculate_threshold_metrics,
    calculate_detailed_metrics
)
from src.utils.config import load_config
from .optimize import optimize_svm, optimize_rf, optimize_xgb

def get_model_config(model_type):
    """Get model configuration from config file."""
    config = load_config()
    return config['models'].get(model_type, {})

def create_svm_pipeline(kernel='rbf', C=1.0, gamma='scale', probability=True, 
                       class_weight='balanced', sampling=None, features=None):
    """Create SVM pipeline with optional SMOTE/ADASYN sampling"""
    config = get_model_config('svm')
    default_params = config.get('default_params', {})
    sampling_config = config.get('sampling', {})
    
    steps = []
    
    if features is not None:
        steps.append(('selector', ColumnSelector(columns=features)))
        
    steps.append(('transform', PowerTransformer(method='yeo-johnson', standardize=False)))
    steps.append(('scaler', StandardScaler()))
    
    # Use provided sampling or default from config
    sampling = sampling or sampling_config.get('method')
    if sampling == 'smote':
        steps.append(('sampler', SMOTE(
            sampling_strategy=sampling_config.get('sampling_strategy', 0.8),
            k_neighbors=sampling_config.get('neighbors', 3),
            random_state=42
        )))
    elif sampling == 'adasyn':
        steps.append(('sampler', ADASYN(
            sampling_strategy=sampling_config.get('sampling_strategy', 0.8),
            n_neighbors=sampling_config.get('neighbors', 3),
            random_state=42
        )))
    
    steps.append(('svm', SVC(
        kernel=kernel or default_params.get('kernel', 'rbf'),
        C=C or default_params.get('C', 1.0),
        gamma=gamma or default_params.get('gamma', 'scale'),
        probability=probability or default_params.get('probability', True),
        class_weight=class_weight or default_params.get('class_weight', 'balanced'),
        random_state=42
    )))
    
    # Use imbalanced-learn pipeline if sampling is enabled, otherwise use sklearn pipeline
    return ImbPipeline(steps) if sampling else Pipeline(steps)

def create_brf_pipeline(features=None, n_estimators=100, max_depth=2, 
                       sampling_strategy=0.7, min_samples_split=15, 
                       min_samples_leaf=20):
    """Create Balanced Random Forest pipeline"""
    config = get_model_config('random_forest')
    default_params = config.get('default_params', {})
    
    steps = []
    
    if features is not None:
        steps.append(('selector', ColumnSelector(columns=features)))
        
    steps.append(('transform', PowerTransformer(method='yeo-johnson', standardize=True)))
    steps.append(('brf', BalancedRandomForestClassifier(
        n_estimators=n_estimators or default_params.get('n_estimators', 100),
        max_depth=max_depth or default_params.get('max_depth', 2),
        sampling_strategy=sampling_strategy or default_params.get('sampling_strategy', 0.7),
        min_samples_split=min_samples_split or default_params.get('min_samples_split', 15),
        min_samples_leaf=min_samples_leaf or default_params.get('min_samples_leaf', 20),
        random_state=42
    )))

    return Pipeline(steps)

def create_xgb_pipeline(features=None, max_depth=2, learning_rate=0.003, 
                       n_estimators=150, subsample=0.4, colsample_bytree=0.3,
                       gamma=5.0, reg_alpha=125.0, reg_lambda=325.0,
                       scale_pos_weight=1.0):
    """Create XGBoost pipeline"""
    config = get_model_config('xgboost')
    default_params = config.get('default_params', {})
    
    steps = []
    
    if features is not None:
        steps.append(('selector', ColumnSelector(columns=features)))
        
    steps.append(('transform', PowerTransformer(method='yeo-johnson', standardize=True)))
    steps.append(('xgb', XGBClassifier(
        max_depth=max_depth or default_params.get('max_depth', 2),
        learning_rate=learning_rate or default_params.get('learning_rate', 0.003),
        n_estimators=n_estimators or default_params.get('n_estimators', 150),
        subsample=subsample or default_params.get('subsample', 0.4),
        colsample_bytree=colsample_bytree or default_params.get('colsample_bytree', 0.3),
        gamma=gamma or default_params.get('gamma', 5.0),
        reg_alpha=reg_alpha or default_params.get('reg_alpha', 125.0),
        reg_lambda=reg_lambda or default_params.get('reg_lambda', 325.0),
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        eval_metric='logloss',
        random_state=42
    )))

    return Pipeline(steps)

def train_with_optimization(X_train, y_train, model_type='svm', cv=5, 
                          n_trials=50, timeout=3600, features=None):
    """Train model with hyperparameter optimization"""
    with mlflow.start_run(nested=True):
        if model_type == 'svm':
            best_params, best_score = optimize_svm(X_train, y_train, cv, n_trials, timeout)
            pipeline = create_svm_pipeline(
                C=best_params['C'],
                gamma=best_params['gamma'],
                sampling=best_params['sampling'],
                features=features
            )
        elif model_type == 'rf':
            best_params, best_score = optimize_rf(X_train, y_train, cv, n_trials, timeout)
            pipeline = create_brf_pipeline(
                features=features,
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                sampling_strategy=best_params['sampling_strategy'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf']
            )
        elif model_type == 'xgb':
            best_params, best_score = optimize_xgb(X_train, y_train, cv, n_trials, timeout)
            # Calculate class weight for XGBoost
            neg_count, pos_count = np.bincount(y_train)
            scale_pos_weight = neg_count / pos_count
            pipeline = create_xgb_pipeline(
                features=features,
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                n_estimators=best_params['n_estimators'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                gamma=best_params['gamma'],
                reg_alpha=best_params['reg_alpha'],
                reg_lambda=best_params['reg_lambda'],
                scale_pos_weight=scale_pos_weight  # Pass calculated weight
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Log optimization results
        mlflow.log_params(best_params)
        mlflow.log_metric("best_validation_score", best_score)
        
        return pipeline, best_params, best_score

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model pipeline."""
    config = load_config()
    beta = config['training']['metrics']['beta']
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    basic_metrics = calculate_basic_metrics(y_test, y_pred, y_prob, beta=beta)
    threshold_metrics = calculate_threshold_metrics(y_test, y_prob, beta=beta)
    detailed_metrics = calculate_detailed_metrics(y_test, y_pred, y_prob)
    
    # Combine metrics
    metrics = {**basic_metrics, **detailed_metrics}
    optimal_threshold = threshold_metrics['optimal_threshold']
    
    # Log metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metrics(basic_metrics)
        mlflow.log_param('optimal_threshold', optimal_threshold)
        
        # Save model
        model_path = f'models/{model_name.lower()}_model.pkl'
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)
    
    return pipeline, metrics, optimal_threshold