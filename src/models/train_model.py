import numpy as np
import mlflow
import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

from src.features.build_features import ColumnSelector, transform_features
from src.utils.config import load_config, get_model_config
from .optimize import optimize_svm, optimize_rf, optimize_xgb

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
                       gamma=5.0, reg_alpha=125.0, reg_lambda=325.0):
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
        eval_metric='logloss',
        random_state=42
    )))

    return Pipeline(steps)

def train_with_optimization(X_train, y_train, model_type='svm', cv=5, 
                          n_trials=50, timeout=3600, features=None):
    """Train model with hyperparameter optimization"""
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
        pipeline = create_xgb_pipeline(
            features=features,
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            gamma=best_params['gamma'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    pipeline.fit(X_train, y_train)
    return pipeline, best_params, best_score

def calculate_metrics(y_true, y_pred, y_prob=None, beta=2.0):
    """Calculate model performance metrics"""
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f_beta': fbeta_score(y_true, y_pred, beta=beta)
    }
    
    if y_prob is not None:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
    return metrics

def find_optimal_threshold(y_true, y_prob, beta=2.0):
    """Find optimal classification threshold using F-beta score"""
    precisions = []
    recalls = []
    thresholds = np.linspace(0, 1, 100)
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Calculate F-beta scores
    f_beta_scores = ((1 + beta**2) * (precisions * recalls) / 
                    ((beta**2 * precisions) + recalls + 1e-8))
    
    optimal_idx = f_beta_scores.argmax()
    return thresholds[optimal_idx]

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, model_name, beta=2.0):
    """Train model and evaluate performance"""
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_train, 
                                            pipeline.predict_proba(X_train)[:, 1],
                                            beta=beta)
    
    # Get predictions using optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob, beta)
    
    # Log with MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(pipeline.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_metric('optimal_threshold', optimal_threshold)
        mlflow.sklearn.log_model(pipeline, "model")
    
    return pipeline, metrics, optimal_threshold