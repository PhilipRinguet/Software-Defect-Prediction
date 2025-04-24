import numpy as np
import mlflow
import joblib
import optuna
from optuna.trial import TrialState
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

def create_model(model_type='svm', params=None):
    """Create a model instance based on the specified type and parameters."""
    model_type = model_type.lower()
    if params is None:
        params = get_default_params(model_type)
    
    # Filter params based on model type
    if model_type == 'svm':
        valid_params = {k: v for k, v in params.items() if k in ['C', 'gamma', 'kernel']}
        return SVC(kernel='rbf', probability=True, random_state=42, **valid_params)
    elif model_type == 'logistic':
        valid_params = {k: v for k, v in params.items() if k in ['C']}
        return LogisticRegression(random_state=42, **valid_params)
    elif model_type == 'balanced_rf':
        valid_params = {k: v for k, v in params.items() if k in ['n_estimators', 'max_depth', 'min_samples_split']}
        return BalancedRandomForestClassifier(random_state=42, **valid_params)
    elif model_type == 'xgboost':
        valid_params = {k: v for k, v in params.items() if k in ['n_estimators', 'learning_rate', 'max_depth']}
        return XGBClassifier(random_state=42, **valid_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various performance metrics."""
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f2_score': fbeta_score(y_true, y_pred, beta=2),
        'average_precision': average_precision_score(y_true, y_prob)
    }
    return metrics

def objective_svm(trial, X_train, X_test, y_train, y_test):
    """Optimization objective for SVM hyperparameters."""
    C = trial.suggest_float('C', 1e-5, 1e5, log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1e5, log=True)
    
    model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Use F-beta score (beta=2) to emphasize recall over precision
    f_beta = fbeta_score(y_test, y_pred, beta=2)
    return f_beta

def objective_rf(trial, X_train, X_test, y_train, y_test):
    """Optimization objective for Balanced Random Forest hyperparameters."""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f_beta = fbeta_score(y_test, y_pred, beta=2)
    return f_beta

def optimize_hyperparameters(X_train, X_test, y_train, y_test, model_type='svm', n_trials=100):
    """Optimize model hyperparameters using Optuna."""
    study = optuna.create_study(direction='maximize')
    
    if model_type == 'svm':
        objective = lambda trial: objective_svm(trial, X_train, X_test, y_train, y_test)
    else:  # balanced_rf
        objective = lambda trial: objective_rf(trial, X_train, X_test, y_train, y_test)
    
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def get_default_params(model_type):
    """Get default parameters for the specified model type."""
    model_type = model_type.lower()
    default_params = {
        'svm': {'C': 1.0, 'gamma': 'scale'},
        'logistic': {'C': 1.0},
        'balanced_rf': {'n_estimators': 100},
        'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}
    }
    return default_params.get(model_type, {})

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='svm', params=None):
    """Train and evaluate a model."""
    if params is None:
        params = get_default_params(model_type)
    
    model = create_model(model_type, params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Log parameters and model
    mlflow.log_params(params)
    mlflow.sklearn.log_model(model, "model")
    
    return model, metrics

if __name__ == "__main__":
    from src.data.make_dataset import load_data, preprocess_data
    from src.features.build_features import select_features, handle_class_imbalance
    
    # Load and preprocess data
    jm1_df, kc1_df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(jm1_df, kc1_df)
    
    # Feature engineering
    X_train_selected, X_test_selected, _ = select_features(X_train, y_train, X_test)
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_selected, y_train)
    
    # Optimize hyperparameters
    best_params, best_score = optimize_hyperparameters(
        X_train_balanced, X_test_selected, 
        y_train_balanced, y_test, 
        model_type='svm'
    )
    print(f"\nBest parameters: {best_params}")
    print(f"Best F-beta score: {best_score}")
    
    # Train and evaluate model
    model, metrics = train_and_evaluate(
        X_train_balanced, X_test_selected,
        y_train_balanced, y_test,
        model_type='svm',
        params=best_params
    )
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")