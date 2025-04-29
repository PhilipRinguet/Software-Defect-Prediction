import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

from src.utils.config import get_model_config

def suggest_value(trial: Trial, param_name: str, param_config: dict):
    """Suggest a value based on the parameter configuration"""
    if param_config['type'] == 'loguniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
    elif param_config['type'] == 'int':
        return trial.suggest_int(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'float':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])
    else:
        raise ValueError(f"Unknown parameter type: {param_config['type']}")

def optimize_svm(X_train, y_train, cv=5, n_trials=50, timeout=3600):
    """Optimize SVM hyperparameters using Optuna"""
    config = get_model_config('svm')
    search_space = config['search_space']
    
    def objective(trial: Trial):
        params = {
            param: suggest_value(trial, param, config)
            for param, config in search_space.items()
        }
        
        steps = [
            ('scaler', StandardScaler()),
        ]
        
        if params['sampling'] == "smote":
            steps.append(('sampler', SMOTE(
                sampling_strategy=config['sampling']['sampling_strategy'],
                k_neighbors=config['sampling']['neighbors'],
                random_state=42
            )))
        elif params['sampling'] == "adasyn":
            steps.append(('sampler', ADASYN(
                sampling_strategy=config['sampling']['sampling_strategy'],
                n_neighbors=config['sampling']['neighbors'],
                random_state=42
            )))
            
        steps.append(('svm', SVC(
            kernel='rbf',
            C=params['C'],
            gamma=params['gamma'],
            probability=True,
            class_weight='balanced',
            random_state=42
        )))
        
        pipeline = ImbPipeline(steps) if params['sampling'] else Pipeline(steps)
        
        try:
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring='average_precision',
                cv=cv,
                n_jobs=-1
            )
            return scores.mean()
        except Exception as e:
            return float('-inf')

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, study.best_value

def optimize_rf(X_train, y_train, cv=5, n_trials=50, timeout=3600):
    """Optimize Random Forest hyperparameters using Optuna"""
    config = get_model_config('random_forest')
    search_space = config['search_space']
    
    def objective(trial: Trial):
        params = {
            param: suggest_value(trial, param, config)
            for param, config in search_space.items()
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', BalancedRandomForestClassifier(
                **params,
                random_state=42
            ))
        ])
        
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring='average_precision',
            cv=cv,
            n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, study.best_value

def optimize_xgb(X_train, y_train, cv=5, n_trials=50, timeout=3600):
    """Optimize XGBoost hyperparameters using Optuna"""
    config = get_model_config('xgboost')
    search_space = config['search_space']
    
    # Calculate class weight for XGBoost
    neg_count, pos_count = np.bincount(y_train)
    scale_pos_weight = neg_count / pos_count
    
    def objective(trial: Trial):
        params = {
            param: suggest_value(trial, param, config)
            for param, config in search_space.items()
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,  # Add class weight balancing
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring='average_precision',
            cv=cv,
            n_jobs=-1
        )
        return scores.mean()
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, study.best_value