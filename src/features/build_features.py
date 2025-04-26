import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.9, k=None):
        self.correlation_threshold = correlation_threshold
        self.k = k
        self.selected_features_ = None
        self.feature_scores_ = None
    
    def fit(self, X, y):
        # Calculate correlation matrix
        correlation_matrix = X.corr(method="spearman")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_dict = dict(zip(X.columns, mi_scores))
        self.feature_scores_ = mi_dict
        
        # Find highly correlated features
        features_to_remove = set()
        for i in correlation_matrix.columns:
            for j in correlation_matrix.columns:
                if i != j and abs(correlation_matrix.loc[i, j]) > self.correlation_threshold:
                    if mi_dict.get(i, 0) >= mi_dict.get(j, 0):
                        features_to_remove.add(j)
                    else:
                        features_to_remove.add(i)
        
        # Get features after correlation filtering
        candidate_features = [f for f in X.columns if f not in features_to_remove]
        
        # Select top k features if specified
        if self.k is not None and self.k < len(candidate_features):
            feature_scores = [(f, mi_dict[f]) for f in candidate_features]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            self.selected_features_ = [f for f, _ in feature_scores[:self.k]]
        else:
            self.selected_features_ = candidate_features
            
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector has not been fitted yet.")
        return X[self.selected_features_]

def select_features(X_train, y_train, X_test=None, correlation_threshold=0.9, k=None):
    """
    Selects features using correlation and mutual information
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame, optional
        Test features
    correlation_threshold : float
        Threshold for correlation filtering
    k : int, optional
        Number of top features to select
    """
    selector = FeatureSelector(correlation_threshold=correlation_threshold, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test) if X_test is not None else None
    return X_train_selected, X_test_selected, selector.selected_features_

def transform_features(X_train, X_test=None):
    """
    Apply Yeo-Johnson transformation to features
    """
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_transformed = pd.DataFrame(
        transformer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    if X_test is not None:
        X_test_transformed = pd.DataFrame(
            transformer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_transformed, X_test_transformed
    
    return X_train_transformed

def handle_class_imbalance(X, y, method='adasyn', sampling_strategy=1.0):
    """
    Handle class imbalance using SMOTE or ADASYN
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    method : str
        Resampling method ('smote' or 'adasyn')
    sampling_strategy : float
        Desired ratio of minority class samples to majority class samples.
        1.0 means perfectly balanced classes.
        
    Returns:
    --------
    X_resampled : pd.DataFrame
        Resampled features
    y_resampled : pd.Series
        Resampled target variable
    """
    if method.lower() == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method.lower() == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        raise ValueError("method must be either 'smote' or 'adasyn'")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)