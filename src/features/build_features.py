import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for feature selection based on mutual information."""
    def __init__(self, k=10):
        self.k = k
        self.selected_features = None
        
    def fit(self, X, y):
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
        # Get indices of top k features
        self.selected_features = np.argsort(mi_scores)[-self.k:]
        return self
        
    def transform(self, X):
        return X.iloc[:, self.selected_features]

def handle_class_imbalance(X, y, method='adasyn', random_state=42):
    """Apply oversampling to handle class imbalance."""
    if method.lower() == 'smote':
        sampler = SMOTE(random_state=random_state)
    else:  # default to ADASYN
        sampler = ADASYN(random_state=random_state)
        
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

def select_features(X_train, y_train, X_test, k=10):
    """Select top k features based on mutual information scores."""
    selector = FeatureSelector(k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.selected_features]
    return X_train_selected, X_test_selected, selected_features

if __name__ == "__main__":
    # This section would typically load the preprocessed data and apply feature engineering
    from src.data.make_dataset import load_data, preprocess_data
    
    # Load and preprocess data
    jm1_df, kc1_df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(jm1_df, kc1_df)
    
    # Apply feature selection
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)
    print("\nSelected Features:")
    print(selected_features.tolist())
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_selected, y_train)
    print("\nClass balance after ADASYN:")
    print(y_train_balanced.value_counts(normalize=True))