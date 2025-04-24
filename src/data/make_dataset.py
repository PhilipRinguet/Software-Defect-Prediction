import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split

def load_data(jm1_path="data/raw/jm1.arff", kc1_path="data/raw/kc1.arff"):
    """Load and preprocess the JM1 and KC1 datasets."""
    # Load ARFF files
    jm1_data, _ = arff.loadarff(jm1_path)
    kc1_data, _ = arff.loadarff(kc1_path)
    
    # Convert to pandas DataFrames
    jm1_df = pd.DataFrame(jm1_data)
    kc1_df = pd.DataFrame(kc1_data)
    
    # Convert column names to lowercase
    jm1_df.columns = jm1_df.columns.str.lower()
    kc1_df.columns = kc1_df.columns.str.lower()
    
    # Rename target columns for consistency
    jm1_df.rename(columns={"label": "target"}, inplace=True)
    kc1_df.rename(columns={"defective": "target"}, inplace=True)
    
    # Decode binary strings and convert target to numeric
    jm1_df["target"] = jm1_df["target"].str.decode("utf-8")
    kc1_df["target"] = kc1_df["target"].str.decode("utf-8")
    jm1_df["target"] = np.where(jm1_df["target"] == "Y", 1, 0)
    kc1_df["target"] = np.where(kc1_df["target"] == "Y", 1, 0)
    
    return jm1_df, kc1_df

def preprocess_data(jm1_df, kc1_df, test_size=0.3, random_state=42):
    """Preprocess the datasets using Yeo-Johnson transformation and standardization."""
    # Combine datasets
    combined_data = pd.concat([jm1_df, kc1_df], ignore_index=True)
    
    # Split features and target
    X = combined_data.drop("target", axis=1)
    y = combined_data["target"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Initialize transformers
    yeo_johnson = PowerTransformer(method="yeo-johnson", standardize=True)
    
    # Fit and transform the training data
    X_train_transformed = yeo_johnson.fit_transform(X_train)
    X_test_transformed = yeo_johnson.transform(X_test)
    
    # Convert back to DataFrames with feature names
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=X.columns)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=X.columns)
    
    return X_train_transformed, X_test_transformed, y_train, y_test, yeo_johnson

if __name__ == "__main__":
    # Load data
    jm1_df, kc1_df = load_data()
    
    # Check for data quality issues
    print("Data Quality Check:")
    print(f"JM1 duplicates: {jm1_df.duplicated().sum()}")
    print(f"KC1 duplicates: {kc1_df.duplicated().sum()}")
    print(f"JM1 null values: {jm1_df.isnull().any().any()}")
    print(f"KC1 null values: {kc1_df.isnull().any().any()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(jm1_df, kc1_df)
    
    print("\nData Preprocessing Complete:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")