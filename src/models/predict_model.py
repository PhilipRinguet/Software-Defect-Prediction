import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)

def predict(model, X, transformer=None):
    """Make predictions using the trained model.
    
    Args:
        model: Trained model instance
        X: Features to predict on
        transformer: Optional preprocessor to transform features
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Transform features if transformer is provided
    if transformer is not None:
        X = transformer.transform(X)
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities

def predict_from_file(input_path, model_path, transformer_path=None, output_path=None):
    """Make predictions on data from a file.
    
    Args:
        input_path: Path to input data file (ARFF format)
        model_path: Path to saved model
        transformer_path: Optional path to saved transformer
        output_path: Optional path to save predictions
    
    Returns:
        DataFrame with original features and predictions
    """
    from scipy.io import arff
    
    # Load data
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    
    # Load model and transformer
    model = load_model(model_path)
    transformer = None
    if transformer_path:
        transformer = load_model(transformer_path)
    
    # Make predictions
    predictions, probabilities = predict(model, df, transformer)
    
    # Add predictions to dataframe
    df['predicted_defect'] = predictions
    df['defect_probability'] = probabilities
    
    # Save results if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    from src.data.make_dataset import load_data, preprocess_data
    
    # Example usage
    model_path = 'models/svm_model.pkl'
    
    # Load test data
    jm1_df, kc1_df = load_data()
    _, X_test, _, y_test, transformer = preprocess_data(jm1_df, kc1_df)
    
    # Load model and make predictions
    model = load_model(model_path)
    predictions, probabilities = predict(model, X_test, transformer)
    
    # Print sample results
    print("\nSample Predictions:")
    results = pd.DataFrame({
        'True_Label': y_test[:5],
        'Predicted': predictions[:5],
        'Probability': probabilities[:5]
    })
    print(results)