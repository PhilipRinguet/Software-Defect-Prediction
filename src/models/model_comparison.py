import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score

def compare_models(models, X_test, y_test, output_dir=None):
    """Compare multiple models on test data"""
    comparison = ModelComparison(reports_dir=output_dir)
    results = comparison.compare_models(models, X_test, y_test)
    comparison.save_results()
    return results

class ModelComparison:
    def __init__(self, reports_dir='reports/model_comparison'):
        self.reports_dir = reports_dir
        self.results = {}
        self.predictions = {}
    
    def evaluate_model(self, model, X_test, y_test, beta=2.0):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F-beta Score': fbeta_score(y_test, y_pred, beta=beta),
            'PR-AUC': average_precision_score(y_test, y_prob)
        }
        
        return metrics
    
    def compare_models(self, models, X_test, y_test):
        """Compare multiple models on test data"""
        results = {}
        predictions = {}
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            predictions[name] = model.predict_proba(X_test)[:, 1]
        
        self.results = results
        self.predictions = predictions
        return pd.DataFrame(results).T
    
    def save_results(self, results=None):
        """Save comparison results"""
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Use provided results or instance results
        results_to_save = results if results is not None else self.results
        
        # Convert to DataFrame if needed
        if not isinstance(results_to_save, pd.DataFrame):
            df = pd.DataFrame(results_to_save).T
        else:
            df = results_to_save.copy()
            
        csv_path = os.path.join(self.reports_dir, 'model_comparison_results.csv')
        json_path = os.path.join(self.reports_dir, 'model_comparison_results.json')
        
        # Save CSV with index
        df.to_csv(csv_path)
        
        # Save JSON in list format but preserve model names
        json_results = []
        for idx, row in df.iterrows():
            result = row.to_dict()
            result['model'] = idx
            json_results.append(result)
            
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return csv_path
    
    def load_results(self):
        """Load saved comparison results"""
        json_path = os.path.join(self.reports_dir, 'model_comparison_results.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Results file not found at {json_path}")
        
        with open(json_path, 'r') as f:
            results = json.load(f)
            
        # Convert list back to DataFrame with proper index
        df = pd.DataFrame(results)
        df.set_index('model', inplace=True)
        df.index.name = None  # Remove index name to match original
        return df