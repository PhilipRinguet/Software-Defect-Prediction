"""Model comparison utilities."""
import os
import json
import pandas as pd
from src.models.metrics import (
    calculate_basic_metrics,
    calculate_threshold_metrics,
    calculate_detailed_metrics
)
from src.visualization.plotting import plot_manager
from src.visualization.visualize import (
    plot_pr_curves,
    plot_metrics_comparison,
    plot_threshold_impact
)
from src.utils.config import load_config

class ModelComparison:
    def __init__(self, reports_dir='reports/model_comparison'):
        self.reports_dir = reports_dir
        self.results = {}
        self.predictions = {}
        self.config = load_config()
        self.beta = self.config['training']['metrics']['beta']
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics using our metrics module
        basic_metrics = calculate_basic_metrics(y_test, y_pred, y_prob, beta=self.beta)
        threshold_metrics = calculate_threshold_metrics(y_test, y_prob, beta=self.beta)
        detailed_metrics = calculate_detailed_metrics(y_test, y_pred, y_prob)
        
        return {
            'basic': basic_metrics,
            'threshold': threshold_metrics,
            'detailed': detailed_metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def compare_models(self, models, X_test, y_test):
        """Compare multiple models on test data."""
        results = {}
        y_probs = {}
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            y_probs[name] = metrics['probabilities']
        
        self.results = results
        self.predictions = y_probs
        
        # Generate and save comparison plots
        plots = {
            'pr_curves': plot_pr_curves(y_test, y_probs),
            'metrics_comparison': plot_metrics_comparison(
                {name: res['basic'] for name, res in results.items()}
            )
        }
        
        # Add individual threshold impact plots
        for name, y_prob in y_probs.items():
            plots[f'threshold_impact_{name}'] = plot_threshold_impact(y_test, y_prob)
        
        # Save all plots
        plot_manager.save_plots(plots, subdir='model_comparison')
        
        # Create results DataFrame with basic metrics and thresholds
        results_df = pd.DataFrame({
            name: {
                'Precision': res['basic']['precision'],
                'Recall': res['basic']['recall'],
                'F-beta Score': res['basic']['f_beta'],
                'PR-AUC': res['basic']['pr_auc'],
                'Optimal Threshold': res['threshold']['optimal_threshold']
            }
            for name, res in results.items()
        }).T
        
        return results_df
    
    def save_results(self, results=None):
        """Save comparison results."""
        os.makedirs(self.reports_dir, exist_ok=True)
        results_to_save = results if results is not None else self.results
        
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
        """Load saved comparison results."""
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