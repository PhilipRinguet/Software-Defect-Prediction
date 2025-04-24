import os
import pandas as pd
import mlflow

class ModelComparison:
    def __init__(self, reports_dir):
        self.reports_dir = reports_dir
        self.results_dir = os.path.join(reports_dir, 'model_comparison')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_results(self):
        """Load and aggregate results from MLflow runs."""
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment by name
        experiment = client.get_experiment_by_name("Default")
        if experiment is None:
            return pd.DataFrame()  # Return empty DataFrame if no experiment found
        
        # Get all runs for the experiment
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        results = []
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            model_type = params.get('model_type', 'unknown')
            
            result = {
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F-beta Score': metrics.get('f2_score', 0),
                'PR-AUC': metrics.get('average_precision', 0)
            }
            results.append((model_type, result))
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame([r[1] for r in results], index=[r[0] for r in results])
            return df
        return pd.DataFrame()

    def save_results(self, results_df):
        """Save results to CSV file."""
        results_path = os.path.join(self.results_dir, 'model_comparison_results.csv')
        results_df.to_csv(results_path)
        return results_path