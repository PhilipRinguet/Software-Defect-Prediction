import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import confusion_matrix, classification_report
from src.visualization.visualize import (
    plot_pr_curves,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_threshold_impact,
    save_plots
)

def evaluate_models(y_true, predictions_dict, output_dir='reports/model_comparison'):
    """Comprehensive evaluation of multiple models"""
    results = {}
    y_probs_dict = {}
    
    for name, preds in predictions_dict.items():
        y_pred, y_prob = preds['predictions'], preds['probabilities']
        
        # Store probabilities for curves
        y_probs_dict[name] = y_prob
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        results[name] = {
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    # Generate plots
    plots = {
        'pr_curves': plot_pr_curves(y_true, y_probs_dict),
        'roc_curves': plot_roc_curves(y_true, y_probs_dict),
        'metrics_comparison': plot_metrics_comparison(
            {name: pred['classification_report'] for name, pred in results.items()}
        )
    }
    
    # Add threshold impact plots for each model
    for name, y_prob in y_probs_dict.items():
        plots[f'threshold_impact_{name}'] = plot_threshold_impact(y_true, y_prob)
    
    # Save plots
    save_plots(plots, output_dir)
    
    # Log results with MLflow
    for name, result in results.items():
        with mlflow.start_run(run_name=f"evaluation_{name}"):
            # Log confusion matrix as a figure
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', ax=ax)
            plt.title(f'Confusion Matrix - {name}')
            mlflow.log_figure(fig, f'confusion_matrix_{name}.png')
            plt.close(fig)
            
            # Log classification report metrics
            report = result['classification_report']
            for metric, value in report['weighted avg'].items():
                mlflow.log_metric(f'weighted_{metric}', value)
    
    return results

def analyze_feature_importance(model, feature_names, output_dir='reports/feature_analysis'):
    """Analyze and visualize feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't provide feature importance information")
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # Log with MLflow
    with mlflow.start_run(run_name="feature_analysis"):
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        
        # Log feature importance values
        for idx, row in importance_df.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
    
    return importance_df

def save_evaluation_results(results, output_dir='reports/model_comparison'):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results for each model
    for name, result in results.items():
        model_dir = f'{output_dir}/{name}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save confusion matrix
        np.save(f'{model_dir}/confusion_matrix.npy', result['confusion_matrix'])
        
        # Save classification report
        pd.DataFrame(result['classification_report']).to_csv(
            f'{model_dir}/classification_report.csv'
        )
        
        # Save predictions
        pd.DataFrame({
            'predictions': result['predictions'],
            'probabilities': result['probabilities']
        }).to_csv(f'{model_dir}/predictions.csv', index=False)
    
    # Save comparative results
    comparative_metrics = {}
    for name, result in results.items():
        comparative_metrics[name] = result['classification_report']['weighted avg']
    
    pd.DataFrame(comparative_metrics).to_csv(f'{output_dir}/model_comparison.csv')