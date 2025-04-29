"""Visualization utilities for data analysis and model evaluation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .plotting import plot_manager
from src.models.metrics import calculate_basic_metrics, calculate_threshold_metrics, calculate_detailed_metrics

def plot_feature_distributions(data, features=None, target=None):
    """Plot distributions of features split by target class."""
    features = features if features is not None else data.columns
    fig = plot_manager.setup_figure('feature_importance')
    
    for feature in features:
        sns.histplot(data=data, x=feature, hue=target, element='step', stat='density')
        
    return fig

def plot_correlation_matrix(data):
    """Plot correlation matrix heatmap."""
    fig = plot_manager.setup_figure('correlation_matrix')
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    return fig

def plot_pr_curves(y_true, y_probs_dict, title="Precision-Recall Curves"):
    """Plot precision-recall curves for multiple models."""
    fig = plot_manager.setup_figure()
    
    for name, y_prob in y_probs_dict.items():
        metrics = calculate_threshold_metrics(y_true, y_prob)
        plt.plot(metrics['recalls'], metrics['precisions'], label=f'{name}')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    return fig

def plot_metrics_comparison(metrics_dict):
    """Plot comparison of model metrics."""
    fig = plot_manager.setup_figure('metrics_comparison')
    metrics = ['precision', 'recall', 'f_beta', 'pr_auc']
    models = list(metrics_dict.keys())
    
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Value': metrics_dict[model].get(metric, metrics_dict[model].get('f_beta'))
            })
    
    sns.barplot(data=pd.DataFrame(data), x='Model', y='Value', hue='Metric')
    plt.xticks(rotation=45)
    plt.title('Model Performance Comparison')
    return fig

def plot_threshold_impact(y_true, y_prob):
    """Plot impact of different classification thresholds."""
    fig = plot_manager.setup_figure()
    metrics = calculate_threshold_metrics(y_true, y_prob)
    
    plt.plot(metrics['thresholds'], metrics['precisions'][:-1], label='Precision')
    plt.plot(metrics['thresholds'], metrics['recalls'][:-1], label='Recall')
    plt.plot(metrics['thresholds'], metrics['f_scores'][:-1], label='F-beta Score')
    plt.axvline(metrics['optimal_threshold'], color='r', linestyle='--', 
                label=f'Optimal Threshold ({metrics["optimal_threshold"]:.2f})')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Classification Threshold')
    plt.legend()
    return fig

def plot_feature_importance(feature_importance, feature_names=None):
    """Plot feature importance."""
    fig = plot_manager.setup_figure('feature_importance')
    features = feature_names if feature_names is not None else [f'Feature {i}' for i in range(len(feature_importance))]
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.barh(y=range(len(importance_df)), width=importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    return fig

def plot_class_distribution(y, title="Class Distribution"):
    """Plot the distribution of classes in the target variable."""
    fig = plot_manager.setup_figure()
    class_counts = pd.Series(y).value_counts()
    
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    
    # Add count labels on top of bars
    for i, count in enumerate(class_counts.values):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    return fig