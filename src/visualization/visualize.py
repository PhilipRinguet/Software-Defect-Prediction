import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, fbeta_score,
    precision_recall_curve, roc_curve, auc
)

def plot_feature_distributions(data, features=None, target=None, n_cols=3):
    """Plot distributions of features split by target class"""
    # Convert data to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Handle features parameter
    if features is None:
        features = [col for col in data.columns if col != 'target']
    elif isinstance(features, str):
        features = [features]
    
    # Handle target column
    if target is None:
        target = 'target'  # Use default target column name
    
    # Validate features
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in data: {missing_features}")
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(data=data, x=target, y=feature, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution by Class')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return plt.gcf()

def plot_pr_curves(y_true, y_probs_dict, title="Precision-Recall Curves"):
    """Plot precision-recall curves for multiple models"""
    plt.figure(figsize=(10, 6))
    
    for name, y_prob in y_probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid
    return plt.gcf()

def plot_metrics_comparison(metrics_dict, beta=2.0):
    """Plot comparison of model metrics"""
    metrics = ['precision', 'recall', f'f{beta}_score', 'pr_auc']
    models = list(metrics_dict.keys())
    
    # Prepare data for plotting
    data = []
    for model in models:
        for metric in metrics:
            value = metrics_dict[model].get(metric, metrics_dict[model].get('f_beta'))
            data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Value': value
            })
    
    plt.figure(figsize=(12, 6))
    plot_data = pd.DataFrame(data)
    sns.barplot(data=plot_data, x='Model', y='Value', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_threshold_impact(y_true, y_prob, thresholds=None, beta=2.0):
    """Plot impact of different classification thresholds"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    precisions = []
    recalls = []
    f_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f_scores.append(fbeta_score(y_true, y_pred, beta=beta))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f_scores, label=f'F{beta}-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Classification Threshold')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def save_plots(plots_dict, output_dir='reports/figures'):
    """Save all plots to specified directory"""
    os.makedirs(output_dir, exist_ok=True)
    for name, fig in plots_dict.items():
        fig.savefig(os.path.join(output_dir, f'{name}.png'))
        plt.close(fig)

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    """Plot precision-recall curve for a single model"""
    if isinstance(y_prob, dict):
        return plot_pr_curves(y_true, y_prob, title)
    else:
        return plot_pr_curves(y_true, {'Model': y_prob}, title)

def plot_class_distribution(y, title="Class Distribution"):
    """Plot the distribution of target classes"""
    plt.figure(figsize=(8, 6))
    class_counts = pd.Series(y).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add count labels on top of each bar
    for i, v in enumerate(class_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    return plt.gcf()

def plot_feature_importance(feature_importance, feature_names=None, title="Feature Importance", figsize=(10, 6)):
    """Plot feature importance scores
    
    Parameters:
    -----------    
    feature_importance : array-like or dict
        Feature importance scores. Can be an array of scores or a dictionary
        mapping feature names to scores
    feature_names : list, optional
        Names of features if feature_importance is an array
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if isinstance(feature_importance, dict):
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
    else:
        scores = feature_importance
        features = feature_names if feature_names is not None else [f"Feature {i}" for i in range(len(scores))]
    
    # Create bar plot
    y_pos = np.arange(len(features))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, features)
    
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_correlation_matrix(data, method='spearman', annot=True, figsize=(12, 8)):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=figsize)
    corr_matrix = data.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f'
    )
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()