import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd

def plot_feature_distributions(df, features, save_path=None):
    """Plot distributions of selected features."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle("Feature Distributions", y=1.02, fontsize=16)
    
    for i, feature in enumerate(features):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        sns.histplot(data=df, x=feature, hue="target", ax=ax)
        ax.set_title(f"Distribution of {feature}")
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix of features."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title("Feature Correlation Matrix", pad=20)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """Plot distribution of target classes."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(feature_importance, feature_names, title="Feature Importance", save_path=None):
    """Plot feature importance scores."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title(title)
    plt.xlabel("Importance Score")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    from src.data.make_dataset import load_data, preprocess_data
    from src.models.predict_model import load_model, predict
    
    # Load data
    jm1_df, kc1_df = load_data()
    
    # Example visualizations
    print("Generating visualizations...")
    
    # Plot class distributions
    plot_class_distribution(jm1_df['target'], title="JM1 Class Distribution",
                          save_path="reports/figures/jm1_class_dist.png")
    plot_class_distribution(kc1_df['target'], title="KC1 Class Distribution",
                          save_path="reports/figures/kc1_class_dist.png")
    
    # Plot feature distributions for a subset of features
    key_features = ['cyclomatic_complexity', 'essential_complexity', 'loc_total']
    plot_feature_distributions(jm1_df, key_features,
                             save_path="reports/figures/feature_distributions.png")
    
    # Plot correlation matrix
    plot_correlation_matrix(jm1_df.drop('target', axis=1),
                          save_path="reports/figures/correlation_matrix.png")