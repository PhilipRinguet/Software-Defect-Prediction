"""Consolidated metrics calculation module for model evaluation."""
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

def calculate_basic_metrics(y_true, y_pred, y_prob=None, beta=2.0):
    """Calculate basic model performance metrics."""
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f_beta': fbeta_score(y_true, y_pred, beta=beta)
    }
    
    if y_prob is not None:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
    return metrics

def calculate_threshold_metrics(y_true, y_prob, beta=2.0):
    """Calculate threshold-based metrics and find optimal threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls)
    optimal_idx = np.nanargmax(f_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    return {
        'optimal_threshold': optimal_threshold,
        'precisions': precisions,
        'recalls': recalls,
        'thresholds': thresholds,
        'f_scores': f_scores
    }

def calculate_detailed_metrics(y_true, y_pred, y_prob=None):
    """Calculate detailed evaluation metrics including confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'confusion_matrix': cm,
        'classification_report': report,
    }
    
    if y_prob is not None:
        metrics['probabilities'] = y_prob
        
    return metrics