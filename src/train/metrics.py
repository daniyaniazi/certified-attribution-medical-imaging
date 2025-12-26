"""Metrics computation: accuracy, AUC, etc."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from typing import Dict, Tuple


class MetricsComputer:
    """Compute train/val metrics."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        task: str = 'binary'
    ) -> Dict[str, float]:
        """
        Compute metrics.
        
        Args:
            y_true: ground truth labels
            y_pred: predicted labels (argmax for multi-class)
            y_proba: predicted probabilities [N, num_classes]
            task: 'binary' or 'multi-class'
        
        Returns:
            dict of metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1
        if task == 'binary':
            metrics['f1'] = f1_score(y_true, y_pred)
        else:
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # AUC (for binary or OvR)
        if y_proba is not None:
            if task == 'binary':
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        
        # Sensitivity & Specificity (binary only)
        if task == 'binary':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics for logging."""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
