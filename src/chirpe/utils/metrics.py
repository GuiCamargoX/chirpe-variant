"""Classification metric utilities for evaluation and reporting."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate weighted and binary-specific metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of scalar metrics. Binary-only metrics (for example AUC,
        specificity) are included only when both classes are present.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Binary-only diagnostics are gated to avoid invalid multiclass behavior.
    if len(np.unique(y_true)) == 2:
        metrics["f1_binary"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision_binary"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        metrics["recall_binary"] = float(recall_score(y_true, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC-AUC
        if y_prob is not None:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))

            # Precision-Recall AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = float(auc(recall_vals, precision_vals))

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print a metric dictionary to stdout.

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key.upper():20s}: {value:.4f}")

    print("=" * 50 + "\n")
