"""Binary classification metrics with an explicit positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Computes metrics; sensitivity treats malignant (1) as the positive class."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")

    y_true = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(probabilities).astype(float).ravel()
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same length.")

    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "threshold": threshold,
    }

