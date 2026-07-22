"""Binary classification metrics and FN/FP export helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, y_score=None, positive_label: int = 1) -> dict:
    """Compute binary classification metrics without assuming clinical label mapping."""
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    if len(labels) == 1:
        labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_score is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["auroc"] = None
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except ValueError:
            metrics["pr_auc"] = None
    return metrics


def export_error_samples(df: pd.DataFrame, y_true, y_pred, output_csv: str, positive_label: int = 1) -> pd.DataFrame:
    """Export FN/FP flags for later error analysis."""
    out = df.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["error_type"] = "correct"
    out.loc[(out["y_true"] == positive_label) & (out["y_pred"] != positive_label), "error_type"] = "fn"
    out.loc[(out["y_true"] != positive_label) & (out["y_pred"] == positive_label), "error_type"] = "fp"
    out.to_csv(output_csv, index=False)
    return out
