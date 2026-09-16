"""Sections 20, 21 — Metrics computation and leaderboard ranking."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba[:, 1]), 4)
            else:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 4
                )
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def build_leaderboard(results: list[dict], problem_type: str) -> list[dict]:
    """results: list of per-model result dicts from model_trainer. Ranks by primary metric,
    breaking ties with CV stability (lower std is better)."""
    valid = [r for r in results if r.get("status") == "success"]

    if problem_type == "classification":
        key = lambda r: (-r["metrics"]["f1"], r["cv"]["std"])
    else:
        key = lambda r: (-r["metrics"]["r2"], r["cv"]["std"])

    ranked = sorted(valid, key=key)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked