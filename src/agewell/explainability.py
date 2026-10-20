"""Feature attribution, missingness-indicator importance separation, and error analysis."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from agewell.models import permutation_importance_np


def compute_feature_explanations(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on held-out test data and flag missingness-indicator features."""
    imp_mean, imp_std = permutation_importance_np(model, X_test, y_test, n_repeats=8, random_state=seed)
    rows = []
    for idx, col in enumerate(X_test.columns):
        is_ind = col.endswith("_is_missing")
        rows.append({
            "feature": col,
            "feature_category": "Missingness Indicator" if is_ind else "Substantive Clinical/Survey",
            "importance_mean": round(float(imp_mean[idx]), 5),
            "importance_std": round(float(imp_std[idx]), 5),
        })
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def summarize_error_Taxonomy(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.28,
) -> pd.DataFrame:
    """Profile True Positives, False Negatives (missed decline), and False Positives across key clinical covariates."""
    work = df_test.copy()
    work["y_true"] = np.asarray(y_true, dtype=int)
    work["y_prob"] = np.asarray(y_prob, dtype=float)
    work["y_pred"] = (work["y_prob"] >= threshold).astype(int)

    conditions = [
        (work["y_true"] == 1) & (work["y_pred"] == 1),
        (work["y_true"] == 1) & (work["y_pred"] == 0),
        (work["y_true"] == 0) & (work["y_pred"] == 1),
        (work["y_true"] == 0) & (work["y_pred"] == 0),
    ]
    labels = ["True Positive (Detected Decline)", "False Negative (Missed Decline)", "False Positive (False Alert)", "True Negative (Stable)"]
    work["outcome_group"] = np.select(conditions, labels, default="Other")

    feature_cols = ["age", "self_rated_health", "prior_falls_count", "gait_speed_mps", "delayed_recall_score"]
    work["missing_feature_count"] = work[feature_cols].isna().sum(axis=1)

    summary = (
        work.groupby("outcome_group")
        .agg(
            count=("y_true", "count"),
            mean_predicted_prob=("y_prob", "mean"),
            mean_age=("age", "mean"),
            mean_self_rated_health=("self_rated_health", "mean"),
            mean_prior_falls=("prior_falls_count", "mean"),
            mean_missing_features=("missing_feature_count", "mean"),
        )
        .reset_index()
    )
    return summary.round(3)
