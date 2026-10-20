"""Comprehensive evaluation metrics, participant-clustered bootstrap CIs, and subgroup audits."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from agewell.models import (
    average_precision_score_np,
    brier_score_loss_np,
    log_loss_np,
    roc_auc_score_np,
)
from agewell.calibration import compute_calibration_slope_intercept, compute_ece


def sensitivity_at_fixed_alert_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alert_rate: float = 0.20,
) -> float:
    """Compute sensitivity (recall) when flagging the top `alert_rate` highest-risk individuals."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    k = max(int(np.round(alert_rate * len(y_prob))), 1)
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(np.sum(y_true[top_k_idx] == 1) / n_pos)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alert_rate: float = 0.20,
) -> Dict[str, float]:
    """Compute primary discrimination, calibration, and operational screening metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "brier": float(brier_score_loss_np(y_true, y_prob)),
            "log_loss": np.nan,
            "ece": np.nan,
            "cal_intercept": np.nan,
            "cal_slope": np.nan,
            "sens_at_top20pct": np.nan,
            "prevalence": float(np.mean(y_true)),
            "n_samples": int(len(y_true)),
        }

    ece_val, _ = compute_ece(y_true, y_prob, n_bins=10)
    cal_int, cal_slope = compute_calibration_slope_intercept(y_true, y_prob)

    return {
        "auroc": round(float(roc_auc_score_np(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score_np(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss_np(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss_np(y_true, y_prob)), 4),
        "ece": round(float(ece_val), 4),
        "cal_intercept": round(float(cal_int), 4),
        "cal_slope": round(float(cal_slope), 4),
        "sens_at_top20pct": round(float(sensitivity_at_fixed_alert_rate(y_true, y_prob, alert_rate)), 4),
        "prevalence": round(float(np.mean(y_true)), 4),
        "n_samples": int(len(y_true)),
    }


def participant_bootstrap_ci(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_prob: np.ndarray,
    n_boot: int = 150,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% confidence intervals via participant-level cluster resampling."""
    rng = np.random.default_rng(seed)
    work = df_eval[["participant_id", y_true_col]].copy()
    work["y_prob"] = np.asarray(y_prob, dtype=float)

    pids = work["participant_id"].unique()
    pid_to_rows = {pid: group for pid, group in work.groupby("participant_id")}

    boot_records = []
    for _ in range(n_boot):
        sampled_pids = rng.choice(pids, size=len(pids), replace=True)
        sampled_df = pd.concat([pid_to_rows[p] for p in sampled_pids], ignore_index=True)
        yt = sampled_df[y_true_col].values
        yp = sampled_df["y_prob"].values
        if len(np.unique(yt)) < 2:
            continue
        ece_v, _ = compute_ece(yt, yp, n_bins=10)
        boot_records.append({
            "auroc": float(roc_auc_score_np(yt, yp)),
            "auprc": float(average_precision_score_np(yt, yp)),
            "brier": float(brier_score_loss_np(yt, yp)),
            "ece": float(ece_v),
            "sens_at_top20pct": float(sensitivity_at_fixed_alert_rate(yt, yp, 0.20)),
        })

    boot_df = pd.DataFrame(boot_records)
    point = compute_all_metrics(work[y_true_col].values, work["y_prob"].values)
    out: Dict[str, Dict[str, float]] = {}
    for metric in ["auroc", "auprc", "brier", "ece", "sens_at_top20pct"]:
        out[metric] = {
            "estimate": point[metric],
            "ci_lower_95": round(float(np.percentile(boot_df[metric], 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(boot_df[metric], 97.5)), 4),
        }
    return out


def evaluate_subgroups(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_prob: np.ndarray,
    subgroup_cols: List[str] = ["age_bracket", "female", "living_alone"],
    min_group_size: int = 25,
) -> pd.DataFrame:
    """Audit model discrimination and calibration across demographic & household subgroups."""
    work = df_eval.copy()
    work["_y_prob"] = np.asarray(y_prob, dtype=float)
    rows = []

    for col in subgroup_cols:
        if col not in work.columns:
            continue
        for val, grp in work.groupby(col):
            if len(grp) < min_group_size:
                continue
            m = compute_all_metrics(grp[y_true_col].values, grp["_y_prob"].values)
            label_val = val
            if col == "female":
                label_val = "Female" if int(val) == 1 else "Male"
            elif col == "living_alone":
                label_val = "Living Alone" if int(val) == 1 else "Co-residing"

            rows.append({
                "subgroup_variable": col,
                "subgroup_level": str(label_val),
                "n_samples": m["n_samples"],
                "prevalence": m["prevalence"],
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "brier": m["brier"],
                "ece": m["ece"],
                "sens_at_top20pct": m["sens_at_top20pct"],
            })

    return pd.DataFrame(rows)
