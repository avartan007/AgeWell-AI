"""Probability calibration (Platt/Sigmoid & Isotonic) and uncertainty-based selective abstention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from agewell.models import (
    IsotonicRegressionNP,
    LogisticRegressionNP,
    brier_score_loss_np,
    roc_auc_score_np,
)


@dataclass(frozen=True)
class CalibrationDiagnostics:
    brier_score: float
    log_loss_val: float
    ece: float
    calibration_intercept: float
    calibration_slope: float


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """Compute Expected Calibration Error (ECE) and bin-level reliability table."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    rows = []
    n_total = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(y_prob[mask].mean())
            obs_freq = float(y_true[mask].mean())
            gap = abs(obs_freq - mean_pred)
            ece += (count / n_total) * gap
            rows.append({
                "bin_index": b,
                "bin_lower": round(float(bins[b]), 2),
                "bin_upper": round(float(bins[b + 1]), 2),
                "count": count,
                "mean_predicted_prob": round(mean_pred, 4),
                "observed_frequency": round(obs_freq, 4),
                "abs_calibration_gap": round(gap, 4),
            })
        else:
            rows.append({
                "bin_index": b,
                "bin_lower": round(float(bins[b]), 2),
                "bin_upper": round(float(bins[b + 1]), 2),
                "count": 0,
                "mean_predicted_prob": np.nan,
                "observed_frequency": np.nan,
                "abs_calibration_gap": np.nan,
            })
    return float(ece), pd.DataFrame(rows)


def compute_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Compute Cox logistic calibration-in-the-large (intercept) and calibration slope."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-5, 1.0 - 1e-5)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)

    if len(np.unique(y_true)) < 2:
        return 0.0, 1.0

    clf = LogisticRegressionNP(C=1e5, max_iter=100, class_weight=None)
    clf.fit(logits, y_true)
    slope = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    return intercept, slope


class PostHocCalibrator:
    """Fit post-hoc probability calibration (sigmoid/Platt or isotonic) strictly on held-out validation data."""

    def __init__(self, method: str = "sigmoid") -> None:
        self.method = method.lower().strip()
        self.platt_lr_: Optional[LogisticRegressionNP] = None
        self.isotonic_: Optional[IsotonicRegressionNP] = None
        self.is_fitted_: bool = False

    def fit(self, y_prob_val: np.ndarray, y_val: np.ndarray) -> "PostHocCalibrator":
        p = np.clip(np.asarray(y_prob_val, dtype=float), 1e-5, 1.0 - 1e-5)
        y = np.asarray(y_val, dtype=int)

        if self.method in ("uncalibrated", "none"):
            self.is_fitted_ = True
            return self

        if self.method in ("sigmoid", "platt"):
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            self.platt_lr_ = LogisticRegressionNP(C=1e5, max_iter=100, class_weight=None)
            self.platt_lr_.fit(logits, y)
            self.is_fitted_ = True
            return self

        if self.method == "isotonic":
            self.isotonic_ = IsotonicRegressionNP(y_min=1e-4, y_max=1.0 - 1e-4)
            self.isotonic_.fit(p, y)
            self.is_fitted_ = True
            return self

        raise ValueError(f"Unsupported calibration method: {self.method}")

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("PostHocCalibrator must be fitted on validation predictions before transform().")
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-5, 1.0 - 1e-5)

        if self.method in ("uncalibrated", "none"):
            return p
        if self.method in ("sigmoid", "platt") and self.platt_lr_ is not None:
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            return np.clip(self.platt_lr_.predict_proba(logits)[:, 1], 1e-5, 1.0 - 1e-5)
        if self.method == "isotonic" and self.isotonic_ is not None:
            return np.clip(self.isotonic_.transform(p), 1e-5, 1.0 - 1e-5)
        return p


def compute_predictive_uncertainty(
    y_prob: np.ndarray,
    missing_fraction: Optional[np.ndarray] = None,
    alpha_missing: float = 0.25,
) -> np.ndarray:
    """Compute composite uncertainty score combining normalized Shannon entropy and record incompleteness."""
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    shannon_entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    if missing_fraction is not None:
        miss_f = np.asarray(missing_fraction, dtype=float)
        return (1.0 - alpha_missing) * shannon_entropy + alpha_missing * miss_f
    return shannon_entropy


def evaluate_selective_abstention(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_scores: np.ndarray,
    coverage_targets: List[float] = [1.0, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60],
    decision_threshold: float = 0.28,
) -> pd.DataFrame:
    """Evaluate selective prediction across coverage fractions (Hypothesis H4)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    u = np.asarray(uncertainty_scores, dtype=float)

    rows = []
    order = np.argsort(u)
    n_total = len(y_true)

    for cov in coverage_targets:
        n_keep = max(int(np.round(cov * n_total)), 10)
        keep_idx = order[:n_keep]
        abst_rate = 1.0 - (n_keep / n_total)

        yt = y_true[keep_idx]
        yp = y_prob[keep_idx]
        yhat = (yp >= decision_threshold).astype(int)

        auroc = float(roc_auc_score_np(yt, yp)) if len(np.unique(yt)) > 1 else np.nan
        brier = float(brier_score_loss_np(yt, yp))
        error_rate = float(np.mean(yhat != yt))
        fn_rate = float(np.sum((yhat == 0) & (yt == 1)) / max(np.sum(yt == 1), 1))
        sens = float(np.sum((yhat == 1) & (yt == 1)) / max(np.sum(yt == 1), 1))

        rows.append({
            "target_coverage": round(float(cov), 2),
            "actual_coverage": round(float(n_keep / n_total), 4),
            "abstention_rate": round(float(abst_rate), 4),
            "retained_n": int(n_keep),
            "retained_auroc": round(auroc, 4),
            "retained_brier": round(brier, 4),
            "retained_error_rate": round(error_rate, 4),
            "retained_sensitivity": round(sens, 4),
            "retained_false_negative_rate": round(fn_rate, 4),
        })

    return pd.DataFrame(rows)
