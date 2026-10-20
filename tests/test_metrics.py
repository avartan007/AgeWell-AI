"""Unit tests for evaluation metrics, calibration diagnostics, and selective abstention."""
import numpy as np
from agewell.calibration import (
    PostHocCalibrator,
    compute_ece,
    compute_predictive_uncertainty,
    evaluate_selective_abstention,
)
from agewell.evaluation import compute_all_metrics, sensitivity_at_fixed_alert_rate


def test_metrics_bounds_and_perfect_ranking():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.75, 0.80, 0.88, 0.95])
    metrics = compute_all_metrics(y_true, y_prob, alert_rate=0.50)

    assert np.isclose(metrics["auroc"], 1.0)
    assert np.isclose(metrics["auprc"], 1.0)
    assert 0.0 <= metrics["brier"] <= 0.25
    assert 0.0 <= metrics["ece"] <= 0.25
    assert np.isclose(sensitivity_at_fixed_alert_rate(y_true, y_prob, alert_rate=0.50), 1.0)


def test_selective_abstention_monotonic_coverage_and_explicit_abstention_rate():
    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(y_true * 0.65 + (1 - y_true) * 0.25 + rng.normal(0, 0.18, size=200), 0.01, 0.99)
    u = compute_predictive_uncertainty(y_prob)
    df_abs = evaluate_selective_abstention(y_true, y_prob, u, coverage_targets=[1.0, 0.80, 0.60])

    assert len(df_abs) == 3
    assert np.allclose(df_abs["actual_coverage"] + df_abs["abstention_rate"], 1.0)
    assert df_abs.iloc[0]["retained_n"] == 200
    assert df_abs.iloc[-1]["retained_n"] == 120
