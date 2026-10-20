"""Unit tests for cohort construction, eligibility criteria, and outcome labeling."""
import pandas as pd
from agewell.cohort import build_wave_pair_cohort
from agewell.data import generate_synthetic_longitudinal_cohort, validate_schema


def test_schema_validation_passes_on_valid_synthetic_cohort():
    df = generate_synthetic_longitudinal_cohort(n_participants=120, waves=(1, 2, 3), seed=11)
    report = validate_schema(df)
    assert report.is_valid is True
    assert len(report.missing_columns) == 0
    assert report.n_participants == 120


def test_cohort_eligibility_excludes_under_65_and_baseline_adl_impaired():
    df = generate_synthetic_longitudinal_cohort(n_participants=200, waves=(1, 2, 3), seed=22)
    cohort_df, summary = build_wave_pair_cohort(df, min_age=65, exclude_baseline_adl_impaired=True)

    assert (cohort_df["age"] >= 65).all(), "Participants under 65 must be excluded"
    assert (cohort_df["baseline_adl_count"] == 0).all(), "Baseline ADL-impaired participants must be excluded"
    assert "followup_adl_count" not in cohort_df.columns, "Follow-up ADL count must not remain in predictor columns"
    assert set(cohort_df["future_adl_decline"].unique()).issubset({0, 1})
    assert summary.final_eligible_observations == len(cohort_df)
