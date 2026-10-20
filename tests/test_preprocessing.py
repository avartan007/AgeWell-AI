"""Unit tests for missingness-aware preprocessing and training-only parameter fitting."""
import numpy as np
import pandas as pd
from agewell.cohort import build_wave_pair_cohort
from agewell.data import CANDIDATE_FEATURES, generate_synthetic_longitudinal_cohort
from agewell.preprocessing import MissingnessAwarePreprocessor, create_participant_grouped_temporal_split


def test_missing_indicators_added_and_no_nans_remain():
    raw_df = generate_synthetic_longitudinal_cohort(n_participants=180, waves=(1, 2, 3), seed=33)
    cohort_df, _ = build_wave_pair_cohort(raw_df)
    split = create_participant_grouped_temporal_split(cohort_df, train_waves=[1], test_wave=2, seed=33)

    prep = MissingnessAwarePreprocessor(features=CANDIDATE_FEATURES, add_missing_indicators=True)
    X_train = prep.fit_transform(split.train_df)
    X_test = prep.transform(split.test_df)

    assert not X_train.isna().any().any(), "Imputed X_train must not contain NaNs"
    assert not X_test.isna().any().any(), "Imputed X_test must not contain NaNs"
    indicator_cols = [c for c in X_train.columns if c.endswith("_is_missing")]
    assert len(indicator_cols) > 0, "At least one missingness indicator column should be created"
    assert list(X_train.columns) == list(X_test.columns)


def test_imputer_statistics_come_strictly_from_train_set():
    train_df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, np.nan],
        "f2": [10.0, 20.0, 30.0, 40.0],
    })
    test_df = pd.DataFrame({
        "f1": [1000.0, 2000.0, np.nan],
        "f2": [500.0, 600.0, 700.0],
    })
    prep = MissingnessAwarePreprocessor(
        features=["f1", "f2"],
        add_missing_indicators=True,
        scale_numeric=False,
        imputation_strategy="median",
    )
    prep.fit(train_df)
    X_test_out = prep.transform(test_df)
    # The missing value in test_df['f1'] must be imputed with train median (2.0), untouched by 1000/2000
    assert np.isclose(X_test_out.loc[2, "f1"], 2.0)
    assert X_test_out.loc[2, "f1_is_missing"] == 1.0
