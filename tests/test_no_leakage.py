"""Critical unit tests verifying zero participant overlap and zero temporal leakage."""
import pandas as pd
from agewell.cohort import build_wave_pair_cohort
from agewell.data import generate_synthetic_longitudinal_cohort
from agewell.preprocessing import (
    create_participant_grouped_random_split,
    create_participant_grouped_temporal_split,
    verify_no_participant_overlap,
)


def test_participant_grouped_splits_have_zero_participant_overlap():
    raw_df = generate_synthetic_longitudinal_cohort(n_participants=250, waves=(1, 2, 3, 4), seed=77)
    cohort_df, _ = build_wave_pair_cohort(raw_df)

    temp_split = create_participant_grouped_temporal_split(
        cohort_df, train_waves=[1, 2], test_wave=3, seed=77
    )
    assert verify_no_participant_overlap(temp_split.train_df, temp_split.val_df, temp_split.test_df)

    rand_split = create_participant_grouped_random_split(cohort_df, seed=77)
    assert verify_no_participant_overlap(rand_split.train_df, rand_split.val_df, rand_split.test_df)


def test_temporal_split_strictly_respects_wave_chronology():
    raw_df = generate_synthetic_longitudinal_cohort(n_participants=250, waves=(1, 2, 3, 4), seed=88)
    cohort_df, _ = build_wave_pair_cohort(raw_df)

    temp_split = create_participant_grouped_temporal_split(
        cohort_df, train_waves=[1, 2], test_wave=3, seed=88
    )
    max_train_wave = int(temp_split.train_df["wave"].max())
    max_val_wave = int(temp_split.val_df["wave"].max())
    min_test_wave = int(temp_split.test_df["wave"].min())

    assert min_test_wave > max_train_wave, "Test wave must be strictly later than training waves"
    assert min_test_wave > max_val_wave, "Test wave must be strictly later than validation waves"


def test_future_wave_perturbations_do_not_change_past_wave_features():
    raw_df = generate_synthetic_longitudinal_cohort(n_participants=100, waves=(1, 2, 3), seed=55)
    cohort_orig, _ = build_wave_pair_cohort(raw_df)

    # Corrupt Wave 3 predictors heavily
    raw_modified = raw_df.copy()
    raw_modified.loc[raw_modified["wave"] == 3, "gait_speed_mps"] = 999.0
    raw_modified.loc[raw_modified["wave"] == 3, "self_rated_health"] = 5
    cohort_mod, _ = build_wave_pair_cohort(raw_modified)

    w1_orig = cohort_orig[cohort_orig["wave"] == 1].reset_index(drop=True)
    w1_mod = cohort_mod[cohort_mod["wave"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(w1_orig, w1_mod)
