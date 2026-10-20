"""Cohort construction and longitudinal wave-pair outcome labeling without temporal leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from agewell.data import CANDIDATE_FEATURES


@dataclass(frozen=True)
class CohortFlowSummary:
    total_raw_records: int
    total_raw_participants: int
    wave_pairs_constructed: int
    excluded_under_age: int
    excluded_baseline_adl_impaired: int
    excluded_missing_followup: int
    final_eligible_observations: int
    final_unique_participants: int
    positive_outcome_count: int
    positive_outcome_prevalence: float


def build_wave_pair_cohort(
    df: pd.DataFrame,
    min_age: int = 65,
    exclude_baseline_adl_impaired: bool = True,
) -> Tuple[pd.DataFrame, CohortFlowSummary]:
    """Construct longitudinal wave-pair records (wave t predictors -> wave t+1 ADL decline outcome)."""
    df_sorted = df.sort_values(["participant_id", "wave"]).reset_index(drop=True)
    total_raw_records = len(df_sorted)
    total_raw_participants = int(df_sorted["participant_id"].nunique())

    next_wave_df = df_sorted[["participant_id", "wave", "baseline_adl_count"]].copy()
    next_wave_df["wave"] = next_wave_df["wave"] - 1
    next_wave_df = next_wave_df.rename(columns={"baseline_adl_count": "followup_adl_count"})

    max_wave = int(df_sorted["wave"].max())
    index_candidates = df_sorted[df_sorted["wave"] < max_wave].copy()
    merged = pd.merge(
        index_candidates,
        next_wave_df,
        on=["participant_id", "wave"],
        how="left",
    )

    wave_pairs_constructed = len(merged)
    missing_followup_mask = merged["followup_adl_count"].isna()
    excluded_missing_followup = int(missing_followup_mask.sum())
    merged = merged[~missing_followup_mask].copy()

    under_age_mask = merged["age"] < min_age
    excluded_under_age = int(under_age_mask.sum())
    merged = merged[~under_age_mask].copy()

    if exclude_baseline_adl_impaired:
        impaired_mask = merged["baseline_adl_count"] > 0
        excluded_baseline_adl = int(impaired_mask.sum())
        merged = merged[~impaired_mask].copy()
    else:
        excluded_baseline_adl = 0

    merged["followup_adl_count"] = merged["followup_adl_count"].astype(int)
    merged["future_adl_decline"] = (
        merged["followup_adl_count"] > merged["baseline_adl_count"]
    ).astype(int)

    merged["age_bracket"] = pd.cut(
        merged["age"],
        bins=[64, 74, 84, 120],
        labels=["65-74", "75-84", "85+"],
    ).astype(str)

    keep_cols = list(dict.fromkeys(
        ["participant_id", "wave", "age_bracket", "baseline_adl_count"]
        + CANDIDATE_FEATURES
        + ["future_adl_decline"]
    ))
    cohort_df = merged[keep_cols].reset_index(drop=True)

    pos_count = int(cohort_df["future_adl_decline"].sum())
    prevalence = float(pos_count / max(len(cohort_df), 1))

    summary = CohortFlowSummary(
        total_raw_records=total_raw_records,
        total_raw_participants=total_raw_participants,
        wave_pairs_constructed=wave_pairs_constructed,
        excluded_under_age=excluded_under_age,
        excluded_baseline_adl_impaired=excluded_baseline_adl,
        excluded_missing_followup=excluded_missing_followup,
        final_eligible_observations=len(cohort_df),
        final_unique_participants=int(cohort_df["participant_id"].nunique()),
        positive_outcome_count=pos_count,
        positive_outcome_prevalence=round(prevalence, 4),
    )
    return cohort_df, summary
