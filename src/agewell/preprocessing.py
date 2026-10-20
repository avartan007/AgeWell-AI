"""Leakage-safe splitting, missingness-indicator feature engineering, and controlled missingness simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from agewell.models import SimpleImputerNP, StandardScalerNP
from agewell.data import CANDIDATE_FEATURES


@dataclass(frozen=True)
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_type: str


def verify_no_participant_overlap(*dfs: pd.DataFrame) -> bool:
    """Assert that participant_id sets across provided DataFrames are mutually disjoint."""
    id_sets = [set(d["participant_id"].unique()) for d in dfs if len(d) > 0]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            if len(id_sets[i].intersection(id_sets[j])) > 0:
                return False
    return True


def create_participant_grouped_temporal_split(
    cohort_df: pd.DataFrame,
    train_waves: List[int] = [1, 2],
    test_wave: int = 3,
    val_participant_fraction: float = 0.25,
    seed: int = 42,
) -> SplitBundle:
    """Split participants into disjoint groups first, and evaluate strictly on later wave `test_wave`."""
    unique_pids = cohort_df["participant_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_pids = rng.permutation(unique_pids)

    n_total = len(shuffled_pids)
    n_test = int(0.30 * n_total)
    n_val = int(val_participant_fraction * (n_total - n_test))

    test_pids = set(shuffled_pids[:n_test])
    val_pids = set(shuffled_pids[n_test : n_test + n_val])
    train_pids = set(shuffled_pids[n_test + n_val :])

    train_df = cohort_df[
        (cohort_df["participant_id"].isin(train_pids)) & (cohort_df["wave"].isin(train_waves))
    ].reset_index(drop=True)
    val_df = cohort_df[
        (cohort_df["participant_id"].isin(val_pids)) & (cohort_df["wave"].isin(train_waves))
    ].reset_index(drop=True)
    test_df = cohort_df[
        (cohort_df["participant_id"].isin(test_pids)) & (cohort_df["wave"] == test_wave)
    ].reset_index(drop=True)

    assert verify_no_participant_overlap(train_df, val_df, test_df), "Participant overlap detected!"
    assert int(test_df["wave"].min()) >= int(train_df["wave"].max()), "Temporal ordering violated!"

    return SplitBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        split_type="participant_grouped_temporal",
    )


def create_participant_grouped_random_split(
    cohort_df: pd.DataFrame,
    seed: int = 42,
) -> SplitBundle:
    """Participant-grouped split across all waves (no participant overlap, same calendar waves)."""
    unique_pids = cohort_df["participant_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_pids = rng.permutation(unique_pids)

    n_total = len(shuffled_pids)
    n_test = int(0.25 * n_total)
    n_val = int(0.25 * (n_total - n_test))

    test_pids = set(shuffled_pids[:n_test])
    val_pids = set(shuffled_pids[n_test : n_test + n_val])
    train_pids = set(shuffled_pids[n_test + n_val :])

    train_df = cohort_df[cohort_df["participant_id"].isin(train_pids)].reset_index(drop=True)
    val_df = cohort_df[cohort_df["participant_id"].isin(val_pids)].reset_index(drop=True)
    test_df = cohort_df[cohort_df["participant_id"].isin(test_pids)].reset_index(drop=True)

    assert verify_no_participant_overlap(train_df, val_df, test_df)
    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df, split_type="participant_grouped_random")


def create_naive_row_random_split_for_audit(
    cohort_df: pd.DataFrame,
    seed: int = 42,
) -> SplitBundle:
    """Intentionally naive row-level random split used ONLY in Hypothesis H3 audit to demonstrate
    how ignoring longitudinal participant clustering and wave structure inflates apparent performance.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cohort_df))
    n_test = int(0.25 * len(cohort_df))
    n_val = int(0.20 * (len(cohort_df) - n_test))

    test_df = cohort_df.iloc[perm[:n_test]].reset_index(drop=True)
    val_df = cohort_df.iloc[perm[n_test : n_test + n_val]].reset_index(drop=True)
    train_df = cohort_df.iloc[perm[n_test + n_val :]].reset_index(drop=True)
    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df, split_type="naive_row_random_leaky")


class MissingnessAwarePreprocessor:
    """Leakage-safe preprocessor that fits imputation and scaling strictly on training data,
    and optionally appends binary missingness indicators M_ij = I(X_ij is NaN).
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        add_missing_indicators: bool = False,
        scale_numeric: bool = True,
        imputation_strategy: str = "median",
        min_missing_rate_for_indicator: float = 0.01,
    ) -> None:
        self.features = features if features is not None else list(CANDIDATE_FEATURES)
        self.add_missing_indicators = add_missing_indicators
        self.scale_numeric = scale_numeric
        self.imputation_strategy = imputation_strategy
        self.min_missing_rate_for_indicator = min_missing_rate_for_indicator

        self.imputer_: Optional[SimpleImputerNP] = None
        self.scaler_: Optional[StandardScalerNP] = None
        self.indicator_features_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, df_train: pd.DataFrame) -> "MissingnessAwarePreprocessor":
        X_raw = df_train[self.features].copy()

        if self.add_missing_indicators:
            miss_rates = X_raw.isna().mean(axis=0)
            self.indicator_features_ = [
                col for col in self.features if miss_rates[col] >= self.min_missing_rate_for_indicator
            ]
        else:
            self.indicator_features_ = []

        self.imputer_ = SimpleImputerNP(strategy=self.imputation_strategy)
        X_imp = self.imputer_.fit_transform(X_raw)

        if self.scale_numeric:
            self.scaler_ = StandardScalerNP()
            self.scaler_.fit(X_imp)

        self.feature_names_out_ = list(self.features) + [
            f"{col}_is_missing" for col in self.indicator_features_
        ]
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_ or self.imputer_ is None:
            raise RuntimeError("MissingnessAwarePreprocessor must be fitted on training data before transform().")

        X_raw = df[self.features].copy()
        X_imp = self.imputer_.transform(X_raw)
        if self.scale_numeric and self.scaler_ is not None:
            X_imp = self.scaler_.transform(X_imp)

        parts = [pd.DataFrame(X_imp, columns=self.features, index=df.index)]
        if self.add_missing_indicators and self.indicator_features_:
            ind_df = pd.DataFrame(
                {f"{col}_is_missing": X_raw[col].isna().astype(float) for col in self.indicator_features_},
                index=df.index,
            )
            parts.append(ind_df)

        out = pd.concat(parts, axis=1)
        return out[self.feature_names_out_]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)


def inject_controlled_missingness(
    df: pd.DataFrame,
    mechanism: str = "MCAR",
    extra_rate: float = 0.20,
    target_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject synthetic additional missingness (MCAR, MAR, or MNAR) into selected features."""
    if extra_rate <= 0.0:
        return df.copy()

    cols = target_cols if target_cols is not None else [
        "gait_speed_mps",
        "grip_strength_kg",
        "delayed_recall_score",
        "cesd_depression_score",
        "social_participation_score",
    ]
    out = df.copy()
    rng = np.random.default_rng(seed)

    for col in cols:
        if col not in out.columns:
            continue
        observed_mask = ~out[col].isna()
        if int(observed_mask.sum()) == 0:
            continue

        if mechanism.upper() == "MCAR":
            probs = np.full(len(out), extra_rate)
        elif mechanism.upper() == "MAR":
            z = 0.08 * (out["age"].values - 72.0) + 0.55 * (out["self_rated_health"].values - 3.0)
            raw_p = 1.0 / (1.0 + np.exp(-z))
            probs = np.clip(raw_p * (extra_rate / max(raw_p.mean(), 1e-6)), 0.0, 0.92)
        elif mechanism.upper() == "MNAR":
            col_vals = out[col].fillna(out[col].median()).values
            std_val = (col_vals - np.nanmean(col_vals)) / (np.nanstd(col_vals) + 1e-6)
            direction = 1.0 if col == "cesd_depression_score" else -1.0
            z = 1.15 * direction * std_val + 0.35 * (out["prior_falls_count"].values)
            raw_p = 1.0 / (1.0 + np.exp(-z))
            probs = np.clip(raw_p * (extra_rate / max(raw_p.mean(), 1e-6)), 0.0, 0.95)
        else:
            raise ValueError(f"Unknown missingness mechanism: {mechanism}")

        drop_draw = (rng.random(len(out)) < probs) & observed_mask.values
        out.loc[drop_draw, col] = np.nan

    return out
