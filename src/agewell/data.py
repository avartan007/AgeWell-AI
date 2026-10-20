"""Data ingestion, schema validation, missingness profiling, and synthetic longitudinal cohort generation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_SCHEMA_COLUMNS: List[str] = [
    "participant_id",
    "wave",
    "age",
    "female",
    "living_alone",
    "education_years",
    "baseline_adl_count",
    "baseline_iadl_count",
    "self_rated_health",
    "chronic_conditions_count",
    "prior_falls_count",
    "bmi",
    "gait_speed_mps",
    "grip_strength_kg",
    "delayed_recall_score",
    "cesd_depression_score",
    "household_income_log",
    "social_participation_score",
]

CANDIDATE_FEATURES: List[str] = [
    "age",
    "female",
    "living_alone",
    "education_years",
    "baseline_iadl_count",
    "self_rated_health",
    "chronic_conditions_count",
    "prior_falls_count",
    "bmi",
    "gait_speed_mps",
    "grip_strength_kg",
    "delayed_recall_score",
    "cesd_depression_score",
    "household_income_log",
    "social_participation_score",
]


@dataclass(frozen=True)
class SchemaValidationReport:
    is_valid: bool
    missing_columns: List[str]
    invalid_types: List[str]
    n_rows: int
    n_participants: int
    waves_present: List[int]
    dataset_sha256: str


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint of a DataFrame."""
    hashed = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def validate_schema(df: pd.DataFrame) -> SchemaValidationReport:
    """Validate longitudinal survey DataFrame against required AgeWell-AI schema."""
    missing_cols = [c for c in REQUIRED_SCHEMA_COLUMNS if c not in df.columns]
    invalid_types: List[str] = []

    if not missing_cols:
        if not pd.api.types.is_integer_dtype(df["wave"]) and not pd.api.types.is_numeric_dtype(df["wave"]):
            invalid_types.append("wave must be numeric/integer")
        if (df["age"].dropna() < 0).any() or (df["age"].dropna() > 120).any():
            invalid_types.append("age contains out-of-range values (<0 or >120)")
        if (df["baseline_adl_count"].dropna() < 0).any():
            invalid_types.append("baseline_adl_count cannot be negative")

    is_valid = len(missing_cols) == 0 and len(invalid_types) == 0
    waves = sorted(df["wave"].dropna().unique().astype(int).tolist()) if "wave" in df.columns else []
    n_parts = int(df["participant_id"].nunique()) if "participant_id" in df.columns else 0

    return SchemaValidationReport(
        is_valid=is_valid,
        missing_columns=missing_cols,
        invalid_types=invalid_types,
        n_rows=len(df),
        n_participants=n_parts,
        waves_present=waves,
        dataset_sha256=compute_dataframe_fingerprint(df) if len(df) > 0 else "empty",
    )


def summarize_missingness(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Summarize feature-level missingness rates and classify likely survey mechanisms."""
    cols = features if features is not None else CANDIDATE_FEATURES
    records = []
    for col in cols:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        rate = float(n_miss / max(len(df), 1))
        if col in ("gait_speed_mps", "grip_strength_kg"):
            mechanism_note = "Informative MNAR/MAR: Physical performance test often skipped when frail or unsafe"
        elif col in ("delayed_recall_score", "cesd_depression_score"):
            mechanism_note = "Informative MNAR/MAR: Cognitive/affective non-response or proxy interview skip"
        elif col in ("household_income_log", "social_participation_score"):
            mechanism_note = "Survey MAR/MNAR: Socioeconomic or social module item non-response"
        else:
            mechanism_note = "Low/sporadic item non-response (MCAR/MAR)"
        records.append({
            "feature": col,
            "missing_count": n_miss,
            "missing_rate": round(rate, 4),
            "mechanism_hypothesis": mechanism_note,
        })
    return pd.DataFrame(records).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def generate_synthetic_longitudinal_cohort(
    n_participants: int = 2000,
    waves: Tuple[int, ...] = (1, 2, 3, 4),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic longitudinal aging survey dataset modeled on harmonized HRS/JAGES structure."""
    rng = np.random.default_rng(seed)
    rows = []

    for pid in range(1, n_participants + 1):
        base_age = int(rng.integers(63, 89))
        female = int(rng.random() < 0.54)
        edu_years = int(np.clip(rng.normal(12.2, 3.1), 4, 20))
        latent_frailty = float(rng.normal(0.0, 1.0)) + 0.045 * (base_age - 70)
        income_base = float(np.clip(rng.normal(10.4 + 0.08 * edu_years - 0.12 * female, 0.65), 8.2, 12.8))

        current_adl = 1 if (latent_frailty + rng.normal(0, 0.7) > 1.65) else 0

        for w_idx, w in enumerate(waves):
            age_w = base_age + 2 * w_idx
            latent_frailty = latent_frailty + 0.18 + rng.normal(0.0, 0.32) + 0.012 * max(age_w - 72, 0)

            living_alone = int(rng.random() < (0.22 + 0.01 * max(age_w - 70, 0) + 0.06 * female))
            chronic_count = int(np.clip(rng.poisson(max(1.4 + 0.45 * latent_frailty, 0.2)), 0, 8))
            prior_falls = int(np.clip(rng.poisson(max(0.28 + 0.35 * max(latent_frailty, 0), 0.05)), 0, 6))
            iadl_count = int(np.clip(rng.poisson(max(0.22 + 0.42 * max(latent_frailty, 0), 0.02)), 0, 5))
            srh = int(np.clip(np.round(2.7 + 0.75 * latent_frailty + rng.normal(0, 0.55)), 1, 5))
            bmi = float(np.clip(rng.normal(23.8 - 0.35 * latent_frailty, 3.4), 15.0, 39.0))

            true_gait = float(np.clip(1.05 - 0.16 * latent_frailty - 0.008 * (age_w - 70) + rng.normal(0, 0.12), 0.25, 1.65))
            true_grip = float(np.clip((26.0 if female else 38.0) - 3.2 * latent_frailty - 0.28 * (age_w - 70) + rng.normal(0, 3.8), 8.0, 58.0))
            true_recall = float(np.clip(5.8 - 0.95 * latent_frailty - 0.06 * (age_w - 70) + 0.12 * (edu_years - 12) + rng.normal(0, 1.3), 0.0, 10.0))
            true_cesd = float(np.clip(1.4 + 0.85 * max(latent_frailty, -0.5) + 0.4 * living_alone + rng.normal(0, 1.1), 0.0, 8.0))
            true_social = float(np.clip(3.2 - 0.55 * latent_frailty - 0.3 * living_alone + rng.normal(0, 0.9), 0.0, 6.0))
            true_income = float(np.clip(income_base - 0.03 * w_idx + rng.normal(0, 0.12), 8.0, 13.0))

            prob_miss_gait = 1.0 / (1.0 + np.exp(-(-2.0 + 1.15 * latent_frailty + 0.45 * prior_falls - 1.2 * (true_gait - 0.8))))
            prob_miss_grip = 1.0 / (1.0 + np.exp(-(-2.2 + 0.95 * latent_frailty + 0.35 * (srh - 3))))
            prob_miss_recall = 1.0 / (1.0 + np.exp(-(-2.3 + 1.05 * latent_frailty - 0.35 * (true_recall - 4.0))))
            prob_miss_cesd = 1.0 / (1.0 + np.exp(-(-2.4 + 0.85 * latent_frailty + 0.25 * (true_cesd - 3.0))))
            prob_miss_social = 1.0 / (1.0 + np.exp(-(-2.5 + 0.55 * latent_frailty + 0.30 * living_alone)))
            prob_miss_income = 0.08 + 0.03 * int(true_income > 11.2)
            prob_miss_bmi = 0.04 + 0.04 * int(latent_frailty > 1.2)

            miss_gait = rng.random() < prob_miss_gait
            miss_grip = rng.random() < prob_miss_grip
            miss_recall = rng.random() < prob_miss_recall
            miss_cesd = rng.random() < prob_miss_cesd
            miss_social = rng.random() < prob_miss_social
            miss_income = rng.random() < prob_miss_income
            miss_bmi = rng.random() < prob_miss_bmi

            if w_idx > 0:
                unmeasured_refusal_risk = 0.65 * int(miss_gait) + 0.55 * int(miss_recall)
                logit_adl = (
                    -2.55
                    + 1.05 * latent_frailty
                    + 0.28 * prior_falls
                    + 0.25 * iadl_count
                    + unmeasured_refusal_risk
                    + 0.90 * int(current_adl > 0)
                )
                prob_adl = 1.0 / (1.0 + np.exp(-logit_adl))
                if rng.random() < prob_adl:
                    current_adl = max(current_adl, 1) + int(rng.random() < 0.35 * prob_adl)
                else:
                    current_adl = max(0, current_adl - int(rng.random() < 0.15))

            rows.append({
                "participant_id": f"P{pid:05d}",
                "wave": int(w),
                "age": int(age_w),
                "female": int(female),
                "living_alone": int(living_alone),
                "education_years": int(edu_years),
                "baseline_adl_count": int(current_adl),
                "baseline_iadl_count": int(iadl_count),
                "self_rated_health": int(srh),
                "chronic_conditions_count": int(chronic_count),
                "prior_falls_count": int(prior_falls),
                "bmi": np.nan if miss_bmi else round(bmi, 2),
                "gait_speed_mps": np.nan if miss_gait else round(true_gait, 3),
                "grip_strength_kg": np.nan if miss_grip else round(true_grip, 2),
                "delayed_recall_score": np.nan if miss_recall else round(true_recall, 1),
                "cesd_depression_score": np.nan if miss_cesd else round(true_cesd, 1),
                "household_income_log": np.nan if miss_income else round(true_income, 3),
                "social_participation_score": np.nan if miss_social else round(true_social, 2),
            })

    return pd.DataFrame(rows)
