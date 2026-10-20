"""AgeWell-AI Read-Only Research Inspection Dashboard (Streamlit).

Reads saved experiment artifacts from `experiments/runs/` and `reports/figures/`.
Never silently retrains models or alters split configurations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
FIGS_DIR = REPO_ROOT / "reports" / "figures"

st.set_page_config(page_title="AgeWell-AI Research Inspector", layout="wide")

st.title("AgeWell-AI: Missingness-Aware & Uncertainty-Calibrated ML Inspector")
st.caption(
    "Read-only scientific dashboard inspecting saved, fingerprinted longitudinal experiments "
    "(Phase 1 Synthetic HRS/JAGES-Harmonized Benchmark)."
)

summary_path = RUNS_DIR / "experiment_results_summary.json"
if not summary_path.exists():
    st.error("No saved experiment artifacts found. Run `python -m agewell.experiments` first.")
    st.stop()

with open(summary_path) as f:
    meta = json.load(f)

st.info(f"**Scientific Governance Notice:** {meta['synthetic_disclaimer']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset SHA-256", meta["schema_validation"]["dataset_sha256"])
col2.metric("Eligible Wave-Pairs", meta["cohort_flow"]["final_eligible_observations"])
col3.metric("Unique Older Adults", meta["cohort_flow"]["final_unique_participants"])
col4.metric("Decline Prevalence", f"{meta['cohort_flow']['positive_outcome_prevalence']*100:.1f}%")

tabs = st.tabs([
    "H1: Missingness Stress Test",
    "H2: Probability Calibration",
    "H3: Temporal Leakage Audit",
    "H4: Selective Abstention",
    "Subgroup Fairness & Explainability",
])

with tabs[0]:
    st.subheader("Hypothesis H1: Missingness-Aware Learning Under MCAR, MAR, and MNAR")
    st.image(str(FIGS_DIR / "fig1_missingness_profile_and_h1.png"))
    st.dataframe(pd.read_csv(RUNS_DIR / "h1_missingness_ablation.csv"), use_container_width=True)

with tabs[1]:
    st.subheader("Hypothesis H2: Post-Hoc Calibration on Out-of-Time Validation Folds")
    st.image(str(FIGS_DIR / "fig2_calibration_reliability_h2.png"))
    st.dataframe(pd.read_csv(RUNS_DIR / "h2_calibration_metrics.csv"), use_container_width=True)

with tabs[2]:
    st.subheader("Hypothesis H3: Why Naive Row-Level Splits Leak Participant Identity")
    st.image(str(FIGS_DIR / "fig3_leakage_split_comparison_h3.png"))
    st.dataframe(pd.read_csv(RUNS_DIR / "h3_split_leakage_audit.csv"), use_container_width=True)

with tabs[3]:
    st.subheader("Hypothesis H4: Uncertainty-Based Abstention & Coverage Trade-offs")
    st.image(str(FIGS_DIR / "fig4_selective_abstention_h4.png"))
    st.dataframe(pd.read_csv(RUNS_DIR / "h4_selective_abstention.csv"), use_container_width=True)

with tabs[4]:
    st.subheader("Subgroup Fairness, Feature Attribution, and Error Taxonomy")
    st.image(str(FIGS_DIR / "fig5_feature_importance_and_subgroups.png"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Subgroup Disaggregated Performance**")
        st.dataframe(pd.read_csv(RUNS_DIR / "subgroup_fairness_audit.csv"), use_container_width=True)
    with c2:
        st.markdown("**Prediction Error Taxonomy**")
        st.dataframe(pd.read_csv(RUNS_DIR / "error_taxonomy.csv"), use_container_width=True)
