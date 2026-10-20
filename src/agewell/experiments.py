"""Controlled experiment runner for Hypotheses H1, H2, H3, H4, Subgroup Fairness, and Publication Figures."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy

from agewell.calibration import (
    PostHocCalibrator,
    compute_ece,
    compute_predictive_uncertainty,
    evaluate_selective_abstention,
)
from agewell.cohort import build_wave_pair_cohort
from agewell.data import (
    CANDIDATE_FEATURES,
    generate_synthetic_longitudinal_cohort,
    summarize_missingness,
    validate_schema,
)
from agewell.evaluation import (
    compute_all_metrics,
    evaluate_subgroups,
    participant_bootstrap_ci,
)
from agewell.explainability import compute_feature_explanations, summarize_error_Taxonomy
from agewell.models import build_model
from agewell.preprocessing import (
    MissingnessAwarePreprocessor,
    create_naive_row_random_split_for_audit,
    create_participant_grouped_random_split,
    create_participant_grouped_temporal_split,
    inject_controlled_missingness,
)

NAVY = "#1B365D"
TEAL = "#008080"
CORAL = "#D9534F"
AMBER = "#D9822B"
SLATE = "#5C6B73"


def run_single_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    add_missing_indicators: bool,
    calibration_method: str = "uncalibrated",
    seed: int = 42,
) -> Dict[str, Any]:
    prep = MissingnessAwarePreprocessor(
        features=CANDIDATE_FEATURES,
        add_missing_indicators=add_missing_indicators,
        scale_numeric=True,
    )
    X_train = prep.fit_transform(train_df)
    y_train = train_df["future_adl_decline"].values

    X_val = prep.transform(val_df)
    y_val = val_df["future_adl_decline"].values

    X_test = prep.transform(test_df)
    y_test = test_df["future_adl_decline"].values

    model = build_model(model_name, seed=seed)
    model.fit(X_train, y_train)

    val_raw_prob = model.predict_proba(X_val)[:, 1]
    test_raw_prob = model.predict_proba(X_test)[:, 1]

    calibrator = PostHocCalibrator(method=calibration_method)
    calibrator.fit(val_raw_prob, y_val)
    test_cal_prob = calibrator.transform(test_raw_prob)

    metrics = compute_all_metrics(y_test, test_cal_prob)
    return {
        "preprocessor": prep,
        "model": model,
        "calibrator": calibrator,
        "X_test": X_test,
        "y_test": y_test,
        "test_prob": test_cal_prob,
        "metrics": metrics,
    }


def generate_publication_figures(
    missing_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    rel_curves: Dict[str, pd.DataFrame],
    h3_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    figures_dir: Path,
) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "#CBD5E1",
    })
    saved_paths: List[Path] = []

    # FIGURE 1: Missingness Profile & H1 Stress Test
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=200)
    top_miss = missing_df.head(10).iloc[::-1]
    colors_m = [CORAL if r > 0.12 else (AMBER if r > 0.05 else SLATE) for r in top_miss["missing_rate"]]
    axes[0].barh(top_miss["feature"], top_miss["missing_rate"] * 100.0, color=colors_m, height=0.68)
    axes[0].set_xlabel("Baseline Survey Missingness Rate (%)")
    axes[0].set_title("A. Informative Missingness in Functional/Cognitive Modules")
    for idx, val in enumerate(top_miss["missing_rate"] * 100.0):
        axes[0].text(val + 0.4, idx, f"{val:.1f}%", va="center", fontsize=8.5, color="#1E293B")
    axes[0].set_xlim(0, max(top_miss["missing_rate"] * 100.0) * 1.18)

    mnar_sub = h1_df[(h1_df["mechanism"] == "MNAR") & (h1_df["model"] == "random_forest")]
    for strat, label, col, marker in [
        ("median_only", "RF Baseline (Imputation Only)", SLATE, "o"),
        ("median_plus_indicators", "RF + Missingness Indicators (Proposed H1)", TEAL, "s"),
    ]:
        sub = mnar_sub[mnar_sub["imputation_strategy"] == strat].sort_values("extra_missing_rate")
        axes[1].plot(
            sub["extra_missing_rate"] * 100.0,
            sub["auroc"],
            label=label,
            color=col,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )
    axes[1].set_xlabel("Additional Simulated MNAR Missingness Rate (%)")
    axes[1].set_ylabel("Held-Out Temporal Test AUROC")
    axes[1].set_title("B. H1 Stress Test: Robustness Under MNAR Missingness")
    axes[1].legend(frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout(pad=2.0)
    p1 = figures_dir / "fig1_missingness_profile_and_h1.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # FIGURE 2: Calibration Reliability Curves & Brier/ECE (H2)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=200)
    axes[0].plot([0, 1], [0, 1], "--", color="#64748B", label="Perfect Calibration (y = x)", linewidth=1.5)
    for key, label, col, marker in [
        ("uncalibrated", "Uncalibrated RF (Class-Balanced)", CORAL, "^"),
        ("sigmoid", "RF + Sigmoid (Platt) Calibration", NAVY, "o"),
        ("isotonic", "RF + Isotonic Calibration", TEAL, "s"),
    ]:
        rc = rel_curves[key].dropna(subset=["mean_predicted_prob", "observed_frequency"])
        axes[0].plot(
            rc["mean_predicted_prob"],
            rc["observed_frequency"],
            label=label,
            color=col,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
        )
    axes[0].set_xlabel("Mean Predicted Probability of Functional Decline")
    axes[0].set_ylabel("Observed 2-Year Functional Decline Frequency")
    axes[0].set_title("A. H2 Reliability Diagram on Out-of-Time Test Wave")
    axes[0].set_xlim(0, 0.95)
    axes[0].set_ylim(0, 0.95)
    axes[0].legend(frameon=True, facecolor="white", fontsize=8.5)

    rf_h2 = h2_df[h2_df["model"] == "random_forest_missing_aware"].copy()
    x = np.arange(len(rf_h2))
    w = 0.35
    axes[1].bar(x - w / 2, rf_h2["brier"], width=w, color=NAVY, label="Brier Score (Lower = Better)")
    axes[1].bar(x + w / 2, rf_h2["ece"], width=w, color=TEAL, label="Expected Calibration Error (ECE)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Uncalibrated", "Sigmoid (Platt)", "Isotonic"])
    axes[1].set_ylabel("Error Metric Value")
    axes[1].set_title("B. H2 Calibration Error Reduction (Validation-Tuned)")
    for i, row in enumerate(rf_h2.itertuples()):
        axes[1].text(i - w / 2, row.brier + 0.004, f"{row.brier:.3f}", ha="center", fontsize=8.5)
        axes[1].text(i + w / 2, row.ece + 0.004, f"{row.ece:.3f}", ha="center", fontsize=8.5)
    axes[1].set_ylim(0, max(rf_h2["brier"].max(), rf_h2["ece"].max()) * 1.25)
    axes[1].legend(frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout(pad=2.0)
    p2 = figures_dir / "fig2_calibration_reliability_h2.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # FIGURE 3: Temporal vs Leaky Split Comparison (H3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
    split_labels = [
        "Naive Row-Random\n(Participant Leakage)",
        "Participant-Grouped\nRandom Split",
        "Participant-Grouped\nTemporal Wave Split",
    ]
    x = np.arange(len(h3_df))
    w = 0.35
    axes[0].bar(x - w / 2, h3_df["auroc"], width=w, color=NAVY, label="AUROC")
    axes[0].bar(x + w / 2, h3_df["auprc"], width=w, color=TEAL, label="AUPRC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_labels, fontsize=9)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A. H3 Audit: How Naive Splits Inflate Apparent Discrimination")
    for i, r in enumerate(h3_df.itertuples()):
        axes[0].text(i - w / 2, r.auroc + 0.02, f"{r.auroc:.3f}", ha="center", fontsize=8.5)
        axes[0].text(i + w / 2, r.auprc + 0.02, f"{r.auprc:.3f}", ha="center", fontsize=8.5)
    axes[0].legend(frameon=True, facecolor="white", fontsize=9)

    axes[1].bar(x, h3_df["ece"], width=0.45, color=[CORAL, AMBER, TEAL])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_labels, fontsize=9)
    axes[1].set_ylabel("Uncalibrated ECE on Held-Out Split")
    axes[1].set_title("B. Temporal Distribution Shift Increases Calibration Drift")
    for i, r in enumerate(h3_df.itertuples()):
        axes[1].text(i, r.ece + 0.004, f"{r.ece:.3f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, h3_df["ece"].max() * 1.28)
    fig.tight_layout(pad=2.0)
    p3 = figures_dir / "fig3_leakage_split_comparison_h3.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # FIGURE 4: Uncertainty-Based Selective Abstention (H4)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
    cov_pct = h4_df["actual_coverage"] * 100.0
    axes[0].plot(cov_pct, h4_df["retained_error_rate"] * 100.0, "-o", color=NAVY, linewidth=2.2, label="Retained Classification Error (%)")
    axes[0].plot(cov_pct, h4_df["retained_false_negative_rate"] * 100.0, "-s", color=CORAL, linewidth=2.2, label="Retained False-Negative Rate (%)")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Retained Cohort Coverage (%) [Right = Stricter Uncertainty Abstention]")
    axes[0].set_ylabel("Error / Miss Rate Among Retained Cases (%)")
    axes[0].set_title("A. H4 Selective Prediction: Error Reduction vs. Coverage")
    axes[0].legend(frameon=True, facecolor="white", fontsize=9)

    abst_pct = h4_df["abstention_rate"] * 100.0
    axes[1].plot(abst_pct, h4_df["retained_auroc"], "-o", color=TEAL, linewidth=2.2, label="Retained AUROC")
    axes[1].plot(abst_pct, h4_df["retained_sensitivity"], "-^", color=AMBER, linewidth=2.2, label="Retained Sensitivity (@ p >= 0.28)")
    axes[1].set_xlabel("Fraction of Cases Flagged for Human Review / Abstention (%)")
    axes[1].set_ylabel("Metric on Retained Predictions")
    axes[1].set_title("B. H4 Explicit Reporting of Abstention Burden vs. Gain")
    axes[1].legend(frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout(pad=2.0)
    p4 = figures_dir / "fig4_selective_abstention_h4.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # FIGURE 5: Explainability & Subgroup Audit
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=200)
    top_f = feat_imp_df.head(12).iloc[::-1]
    f_cols = [TEAL if c == "Missingness Indicator" else NAVY for c in top_f["feature_category"]]
    axes[0].barh(top_f["feature"], top_f["importance_mean"], xerr=top_f["importance_std"], color=f_cols, height=0.68, capsize=2)
    axes[0].set_xlabel("Permutation Importance (Mean AUROC Drop on Held-Out Test Set)")
    axes[0].set_title("A. Feature Attribution: Clinical vs. Missingness Indicators")
    legend_elements = [
        Line2D([0], [0], color=NAVY, lw=6, label="Substantive Survey/Clinical Feature"),
        Line2D([0], [0], color=TEAL, lw=6, label="Informative Missingness Indicator (M_ij)"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    y_pos = np.arange(len(subgroup_df))
    labels_sg = [f"{r.subgroup_variable}: {r.subgroup_level}\n(n={r.n_samples})" for r in subgroup_df.itertuples()]
    w = 0.36
    axes[1].barh(y_pos - w / 2, subgroup_df["auroc"], height=w, color=NAVY, label="Subgroup AUROC")
    axes[1].barh(y_pos + w / 2, 1.0 - subgroup_df["brier"], height=w, color=TEAL, label="1 - Brier Score (Reliability)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels_sg, fontsize=8.5)
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel("Performance Metric")
    axes[1].set_title("B. Subgroup Fairness & Reliability Audit (Held-Out Wave)")
    axes[1].legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)
    fig.tight_layout(pad=2.0)
    p5 = figures_dir / "fig5_feature_importance_and_subgroups.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    # FIGURE 6: System Architecture & 2-Year MEXT Research Progression Diagram
    fig, ax = plt.subplots(figsize=(13.0, 5.2), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    ax.text(50, 39.5, "AgeWell-AI: Leakage-Safe Architecture & 2-Year MEXT Master's Research Roadmap",
            ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)

    boxes = [
        (3, 21, 17, 14, NAVY, "1. Longitudinal Ingestion\n& Schema Audit",
         "• HRS (Phase 1 Pilot)\n• JAGES (Phase 2 Japan)\n• Wave t -> t+1 Pairing\n• SHA-256 Fingerprinting"),
        (23, 21, 17, 14, "#2B4C7E", "2. Leakage-Safe\nCohort & Splits",
         "• Age >= 65, ADL(t)=0\n• Disjoint Participant IDs\n• Temporal Wave Split\n• Zero Future Contamination"),
        (43, 21, 17, 14, TEAL, "3. Missingness-Aware\nFeature Engineering",
         "• Train-Only Imputation\n• Informative Indicators M_ij\n• MCAR / MAR / MNAR\n  Stress-Test Suite (H1)"),
        (63, 21, 17, 14, "#0F766E", "4. Calibrated ML &\nSelective Abstention",
         "• LogReg / RF / GBDT\n• Val-Set Platt & Isotonic\n• Entropy + Incompleteness\n  Abstention Rule (H2, H4)"),
        (83, 21, 14, 14, AMBER, "5. TRIPOD+AI Audit\n& Dashboard",
         "• Clustered Bootstrap CIs\n• Subgroup Fairness\n• Permutation Attribution\n• Read-Only Streamlit UI"),
    ]
    for bx, by, bw, bh, col, title, body in boxes:
        rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.6", facecolor="white", edgecolor=col, linewidth=2.2)
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch((bx, by + bh - 3.8), bw, 3.8, boxstyle="round,pad=0.3", facecolor=col, edgecolor=col, linewidth=1)
        ax.add_patch(hdr)
        ax.text(bx + bw / 2, by + bh - 1.9, title, ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        ax.text(bx + 0.8, by + bh - 5.0, body, ha="left", va="top", color="#1E293B", fontsize=8.0, linespacing=1.38)

    for arrow_x in [20.3, 40.3, 60.3, 80.3]:
        ax.annotate("", xy=(arrow_x + 2.3, 28), xytext=(arrow_x, 28),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.0))

    phases = [
        (3, 3, 29, 13, "#E2E8F0", NAVY, "Semesters 1–2 (Year 1 in Japan): Methodological Core",
         "• Formalize Missingness-Aware + Conformal/Post-Hoc Calibration on HRS\n• Submit Ethics/Data Application for Japanese Cohort (JAGES / Municipal LTCI)\n• Publish Workshop/Conference Paper on Informative Missingness Drift"),
        (35, 3, 30, 13, "#E6F4F1", TEAL, "Semester 3 (Year 2 Fall): Cross-National JAGES Validation",
         "• Harmonize ADL / Long-Term Care Insurance (LTCI) Outcomes in Japan\n• Evaluate Cross-Wave Transportability of Missingness Indicators\n• Audit Subgroup Calibration Across Age (65-74, 75-84, 85+) & Living Alone"),
        (68, 3, 29, 13, "#FEF3C7", AMBER, "Semester 4 (Year 2 Spring): Master's Thesis & Clinical Triage",
         "• Optimize Selective Abstention Under Municipal Public-Health Nurse Budgets\n• Release Open-Source TRIPOD+AI Compliant Benchmark Package\n• Defend Master's Thesis & Submit Journal Manuscript (JAMIA / J Gen Intern Med)"),
    ]
    for px, py, pw, ph, bg_c, edge_c, ptitle, pbody in phases:
        prect = patches.FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.5", facecolor=bg_c, edgecolor=edge_c, linewidth=1.8)
        ax.add_patch(prect)
        ax.text(px + 1.0, py + ph - 1.8, ptitle, ha="left", va="center", color=edge_c, fontsize=8.8, fontweight="bold")
        ax.text(px + 1.0, py + ph - 3.6, pbody, ha="left", va="top", color="#1E293B", fontsize=7.8, linespacing=1.35)

    p6 = figures_dir / "fig6_system_architecture_and_roadmap.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p6)

    return saved_paths


def run_all_experiments(
    output_dir: Path,
    figures_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_df = generate_synthetic_longitudinal_cohort(n_participants=2000, waves=(1, 2, 3, 4), seed=seed)
    schema_report = validate_schema(raw_df)
    assert schema_report.is_valid, f"Schema invalid: {schema_report}"

    cohort_df, cohort_flow = build_wave_pair_cohort(raw_df, min_age=65, exclude_baseline_adl_impaired=True)
    missing_summary_df = summarize_missingness(cohort_df)
    missing_summary_df.to_csv(output_dir / "missingness_profile.csv", index=False)

    temporal_split = create_participant_grouped_temporal_split(
        cohort_df, train_waves=[1, 2], test_wave=3, val_participant_fraction=0.25, seed=seed
    )

    h2_records = []
    rel_curves: Dict[str, pd.DataFrame] = {}
    best_pipeline_out: Dict[str, Any] = {}

    for model_name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
        for add_ind in [False, True]:
            tag = f"{model_name}_{'missing_aware' if add_ind else 'imputation_only'}"
            for cal_method in ["uncalibrated", "sigmoid", "isotonic"]:
                res = run_single_pipeline(
                    temporal_split.train_df,
                    temporal_split.val_df,
                    temporal_split.test_df,
                    model_name=model_name,
                    add_missing_indicators=add_ind,
                    calibration_method=cal_method,
                    seed=seed,
                )
                m = res["metrics"]
                h2_records.append({
                    "model_family": model_name,
                    "model": tag,
                    "missing_indicators": add_ind,
                    "calibration": cal_method,
                    **m,
                })
                if tag == "random_forest_missing_aware":
                    _, bin_df = compute_ece(res["y_test"], res["test_prob"], n_bins=8)
                    rel_curves[cal_method] = bin_df
                    if cal_method == "sigmoid":
                        best_pipeline_out = res

    h2_df = pd.DataFrame(h2_records)
    h2_df.to_csv(output_dir / "h2_calibration_metrics.csv", index=False)

    h1_records = []
    for mech in ["MCAR", "MAR", "MNAR"]:
        for rate in [0.0, 0.10, 0.20, 0.30, 0.40]:
            tr_pert = inject_controlled_missingness(temporal_split.train_df, mechanism=mech, extra_rate=rate, seed=seed)
            va_pert = inject_controlled_missingness(temporal_split.val_df, mechanism=mech, extra_rate=rate, seed=seed + 1)
            te_pert = inject_controlled_missingness(temporal_split.test_df, mechanism=mech, extra_rate=rate, seed=seed + 2)
            for model_name in ["logistic_regression", "random_forest"]:
                for add_ind, strat_name in [(False, "median_only"), (True, "median_plus_indicators")]:
                    res = run_single_pipeline(
                        tr_pert, va_pert, te_pert,
                        model_name=model_name,
                        add_missing_indicators=add_ind,
                        calibration_method="sigmoid",
                        seed=seed,
                    )
                    h1_records.append({
                        "mechanism": mech,
                        "extra_missing_rate": rate,
                        "model": model_name,
                        "imputation_strategy": strat_name,
                        **res["metrics"],
                    })
    h1_df = pd.DataFrame(h1_records)
    h1_df.to_csv(output_dir / "h1_missingness_ablation.csv", index=False)

    h3_records = []
    for split_bundle in [
        create_naive_row_random_split_for_audit(cohort_df, seed=seed),
        create_participant_grouped_random_split(cohort_df, seed=seed),
        temporal_split,
    ]:
        res = run_single_pipeline(
            split_bundle.train_df,
            split_bundle.val_df,
            split_bundle.test_df,
            model_name="random_forest",
            add_missing_indicators=True,
            calibration_method="uncalibrated",
            seed=seed,
        )
        overlap_flag = not (
            len(set(split_bundle.train_df["participant_id"]).intersection(set(split_bundle.test_df["participant_id"]))) == 0
        )
        h3_records.append({
            "split_strategy": split_bundle.split_type,
            "has_participant_leakage": overlap_flag,
            "train_n": len(split_bundle.train_df),
            "test_n": len(split_bundle.test_df),
            **res["metrics"],
        })
    h3_df = pd.DataFrame(h3_records)
    h3_df.to_csv(output_dir / "h3_split_leakage_audit.csv", index=False)

    test_missing_frac = temporal_split.test_df[CANDIDATE_FEATURES].isna().mean(axis=1).values
    u_scores = compute_predictive_uncertainty(
        best_pipeline_out["test_prob"],
        missing_fraction=test_missing_frac,
        alpha_missing=0.25,
    )
    h4_df = evaluate_selective_abstention(
        best_pipeline_out["y_test"],
        best_pipeline_out["test_prob"],
        u_scores,
        coverage_targets=[1.0, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60],
        decision_threshold=0.28,
    )
    h4_df.to_csv(output_dir / "h4_selective_abstention.csv", index=False)

    subgroup_df = evaluate_subgroups(
        temporal_split.test_df,
        y_true_col="future_adl_decline",
        y_prob=best_pipeline_out["test_prob"],
    )
    subgroup_df.to_csv(output_dir / "subgroup_fairness_audit.csv", index=False)

    feat_imp_df = compute_feature_explanations(
        best_pipeline_out["model"],
        best_pipeline_out["X_test"],
        best_pipeline_out["y_test"],
        seed=seed,
    )
    feat_imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    error_tax_df = summarize_error_Taxonomy(
        temporal_split.test_df,
        best_pipeline_out["y_test"],
        best_pipeline_out["test_prob"],
        threshold=0.28,
    )
    error_tax_df.to_csv(output_dir / "error_taxonomy.csv", index=False)

    baseline_res = run_single_pipeline(
        temporal_split.train_df,
        temporal_split.val_df,
        temporal_split.test_df,
        model_name="logistic_regression",
        add_missing_indicators=False,
        calibration_method="uncalibrated",
        seed=seed,
    )
    ci_proposed = participant_bootstrap_ci(
        temporal_split.test_df, "future_adl_decline", best_pipeline_out["test_prob"], n_boot=60, seed=seed
    )
    ci_baseline = participant_bootstrap_ci(
        temporal_split.test_df, "future_adl_decline", baseline_res["test_prob"], n_boot=60, seed=seed
    )

    fig_paths = generate_publication_figures(
        missing_summary_df, h1_df, h2_df, rel_curves, h3_df, h4_df, feat_imp_df, subgroup_df, figures_dir
    )

    summary_payload = {
        "project": "AgeWell-AI",
        "version": "0.2.0",
        "synthetic_benchmark": True,
        "synthetic_disclaimer": (
            "All metrics in this run derive from the synthetic HRS/JAGES-harmonized benchmark generator "
            "and represent reproducible pipeline verification, NOT clinical evidence from real human patients."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "environment": {
            "python_version": platform.python_version(),
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "schema_validation": {
            "is_valid": schema_report.is_valid,
            "dataset_sha256": schema_report.dataset_sha256,
            "raw_rows": schema_report.n_rows,
            "raw_participants": schema_report.n_participants,
        },
        "cohort_flow": cohort_flow.__dict__,
        "primary_comparison_95ci": {
            "conventional_logreg_baseline": ci_baseline,
            "proposed_missing_aware_calibrated_rf": ci_proposed,
        },
        "generated_figures": [str(p.name) for p in fig_paths],
    }

    with open(output_dir / "experiment_results_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload
