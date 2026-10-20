# AgeWell-AI: Missingness-Aware and Uncertainty-Calibrated Machine Learning for Early Identification of Functional Decline in Older Adults

[![Reproducibility & Leakage CI](https://img.shields.io/badge/tests-9%2F9%20passing-brightgreen)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![TRIPOD+AI Aligned](https://img.shields.io/badge/Reporting-TRIPOD%2BAI-navy)](#)

> **MEXT Master's Research Portfolio (October 2027 Intake)**  
> **Author:** Avartan (B.E. in Computer Engineering, Savitribai Phule Pune University, June 2026; CGPA 8.05/10.00 — First Class with Distinction)

---

## 1. Scientific Motivation & Research Question

Rapid population aging in Japan and worldwide has made early identification of **functional decline**—new or worsening limitations in Activities of Daily Living (ADLs) or incident Long-Term Care Insurance (LTCI) certification—a critical public-health priority. While machine learning models trained on longitudinal cohorts such as the **US Health and Retirement Study (HRS)** and the **Japan Gerontological Evaluation Study (JAGES)** achieve strong rank discrimination (`AUROC ~ 0.74–0.82`), two methodological barriers limit their clinical and municipal deployment:

1. **Informative Survey Missingness (MAR / MNAR):** Physical performance tests (gait speed, grip strength) and cognitive recall modules are disproportionately missing among frail, unsteady, or cognitively impaired older adults. Standard single imputation discards the predictive signal carried by the missingness pattern itself.
2. **Miscalibrated Confidence & Unchecked Predictions:** Class-balanced classifiers produce systematically distorted probability estimates under temporal wave shift and lack principled abstention rules to flag ambiguous or incomplete records for municipal public-health nurse review.

**Primary Research Question:**  
*Can a leakage-safe machine learning pipeline that explicitly incorporates informative missingness indicators ($M_{ij}$) and validation-tuned probability calibration identify 2-year functional decline more reliably than conventional imputation-only models across longitudinal survey waves?*

---

## 2. Repository Structure

```text
agewell-ai/
├── README.md
├── pyproject.toml
├── LICENSE
├── configs/
│   ├── baseline.yaml
│   └── experiments.yaml
├── data/
│   ├── README.md                  # HRS & JAGES variable harmonization & access guide
│   └── processed/.gitkeep
├── notebooks/
│   ├── 01_data_audit.ipynb        # Executed notebook: schema, cohort flow & missingness
│   └── 02_exploratory_analysis.ipynb # Executed notebook: H1-H4 experiments & plots
├── src/agewell/
│   ├── __init__.py
│   ├── data.py                    # Schema validator, SHA-256 fingerprinting & benchmark generator
│   ├── cohort.py                  # Wave t -> t+1 cohort builder & exclusion tracker
│   ├── preprocessing.py           # Participant-grouped & temporal splits + M_ij indicators
│   ├── models.py                  # L2 Logistic Regression, Random Forest & Gradient Boosting
│   ├── calibration.py             # Platt/Sigmoid, Isotonic (PAVA) & selective abstention
│   ├── evaluation.py              # AUROC, AUPRC, Brier, ECE, Clustered Bootstrap CIs & Subgroups
│   ├── explainability.py          # Permutation attribution & error taxonomy
│   └── experiments.py             # Deterministic end-to-end runner & 200-DPI figure generator
├── app/
│   └── dashboard.py               # Read-only Streamlit inspection dashboard
├── tests/
│   ├── test_cohort.py
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   └── test_no_leakage.py
├── experiments/
│   ├── README.md
│   └── runs/                      # Saved CSV tables & experiment_results_summary.json
└── reports/
    ├── figures/                   # 6 publication-quality PNG figures
    ├── paper.md                   # Preliminary technical research manuscript
    └── skeptical_ml_review.md     # 7-point methodological & leakage audit
```

---

## 3. Summary of Pilot Benchmark Findings (`dataset_sha256: f8180c83baf3fa39`)

> **Disclaimer:** Metrics below are recorded from the deterministic synthetic HRS/JAGES-harmonized benchmark (`n = 3,892` eligible wave-pairs across `1,734` older adults; out-of-time Wave 3→4 test set `n = 284`, decline prevalence `29.58%`) to validate pipeline integrity and statistical methodology prior to credentialed execution on RAND HRS and JAGES.

| Model & Specification | Missing Indicators ($M_{ij}$) | Calibration | AUROC (95% CI) | AUPRC (95% CI) | Brier Score $\downarrow$ | ECE $\downarrow$ | Sens @ Top 20% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | No | Uncalibrated | 0.7468 (0.6855–0.8066) | 0.5454 (0.4416–0.6501) | 0.2057 | 0.1653 | 0.4286 |
| **Logistic Regression (Calibrated)** | No | Sigmoid (Platt) | 0.7468 | 0.5454 | 0.1770 | 0.0619 | 0.4286 |
| **Logistic Regression + Missing-Aware** | **Yes** | **Sigmoid (Platt)** | **0.7511** | **0.5675** | **0.1741** | **0.0335** | 0.4048 |
| **Random Forest (Baseline)** | No | Uncalibrated | 0.7381 | 0.5510 | 0.2247 | 0.2047 | 0.3929 |
| **Random Forest + Missing-Aware** | **Yes** | Uncalibrated | 0.7440 | 0.5682 | 0.2220 | 0.2055 | 0.4048 |
| **Random Forest + Missing-Aware (Proposed)** | **Yes** | **Sigmoid (Platt)** | **0.7440 (0.6771–0.7963)** | **0.5682 (0.4582–0.6690)** | **0.1770** | **0.0505** | 0.4048 |
| **HistGradientBoosting + Missing-Aware** | **Yes** | **Sigmoid (Platt)** | 0.7314 | 0.5567 | 0.1796 | 0.0460 | 0.3810 |

### Key Scientific Takeaways Across Hypotheses H1–H4
- **H1 (Informative Missingness vs. MCAR Noise):** Adding missingness indicators ($M_{ij}$) improves baseline AUPRC (`+2.21%` for LogReg, `+1.72%` for RF) and preserves discrimination under severe simulated **MNAR (+40% extra missingness)** (`LogReg AUROC 0.7523 vs. 0.7327`; `RF AUROC 0.7251 vs. 0.7143`, `Sens@Top20% 0.4048 vs. 0.3452`). Conversely, under pure **MCAR (+40%)**, indicators slightly degrade Random Forest AUROC (`0.7255 → 0.7167`), confirming that indicators should be restricted to modules with domain-plausible MAR/MNAR mechanisms.
- **H2 (Calibration Reliability):** Validation-tuned Platt scaling reduces Missing-Aware Random Forest **Brier score by 20.3%** (`0.2220 → 0.1770`) and **ECE by 75.4%** (`0.2055 → 0.0505`), and reduces Missing-Aware Logistic Regression **ECE by 77.9%** (`0.1519 → 0.0335`).
- **H3 (Temporal vs. Random Splits):** Same-period participant-grouped random splits overestimate AUROC (`0.7716`) compared to strict out-of-time temporal wave splits (`0.7440`), demonstrating a **+2.76 percentage-point optimism bias**.
- **H4 (Selective Abstention Nuance):** Abstaining on the 25% highest-uncertainty cases reduces overall retained classification error (`34.15% → 29.58%`) and Brier score (`0.1770 → 0.1585`), but naive symmetric entropy abstention reduces retained sensitivity (`66.67% → 40.43%`) because borderline high-risk patients sit near $p \approx 0.50$. Consequently, deferred records must be routed to **human clinical/nurse review** rather than dismissed as low risk.

---

## 4. Quickstart & Reproducibility

```bash
# 1. Clone and install in editable mode
git clone https://github.com/<your-username>/agewell-ai.git
cd agewell-ai
pip install -e .

# 2. Run the leakage, cohort, preprocessing, and metric test suite (9 tests)
pytest tests/ -v

# 3. Reproduce all experiments, CSV tables, bootstrap CIs, and 6 figures (seed=42)
PYTHONPATH=src python3 -m agewell.experiments

# 4. Launch the read-only Streamlit inspection dashboard
streamlit run app/dashboard.py
```
# AgeWell-AI
