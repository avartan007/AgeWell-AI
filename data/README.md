# Data Access & Harmonization Guide (HRS & JAGES)

##Important Governance Notice
This repository **never** commits restricted individual-level survey records from the US Health and Retirement Study (HRS) or the Japan Gerontological Evaluation Study (JAGES).

## 1. Phase 1 Data Source: US Health and Retirement Study (HRS)
- **Provider**: University of Michigan Institute for Social Research & NIA (`https://hrsdata.isr.umich.edu/`)
- **Recommended File**: RAND HRS Longitudinal File (public-use registration required)
- **Core Variables Mapped into `agewell.data`**:
  - `HHIDPN`: Unique longitudinal participant identifier
  - `RwAGEY_E`: Age at interview in Wave $w$
  - `RwADLA`: Count of 5 core Activities of Daily Living limitations (bathing, dressing, eating, getting in/out of bed, walking across a room; range 0–5)
  - `RwIADLZA`: Instrumental ADLs (managing money, medication, shopping, meals, phone)
  - `RwSHLT`: Self-rated health (1=Excellent to 5=Poor)
  - `RwCESD`: CES-D depressive symptom score
  - `RwMOBILA` / `RwLGMUSA`: Mobility and large-muscle index
  - `RwGaitSpeed` / `RwGripStrength`: Objective physical performance modules (administered to half-samples or subject to physical-inability non-response, creating informative MNAR missingness)

## 2. Phase 2 Target Cohort (Master's Research in Japan): JAGES
- **Provider**: Japan Gerontological Evaluation Study (`https://www.jages.net/About-JAGES/?lang=english`)
- **Eligibility**: Community-dwelling Japanese older adults aged $\ge 65$ without baseline Long-Term Care Insurance (LTCI) certification.
- **Outcome Definition**: Incident certification of Support Level 1–2 or Care Level 1–5 in municipal Long-Term Care Insurance registries over a 3-to-5 year follow-up (following Lu, Sato, Kondo et al., *J Gen Intern Med* 2023; DOI: `10.1007/s11606-023-08215-2`).

## 3. Synthetic Harmonized Benchmark Generator
For automated testing (`pytest`), continuous integration, and preliminary method verification prior to credentialed data extraction, `src/agewell/data.py` provides `generate_synthetic_longitudinal_cohort()`. All results generated from this synthetic cohort are explicitly tagged `"synthetic_benchmark": true` in run metadata so they are never confused with empirical clinical findings.
