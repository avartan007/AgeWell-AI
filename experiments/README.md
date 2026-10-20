# Experiment Logs & Reproducibility Metadata

Every execution of `python -m agewell.experiments` writes a timestamped, fingerprinted summary to `experiments/runs/`:
- `experiment_results_summary.json`: Complete configuration, SHA-256 dataset fingerprint, random seed, split statistics, and bootstrap 95% confidence intervals.
- `h1_missingness_ablation.csv`: Performance across MCAR, MAR, and MNAR missingness stress tests.
- `h2_calibration_metrics.csv`: Pre- vs. post-calibration Brier score, Log Loss, ECE, and calibration slope/intercept.
- `h3_split_leakage_audit.csv`: Empirical comparison of naive row-level random split (leaky) vs. participant-grouped split vs. temporal wave split.
- `h4_selective_abstention.csv`: Risk-coverage and sensitivity trade-offs under uncertainty-based abstention.
- `subgroup_fairness_audit.csv`: Disaggregated metrics across age strata (65-74, 75-84, 85+), sex, and household composition.
