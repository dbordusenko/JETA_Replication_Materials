# POFC Framework — Empirical Test Suite and Synthetic Dataset (v3)

Reproducibility package for:

> **Decision Latency and the Predictive Operational Financial Control Framework in Capital-Intensive Industries**
> Dmytro Bordusenko, American Bureau of Shipping (ABS)
> ORCID: https://orcid.org/0009-0007-4643-2457
> *Journal of Emerging Technologies in Accounting* (under review)

This deposit contains everything needed to regenerate the empirical results reported in the paper: the Python test suite, the fully **synthetic** calibrated dataset (N = 420), the model-results and covariate-balance tables, and the publication figures. The pipeline is seeded (`RNG_SEED = 2025`), so it is deterministic within a fixed software environment.

---

## Contents

```
POFC-Empirical-Suite-v3/
├── pofc_empirical_test_v3.py     # Full empirical pipeline (data generation + analysis + figures)
├── requirements.txt              # Pinned Python environment
├── LICENSE                       # MIT
├── CITATION.cff                  # Citation metadata (Zenodo / GitHub)
├── data/
│   └── pofc_dataset_N420_v3.csv  # Synthetic project portfolio (N = 420, 29 columns)
├── results/
│   ├── pofc_model_results_v3.csv # Regression + classification metrics (Table 3)
│   ├── pofc_balance_table.csv    # Covariate balance / standardized mean differences
│   ├── pofc_summary_v3.json      # Machine-readable summary of key results
│   └── sampling_rationale.txt    # Six-sentence sampling rationale (as in the paper)
└── figures/
    └── fig1..fig9 .png           # 300-dpi analytical figures
```

The `data/`, `results/`, and `figures/` folders are a pre-organized copy of the script's output. When you run the script it writes the same files to a fresh `./outputs_v3/` folder next to the script.

---

## Requirements

- Python ≥ 3.10 (reference environment: Python 3.12)
- Packages pinned in `requirements.txt`: NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, seaborn 0.13.2

The manuscript's original run used Python 3.10 with scikit-learn 1.8 and NumPy 2.4, consistent with the pinned environment above.

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python pofc_empirical_test_v3.py
```

The script runs end-to-end in roughly one minute on a standard machine and writes the dataset, result tables, summary JSON, sampling-rationale note, and all figures to `./outputs_v3/`.

---

## The dataset is synthetic

`data/pofc_dataset_N420_v3.csv` is **fully synthetic**. It is produced by the included script and contains no real, proprietary, or personally identifiable data from any organization. It was generated to replicate the *selection mechanism* observed in real capital-intensive project portfolios — POFC adoption is driven by observable factors (larger budgets, longer durations, ERP legacy readiness), not randomization — so that selection bias can be explicitly modeled and corrected via propensity scoring and inverse-probability weighting (IPW). The synthetic study is intended to stress-test the Decision Latency construct under controlled, replicable conditions, **not** to substitute for field evidence.

### Key columns

| Column | Meaning |
|---|---|
| `forecast_model` | Control regime: `Manual` or `POFC` (the "treatment") |
| `decision_latency_days` | Decision Latency, DT = Tr − Td (days) |
| `overrun_ratio` / `overrun_flag` | Continuous cost-overrun ratio / binary overrun (>5%) label |
| `budget_planned_kUSD`, `duration_months` | Project scale covariates |
| `erp_legacy_score`, `erp_real_time` | ERP maturity (instrument / moderator) |
| `propensity_score` | Estimated probability of POFC adoption (for IPW) |
| `cpi_t0`, `spi_t0`, `cv_index`, `sv_index` | Earned-value indices |
| `mape_manual`, `mape_pofc`, `report_days_*`, `opex_*` | KPI fields for the aggregate comparison |

---

## Reference results and reproducibility

Running the script regenerates the files in `results/` and `figures/`. Because the pipeline is seeded, results are **deterministic within a fixed environment**. Two categories of output behave differently across environments:

- **Fully deterministic across environments** (pure NumPy / linear solvers): the synthetic dataset, the covariate-balance table, the Decision Latency analysis (Manual 37.0 d vs. POFC 11.4 d; −69.2%; Cohen's d = 3.77; r = 0.54), the Monte Carlo projections (P(overrun > 5%): 74.5% manual vs. 2.1% POFC; P80 EAC USD 9,402k vs. 7,846k; −15.7% mean EAC), and the linear models (Ridge R² = 0.222; Logistic Regression ROC-AUC = 0.831). These reproduce exactly.

- **Reproducible within a small tolerance** (tree ensembles): Random Forest and Gradient Boosting metrics may differ by up to roughly 0.02 in R²/ROC-AUC across scikit-learn and operating-system versions, owing to low-level floating-point and threading differences. This is expected behavior for these estimators and does not affect any conclusion in the paper. The exact values reported in the manuscript (Table 3) correspond to the author's original environment (documented above); the reference values in this deposit's `results/pofc_model_results_v3.csv` were produced with the pinned environment and agree within this tolerance.

To reproduce the manuscript's Table 3 values as closely as possible, install the pinned `requirements.txt` before running.

---

## License

Released under the MIT License (see `LICENSE`). The synthetic dataset and generated outputs are covered by the same permissive terms.

## How to cite

Please cite both the archived materials and the article:

> Bordusenko, D. (2026). *POFC Framework — Empirical Test Suite and Synthetic Dataset (v3)* [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

> Bordusenko, D. (2026). Decision Latency and the Predictive Operational Financial Control Framework in Capital-Intensive Industries. *Journal of Emerging Technologies in Accounting.*

Replace `10.5281/zenodo.XXXXXXX` with the DOI assigned by Zenodo on publication, and enter the same DOI in the manuscript's Data Availability Statement.
