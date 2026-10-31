# mimiciv-multimorbidity-prediction

A self-contained, re-runnable MIMIC-IV multimorbidity pipeline (Charlson comorbidity network, escalation prediction, next-visit diagnosis prediction and SHAP explainability). Every path is computed from each script's own location (`ROOT = parents[2]` of the script), so the repository can be cloned to any machine or folder and run from there.

## Setup: read this first

**The MIMIC-IV data is not in this repository.** MIMIC-IV is restricted data: access needs PhysioNet credentialing and a signed Data Use Agreement, and its files may not be redistributed. The `raw/` folder is therefore **empty** (it contains only a `.gitkeep` placeholder) and `.gitignore` stops anything placed in it, and every patient-level output the scripts create, from being committed.

1. **Get the data.** Request access at https://physionet.org/content/mimiciv/ and download these six files:

   | File | MIMIC-IV module |
   |---|---|
   | `admissions.csv` | hosp |
   | `patients.csv` | hosp |
   | `diagnoses_icd.csv` | hosp |
   | `procedures_icd.csv` | hosp |
   | `drgcodes.csv` | hosp |
   | `icustays.csv` | icu |

   The dataset version used for the original results was not recorded; the files have the `anchor_year_group` and `edregtime` columns (MIMIC-IV 2.x or later). Approximate original row counts (so you can sanity-check your download): `diagnoses_icd.csv` 6,364,488 rows, `admissions.csv` 546,028, `drgcodes.csv` 761,856, `procedures_icd.csv` 859,655, `patients.csv` 364,627, `icustays.csv` 94,458.
2. **Place the six `.csv` files directly in `raw/`** (not in a sub-folder), keeping their names. Nothing else needs editing; the scripts read from `raw/` by relative path.
3. **Create a virtual environment anywhere you like**, activate it, and install the pinned packages:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)   or   source venv/bin/activate   (Linux/macOS)
   pip install -r requirements.txt
   ```
   The results were produced with Python 3.13; the package versions are pinned on purpose (an unpinned scikit-learn once silently changed behaviour during the original project).
4. **Run the stages in order** (see "How to run" below). Every stage writes into its own folder under `work/`; these folders exist but are empty apart from a few small aggregate results and figures kept for reference. Re-running overwrites them. The full run took roughly 1.5 hours in total on the 8-core laptop CPU used here (no GPU needed).

**What is and is not in the repository:** the scripts, `reference/ccs/` (public AHRQ CCS lookup tables), `requirements.txt`, this README, and aggregate result files (JSON, PNG, logs) without patient-level content. The following are left out and are recreated when you run the pipeline: the raw data, all `.parquet`, `.npy` and `.npz` files, the per-patient case studies and trajectories, the SHAP HTML report and its interactive force plots (they show individual patients), and the per-patient waterfall plots. Comparisons in this README with "shipped" results refer to the original thesis package (`mimic_thesis_full_code_and_data`), which is not part of this repository.

## Overview

The work behind the thesis "Clinically Interpretable Multimorbidity Prediction Using Charlson Comorbidities and Network-Based Clustering". It builds a network of the 17 Charlson comorbidity categories from MIMIC-IV admissions, groups the categories into disease modules with Louvain community detection, and uses those modules and richer admission features to predict how a patient's comorbidity burden develops. Two prediction tasks are used:
- **Escalation** (Stages 02 to 07): a single yes/no question per admission pair, whether the Charlson score rises by 2 or more at the next admission.
- **Next-visit diagnoses** (Stages 08 to 12); Stage 13 explains the models with SHAP: which of the 17 categories will be present at the next admission (a multi-label task, see the glossary), compared with a persistence baseline and with the TCN-DM method of Liang et al. (2024).

**Stages 01 to 09 replicate the original package**, so shipped results exist to compare against. **Stages 10 to 13 are extensions added in this project** and have no shipped counterpart. The 17 Charlson categories are the main representation because they are clinically interpretable; CCS (Stage 03) is a side comparison that predicts better but is not carried past Stage 04.

## Status

| Stage | Description | Status |
|---|---|---|
| 01 Charlson mapping | `charlson_map.py` library (ICD-9/10 -> 17 Charlson categories) | Done (library, no outputs) |
| 02 Network construction | Admission features, Charlson network, Louvain communities, silhouette validation, escalation model, case studies | Done, all 8 scripts run |
| 03 CCS extension | Map diagnoses to AHRQ CCS (Clinical Classifications Software) categories, rebuild network and Louvain at CCS level, compare Charlson vs CCS prediction | Done, all 5 scripts run |
| 04 Enrichment features | Add age, ICU, procedures, DRG severity, age-adjusted Charlson (ACCI) and multi-hop trajectory features; build the final modelling table | Done, all 4 scripts run |
| 05 Escalation prediction | Compare v1, v2 and v3 feature sets with fixed hyperparameters to isolate the feature-engineering effect | Done, 1 script runs |
| 06 Hyperparameter tuning | Random search with patient-grouped cross-validation and early stopping on the v3 feature set; one final test evaluation | Done, 2 scripts run |
| 07 Imbalance and SMOTE | Patient-grouped 3-way split, regularized logistic regression, XGBoost/LightGBM, calibration, ensembling, SMOTE/SMOTENC/hybrid resampling, all scored with validation-selected thresholds | Done, 5 original scripts and 1 merge script run (`smote_analysis_part2.py` skipped) |
| 08 HF cohort and Liang replication | Heart-failure cohort, network collapse, population transition network (TCN), TCN-DM and baselines | Done, all 6 scripts run |
| 09 Multi-label persistence model, heart-failure cohort | 17 per-category XGBoost classifiers predicting the next-visit diagnosis set, compared with a persistence baseline | Done, 2 original scripts and 1 evaluation script run |
| 10 Multi-label, general population (extension) | The Stage 09 model on the full 223,291-patient cohort; no shipped counterpart | Done, 3 scripts run (added in this project) |
| 11 Generalisation experiments (extension) | History features, onset-only scoring and subgroup breakdown on the full cohort | Done, 3 scripts and a shared library run (added in this project) |
| 12 Other disease cohorts (extension) | The Stage 09 experiment on renal disease, chronic pulmonary disease and diabetes cohorts | Done, 1 script run (added in this project) |
| 13 SHAP explainability (extension) | SHAP for the escalation models and the heart-failure next-visit models, a leave-one-group-out ablation, and one HTML report | Done, 4 scripts and a shared library run (added in this project) |

## Glossary

- **Charlson categories**: the 17 comorbidity groups of the Charlson index (for example congestive heart failure, renal disease, diabetes), each with a weight. The **Charlson score** of an admission is the weighted sum of the categories present.
- **CCS (Clinical Classifications Software)**: a tool from AHRQ (the US Agency for Healthcare Research and Quality) that groups ICD codes into about 280 diagnosis categories. Used only in Stages 03 and 04.
- **Louvain communities**: groups of categories that co-occur and follow each other often; here they act as disease modules.
- **Escalation**: the Charlson score rises by 2 or more at the next admission (about 13% of admission pairs in the general population).
- **Multi-label prediction**: each patient can have several of the 17 categories at once, so each category is its own yes/no label. We train one binary XGBoost classifier per category and keep the highest-probability ones. (Multi-class would mean exactly one class per patient.)
- **L**: how many categories the model may return per patient (3, 5 or 10). L=10 is 10 of only 17 categories, so recall is high almost automatically; do not compare these L values with Liang's L = 5, 10 and 20 over raw ICD-9 codes.
- **Persistence baseline**: predict that the next admission has the same categories as the current one. It scores well because a predicted category counts as correct whether it persisted or is new.
- **Last transition, no padding**: only the last admission pair of each test patient is scored, and persistence is not padded up to L with common categories.
- **Rule A / rule B / rule "ever"**: how many categories the model returns. Rule A returns at least one even if the patient has no current diagnosis (the rule inferred for the shipped Stage 09 result); rule B returns as many as the patient currently has (none if none), matching persistence; rule "ever" returns as many as distinct categories ever diagnosed.
- **TCN / TCN-DM**: the transition network and the distance-matched nearest-neighbour method of Liang et al.
- **Extension stage**: added in this project, with no shipped result to compare against.

## Project structure

```
mimiciv-multimorbidity-prediction/
├── README.md
├── requirements.txt                    pinned packages
├── reference/
│   └── ccs/                            CCS lookup tables: ccs_mapping_raw.parquet, ccs_categories_raw.parquet (diagnoses),
│                                       CCS_PCS_mapping.rda (procedures, from github.com/vcastro/CCS, commit 0a999b6, 2025-03-05)
├── raw/                                input data: EMPTY in the repository; you place the 6 MIMIC-IV CSVs here
│   ├── admissions.csv
│   ├── diagnoses_icd.csv
│   ├── patients.csv
│   ├── icustays.csv
│   ├── procedures_icd.csv
│   └── drgcodes.csv
├── scripts/
│   ├── 01_charlson_mapping/
│   │   └── charlson_map.py
│   ├── 02_network_construction/
│   │   ├── analyze.py
│   │   ├── progression.py
│   │   ├── build_network.py
│   │   ├── compute_silhouette.py
│   │   ├── build_features.py
│   │   ├── train_model.py
│   │   ├── case_studies.py
│   │   └── build_trajectories.py
│   ├── 03_ccs_extension/
│   │   ├── 00_build_ccs_mapping.py
│   │   ├── build_ccs_matrix.py
│   │   ├── build_ccs_network.py
│   │   ├── build_ccs_features.py
│   │   └── compare_models.py
│   ├── 04_enrichment_features/
│   │   ├── build_enrichment.py
│   │   ├── build_multihop_features.py
│   │   ├── build_final_charlson_dataset.py
│   │   └── train_enriched_models.py
│   ├── 05_escalation_prediction/
│   │   └── compare_feature_engineering.py
│   ├── 06_hyperparameter_tuning/
│   │   ├── 00_build_search_subsample.py
│   │   └── tune_final_model.py
│   ├── 07_imbalance_smote/
│   │   ├── improve_f1_linear.py
│   │   ├── improve_f1_xgb.py
│   │   ├── improve_f1_round2.py
│   │   ├── smote_analysis.py
│   │   ├── smote_analysis_part2.py       (copied, not run; see Stage 07)
│   │   ├── smote_analysis_part2b.py
│   │   └── 99_merge_smote_results.py
│   ├── 08_hf_cohort_liang_replication/
│   │   ├── build_hf_network.py
│   │   ├── build_hf_dataset.py
│   │   ├── train_hf_model.py
│   │   ├── build_population_tcn.py
│   │   ├── build_freq_vectors.py
│   │   └── run_tcn_predictions.py
│   ├── 09_hf_multilabel_persistence_model/
│   │   ├── build_multilabel_dataset.py
│   │   ├── train_multilabel_model.py
│   │   └── evaluate_lasttrans_nopad.py
│   ├── 10_multilabel_general_population/   (extension, copies of the Stage 09 scripts)
│   │   ├── build_multilabel_dataset.py
│   │   ├── train_multilabel_model.py
│   │   └── evaluate_lasttrans_nopad.py
│   ├── 11_generalisation_experiments/      (extension)
│   │   ├── multilabel_lib.py                shared helpers, also used by Stage 12
│   │   ├── 01_history_features.py
│   │   ├── 02_onset_analysis.py
│   │   └── 03_subgroup_analysis.py
│   ├── 12_other_disease_cohorts/           (extension)
│   │   └── run_cohort_experiments.py
│   └── 13_shap_explainability/             (extension)
│       ├── shap_common.py                   feature groups, model refits, plot helpers
│       ├── 01_shap_escalation.py
│       ├── 02_shap_multilabel_hf.py
│       ├── 03_ablation_by_group.py
│       └── 04_build_html_report.py
└── work/
    ├── 01_charlson_mapping/            empty (Stage 01 writes no files)
    ├── 02_network_construction/        outputs and log_<script>.txt files
    ├── 03_ccs_extension/               outputs and log_<script>.txt files
    ├── 04_enrichment_features/         outputs and log_<script>.txt files
    ├── 05_escalation_prediction/       outputs and log_<script>.txt files
    ├── 06_hyperparameter_tuning/       outputs and log_<script>.txt files
    ├── 07_imbalance_smote/             outputs, probability arrays (.npy) and log_<script>.txt files
    ├── 08_hf_cohort_liang_replication/ outputs, frequency vectors (.npy) and log_<script>.txt files
    ├── 09_hf_multilabel_persistence_model/ outputs and log_<script>.txt files
    ├── 10_multilabel_general_population/ outputs and log_<script>.txt files
    ├── 11_generalisation_experiments/  outputs and log_<script>.txt files
    ├── 12_other_disease_cohorts/       per-cohort results, cohort_summary.json and log
    └── 13_shap_explainability/         SHAP json, figures/ (PNG), force-plot fragments and shap_report.html
```

## Stage 01: Charlson mapping

`scripts/01_charlson_mapping/charlson_map.py` is a library, not a runnable step.
- Exports `map_code_to_category(code, version)`, `CHARLSON_WEIGHTS` and `CATEGORY_ORDER` (17 categories).
- Imported by every Stage 02 script through `sys.path` (set from `ROOT`).

## Stage 02: Network construction

Run in this order. Inputs and outputs are under `raw/` and `work/02_network_construction/`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `analyze.py` | `raw/admissions.csv`, `raw/diagnoses_icd.csv` | `adm_features.parquet` |
| 2 | `progression.py` | `adm_features.parquet` | `report_data.json`, `score_by_patient.parquet` |
| 3 | `build_network.py` | `adm_features.parquet` | `network_data.json`, `cat_to_comm.json` |
| 4 | `compute_silhouette.py` | `adm_features.parquet` | `silhouette_results.json` |
| 5 | `build_features.py` | `adm_features.parquet`, `cat_to_comm.json` | `model_df.parquet`, `adm_features_with_comm.parquet` |
| 6 | `train_model.py` | `model_df.parquet` | `model_results.json`, `escalation_model.json` |
| 7 | `case_studies.py` | `adm_features.parquet`, `score_by_patient.parquet` | `case_studies.json` |
| 8 | `build_trajectories.py` | `adm_features_with_comm.parquet`, `cat_to_comm.json`, `case_studies.json` | `network_trajectories.json` |

What each does:
1. `analyze.py`: maps diagnoses to Charlson categories, orders each patient's admissions (`admit_seq`), builds 17 flags per admission, applies override rules, and computes `charlson_score` and `n_categories`.
2. `progression.py`: score by admission number; onset, persistence and resolution per category; patient trajectory archetypes.
3. `build_network.py`: 17-node graph with edges = 0.5 co-occurrence + 0.5 next-admission transition; Louvain (seed 42) gives 3 communities.
4. `compute_silhouette.py`: scores the partition with silhouette across Louvain resolutions.
5. `build_features.py`: community counts, active flags and dominant community per admission; consecutive-admission pairs; label `escalation_label` (Charlson score rises by 2 or more at the next admission).
6. `train_model.py`: XGBoost escalation classifier on a patient-grouped split.
7. `case_studies.py`: picks a representative patient for each archetype.
8. `build_trajectories.py`: turns case patients' timelines into community-level paths.

## Stage 03: CCS extension

Same idea as Stage 02 but using CCS (Clinical Classifications Software, from AHRQ, the US Agency for Healthcare Research and Quality), which groups ICD codes into about 280 diagnosis categories instead of 17 Charlson categories. Run in this order. Outputs go to `work/03_ccs_extension/`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `00_build_ccs_mapping.py` | `raw/diagnoses_icd.csv`, `reference/ccs/ccs_mapping_raw.parquet`, `reference/ccs/ccs_categories_raw.parquet` | `dx_with_ccs.parquet` |
| 2 | `build_ccs_matrix.py` | `dx_with_ccs.parquet`, `reference/ccs/ccs_categories_raw.parquet`, `work/02_network_construction/adm_features.parquet` | `ccs_matrix.npz`, `adm_base_for_ccs.parquet`, `ccs_category_list.json`, `coverage_comparison.json` |
| 3 | `build_ccs_network.py` | `ccs_matrix.npz`, `adm_base_for_ccs.parquet`, `ccs_category_list.json` | `ccs_network_data.json`, `ccs_cat_to_comm.json` |
| 4 | `build_ccs_features.py` | `ccs_matrix.npz`, `adm_base_for_ccs.parquet`, `ccs_category_list.json`, `ccs_cat_to_comm.json` | `ccs_model_df.parquet`, `ccs_comm_meta.json`, `adm_with_ccs_comm.parquet` |
| 5 | `compare_models.py` | `work/02_network_construction/model_df.parquet`, `ccs_model_df.parquet`, `ccs_comm_meta.json` | `comparison_results.json` |

What each does:
1. `00_build_ccs_mapping.py`: maps every diagnosis row (ICD-9 and ICD-10) to a CCS category using the vendored lookup tables. No internet access is needed.
2. `build_ccs_matrix.py`: builds a sparse admission-by-category matrix and compares mapping coverage with Charlson.
3. `build_ccs_network.py`: builds the CCS co-occurrence and transition network and runs Louvain.
4. `build_ccs_features.py`: community counts and flags per admission, with the same escalation label as Stage 02.
5. `compare_models.py`: trains XGBoost on Charlson, CCS and combined feature sets and compares them.

## Stage 04: Enrichment features

Adds clinical context to each admission (age, ICU use, procedures, DRG severity), builds the age-adjusted Charlson score and multi-hop history features, and writes the final modelling table used by Stages 05 to 09. Run in this order. Outputs go to `work/04_enrichment_features/`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `build_enrichment.py` | `work/02_network_construction/adm_features.parquet`, `raw/patients.csv`, `raw/icustays.csv`, `raw/procedures_icd.csv`, `raw/drgcodes.csv`, `reference/ccs/CCS_PCS_mapping.rda` | `enrichment_features.parquet` |
| 2 | `build_multihop_features.py` | `work/02_network_construction/adm_features.parquet`, `enrichment_features.parquet` | `adm_full_enriched_multihop.parquet` |
| 3 | `build_final_charlson_dataset.py` | `adm_full_enriched_multihop.parquet`, `work/02_network_construction/cat_to_comm.json` | `final_charlson_model_df.parquet`, `final_feature_sets.json` |
| 4 | `train_enriched_models.py` | `work/02_network_construction/model_df.parquet`, `work/03_ccs_extension/ccs_model_df.parquet`, `work/03_ccs_extension/ccs_comm_meta.json`, `enrichment_features.parquet` | `enriched_comparison_results.json` |

What each does:
1. `build_enrichment.py`: age at admission (from `anchor_age` and `anchor_year`), gender, ICU stay count and length of stay, procedure counts and distinct procedure categories (through the CCS procedure mapping), and APR-DRG severity and mortality (median-imputed, with a `has_apr_drg` flag).
2. `build_multihop_features.py`: age-adjusted Charlson score (ACCI = Charlson score + 1 point per decade over 40), lag 1 to 3 scores, a 3-admission trend slope, days since the previous admission, admission frequency, and cumulative ICU and procedure history.
3. `build_final_charlson_dataset.py`: builds consecutive-admission pairs with the escalation label and defines three feature sets: `charlson_full_v1_baseline`, `charlson_enriched_v2` and `charlson_acci_multihop_v3`.
4. `train_enriched_models.py`: trains XGBoost on Charlson, CCS and combined features with and without the enrichment features, and compares them.

## Stage 05: Escalation prediction (feature-engineering comparison)

Measures how much each round of feature engineering improves escalation prediction, with the model settings held fixed so that only the features change. Output goes to `work/05_escalation_prediction/`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `compare_feature_engineering.py` | `work/04_enrichment_features/final_charlson_model_df.parquet`, `work/04_enrichment_features/final_feature_sets.json` | `feature_engineering_comparison.json` |

What it does:
1. Splits the admission pairs once, 80/20 by patient (`GroupShuffleSplit`, seed 42): 257,804 training pairs (80,045 patients) and 64,402 test pairs (20,012 patients).
2. Trains the same XGBoost model (300 trees, depth 4, learning rate 0.05, class weighting) on three feature sets.
3. Records AUROC, AUPRC, Brier score, accuracy, precision, recall, specificity and F1 at the 0.5 threshold and at the F1-optimal threshold, plus the top 15 feature importances.

Feature sets: **v1** baseline (17 Charlson flags, community features, score, trend); **v2** adds age, gender, ICU, procedures and DRG; **v3** adds the age-adjusted Charlson score, lag 1 to 3 scores, a 3-admission trend slope, admission frequency and cumulative ICU and procedure counts.

Run time: about 35 seconds.

Comparison with the shipped `data/results_json/feature_engineering_comparison.json`:

| Feature set | AUROC (this run / shipped) | AUPRC | Brier | Best F1 |
|---|---|---|---|---|
| v1 baseline | 0.7068 / 0.7067 | 0.3208 / 0.3201 | 0.2117 / 0.2116 | 0.3527 / 0.3521 |
| v2 enriched | 0.7490 / 0.7492 | 0.3508 / 0.3511 | 0.2030 / 0.2029 | 0.3754 / 0.3751 |
| v3 ACCI + multi-hop | 0.7630 / 0.7632 | 0.3804 / 0.3806 | 0.1971 / 0.1970 | 0.3923 / 0.3923 |

Split sizes and the test base rate (0.1308) are identical. All metrics agree to within 0.0007. The top-15 feature importance lists share 12 to 14 of 15 features with the shipped ones; the lowest-ranked features swap places (XGBoost multi-threading is not bit-reproducible). For v3 the top features are `score_trend`, `age_points`, `age_at_admission`, `charlson_trend_slope_3adm` and `charlson_score`.

## Stage 06: Hyperparameter tuning

Tunes XGBoost on the v3 feature set (57 features) and evaluates the winning model once on the held-out test set. Outputs go to `work/06_hyperparameter_tuning/`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `00_build_search_subsample.py` | `work/04_enrichment_features/final_charlson_model_df.parquet` | `search_subsample.parquet` |
| 2 | `tune_final_model.py` | `work/04_enrichment_features/final_charlson_model_df.parquet`, `work/04_enrichment_features/final_feature_sets.json`, `search_subsample.parquet` | `hyperparam_search_results.json`, `final_tuned_model_results.json` |

What each does:
1. `00_build_search_subsample.py` (added in this project, not in the original package): `tune_final_model.py` reads `search_subsample.parquet`, but no original script creates it. This script recreates it as the script's comments describe: a 35% patient-level random sample of the training set (28,016 of 80,045 patients, 90,408 pairs), using the same outer 80/20 split as every other stage. **The original sampling seed is not recorded; `SEED = 7` is an assumption.**
2. `tune_final_model.py`: random search over 20 hyperparameter combinations (depth, learning rate, min child weight, subsample, column sample, gamma, L1/L2), scored by mean AUROC over patient-grouped cross-validation folds with early stopping. The best combination is refit on the full training set (600 trees max, early stopping) and evaluated once on the test set. The script prints "5-fold" but the code uses `GroupKFold(n_splits=2)`.

Run time: about 4 minutes.

Comparison with the shipped `data/results_json/final_tuned_model_results.json`:

| Metric | This run | Shipped |
|---|---|---|
| Test AUROC | 0.7624 | 0.7626 |
| Test AUPRC | 0.3793 | 0.3794 |
| Test Brier | 0.1972 | 0.1961 |
| Best F1 (test) | 0.3931 | 0.3903 |
| CV AUROC (mean) | 0.7547 | 0.7538 |

Split sizes are identical (257,804 train and 64,402 test pairs, 57 features). The test metrics agree to within 0.003 and the top-15 feature lists share 12 of 15 features.

**The selected hyperparameters differ from the shipped ones**, because the search subsample is a different random sample:
- This run: depth 4, learning rate 0.02, min child weight 10, column sample 0.8, gamma 0.5, L1 0, L2 5.0 (best iteration 597 of 600).
- Shipped: depth 4, learning rate 0.1, min child weight 20, column sample 1.0, gamma 1.0, L1 1.0, L2 10.0 (best iteration 229).

Many combinations score almost the same (the 20 candidates span CV AUROC 0.747 to 0.755), so different winners give nearly identical test results. Stages 07 to 09 read `best_params` from `final_tuned_model_results.json`, so they will use this run's parameters.

## Stage 07: Class imbalance, F1 improvement and SMOTE

Tries to raise the F1 score of the escalation model (the outcome is rare, about 13%) and tests whether SMOTE helps. Every model is trained on a training split, the F1-maximising threshold is chosen on a separate validation split, and the test split is touched only for the final numbers. Uses the `best_params` from Stage 06 of this run. Outputs go to `work/07_imbalance_smote/`.

Run in this order (total about 18 minutes):

| # | Script | Input | Output | Time |
|---|---|---|---|---|
| 1 | `improve_f1_linear.py` | `work/04_enrichment_features/final_charlson_model_df.parquet`, `final_feature_sets.json` | `split_train/val/test.parquet`, `split_info.json`, `split_y_val/test.npy`, `proba_{val,test}_<model>.npy`, `linear_model_results.json` | 1.5 min |
| 2 | `improve_f1_xgb.py` | the three splits, `final_feature_sets.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json`, `linear_model_results.json` and the linear `proba_*.npy` | `proba_{val,test}_xgb_best.npy`, `f1_improvement_results.json` | 3 min |
| 3 | `improve_f1_round2.py` | the three splits, `final_feature_sets.json`, `final_tuned_model_results.json`, `proba_*_xgb_best.npy` | `proba_*_lgb_nl*.npy`, `proba_*_xgb_slow.npy`, `f1_round2_results.json` | 2.5 min |
| 4 | `smote_analysis.py` | the three splits, `final_feature_sets.json`, `final_tuned_model_results.json` | `smote_results_partial1.json` | 7 min |
| 5 | `smote_analysis_part2b.py` | same as 4 | `smote_results_partial2.json` | 4 min |
| 6 | `99_merge_smote_results.py` | both partial files | `smote_all_results.json` | instant |

What each does:
1. `improve_f1_linear.py`: builds the 3-way patient-grouped split (192,825 train, 64,979 validation, 64,402 test pairs; the test set equals the one used in every earlier stage). Trains nine regularized logistic regressions (Lasso, Ridge, ElasticNet and a weakly regularized one) and saves their probabilities.
2. `improve_f1_xgb.py`: XGBoost with different `scale_pos_weight` values, isotonic calibration, and ensembles with the best linear model.
3. `improve_f1_round2.py`: LightGBM (31 and 63 leaves), XGBoost on Lasso-selected features, a slow-learning-rate XGBoost, and XGBoost+LightGBM ensembles.
4. `smote_analysis.py`: baseline, class weighting, plain SMOTE and SMOTENC at ratios 0.3, 0.5 and 1.0.
5. `smote_analysis_part2b.py`: SMOTENC at ratio 1.0 and a hybrid (SMOTENC to 30%, then random undersampling to 1:1).
6. `99_merge_smote_results.py`: added in this project. Merges the two partial files into `smote_all_results.json`, the name used by the shipped results; no original script writes that file.

Notes on what was not run or not reproduced:
- `smote_analysis_part2.py` is copied but **not run**. It repeats SMOTENC 1.0 and adds a slow Tomek-links hybrid; the shipped results contain no `smotenc_tomek` entry, and `part2b` replaces it with the faster random-undersampling hybrid.
- `f1_final_embedded.json` (in the shipped results) is not written by any script, so it is not reproduced.

Comparison with the shipped `data/results_json/` files (test-set F1):
- **Splits:** identical rows, same order, same `split_info.json`.
- **Linear models:** identical for all nine configurations (for example best ElasticNet C=1e-3 F1 0.3632, AUROC 0.7355; Lasso C=1e-5 collapses to AUROC 0.5). The README's note about sklearn 1.8 silently disabling `penalty=` does not apply here: the nine configurations give clearly different results.
- **XGBoost and LightGBM models:** AUROC within 0.0015 and F1 within 0.003 of shipped, for example `xgb_spw_1p00` 0.3911 (shipped 0.3886) and `lightgbm_nl31` 0.3857 (0.3869). The best validation-selected XGBoost variant differs (`xgb_spw_6p71` here, `xgb_spw_1p00` shipped) because several variants are nearly tied. These runs use Stage 06 parameters that differ from the shipped ones (see Stage 06).
- **SMOTE:**

| Configuration | This run | Shipped |
|---|---|---|
| Baseline (imputed, no resampling) | 0.3879 | 0.3873 |
| Class weighting | 0.3875 | 0.3873 |
| SMOTE 0.3 / 0.5 / 1.0 | 0.3860 / 0.3747 / 0.3596 | 0.3844 / 0.3775 / 0.3716 |
| SMOTENC 0.3 / 0.5 / 1.0 | 0.3821 / 0.3674 / 0.3539 | 0.3856 / 0.3703 / 0.3629 |
| SMOTENC + undersample hybrid | 0.3822 | 0.3855 |

SMOTE never beats the baseline in either run, and F1 falls as the oversampling ratio rises. The conclusion matches the shipped one.

## Stage 08: Heart-failure cohort and TCN-DM replication

Restricts to heart-failure (HF) patients, checks whether the Charlson network survives in a single-disease cohort, and re-implements the method of Liang et al. (2024): a statistically filtered transition network (TCN) and TCN-DM, which predicts a patient's last-visit diagnoses from a distance-matched similar patient. Outputs go to `work/08_hf_cohort_liang_replication/`. Run in this order (total about 1.5 minutes):

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `build_hf_network.py` | `work/04_enrichment_features/adm_full_enriched_multihop.parquet` | `hf_cohort_adm.parquet`, `hf_cat_to_comm.json`, `hf_network_summary.json` |
| 2 | `build_hf_dataset.py` | `hf_cohort_adm.parquet`, `work/02_network_construction/cat_to_comm.json` | `hf_model_df.parquet`, `hf_feature_list.json` |
| 3 | `train_hf_model.py` | `hf_model_df.parquet`, `hf_feature_list.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json` | `hf_model_results.json` |
| 4 | `build_population_tcn.py` | `hf_cohort_adm.parquet` | `hf_population_tcn.json` |
| 5 | `build_freq_vectors.py` | `hf_cohort_adm.parquet`, `hf_population_tcn.json` | `cand_node_freq.npy`, `cand_edge_freq.npy`, `targ_node_freq.npy`, `targ_edge_freq.npy`, `tcn_dm_patient_lists.json` |
| 6 | `run_tcn_predictions.py` | `hf_cohort_adm.parquet`, `hf_population_tcn.json`, `tcn_dm_patient_lists.json` and the four `.npy` files | `tcn_prediction_results.json` |

What each does:
1. `build_hf_network.py`: keeps patients with a congestive heart failure diagnosis and 2 or more admissions (21,975 patients, 130,206 admissions) and rebuilds the Charlson network on them. Louvain finds 3 communities with modularity 0.000, so the structure collapses when every patient has the same disease.
2. `build_hf_dataset.py`: builds the admission-pair table for this cohort with the v3 features and the general-population communities (108,231 pairs).
3. `train_hf_model.py`: XGBoost escalation model on the HF pairs with the Stage 06 parameters, so only the cohort changes.
4. `build_population_tcn.py`: chi-square test for every ordered category pair (visit t to visit t+1), then a binomial test for direction; 196 significant edges remain.
5. `build_freq_vectors.py`: per-patient node and edge frequency vectors; full histories for the candidate pool and truncated histories (last visit removed) for a random sample of 5,000 targets with 3 or more visits (seed 42).
6. `run_tcn_predictions.py`: TCN-ALL, TCN-CN, TCN-JI and TCN-DM predict the last visit's diagnoses from the visit before; scored with Liang's micro-averaged precision, recall and F1 at L = 3, 5 and 10. The demographics term of Liang's distance is skipped, and L is capped at 10 because there are only 17 categories. A diagnosis counts as correct whether it persisted from the previous visit or is new.

Comparison with the shipped `data/`: `hf_cohort_adm`, `hf_model_df`, `hf_network_summary.json` and `hf_population_tcn.json` are **identical**, and `tcn_prediction_results.json` is **identical** for all four methods and all L. The HF escalation model gives AUROC 0.7503 (shipped 0.7507), AUPRC 0.4412 (0.4407) and Brier 0.1315 (0.1315), the usual XGBoost drift.

TCN-DM F1 (this run = shipped): 0.523 at L=3, 0.586 at L=5, 0.595 at L=10. For comparison, Liang et al. report F1 0.454, 0.581 and 0.612 at L = 5, 10 and 20 on raw ICD-9 codes. These are not directly comparable: Liang's L is out of a far larger label space, while L=10 here is 10 of 17 categories.

## Stage 09: Multi-label persistence model, heart-failure cohort

Predicts the full set of diagnoses at the next admission for the HF cohort, with one XGBoost classifier per Charlson category (17 models, Stage 06 parameters), and compares it with a persistence baseline that repeats the current diagnoses. Outputs go to `work/09_hf_multilabel_persistence_model/`. Run time about 3 minutes.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `build_multilabel_dataset.py` | `work/08_hf_cohort_liang_replication/hf_cohort_adm.parquet`, `work/02_network_construction/cat_to_comm.json` | `hf_multilabel_df.parquet`, `hf_multilabel_meta.json` |
| 2 | `train_multilabel_model.py` | `hf_multilabel_df.parquet`, `hf_multilabel_meta.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json` | `multilabel_proba_test.npy`, `multilabel_results.json` |
| 3 | `evaluate_lasttrans_nopad.py` | `multilabel_proba_test.npy`, `hf_multilabel_df.parquet`, `hf_multilabel_meta.json` | `multilabel_lasttrans_nopad_results.json` |

What each does:
1. `build_multilabel_dataset.py`: same pair table as Stage 08 plus the 17 next-visit category flags as labels (108,231 pairs, 77 columns).
2. `train_multilabel_model.py`: 80/20 patient split, inner train/validation split for early stopping; test set is 21,911 pairs from 4,395 patients. Ranks the 17 categories by predicted probability and scores the top L (3, 5, 10) with Liang's metric.
3. `evaluate_lasttrans_nopad.py` (added in this project): the packaged script scores **all** test pairs and **pads** the persistence baseline up to L with common categories. The shipped result `multilabel_lasttrans_nopad_results.json` used a different protocol: only the last transition per test patient (4,395 patients) and no padding. This script applies that protocol. The persistence numbers reproduce the shipped ones **exactly** (F1 0.6495, 0.7038, 0.7086). The model rule (predict the min(L, number of current diagnoses) most probable categories) is **inferred**, because the original evaluation script is not in the package.

Results (F1 at L = 3 / 5 / 10):

| Protocol | Model | Persistence |
|---|---|---|
| Shipped (last transition, no padding) | 0.681 / 0.714 / 0.715 | 0.650 / 0.704 / 0.709 |
| This run, same protocol (script 3) | 0.676 / 0.711 / 0.713 | 0.650 / 0.704 / 0.709 (identical) |
| This run, packaged script (all pairs, padded) | 0.628 / 0.564 / 0.385 | 0.620 / 0.554 / 0.381 |

The model beats persistence by 0.031, 0.010 and 0.006 F1 in the shipped results, by 0.026, 0.007 and 0.005 in this run under the same protocol, and by 0.008, 0.010 and 0.003 under the packaged protocol, so the margin is small in every case. The model differs from the shipped one by at most 0.005 F1 because the Stage 06 parameters of this run differ from the shipped ones. The packaged protocol's F1 falls as L grows because padding adds low-value guesses to the list.

## Stage 10: Multi-label model on the general population (extension)

Not part of the original package. Stage 09 only covers the heart-failure cohort, so this stage runs the same multi-label model on the full cohort (all 322,206 admission pairs, 223,291 patients) to see whether the heart-failure result carries over. The scripts are copies of the Stage 09 scripts that read the full admission table instead of the heart-failure cohort. There is no shipped result to compare against. Outputs go to `work/10_multilabel_general_population/`. Run time about 10 minutes.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `build_multilabel_dataset.py` | `work/04_enrichment_features/adm_full_enriched_multihop.parquet`, `work/02_network_construction/cat_to_comm.json` | `general_multilabel_df.parquet`, `general_multilabel_meta.json` |
| 2 | `train_multilabel_model.py` | `general_multilabel_df.parquet`, `general_multilabel_meta.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json` | `multilabel_proba_test.npy`, `multilabel_results.json` |
| 3 | `evaluate_lasttrans_nopad.py` | `multilabel_proba_test.npy`, `general_multilabel_df.parquet`, `general_multilabel_meta.json` | `multilabel_lasttrans_nopad_results.json` |

The split is the same as in every other stage (192,825 train, 64,979 validation and 64,402 test pairs; 20,012 test patients). Scripts 1 and 2 are the Stage 09 scripts unchanged apart from paths. Script 3 evaluates only the last transition of each test patient with no padding, as in Stage 09, and reports the model under two rules, because the choice matters here:
- **Rule A** (the rule used in Stage 09): at least one prediction per patient, even if the patient has no current diagnosis.
- **Rule B** (symmetric with persistence): exactly min(L, number of current diagnoses) predictions, none for a patient with no current diagnosis.

38.8% of the general-population test patients (7,764 of 20,012) have no current Charlson diagnosis, against 7.0% (306 of 4,395) in the heart-failure cohort. The persistence baseline predicts nothing for them, so rule A forces the model to make a low-precision prediction that persistence avoids.

Results, last transition per test patient, F1 (precision / recall) at L = 3 / 5 / 10:

| Cohort and method | L=3 | L=5 | L=10 |
|---|---|---|---|
| General, persistence | 0.642 (0.706 / 0.588) | 0.672 (0.710 / 0.638) | 0.674 (0.710 / 0.642) |
| General, multi-label XGBoost, rule A | 0.577 | 0.597 | 0.598 |
| General, multi-label XGBoost, rule B | 0.649 (0.714 / 0.594) | 0.669 (0.707 / 0.635) | 0.670 (0.706 / 0.638) |
| Heart failure, persistence | 0.650 | 0.704 | 0.709 |
| Heart failure, multi-label XGBoost, rule A (shipped: 0.681 / 0.714 / 0.715) | 0.676 | 0.711 | 0.713 |

Under the packaged all-pairs, padded protocol of `train_multilabel_model.py`, the general-population model scores 0.478 / 0.389 / 0.250 against persistence 0.452 / 0.371 / 0.243.

Reading the results: in the general population the multi-label model **does not clearly beat persistence**. Under rule B it ties (+0.007 at L=3, -0.003 at L=5, -0.004 at L=10); under rule A it loses by 0.06 to 0.08; only the padded all-pairs protocol shows a small win (+0.026, +0.019, +0.007). The heart-failure advantage of 0.005 to 0.026 F1 does not carry over. The general cohort has few diagnoses per visit (mean 1.2 current, 1.3 next), so persistence is already strong and there is little for the model to add.

## Stage 11: Generalisation experiments on the full cohort (extension)

Not part of the original package. Stage 10 showed the multi-label model only ties the persistence baseline in the general population, where 38.8% of patients have no Charlson diagnosis at the current visit. This stage tests three ideas on the same split, model and last-transition protocol. Shared helpers (pair building, split, training, scoring, prediction rules) are in `multilabel_lib.py`; script 1 checks that its test rows and labels equal Stage 10's before reusing Stage 10's probabilities. Outputs go to `work/11_generalisation_experiments/`. Run time about 11 minutes.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `01_history_features.py` | `work/04_enrichment_features/adm_full_enriched_multihop.parquet`, `work/02_network_construction/cat_to_comm.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json`, Stage 10 files | `history_proba_test.npy`, `history_features_results.json` |
| 2 | `02_onset_analysis.py` | `history_proba_test.npy`, Stage 10 `multilabel_proba_test.npy` | `onset_analysis_results.json` |
| 3 | `03_subgroup_analysis.py` | same as 2 | `subgroup_analysis_results.json` |
| - | (computed separately) | both probability files | `multilabel_auroc.json` |

1. **History features.** Adds `cum_<category>` (admissions up to and including the current one with that category) as features, and adds a persistence baseline that repeats every category ever diagnosed. 33.7% of last-transition patients have no diagnosis at all so far, 38.8% none at the current visit.
2. **Onset.** Persistence earns no credit for a condition absent at the current visit, so this scores the cases it cannot get right: among pairs where a category is absent at the current visit (or never seen so far), how well does the predicted probability rank those present at the next visit?
3. **Subgroups.** The last-transition F1 split by number of current diagnoses and by length of history.

Experiment 1, last-transition F1 at L = 3 / 5 / 10 (rule A: at least one prediction per patient; rule B: as many predictions as current diagnoses; rule "ever": as many as distinct categories ever diagnosed):

| Method | L=3 | L=5 | L=10 |
|---|---|---|---|
| Persistence, current visit | 0.642 | 0.672 | 0.674 |
| Persistence, whole history | 0.638 | 0.654 | 0.650 |
| Stage 10 model (current features), rule A | 0.577 | 0.597 | 0.598 |
| Stage 10 model (current features), rule B | 0.649 | 0.669 | 0.670 |
| History model, rule A | 0.594 | 0.610 | 0.611 |
| History model, rule B | 0.659 | 0.676 | 0.676 |
| History model, rule "ever" | 0.653 | 0.659 | 0.652 |

Experiment 2, pooled over the 17 categories (all test pairs):

| Cases scored | Positives | Model, current features (AUROC / AP) | Model, history features (AUROC / AP) | Recurrence baseline (AUROC / AP) |
|---|---|---|---|---|
| Absent at current visit, present at next | 30,621 of 1,005,355 (3.0%) | 0.817 / 0.153 | 0.856 / 0.277 | 0.662 / 0.137 |
| Never diagnosed so far, present at next | 19,740 of 960,741 (2.1%) | 0.790 / 0.069 | 0.806 / 0.100 | 0.500 / 0.021 |

Overall multi-label AUROC on all test pairs (pooled over all pair-category cells / mean of per-category AUROC): 0.929 / 0.892 with current features, 0.943 / 0.911 with history features. These are high because persistence of conditions the patient already has is easy to rank; the onset table above is the harder test.

Experiment 3, last-transition F1 at L=5:

| Subgroup | Patients | Persistence, current | Persistence, history | Stage 10 model (rule B) | History model ("ever") |
|---|---|---|---|---|---|
| No current diagnosis | 7,764 | 0.000 | 0.237 | 0.000 | 0.239 |
| 1 current diagnosis | 5,690 | 0.622 | 0.611 | 0.622 | 0.611 |
| 2 or more current diagnoses | 6,558 | 0.736 | 0.712 | 0.732 | 0.720 |
| 2 admissions in total | 8,897 | 0.620 | 0.620 | 0.618 | 0.621 |
| 3 admissions | 3,920 | 0.666 | 0.662 | 0.661 | 0.666 |
| 4-5 admissions | 3,569 | 0.686 | 0.668 | 0.683 | 0.672 |
| 6+ admissions | 3,626 | 0.727 | 0.669 | 0.724 | 0.681 |

Reading the results: history features lift the model slightly above current-visit persistence under rule B (+0.017, +0.004, +0.002 F1), which is still close to a tie. The model does carry real signal for new diagnoses (onset AUROC 0.82 to 0.86 against 0.66 for simple recurrence), which persistence cannot produce. In every subgroup with at least one current diagnosis the models only match persistence; they do not beat it.

## Stage 12: Other disease cohorts (extension)

Not part of the original package. Liang et al. use a heart-failure cohort, and Stage 09 follows them. This stage repeats the Stage 09 multi-label experiment, unchanged, on other cohorts built exactly like the heart-failure cohort (Stage 08): patients with the disease on any admission and 2 or more admissions in total, keeping all their admissions. It also trains the binary escalation model on each cohort to report AUROC. The TCN-DM replication is not repeated. Outputs go to `work/12_other_disease_cohorts/`. Run time about 24 minutes.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `run_cohort_experiments.py` | `work/04_enrichment_features/adm_full_enriched_multihop.parquet`, `work/02_network_construction/cat_to_comm.json`, `work/06_hyperparameter_tuning/final_tuned_model_results.json`, `scripts/11_generalisation_experiments/multilabel_lib.py` | `<cohort>_results.json` (one per cohort), `cohort_summary.json` |

For each cohort it trains 17 per-category XGBoost classifiers twice (current-visit features, and current plus history features, as in Stage 11) and scores them with the persistence baselines, on the last transition of each test patient. The heart-failure cohort is included as a reference and **reproduces Stage 09 exactly** (current-features model, rule A: 0.676 / 0.711 / 0.713).

| Cohort | Patients | Pairs | Test patients | No current dx | Mean current / next dx | Escalation AUROC (base rate) |
|---|---|---|---|---|---|---|
| Heart failure (reference) | 21,975 | 108,231 | 4,395 | 7.0% | 2.51 / 2.81 | 0.750 (0.189) |
| Renal disease | 20,466 | 105,029 | 4,094 | 5.7% | 2.59 / 2.85 | 0.791 (0.214) |
| Chronic pulmonary disease | 25,074 | 111,081 | 5,015 | 12.5% | 1.98 / 2.17 | 0.762 (0.149) |
| Diabetes | 26,625 | 118,565 | 5,325 | 7.0% | 2.28 / 2.49 | 0.756 (0.171) |
| General population (Stages 06 and 10, for reference) | 223,291 | 322,206 | 20,012 | 38.8% | 1.19 / 1.32 | 0.762 (0.131) |

Last-transition F1 at L = 3 / 5 / 10:

| Cohort | Persistence (current) | Current model, rule A | Current model, rule B | History model, rule A | History model, rule B |
|---|---|---|---|---|---|
| Heart failure | 0.649 / 0.704 / 0.709 | 0.676 / 0.711 / 0.713 | 0.666 / 0.702 / 0.705 | 0.692 / 0.723 / 0.724 | 0.681 / 0.714 / 0.715 |
| Renal disease | 0.649 / 0.708 / 0.713 | 0.683 / 0.717 / 0.719 | 0.672 / 0.708 / 0.710 | 0.693 / 0.726 / 0.727 | 0.682 / 0.716 / 0.718 |
| Chronic pulmonary disease | 0.650 / 0.695 / 0.698 | 0.690 / 0.716 / 0.716 | 0.665 / 0.693 / 0.694 | 0.696 / 0.722 / 0.722 | 0.671 / 0.699 / 0.700 |
| Diabetes | 0.636 / 0.686 / 0.689 | 0.658 / 0.689 / 0.691 | 0.650 / 0.682 / 0.683 | 0.671 / 0.699 / 0.700 | 0.662 / 0.691 / 0.692 |

Reading the results: in all four single-disease cohorts the model is above the persistence baseline, and history features add a further 0.006 to 0.016 F1. The margin depends on the prediction rule: with rule A (at least one prediction, which gives the model credit for patients persistence cannot predict for) it is +0.002 to +0.040 F1; with rule B (as many predictions as current diagnoses) the current-features model ties persistence at L=5 and L=10 (-0.006 to 0.000) and leads only at L=3 (+0.014 to +0.023), while the history model leads by +0.002 to +0.033. So the heart-failure result is not specific to heart failure, but the advantage over a trivial baseline is small and concentrated at small L. Escalation AUROC is similar across cohorts (0.750 to 0.791). Neither multi-label AUROC nor the onset analysis was computed per cohort, because the per-category probabilities were not saved.

## Stage 13: SHAP explainability (extension)

Not part of the original package, which lists SHAP as not implemented. SHAP splits each prediction into one contribution per feature (log-odds units), so it shows which inputs push a patient's predicted risk up or down. The final models of earlier stages were never saved, so they are refit with the same splits, parameters and recipes; the refit AUROCs equal the earlier stages' (0.7624 for Stage 06, 0.7503 for Stage 08). `shap.TreeExplainer` is used on a random sample of up to 5,000 held-out test rows, never training data. Because the score, ACCI, lag and trend features are strongly correlated and share SHAP credit, importance is also summed by feature group (defined in `shap_common.py`). Outputs go to `work/13_shap_explainability/`. Run time about 10 minutes. Packages added: `shap==0.54.1`, `matplotlib==3.11.2`.

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `01_shap_escalation.py` | `work/04_enrichment_features/final_charlson_model_df.parquet`, `final_feature_sets.json`, `work/08_hf_cohort_liang_replication/hf_model_df.parquet`, `hf_feature_list.json`, Stage 06 parameters, `cat_to_comm.json` | `escalation_<cohort>_shap.json`, `figures/escalation_<cohort>_*.png`, `force_escalation_<cohort>.html` |
| 2 | `02_shap_multilabel_hf.py` | `work/04_enrichment_features/adm_full_enriched_multihop.parquet`, Stage 06 parameters, `scripts/11_generalisation_experiments/multilabel_lib.py` | `multilabel_hf_shap.json`, `figures/multilabel_hf_*.png` |
| 3 | `03_ablation_by_group.py` | `final_charlson_model_df.parquet`, `final_feature_sets.json`, Stage 06 parameters | `ablation_by_group.json` |
| 4 | `04_build_html_report.py` | everything above | `shap_report.html` (single self-contained file, about 6 MB) |

1. **Escalation models** (general population and heart-failure cohort): mean |SHAP| per feature and per group, beeswarm and dependence plots, waterfall plots for three patients (highest-risk escalation, a missed escalation, lowest-risk non-escalation), an interactive force plot of 500 patients, rank agreement with XGBoost's built-in importance and bootstrap stability.
2. **Next-visit diagnoses** (the 17 Stage 09 models, heart-failure cohort): top features and group importance per category, a category-by-group heatmap, and beeswarm plots for the three most prevalent categories.
3. **Ablation**: SHAP shows attribution, not causation, so the general-population escalation model is refit with each feature group removed in turn.
4. **Report**: `shap_report.html` embeds all figures as base64 and the force plots with the SHAP JavaScript, so it opens in any browser and can be moved or emailed on its own.

Findings, share of total mean |SHAP| by feature group:

| Group | General population | Heart failure |
|---|---|---|
| Burden score and trajectory | 40.8% | 44.8% |
| Age and sex | 26.1% | 12.2% |
| Charlson category flags | 14.6% | 20.6% |
| Admission history and timing | 9.7% | 11.0% |
| ICU, procedures and DRG | 6.0% | 7.4% |
| Network communities (Louvain) | 2.8% | 4.0% |

- The top features are `age_at_admission`, `score_trend`, `charlson_trend_slope_3adm`, `charlson_score` and `admissions_per_year_so_far` in the general population, and `score_trend`, `charlson_score`, `charlson_trend_slope_3adm` and `age_at_admission` in heart failure. SHAP importance and XGBoost's built-in importance agree (Spearman 0.83 and 0.76); the SHAP ranking is stable over bootstrap resamples of the explained rows (Spearman 0.9995; this does not measure refit variability).
- Direction: a high current Charlson score or score trend pushes the predicted risk down, probably because the outcome is a further rise of 2 or more (regression to the mean; an interpretation, not tested by SHAP). Older age pushes it up.
- Ablation (general population, AUROC 0.7624 for the full model): removing burden score and trajectory costs 0.062; removing the Charlson flags 0.006; age and sex 0.004; admission history 0.002; ICU, procedures and DRG 0.0007; **removing the Louvain community features changes nothing (+0.0002, within refit noise of about 0.0005)**. On the 17 flags alone AUROC is 0.611; adding the 7 community features gives 0.618 (+0.007).
- Next-visit models: for 7 of 17 categories the category's own current flag is the top feature (it is second for the other 10), that is, the persistence signal; the Charlson flags group carries 30% to 87% of importance per category, and the community features 0% to 19%.

## Results so far

- Charlson mapping coverage of diagnosis rows: 14.5%
- Louvain communities: 3, modularity 0.039
- Silhouette (chosen partition): 0.075
- Escalation model, full features: AUROC 0.707, AUPRC 0.321, Brier 0.212 (base rate 0.131)
- Dataset sizes: 545,497 admissions, 322,206 admission pairs (13.09% escalation)
- CCS mapping coverage of diagnosis rows: 95.2% (279 CCS categories observed); the CCS network gives 10 Louvain communities, modularity 0.051
- Charlson vs CCS escalation prediction (AUROC): Charlson full 0.707, CCS community-only 0.721, CCS full 0.754, combined 0.759 (CCS beats Charlson)
- Enrichment coverage: age 100%, ICU stay in 15.6% of admissions, procedure CCS mapping 99.9%, APR-DRG 67.3%
- Effect of enrichment (AUROC): Charlson 0.707 to 0.749, CCS 0.754 to 0.766, combined 0.772; enrichment features alone 0.737
- Feature engineering with fixed hyperparameters (Charlson, AUROC / best F1): v1 0.707 / 0.353, v2 0.749 / 0.375, v3 0.763 / 0.392
- Best escalation F1 after imbalance experiments: about 0.39 (XGBoost 0.391, LightGBM 0.386, best logistic regression 0.363). SMOTE and SMOTENC do not beat the plain baseline (0.388) at any ratio
- Heart-failure cohort (21,975 patients): Charlson network modularity 0.000; TCN-DM F1 0.523 / 0.586 / 0.595 at L = 3 / 5 / 10, below the persistence baseline (0.650 / 0.704 / 0.709); HF escalation AUROC 0.750, F1 0.456 (validation-selected threshold, base rate 0.189)
- Multi-label XGBoost vs persistence (last-transition protocol, F1 at L = 3 / 5 / 10): 0.676 / 0.711 / 0.713 vs 0.650 / 0.704 / 0.709
- Multi-label on the full cohort (Stage 10, extension): F1 at L = 3 / 5 / 10 is 0.649 / 0.669 / 0.670 (symmetric rule) against persistence 0.642 / 0.672 / 0.674, so a tie; the heart-failure advantage does not carry over
- History features (Stage 11, extension): lift the multi-label model to 0.659 / 0.676 / 0.676 F1 (persistence 0.642 / 0.672 / 0.674); onset AUROC 0.856 for new diagnoses against 0.662 for simple recurrence; multi-label AUROC 0.943 (pooled)
- Other disease cohorts (Stage 12, extension): the model beats persistence in renal, pulmonary and diabetes cohorts as in heart failure, by small margins (history model, rule A: +0.04 to +0.05 F1 at L=3); escalation AUROC 0.750 to 0.791
- SHAP (Stage 13, extension): burden score and trajectory features carry about 41% (general) to 45% (heart failure) of the escalation model's attributions; the Louvain community features carry 2.8% to 4.0% and removing them leaves AUROC unchanged
- Tuned v3 model on the held-out test set: AUROC 0.762, AUPRC 0.379, best F1 0.393. Tuning does not improve on the default v3 model (AUROC 0.763)

Checked against the original `thesis_package` outputs: `adm_features`, `model_df`, `case_studies`, `network_trajectories`, `silhouette_results` and `coverage_comparison` are identical. `ccs_model_df.parquet` has identical values and columns; two columns (`ccs_76`, `ccs_182`) are in swapped order. `final_charlson_model_df.parquet` has identical columns and values except `charlson_trend_slope_3adm`, which differs by at most 6e-15 (floating-point noise). `model_results.json`, `comparison_results.json`, `enriched_comparison_results.json`, `feature_engineering_comparison.json` and `final_tuned_model_results.json` differ only in the 3rd to 4th decimal (multi-threaded XGBoost is not bit-reproducible, and this run used Python 3.13). Stage 08 is identical to the shipped results except the HF model's AUROC (0.7503 vs 0.7507). The Stage 07 splits and linear-model results are identical; its XGBoost/LightGBM F1 values are within 0.003 of shipped, and the SMOTE/SMOTENC F1 values are within 0.012 (the largest gap is SMOTE at ratio 1.0: 0.3596 here, 0.3716 shipped), with the same conclusion.

## Limitations and how to read the results

- **Not comparable with Liang et al.** Their F1 (0.454 at L=5, 0.581 at L=10, 0.612 at L=20) is over raw ICD-9 codes in MIMIC-III; ours is over 17 categories in MIMIC-IV, so a higher F1 here does not show a better method. Compare methods within our own setup: persistence, our TCN-DM replication and the multi-label model.
- **Our TCN-DM is a simplified re-implementation** (no demographics term, L capped at 10, 5,000-patient target sample). It loses to persistence here; that does not show Liang's published result is wrong.
- **The multi-label advantage is small.** The model beats persistence in the four single-disease cohorts (heart failure, renal disease, chronic pulmonary disease, diabetes) by 0.002 to 0.040 F1 depending on L and on the prediction rule, and the current-features model ties it at larger L under rule B. In the general population it ties persistence. A fair summary is "above the persistence baseline in single-disease cohorts, equal to it in the full population".
- **No statistical uncertainty is reported.** Each result is one split and one seed, with 4,000 to 5,300 test patients per disease cohort, so differences of a few thousandths of F1 may be noise. The disease cohorts also overlap (a patient with heart failure and diabetes is in both), so they are not independent replications.
- **Where the model does show clear signal** is in ranking new diagnoses: in the general population the AUROC for a category appearing at the next visit when absent at the current one is 0.82 (current features) to 0.86 (with history features), against 0.66 for simple recurrence. This was not repeated per disease cohort.
- **The model-selection rule of the shipped Stage 09 result is inferred.** The original evaluation script is missing; persistence reproduces exactly, the model within 0.005 F1.
- **Stage 06 differs from the shipped run** because the sampling seed of the search subsample is not recorded; the selected hyperparameters differ, the test metrics agree to within 0.003. Stages 07 to 12 use this run's parameters.
- **Thresholds.** The honest escalation F1 is the validation-selected one (Stage 07, about 0.39). F1 values quoted for Stages 05 and 06 pick the best threshold on the test set and are slightly optimistic. The heart-failure escalation F1 (0.456, Stage 08) also uses a validation-selected threshold; it is higher than the general population's 0.39 partly because the base rate is higher (18.9% against 13.1%).
- **The Louvain network modules add little predictive information once the burden-score features are present.** SHAP gives them 2.8% (general population) and 4.0% (heart failure) of the importance, and the leave-one-group-out ablation shows no change in AUROC when they are removed; only on the 17 Charlson flags alone do they help (+0.007 AUROC). Their value in the thesis is interpretability (disease modules), not prediction.
- **SHAP is attribution, not causation**, and correlated features share credit; read the feature-group results first. SHAP was computed for the escalation models and the heart-failure next-visit models only, not for the general-population multi-label model or the Stage 12 cohorts.
- **Charlson covers only 14.5% of diagnosis rows**, so 38.8% of general-population patients have no Charlson diagnosis at the current visit (6% to 13% in the disease cohorts). This drives the weaker persistence-vs-model results in the full population.
- **CCS predicts better than Charlson** (escalation AUROC 0.754 against 0.707, and 0.766 against 0.749 with enrichment) but is not used after Stage 04, because the thesis is built around interpretable Charlson categories.
- **Not computed:** multi-label AUROC and the onset analysis for the Stage 12 cohorts.

## How to run

1. Put the six CSVs in `raw/` (see Setup above).
2. Create a venv anywhere, activate it, then `pip install -r requirements.txt`.
3. Run the scripts from any directory, in the order of the stage sections below (each stage's table lists its scripts in run order), for example:
   `python scripts/02_network_construction/analyze.py`
   Stage 01 is a library and needs no run. Stages 03 and 04 need the files in `reference/ccs/`, which are included.

## Notes

- The MIMIC dates (for example 2160) are shifted per patient for privacy. Gaps between a patient's admissions and ages are real; years are not.
- `requirements.txt` covers Stages 02 to 13 (new packages after Stage 07: `shap` and `matplotlib` in Stage 13).
- Stage 09 (heart failure) and Stage 10 (general population) are the same experiment on different cohorts; Stage 12 repeats it on three other disease cohorts using the shared `scripts/11_generalisation_experiments/multilabel_lib.py`.
- The scripts for Stages 01 to 09 are path-adjusted copies of the original thesis package's scripts; Stages 10 to 13 were added in this project.
