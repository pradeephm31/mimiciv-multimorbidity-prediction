from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 12 (extension, no shipped counterpart): is the heart-failure result specific to heart failure?

Liang et al. study a heart-failure cohort, and Stage 09 follows them. This stage repeats the Stage 09 multi-label
experiment, unchanged, on other single-disease cohorts built the same way as the heart-failure cohort (Stage 08):
patients with the disease on ANY admission and 2 or more admissions in total, keeping ALL of their admissions.

Cohorts: heart failure (reference, should reproduce Stage 09), renal disease, chronic pulmonary disease, diabetes
(with or without complication).

For every cohort, with the same split, model and protocol as Stages 09/10 and 11:
  * 17 per-category XGBoost classifiers on current-visit features ("current") and on current + history features
    ("history"), scored on the last transition of each test patient at L = 3, 5, 10
  * persistence on the current visit and on the whole history
  * the binary escalation model (Charlson score rises by 2 or more) with the current-visit features, for AUROC
The TCN-DM replication (Stage 08) is not repeated.
Input : work/04_enrichment_features/adm_full_enriched_multihop.parquet, work/02_network_construction/cat_to_comm.json,
        work/06_hyperparameter_tuning/final_tuned_model_results.json
Output: work/12_other_disease_cohorts/<cohort>_results.json, cohort_summary.json
"""
import json, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/11_generalisation_experiments")
from charlson_map import CATEGORY_ORDER
import multilabel_lib as ml

OUT = ROOT + "/work/12_other_disease_cohorts/"
adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet")
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
params = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]

COHORTS = {
    "heart_failure": ["Congestive heart failure"],
    "renal_disease": ["Renal disease"],
    "chronic_pulmonary_disease": ["Chronic pulmonary disease"],
    "diabetes": ["Diabetes without complication", "Diabetes with complication"],
}
n_adm = adm.groupby("subject_id")["hadm_id"].transform("count")
summary = {}
for cname, flags in COHORTS.items():
    t0 = time.time()
    has_dx = adm[flags].sum(axis=1) > 0
    patients = set(adm.loc[has_dx, "subject_id"]) & set(adm.loc[n_adm >= 2, "subject_id"])
    pairs, feats, hist_cols, label_cols = ml.make_pairs(adm, cat_to_comm, CATEGORY_ORDER, patient_mask=patients)
    tr, va, te = ml.split_pairs(pairs)
    Y = te[label_cols].values.astype(bool)
    cur = te[CATEGORY_ORDER].values.astype(bool)
    cum = te[hist_cols].values
    rows = ml.last_transition_rows(te)
    print(f"\n=== {cname}: {len(patients):,} patients, {pairs.shape[0]:,} pairs; test {len(rows):,} patients "
          f"(train {len(tr):,} / val {len(va):,} / test {len(te):,} pairs)", flush=True)

    P_cur = ml.train_multilabel(tr, va, te, feats, label_cols, params, log=lambda s: None)
    P_hist = ml.train_multilabel(tr, va, te, feats + hist_cols, label_cols, params, log=lambda s: None)
    R_cur, R_hist = ml.Rankers(cur, cum, P_cur), ml.Rankers(cur, cum, P_hist)

    methods = {
        "persistence_current": lambda L: R_cur.persistence_current(L),
        "persistence_history": lambda L: R_cur.persistence_history(L),
        "current_model_rule_a": lambda L: R_cur.model_rule_a(L),
        "current_model_rule_b": lambda L: R_cur.model_rule_b(L),
        "history_model_rule_a": lambda L: R_hist.model_rule_a(L),
        "history_model_rule_b": lambda L: R_hist.model_rule_b(L),
        "history_model_rule_ever": lambda L: R_hist.model_rule_ever(L),
    }
    res = {m: {L: ml.score_sets(Y, rows, mk(L)) for L in ml.L_VALUES} for m, mk in methods.items()}

    # binary escalation model, current-visit features, same recipe as Stage 08 (train_hf_model.py)
    m = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=1.0, early_stopping_rounds=30,
                          random_state=42, n_jobs=4, **params)
    m.fit(tr[feats], tr["escalation_label"].values, eval_set=[(va[feats], va["escalation_label"].values)], verbose=False)
    pe = m.predict_proba(te[feats])[:, 1]
    esc = {"auroc": round(float(roc_auc_score(te["escalation_label"], pe)), 4),
           "auprc": round(float(average_precision_score(te["escalation_label"], pe)), 4),
           "base_rate_test": round(float(te["escalation_label"].mean()), 4)}

    info = {"flags": flags, "n_patients": len(patients), "n_admissions": int(len(adm[adm.subject_id.isin(patients)])),
            "n_pairs": int(len(pairs)), "n_test_patients": int(len(rows)),
            "frac_no_current_dx": round(float((R_cur.n_cur[rows] == 0).mean()), 4),
            "mean_current_dx": round(float(R_cur.n_cur[rows].mean()), 2), "mean_next_dx": round(float(Y[rows].sum(1).mean()), 2),
            "escalation": esc, "multilabel": res, "seconds": round(time.time() - t0)}
    json.dump(info, open(OUT + f"{cname}_results.json", "w"), indent=2)
    summary[cname] = info
    print(f"   no current dx: {info['frac_no_current_dx']:.1%}; mean current {info['mean_current_dx']}, next {info['mean_next_dx']}; escalation {esc}")
    for mname in methods:
        print(f"   {mname:24}", {L: res[mname][L]["f1"] for L in ml.L_VALUES})
    print(f"   done in {info['seconds']}s", flush=True)

json.dump(summary, open(OUT + "cohort_summary.json", "w"), indent=2)
print("\nSaved cohort_summary.json")
