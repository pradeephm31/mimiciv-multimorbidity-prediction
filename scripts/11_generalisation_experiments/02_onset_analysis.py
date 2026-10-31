from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 11, experiment 2 (extension, no shipped counterpart): can the model predict NEW diagnoses?

Persistence earns no credit for a condition that is not present at the current visit, so the real test of
whether the model adds anything is on the cases persistence cannot get right. For each category:
  * "new vs current visit": among test pairs where the category is ABSENT at the current visit, rank pairs by
    predicted probability and measure AUROC / average precision for "present at the next visit"
  * "first-ever": among pairs where the patient has NEVER had the category up to the current visit
Compared scores: the Stage 10 model (current-visit features), the history model (Stage 11, script 1), and a
simple history baseline that scores a pair by how many earlier admissions had the category (recurrence).
Pooled ("micro") figures stack all categories.
Input : work/11_generalisation_experiments/history_proba_test.npy (script 1) and the Stage 10 files it relies on
Output: work/11_generalisation_experiments/onset_analysis_results.json
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/11_generalisation_experiments")
from charlson_map import CATEGORY_ORDER
import multilabel_lib as ml

OUT = ROOT + "/work/11_generalisation_experiments/"
adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet")
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
pairs, feats, hist_cols, label_cols = ml.make_pairs(adm, cat_to_comm, CATEGORY_ORDER)
_, _, te = ml.split_pairs(pairs)
P_hist = np.load(OUT + "history_proba_test.npy")
P_cur = np.load(ROOT + "/work/10_multilabel_general_population/multilabel_proba_test.npy")
Y = te[label_cols].values.astype(int)
cur = te[CATEGORY_ORDER].values.astype(bool)
cum = te[hist_cols].values

def metrics(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return {"auroc": round(float(roc_auc_score(y, s)), 4), "ap": round(float(average_precision_score(y, s)), 4)}

def analyse(mask_fn, title):
    out = {"per_category": {}}
    pooled = {"y": [], "cur": [], "hist": [], "recur": []}
    for i, c in enumerate(CATEGORY_ORDER):
        m = mask_fn(i)
        y = Y[m, i]
        if len(y) == 0:
            continue
        res = {"n_pairs": int(m.sum()), "n_positive": int(y.sum()), "rate": round(float(y.mean()), 4),
               "model_current_features": metrics(y, P_cur[m, i]),
               "model_history_features": metrics(y, P_hist[m, i]),
               "recurrence_baseline": metrics(y, cum[m, i])}
        out["per_category"][c] = res
        pooled["y"].append(y); pooled["cur"].append(P_cur[m, i]); pooled["hist"].append(P_hist[m, i]); pooled["recur"].append(cum[m, i])
    y = np.concatenate(pooled["y"])
    out["pooled"] = {"n_pairs": int(len(y)), "n_positive": int(y.sum()), "rate": round(float(y.mean()), 4),
                     "model_current_features": metrics(y, np.concatenate(pooled["cur"])),
                     "model_history_features": metrics(y, np.concatenate(pooled["hist"])),
                     "recurrence_baseline": metrics(y, np.concatenate(pooled["recur"]))}
    print(f"\n== {title}  (pooled, {out['pooled']['n_pairs']:,} category-pairs, {out['pooled']['n_positive']:,} positives)")
    for k in ["model_current_features", "model_history_features", "recurrence_baseline"]:
        print(f"   {k:24} {out['pooled'][k]}")
    return out

results = {
    "new_vs_current_visit": analyse(lambda i: ~cur[:, i], "category absent at the current visit, present at the next"),
    "first_ever": analyse(lambda i: cum[:, i] == 0, "patient has never had the category so far, present at the next visit"),
}
json.dump(results, open(OUT + "onset_analysis_results.json", "w"), indent=2)
print("\nSaved onset_analysis_results.json")
