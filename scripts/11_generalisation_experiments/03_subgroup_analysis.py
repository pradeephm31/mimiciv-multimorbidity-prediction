from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 11, experiment 3 (extension, no shipped counterpart): where does the model help? Results by subgroup.

Scores the last transition of each test patient, split by
  * number of current Charlson diagnoses (0, 1, 2 or more)
  * length of the patient's history: admit_seq of the last transition (1 = the patient has 2 admissions in total,
    2 = 3 admissions, 3-4, 5 or more)
Methods: persistence on the current visit, persistence on the whole history, the Stage 10 model (rule B, current-visit
features) and the history model (Stage 11, script 1) with as many predictions as distinct categories ever diagnosed.
Input : work/11_generalisation_experiments/history_proba_test.npy (script 1) and the Stage 10 probabilities
Output: work/11_generalisation_experiments/subgroup_analysis_results.json
"""
import json, sys
import numpy as np
import pandas as pd
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
Y = te[label_cols].values.astype(bool)
cur = te[CATEGORY_ORDER].values.astype(bool)
cum = te[hist_cols].values
R = ml.Rankers(cur, cum, P_hist)
R10 = ml.Rankers(cur, cum, P_cur)
rows = ml.last_transition_rows(te)
seq = te["admit_seq"].values

groups = {
    "current dx = 0": rows[R.n_cur[rows] == 0],
    "current dx = 1": rows[R.n_cur[rows] == 1],
    "current dx >= 2": rows[R.n_cur[rows] >= 2],
    "2 admissions in total": rows[seq[rows] == 1],
    "3 admissions in total": rows[seq[rows] == 2],
    "4-5 admissions in total": rows[(seq[rows] >= 3) & (seq[rows] <= 4)],
    "6+ admissions in total": rows[seq[rows] >= 5],
    "all": rows,
}
methods = {
    "persistence_current": lambda L: R.persistence_current(L),
    "persistence_history": lambda L: R.persistence_history(L),
    "stage10_model_rule_b": lambda L: R10.model_rule_b(L),
    "history_model_rule_ever": lambda L: R.model_rule_ever(L),
}
results = {}
for gname, grows in groups.items():
    results[gname] = {"n_patients": int(len(grows)), "methods": {}}
    for mname, mk in methods.items():
        results[gname]["methods"][mname] = {L: ml.score_sets(Y, grows, mk(L)) for L in ml.L_VALUES}
    f = {m: results[gname]["methods"][m][5]["f1"] for m in methods}
    print(f"{gname:26} n={len(grows):6,}  F1@L=5  " + "  ".join(f"{m.replace('_rule_ever','').replace('_rule_b','')}={v:.3f}" for m, v in f.items()))
json.dump(results, open(OUT + "subgroup_analysis_results.json", "w"), indent=2)
print("Saved subgroup_analysis_results.json")
