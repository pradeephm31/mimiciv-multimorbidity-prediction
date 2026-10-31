from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 10 (added in this project, general-population extension of Stage 09; no shipped counterpart).
Applies the same protocol as Stage 09's evaluate_lasttrans_nopad.py so the scores are comparable:
  * only the LAST transition of each test patient
  * NO padding: persistence predicts the patient's current categories, at most L of them
  * model rule A (as Stage 09): the min(L, max(number of current categories, 1)) most probable categories
  * model rule B (symmetric with persistence): the min(L, number of current categories) most probable categories,
    i.e. nothing for a patient with no current diagnosis. In this cohort 39% of patients have none.
Input : multilabel_proba_test.npy, general_multilabel_df.parquet, general_multilabel_meta.json
Output: multilabel_lasttrans_nopad_results.json
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

W = ROOT + "/work/10_multilabel_general_population/"
df = pd.read_parquet(W + "general_multilabel_df.parquet")
meta = json.load(open(W + "general_multilabel_meta.json"))
# same outer split as train_multilabel_model.py
_, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, df.escalation_label.values, df.subject_id.values))
te = df.iloc[test_idx].reset_index(drop=True)
proba = np.load(W + "multilabel_proba_test.npy")
Y = te[meta["label_cols"]].values.astype(bool)
cur = te[CATEGORY_ORDER].values.astype(bool)
rows = np.where(te.groupby("subject_id")["admit_seq"].transform("max") == te["admit_seq"])[0]   # last transition per patient

def score(rank):
    ni = npred = ntrue = 0
    for r in rows:
        R = rank(r); T = set(np.where(Y[r])[0])
        ni += len(R & T); npred += len(R); ntrue += len(T)
    p = ni / npred; rc = ni / ntrue
    return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / (p + rc), 4)}

model_res, model_sym, pers_res = {}, {}, {}
for L in [3, 5, 10]:
    # rule A (same as Stage 09): at least 1 prediction, even if the patient has no current diagnosis
    model_res[L] = score(lambda r: set(np.argsort(-proba[r])[:min(L, max(int(cur[r].sum()), 1))].tolist()))
    # rule B (symmetric with persistence): exactly min(L, number of current diagnoses) predictions, none if there are none
    model_sym[L] = score(lambda r: set(np.argsort(-proba[r])[:min(L, int(cur[r].sum()))].tolist()))
    pers_res[L] = score(lambda r: set(np.where(cur[r])[0][:L].tolist()))
    print(f"L={L}: model(rule A) {model_res[L]}  model(rule B) {model_sym[L]}  persistence {pers_res[L]}")
n_empty = int((cur[rows].sum(1) == 0).sum())
print(f"patients with no current diagnosis: {n_empty} of {len(rows)} ({n_empty / len(rows):.1%})")
json.dump({"n_patients": int(len(rows)), "n_patients_no_current_dx": n_empty,
           "model_results": model_res, "model_results_symmetric": model_sym, "persistence_results": pers_res},
          open(W + "multilabel_lasttrans_nopad_results.json", "w"), indent=2)
