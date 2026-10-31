from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Added in this project (not in the original package). The packaged train_multilabel_model.py scores ALL
test pairs and PADS the persistence baseline up to L with the most common categories. The shipped
result file (multilabel_lasttrans_nopad_results.json) was produced by a different protocol:
  * only the LAST transition of each test patient (one pair per patient, 4,395 patients)
  * NO padding: persistence predicts the patient's current categories, at most L of them
  * the model predicts the min(L, number of current categories) most probable categories
The persistence part reproduces the shipped numbers exactly. The model rule (k = min(L, |current|)) is
inferred, because the original evaluation script is not in the package.
Input : multilabel_proba_test.npy (from train_multilabel_model.py), hf_multilabel_df.parquet, hf_multilabel_meta.json
Output: multilabel_lasttrans_nopad_results.json
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

W = ROOT + "/work/09_hf_multilabel_persistence_model/"
df = pd.read_parquet(W + "hf_multilabel_df.parquet")
meta = json.load(open(W + "hf_multilabel_meta.json"))
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

model_res, pers_res = {}, {}
for L in [3, 5, 10]:
    model_res[L] = score(lambda r: set(np.argsort(-proba[r])[:min(L, max(int(cur[r].sum()), 1))].tolist()))
    pers_res[L] = score(lambda r: set(np.where(cur[r])[0][:L].tolist()))
    print(f"L={L}: model {model_res[L]}  persistence {pers_res[L]}")
json.dump({"n_patients": int(len(rows)), "model_results": model_res, "persistence_results": pers_res},
          open(W + "multilabel_lasttrans_nopad_results.json", "w"), indent=2)
