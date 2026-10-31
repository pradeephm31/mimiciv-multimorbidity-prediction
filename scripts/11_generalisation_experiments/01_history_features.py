from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 11, experiment 1 (extension, no shipped counterpart): use the whole admission history, not just the current visit.

Why: in the general population 39% of patients have no Charlson diagnosis at the current visit, and chronic
conditions are often coded at one admission and not the next, so "current visit only" is a weak summary.

What it does:
  * adds cum_<category> (number of admissions up to and including the current one with that category) as features
  * trains the same 17 per-category XGBoost classifiers as Stage 10 on current + history features
  * scores the last transition of each test patient at L = 3, 5, 10 and compares:
      - persistence on the current visit            (the Stage 09/10 baseline)
      - persistence on the whole history            (repeat every category ever diagnosed, most frequent first)
      - Stage 10 model (current-visit features)     (probabilities reused from Stage 10)
      - history model                               (this script), under three prediction-count rules
Input : work/04_enrichment_features/adm_full_enriched_multihop.parquet, work/02_network_construction/cat_to_comm.json,
        work/06_hyperparameter_tuning/final_tuned_model_results.json, work/10_multilabel_general_population/*
Output: work/11_generalisation_experiments/history_proba_test.npy, history_features_results.json
"""
import json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/11_generalisation_experiments")
from charlson_map import CATEGORY_ORDER
import multilabel_lib as ml

OUT = ROOT + "/work/11_generalisation_experiments/"
adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet")
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
params = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]

pairs, feats, hist_cols, label_cols = ml.make_pairs(adm, cat_to_comm, CATEGORY_ORDER)
tr, va, te = ml.split_pairs(pairs)
print(f"pairs {len(pairs):,}; train {len(tr):,} val {len(va):,} test {len(te):,} ({te.subject_id.nunique():,} patients)")

# consistency check against Stage 10: same test rows and labels, so Stage 10 probabilities can be reused
d10 = pd.read_parquet(ROOT + "/work/10_multilabel_general_population/general_multilabel_df.parquet")
_, _, te10 = ml.split_pairs(d10)
assert (te10.hadm_id.values == te.hadm_id.values).all() and (te10[label_cols].values == te[label_cols].values).all(), "test rows differ from Stage 10"
P10 = np.load(ROOT + "/work/10_multilabel_general_population/multilabel_proba_test.npy")
assert P10.shape == (len(te), len(CATEGORY_ORDER))
print("Consistency check with Stage 10 passed (same test rows and labels).")

t0 = time.time()
P_hist = ml.train_multilabel(tr, va, te, feats + hist_cols, label_cols, params)
print(f"trained 17 history models in {time.time() - t0:.0f}s")
np.save(OUT + "history_proba_test.npy", P_hist)

Y = te[label_cols].values.astype(bool)
cur = te[CATEGORY_ORDER].values.astype(bool)
cum = te[hist_cols].values
rows = ml.last_transition_rows(te)
R = ml.Rankers(cur, cum, P_hist)
R10 = ml.Rankers(cur, cum, P10)
print(f"last-transition patients: {len(rows):,}; no current dx: {(R.n_cur[rows] == 0).mean():.1%}; no dx ever so far: {(R.n_ever[rows] == 0).mean():.1%}")

methods = {
    "persistence_current": lambda L: R.persistence_current(L),
    "persistence_history": lambda L: R.persistence_history(L),
    "stage10_model_rule_a": lambda L: R10.model_rule_a(L),
    "stage10_model_rule_b": lambda L: R10.model_rule_b(L),
    "history_model_rule_a": lambda L: R.model_rule_a(L),
    "history_model_rule_b": lambda L: R.model_rule_b(L),
    "history_model_rule_ever": lambda L: R.model_rule_ever(L),
}
results = {}
for name, mk in methods.items():
    results[name] = {L: ml.score_sets(Y, rows, mk(L)) for L in ml.L_VALUES}
    print(f"{name:26}", {L: results[name][L]["f1"] for L in ml.L_VALUES})

json.dump({"n_test_patients": int(len(rows)),
           "frac_no_current_dx": float((R.n_cur[rows] == 0).mean()),
           "frac_no_dx_ever_so_far": float((R.n_ever[rows] == 0).mean()),
           "results": results}, open(OUT + "history_features_results.json", "w"), indent=2)
print("Saved history_features_results.json")
