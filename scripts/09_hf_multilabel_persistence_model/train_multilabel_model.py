from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

df = pd.read_parquet(ROOT + "/work/09_hf_multilabel_persistence_model/hf_multilabel_df.parquet")
meta = json.load(open(ROOT + "/work/09_hf_multilabel_persistence_model/hf_multilabel_meta.json"))
FEAT = meta["features"]
LABEL_COLS = meta["label_cols"]

y_dummy = df["escalation_label"].values
groups = df["subject_id"].values

# SAME 3-way split methodology/seeds as every prior HF-cohort experiment
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss_outer.split(df, y_dummy, groups))
df_tv = df.iloc[trainval_idx].reset_index(drop=True)
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
tr_i, val_i = next(gss_inner.split(df_tv, df_tv["escalation_label"].values, df_tv["subject_id"].values))

X_train, X_val, X_test = df_tv.iloc[tr_i][FEAT], df_tv.iloc[val_i][FEAT], df.iloc[test_idx][FEAT]
Y_train, Y_val, Y_test = df_tv.iloc[tr_i][LABEL_COLS], df_tv.iloc[val_i][LABEL_COLS], df.iloc[test_idx][LABEL_COLS]
current_test = df.iloc[test_idx][CATEGORY_ORDER]   # for persistence baseline

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"Test patients: {df.iloc[test_idx].subject_id.nunique():,}")

BEST_PARAMS = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]

# ---------------------------------------------------------------------------
# Train one binary classifier per Charlson category
# ---------------------------------------------------------------------------
proba_test = np.zeros((len(X_test), len(CATEGORY_ORDER)))
print("\nTraining 17 per-category classifiers...")
for i, cat in enumerate(CATEGORY_ORDER):
    y_tr = Y_train[f"next__{cat}"].values
    y_va = Y_val[f"next__{cat}"].values
    pos = y_tr.sum()
    if pos < 5 or pos > len(y_tr) - 5:
        # degenerate label (near-constant) -- fall back to prevalence
        proba_test[:, i] = y_tr.mean()
        continue
    neg = len(y_tr) - pos
    model = xgb.XGBClassifier(n_estimators=400, eval_metric="aucpr", scale_pos_weight=1.0,
                               early_stopping_rounds=25, random_state=42, n_jobs=4, **BEST_PARAMS)
    model.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
    proba_test[:, i] = model.predict_proba(X_test)[:, 1]
    print(f"  [{i+1}/17] {cat}: pos_rate_train={pos/len(y_tr):.3f}, best_iter={model.best_iteration}")

np.save(ROOT + "/work/09_hf_multilabel_persistence_model/multilabel_proba_test.npy", proba_test)

# ---------------------------------------------------------------------------
# Evaluate at L=3,5,10 with Liang's exact micro-averaged Precision/Recall/F1
# ---------------------------------------------------------------------------
Y_test_mat = Y_test.values  # (n_test, 17)
current_mat = current_test.values.astype(bool)

def evaluate_predictions(rank_fn, label):
    results = {}
    for L in [3, 5, 10]:
        num_inter, num_pred, num_true = 0, 0, 0
        for row in range(len(Y_test_mat)):
            R = rank_fn(row, L)
            T = set(np.where(Y_test_mat[row])[0])
            num_inter += len(R & T)
            num_pred += len(R)
            num_true += len(T)
        p = num_inter / num_pred if num_pred > 0 else 0
        r = num_inter / num_true if num_true > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[L] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        print(f"[{label}] L={L}: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return results

def model_rank(row, L):
    order = np.argsort(-proba_test[row])
    return set(order[:L].tolist())

def persistence_rank(row, L):
    active = np.where(current_mat[row])[0]
    if len(active) >= L:
        return set(active[:L].tolist())
    # pad with most-prevalent-overall categories not already included
    overall_prev = current_mat.sum(axis=0)
    pad_order = np.argsort(-overall_prev)
    result = set(active.tolist())
    for c in pad_order:
        if len(result) >= L:
            break
        result.add(int(c))
    return result

print("\n=== Multi-label model (rich XGBoost features) ===")
model_results = evaluate_predictions(model_rank, "Multi-label XGBoost")
print("\n=== Persistence baseline (same test set) ===")
persist_results = evaluate_predictions(persistence_rank, "Persistence")

out = {
    "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
    "n_test_patients": int(df.iloc[test_idx].subject_id.nunique()),
    "model_results": model_results, "persistence_results": persist_results,
}
with open(ROOT + "/work/09_hf_multilabel_persistence_model/multilabel_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved multilabel_results.json")
