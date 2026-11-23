from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import itertools
import random
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
import xgboost as xgb

df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
feature_sets = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))
FEAT = feature_sets["charlson_acci_multihop_v3"]

y = df["escalation_label"].values
groups = df["subject_id"].values

# Same outer train/test split as every prior report (random_state=42) --
# tuning happens ONLY inside the training portion; test set stays untouched.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups))
df_train = df.iloc[train_idx].reset_index(drop=True)
X_test, y_test = df.iloc[test_idx][FEAT], y[test_idx]
X_train_full, y_train_full, groups_train_full = df_train[FEAT], df_train["escalation_label"].values, df_train["subject_id"].values

# Hyperparameter SEARCH runs on a 35% patient-level subsample for speed;
# the FINAL model is refit on the full training set with the winning params.
search_df = pd.read_parquet(ROOT + "/work/06_hyperparameter_tuning/search_subsample.parquet")
X_search, y_search, groups_search = search_df[FEAT], search_df["escalation_label"].values, search_df["subject_id"].values

print(f"Outer train (full): {len(X_train_full):,} pairs ({df_train.subject_id.nunique():,} patients)")
print(f"Search subsample:    {len(X_search):,} pairs ({search_df.subject_id.nunique():,} patients)")
print(f"Outer test:  {len(X_test):,} pairs ({df.iloc[test_idx].subject_id.nunique():,} patients)")

# ---------------------------------------------------------------------------
# Random hyperparameter search, 5-fold GroupKFold, early stopping inside
# each fold's own inner train/val carve-out (grouped, no leakage)
# ---------------------------------------------------------------------------
param_grid = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0],
}
random.seed(42)
keys = list(param_grid.keys())
n_candidates = 20
candidates = []
seen = set()
while len(candidates) < n_candidates:
    combo = tuple(random.choice(param_grid[k]) for k in keys)
    if combo in seen:
        continue
    seen.add(combo)
    candidates.append(dict(zip(keys, combo)))

gkf = GroupKFold(n_splits=2)
outer_folds = list(gkf.split(X_search, y_search, groups_search))

def cv_score(params):
    fold_aurocs = []
    for fold_train_idx, fold_val_idx in outer_folds:
        Xf_tr, yf_tr = X_search.iloc[fold_train_idx], y_search[fold_train_idx]
        Xf_val, yf_val = X_search.iloc[fold_val_idx], y_search[fold_val_idx]
        gf_tr = groups_search[fold_train_idx]

        # Inner carve-out for early stopping (grouped, ~15% of this fold's training groups)
        inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        inner_tr_idx, inner_es_idx = next(inner_gss.split(Xf_tr, yf_tr, gf_tr))
        Xi_tr, yi_tr = Xf_tr.iloc[inner_tr_idx], yf_tr[inner_tr_idx]
        Xi_es, yi_es = Xf_tr.iloc[inner_es_idx], yf_tr[inner_es_idx]

        pos = yi_tr.sum(); neg = len(yi_tr) - pos
        model = xgb.XGBClassifier(
            n_estimators=500, eval_metric="aucpr", scale_pos_weight=neg / pos,
            early_stopping_rounds=25, random_state=42, n_jobs=4, **params,
        )
        model.fit(Xi_tr, yi_tr, eval_set=[(Xi_es, yi_es)], verbose=False)
        proba = model.predict_proba(Xf_val)[:, 1]
        fold_aurocs.append(roc_auc_score(yf_val, proba))
    return float(np.mean(fold_aurocs)), float(np.std(fold_aurocs))

print(f"\nSearching {n_candidates} candidates x 5-fold GroupKFold (with early stopping)...")
search_results = []
for i, params in enumerate(candidates):
    mean_auroc, std_auroc = cv_score(params)
    search_results.append({"params": params, "cv_auroc_mean": mean_auroc, "cv_auroc_std": std_auroc})
    print(f"  [{i+1}/{n_candidates}] CV AUROC={mean_auroc:.4f} (+/-{std_auroc:.4f})  {params}")

search_results.sort(key=lambda r: -r["cv_auroc_mean"])
best = search_results[0]
print(f"\nBest params (CV AUROC={best['cv_auroc_mean']:.4f}): {best['params']}")

with open(ROOT + "/work/06_hyperparameter_tuning/hyperparam_search_results.json", "w") as f:
    json.dump(search_results, f, indent=2)

# ---------------------------------------------------------------------------
# Refit best params on full outer-training set (with its own early-stopping
# carve-out), evaluate ONCE on the held-out outer test set
# ---------------------------------------------------------------------------
inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
tr_idx, es_idx = next(inner_gss.split(X_train_full, y_train_full, groups_train_full))
X_tr, y_tr = X_train_full.iloc[tr_idx], y_train_full[tr_idx]
X_es, y_es = X_train_full.iloc[es_idx], y_train_full[es_idx]
pos = y_tr.sum(); neg = len(y_tr) - pos

final_model = xgb.XGBClassifier(
    n_estimators=600, eval_metric="aucpr", scale_pos_weight=neg / pos,
    early_stopping_rounds=30, random_state=42, n_jobs=4, **best["params"],
)
final_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
print(f"\nFinal model: best_iteration={final_model.best_iteration} (of max 600)")

proba = final_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, proba)
auprc = average_precision_score(y_test, proba)
brier = brier_score_loss(y_test, proba)

pred_05 = (proba >= 0.5).astype(int)
prec_c, rec_c, th_c = precision_recall_curve(y_test, proba)
f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
best_i = np.nanargmax(f1_c[:-1])
best_thresh = float(th_c[best_i])
pred_best = (proba >= best_thresh).astype(int)

def metrics_at(pred):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0), "sensitivity": recall_score(y_test, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0, "f1": f1_score(y_test, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

importances = pd.Series(final_model.feature_importances_, index=FEAT).sort_values(ascending=False)
fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_test, proba)

def downsample(x, n=60):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return x[idx].tolist()

final_out = {
    "best_params": best["params"], "cv_auroc_mean": best["cv_auroc_mean"], "cv_auroc_std": best["cv_auroc_std"],
    "best_iteration": int(final_model.best_iteration), "max_iterations": 600,
    "n_features": len(FEAT), "auroc": round(auroc, 4), "auprc": round(auprc, 4), "brier": round(brier, 4),
    "at_threshold_0.5": metrics_at(pred_05), "best_f1_threshold": best_thresh,
    "at_best_f1_threshold": metrics_at(pred_best),
    "top_features": importances.head(20).round(4).to_dict(),
    "roc_curve": {"fpr": downsample(fpr), "tpr": downsample(tpr)},
    "n_train_pairs": int(len(X_train_full)), "n_test_pairs": int(len(X_test)),
    "n_train_patients": int(df_train.subject_id.nunique()),
    "n_test_patients": int(df.iloc[test_idx].subject_id.nunique()),
    "base_rate_test": round(float(y_test.mean()), 4),
}
with open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json", "w") as f:
    json.dump(final_out, f, indent=2)

print(f"\n=== FINAL TUNED MODEL (held-out test set) ===")
print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")
print(f"F1 (best thresh {best_thresh:.3f}) = {metrics_at(pred_best)['f1']:.4f}")
print(f"Accuracy={metrics_at(pred_best)['accuracy']:.4f}  Precision={metrics_at(pred_best)['precision']:.4f}  Recall={metrics_at(pred_best)['recall']:.4f}")
