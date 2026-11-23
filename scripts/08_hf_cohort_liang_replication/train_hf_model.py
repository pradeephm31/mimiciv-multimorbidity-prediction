from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix, roc_curve)
import xgboost as xgb

df = pd.read_parquet(ROOT + "/work/08_hf_cohort_liang_replication/hf_model_df.parquet")
FEAT = json.load(open(ROOT + "/work/08_hf_cohort_liang_replication/hf_feature_list.json"))

y = df["escalation_label"].values
groups = df["subject_id"].values

# Same 3-way split methodology as full-cohort: outer 80/20 train+val/test,
# then inner 75/25 train/val split -- all patient-grouped, no leakage.
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss_outer.split(df, y, groups))
df_tv = df.iloc[trainval_idx].reset_index(drop=True)

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
tr_i, val_i = next(gss_inner.split(df_tv, df_tv["escalation_label"].values, df_tv["subject_id"].values))

X_train, y_train = df_tv.iloc[tr_i][FEAT], df_tv.iloc[tr_i]["escalation_label"].values
X_val, y_val = df_tv.iloc[val_i][FEAT], df_tv.iloc[val_i]["escalation_label"].values
X_test, y_test = df.iloc[test_idx][FEAT], y[test_idx]

split_info = {
    "n_train_pairs": int(len(X_train)), "n_val_pairs": int(len(X_val)), "n_test_pairs": int(len(X_test)),
    "n_train_patients": int(df_tv.iloc[tr_i].subject_id.nunique()),
    "n_val_patients": int(df_tv.iloc[val_i].subject_id.nunique()),
    "n_test_patients": int(df.iloc[test_idx].subject_id.nunique()),
    "base_rate_train": round(float(y_train.mean()), 4),
    "base_rate_val": round(float(y_val.mean()), 4),
    "base_rate_test": round(float(y_test.mean()), 4),
}
print(json.dumps(split_info, indent=2))

s_tr = set(df_tv.iloc[tr_i].subject_id); s_val = set(df_tv.iloc[val_i].subject_id); s_te = set(df.iloc[test_idx].subject_id)
assert not (s_tr & s_val) and not (s_tr & s_te) and not (s_val & s_te), "LEAKAGE"
print("Leakage check passed.\n")

def metrics_at(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0), "sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn/(tn+fp) if (tn+fp)>0 else 0, "f1": f1_score(y_true, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def best_threshold_on(y_true, proba):
    prec, rec, th = precision_recall_curve(y_true, proba)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    i = np.nanargmax(f1[:-1])
    return float(th[i]), float(f1[i])

# Use the SAME tuned hyperparameters found for the full-cohort model, for a
# clean comparison isolating cohort effect rather than re-introducing a
# hyperparameter-search confound.
BEST_PARAMS = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
print(f"Using previously-tuned params: {BEST_PARAMS}\n")

pos = y_train.sum(); neg = len(y_train) - pos
model = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=1.0,
                           early_stopping_rounds=30, random_state=42, n_jobs=4, **BEST_PARAMS)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

proba_val = model.predict_proba(X_val)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

auroc = roc_auc_score(y_test, proba_test)
auprc = average_precision_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)
thr, val_f1 = best_threshold_on(y_val, proba_val)
pred_05 = (proba_test >= 0.5).astype(int)
pred_best = (proba_test >= thr).astype(int)

m5 = metrics_at(y_test, pred_05)
mbest = metrics_at(y_test, pred_best)

importances = pd.Series(model.feature_importances_, index=FEAT).sort_values(ascending=False)

print(f"\n=== HF-cohort model (held-out test set) ===")
print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")
print(f"At threshold 0.5: F1={m5['f1']:.4f} P={m5['precision']:.4f} R={m5['recall']:.4f} Acc={m5['accuracy']:.4f}")
print(f"At best (val) threshold {thr:.3f}: F1={mbest['f1']:.4f} P={mbest['precision']:.4f} R={mbest['recall']:.4f} Acc={mbest['accuracy']:.4f}")
print(f"\nTop 15 features:")
print(importances.head(15))

fpr, tpr, _ = roc_curve(y_test, proba_test)
def ds(x, n=60):
    x = np.asarray(x); idx = np.linspace(0, len(x)-1, min(n,len(x))).astype(int)
    return x[idx].tolist()

out = {
    "split_info": split_info,
    "auroc": round(auroc,4), "auprc": round(auprc,4), "brier": round(brier,4),
    "best_iteration": int(model.best_iteration),
    "at_threshold_0.5": m5, "val_selected_threshold": round(thr,4), "at_best_f1_threshold": mbest,
    "top_features": importances.head(20).round(4).to_dict(),
    "roc_curve": {"fpr": ds(fpr), "tpr": ds(tpr)},
}
with open(ROOT + "/work/08_hf_cohort_liang_replication/hf_model_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved hf_model_results.json")
