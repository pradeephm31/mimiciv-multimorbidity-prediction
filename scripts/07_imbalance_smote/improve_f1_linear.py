from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix, roc_curve)

df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
feature_sets = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))
FEAT = feature_sets["charlson_acci_multihop_v3"]

y = df["escalation_label"].values
groups = df["subject_id"].values

# ---------------------------------------------------------------------------
# THREE-way patient-level split.
# Test set is identical to every prior report (GroupShuffleSplit
# test_size=0.2, random_state=42) so results stay comparable; the previous
# "train" portion is now sub-split into train (75%) + validation (25%).
# Thresholds and model selection use VALIDATION only -- test stays untouched.
# ---------------------------------------------------------------------------
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss_outer.split(df, y, groups))
df_tv = df.iloc[trainval_idx].reset_index(drop=True)

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
tr_i, val_i = next(gss_inner.split(df_tv, df_tv["escalation_label"].values, df_tv["subject_id"].values))

X_train, y_train = df_tv.iloc[tr_i][FEAT], df_tv.iloc[tr_i]["escalation_label"].values
X_val,   y_val   = df_tv.iloc[val_i][FEAT], df_tv.iloc[val_i]["escalation_label"].values
X_test,  y_test  = df.iloc[test_idx][FEAT], y[test_idx]

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

# Sanity: no patient appears in more than one split
s_tr = set(df_tv.iloc[tr_i].subject_id); s_val = set(df_tv.iloc[val_i].subject_id); s_te = set(df.iloc[test_idx].subject_id)
assert not (s_tr & s_val) and not (s_tr & s_te) and not (s_val & s_te), "PATIENT LEAKAGE DETECTED"
print("\nLeakage check passed: train/val/test patient sets are disjoint.")

np.save(ROOT + "/work/07_imbalance_smote/split_y_val.npy", y_val)
np.save(ROOT + "/work/07_imbalance_smote/split_y_test.npy", y_test)
df_tv.iloc[tr_i].to_parquet(ROOT + "/work/07_imbalance_smote/split_train.parquet")
df_tv.iloc[val_i].to_parquet(ROOT + "/work/07_imbalance_smote/split_val.parquet")
df.iloc[test_idx].to_parquet(ROOT + "/work/07_imbalance_smote/split_test.parquet")
with open(ROOT + "/work/07_imbalance_smote/split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)

# ---------------------------------------------------------------------------
# Helper: pick the F1-maximizing threshold on VALIDATION, apply it to TEST
# ---------------------------------------------------------------------------
def metrics_at(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0), "sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0, "f1": f1_score(y_true, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def best_threshold_on(y_true, proba):
    prec, rec, th = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = np.nanargmax(f1[:-1])
    return float(th[i]), float(f1[i])

def evaluate(name, proba_val, proba_test, extra=None):
    thr, val_f1 = best_threshold_on(y_val, proba_val)
    pred_test = (proba_test >= thr).astype(int)
    m = metrics_at(y_test, pred_test)
    out = {
        "label": name,
        "auroc": round(float(roc_auc_score(y_test, proba_test)), 4),
        "auprc": round(float(average_precision_score(y_test, proba_test)), 4),
        "brier": round(float(brier_score_loss(y_test, proba_test)), 4),
        "val_selected_threshold": round(thr, 4),
        "val_f1_at_threshold": round(val_f1, 4),
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
    }
    if extra:
        out.update(extra)
    print(f"[{name}] AUROC={out['auroc']:.4f} AUPRC={out['auprc']:.4f} "
          f"| thr(val)={thr:.3f} -> TEST F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f}")
    return out

results = {}

# ---------------------------------------------------------------------------
# Regularized logistic regression: Lasso (L1), Ridge (L2), ElasticNet
# Standardized + median-imputed (multi-hop lag features have genuine NaNs).
# class_weight='balanced' mirrors XGBoost's scale_pos_weight handling.
# ---------------------------------------------------------------------------
def make_pipe(l1_ratio, C):
    # sklearn >=1.8: penalty= is deprecated and IGNORED; regularization type is
    # controlled purely by l1_ratio (0.0 = pure L2/Ridge, 1.0 = pure L1/Lasso,
    # in between = ElasticNet). Using the old penalty= arg silently gives Ridge
    # for every config.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(l1_ratio=l1_ratio, C=C, solver="saga", max_iter=3000,
                                    class_weight="balanced", random_state=42, tol=1e-3)),
    ])

linear_configs = [
    # NOTE on C: sklearn's objective is C * sum(log_loss) + penalty, so with
    # ~193k training rows the data term dominates unless C is very small.
    # C=0.05 is effectively UNREGULARIZED here; real sparsity needs C ~1e-4 or less.
    ("Unregularized-ish logistic (C=1.0)", 0.0, 1.0),
    ("Ridge (L2) logistic, C=1e-3", 0.0, 1e-3),
    ("Ridge (L2) logistic, C=1e-4", 0.0, 1e-4),
    ("Ridge (L2) logistic, C=1e-5", 0.0, 1e-5),
    ("Lasso (L1) logistic, C=1e-3", 1.0, 1e-3),
    ("Lasso (L1) logistic, C=1e-4", 1.0, 1e-4),
    ("Lasso (L1) logistic, C=1e-5", 1.0, 1e-5),
    ("ElasticNet logistic, C=1e-4, l1_ratio=0.5", 0.5, 1e-4),
    ("ElasticNet logistic, C=1e-3, l1_ratio=0.5", 0.5, 1e-3),
]

for name, l1r, C in linear_configs:
    pipe = make_pipe(l1r, C)
    pipe.fit(X_train, y_train)
    pv = pipe.predict_proba(X_val)[:, 1]
    pt = pipe.predict_proba(X_test)[:, 1]
    coefs = pipe.named_steps["clf"].coef_[0]
    n_nonzero = int((np.abs(coefs) > 1e-8).sum())
    top = pd.Series(coefs, index=FEAT).reindex(
        pd.Series(np.abs(coefs), index=FEAT).sort_values(ascending=False).index[:15])
    key = (name.replace(" ", "_").replace("(", "").replace(")", "")
               .replace(",", "").replace("=", "").replace(".", "").lower())
    results[key] = evaluate(name, pv, pt, extra={
        "n_features_used": n_nonzero, "n_features_total": len(FEAT),
        "top_coefficients": {k: round(float(v), 4) for k, v in top.items()},
    })
    print(f"    -> non-zero coefficients: {n_nonzero}/{len(FEAT)}")
    np.save(ROOT + f"/work/07_imbalance_smote/proba_val_{key}.npy", pv)
    np.save(ROOT + f"/work/07_imbalance_smote/proba_test_{key}.npy", pt)

with open(ROOT + "/work/07_imbalance_smote/linear_model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved linear_model_results.json")
