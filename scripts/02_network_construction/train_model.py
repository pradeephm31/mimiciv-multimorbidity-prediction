from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, roc_curve)
import xgboost as xgb

df = pd.read_parquet(ROOT + "/work/02_network_construction/model_df.parquet")
COMM_IDS = [0, 1, 2]
COMM_NAMES = {0: "Chronic-systemic/oncologic", 1: "Cardiorenal-metabolic", 2: "Neuro-frailty"}

feature_cols = (
    CATEGORY_ORDER
    + [f"comm{cid}_count" for cid in COMM_IDS]
    + [f"comm{cid}_active" for cid in COMM_IDS]
    + ["charlson_score", "n_categories", "admit_seq", "score_trend",
       "days_since_prior_admission", "dominant_community", "hospital_expire_flag"]
)

df["admission_type"] = df["admission_type"].astype("category")
X = df[feature_cols + ["admission_type"]].copy()
X["admission_type"] = X["admission_type"].cat.codes
y = df["escalation_label"].values
groups = df["subject_id"].values

# ---------------------------------------------------------------------------
# Patient-level split (80/20) -- a patient's admissions never span both sets
# ---------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(X_train):,} pairs ({df.iloc[train_idx].subject_id.nunique():,} patients)")
print(f"Test:  {len(X_test):,} pairs ({df.iloc[test_idx].subject_id.nunique():,} patients)")
print(f"Train escalation rate: {y_train.mean():.2%} | Test escalation rate: {y_test.mean():.2%}")

# ---------------------------------------------------------------------------
# Two models: (A) community/network features only  (B) full feature set
# -- to see how much the Louvain-community projection alone can capture
# ---------------------------------------------------------------------------
comm_feature_cols = [f"comm{cid}_count" for cid in COMM_IDS] + [f"comm{cid}_active" for cid in COMM_IDS] + \
    ["charlson_score", "n_categories", "admit_seq", "score_trend", "days_since_prior_admission", "dominant_community"]

def train_eval(feat_cols, label):
    Xtr, Xte = X_train[feat_cols], X_test[feat_cols]
    pos = y_train.sum(); neg = len(y_train) - pos
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        random_state=42, n_jobs=4,
    )
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"\n[{label}] AUROC={auroc:.3f}  AUPRC={auprc:.3f} (base rate {y_test.mean():.3f})  Brier={brier:.3f}")
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(importances.head(12))
    return model, proba, auroc, auprc, brier, importances

model_comm, proba_comm, auroc_comm, auprc_comm, brier_comm, imp_comm = train_eval(comm_feature_cols, "Community-features-only model")
model_full, proba_full, auroc_full, auprc_full, brier_full, imp_full = train_eval(feature_cols, "Full-feature model (17 categories + community features)")

# ---------------------------------------------------------------------------
# Save ROC / PR curves + everything needed for the report
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, proba_full)
prec, rec, _ = precision_recall_curve(y_test, proba_full)

def downsample_curve(x, y, n=60):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return x[idx].tolist(), y[idx].tolist()

fpr_ds, tpr_ds = downsample_curve(fpr, tpr)
rec_ds, prec_ds = downsample_curve(rec, prec)

model_out = {
    "n_train_pairs": int(len(X_train)), "n_test_pairs": int(len(X_test)),
    "n_train_patients": int(df.iloc[train_idx].subject_id.nunique()),
    "n_test_patients": int(df.iloc[test_idx].subject_id.nunique()),
    "base_rate_train": round(float(y_train.mean()), 4),
    "base_rate_test": round(float(y_test.mean()), 4),
    "community_model": {
        "auroc": round(auroc_comm, 4), "auprc": round(auprc_comm, 4), "brier": round(brier_comm, 4),
        "top_features": imp_comm.head(10).round(4).to_dict(),
    },
    "full_model": {
        "auroc": round(auroc_full, 4), "auprc": round(auprc_full, 4), "brier": round(brier_full, 4),
        "top_features": imp_full.head(15).round(4).to_dict(),
    },
    "roc_curve": {"fpr": fpr_ds, "tpr": tpr_ds},
    "pr_curve": {"recall": rec_ds, "precision": prec_ds},
    "community_names": COMM_NAMES,
}
with open(ROOT + "/work/02_network_construction/model_results.json", "w") as f:
    json.dump(model_out, f, indent=2)
print("\nSaved model_results.json")

model_full.save_model(ROOT + "/work/02_network_construction/escalation_model.json")
print("Saved escalation_model.json (XGBoost model file)")
