from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, roc_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
import xgboost as xgb

charlson_df = pd.read_parquet(ROOT + "/work/02_network_construction/model_df.parquet")
ccs_df = pd.read_parquet(ROOT + "/work/03_ccs_extension/ccs_model_df.parquet")
ccs_meta = json.load(open(ROOT + "/work/03_ccs_extension/ccs_comm_meta.json"))

# Align on hadm_id (both built from the same 322,206 admission-pairs)
merged = charlson_df.merge(
    ccs_df.drop(columns=["escalation_label", "admission_type", "charlson_score",
                          "admit_seq", "score_trend", "days_since_prior_admission",
                          "hospital_expire_flag"]),
    on=["subject_id", "hadm_id"], suffixes=("", "_ccsdup"), how="inner"
)
print(f"Merged modeling table: {merged.shape}")

COMM_IDS_CHARLSON = [0, 1, 2]
charlson_full_cols = (
    CATEGORY_ORDER
    + [f"comm{cid}_count" for cid in COMM_IDS_CHARLSON]
    + [f"comm{cid}_active" for cid in COMM_IDS_CHARLSON]
    + ["charlson_score", "n_categories", "admit_seq", "score_trend",
       "days_since_prior_admission", "dominant_community", "hospital_expire_flag"]
)
charlson_comm_cols = [f"comm{cid}_count" for cid in COMM_IDS_CHARLSON] + [f"comm{cid}_active" for cid in COMM_IDS_CHARLSON] + \
    ["charlson_score", "n_categories", "admit_seq", "score_trend", "days_since_prior_admission", "dominant_community"]

ccs_comm_cols = ccs_meta["comm_feature_cols"] + \
    ["charlson_score", "n_ccs_categories", "admit_seq", "score_trend", "days_since_prior_admission", "dominant_ccs_community"]
ccs_raw_cols = ccs_meta["raw_ccs_cols"]
ccs_full_cols = ccs_raw_cols + ccs_comm_cols

combined_full_cols = list(dict.fromkeys(charlson_full_cols + ccs_full_cols))

merged["admission_type"] = merged["admission_type"].astype("category")
merged["admission_type_code"] = merged["admission_type"].cat.codes
y = merged["escalation_label"].values
groups = merged["subject_id"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(merged, y, groups))
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx):,} pairs ({merged.iloc[train_idx].subject_id.nunique():,} patients)")
print(f"Test:  {len(test_idx):,} pairs ({merged.iloc[test_idx].subject_id.nunique():,} patients)")

def train_eval(feat_cols, label):
    X = merged[feat_cols].copy()
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
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

    pred_05 = (proba >= 0.5).astype(int)
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, proba)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12)
    best_idx = np.nanargmax(f1_curve[:-1])
    best_thresh = float(thresh_curve[best_idx])
    pred_best = (proba >= best_thresh).astype(int)

    def metrics_at(pred):
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0), "sensitivity": recall_score(y_test, pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0, "f1": f1_score(y_test, pred, zero_division=0),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"[{label}] n_features={len(feat_cols)}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")
    return {
        "label": label, "n_features": len(feat_cols),
        "auroc": round(auroc, 4), "auprc": round(auprc, 4), "brier": round(brier, 4),
        "at_threshold_0.5": metrics_at(pred_05),
        "best_f1_threshold": best_thresh, "at_best_f1_threshold": metrics_at(pred_best),
        "top_features": importances.head(15).round(4).to_dict(),
    }, model, proba

results = {}
model_objs = {}
for key, cols, label in [
    ("charlson_community", charlson_comm_cols, "Charlson: community-only (6 feat)"),
    ("charlson_full", charlson_full_cols, "Charlson: full (17 raw + community)"),
    ("ccs_community", ccs_comm_cols, "CCS: community-only (5 modules)"),
    ("ccs_full", ccs_full_cols, "CCS: full (251 raw categories + community)"),
    ("combined_full", combined_full_cols, "Combined: Charlson full + CCS full"),
]:
    res, model, proba = train_eval(cols, label)
    results[key] = res
    model_objs[key] = (model, proba)

fpr, tpr, _ = roc_curve(y_test, model_objs["ccs_full"][1])
fpr_ch, tpr_ch, _ = roc_curve(y_test, model_objs["charlson_full"][1])

def downsample_curve(x, n=60):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return x[idx].tolist()

comparison_out = {
    "n_train_pairs": int(len(train_idx)), "n_test_pairs": int(len(test_idx)),
    "n_train_patients": int(merged.iloc[train_idx].subject_id.nunique()),
    "n_test_patients": int(merged.iloc[test_idx].subject_id.nunique()),
    "base_rate_test": round(float(y_test.mean()), 4),
    "models": results,
    "roc_curves": {
        "ccs_full": {"fpr": downsample_curve(fpr), "tpr": downsample_curve(tpr)},
        "charlson_full": {"fpr": downsample_curve(fpr_ch), "tpr": downsample_curve(tpr_ch)},
    },
}
with open(ROOT + "/work/03_ccs_extension/comparison_results.json", "w") as f:
    json.dump(comparison_out, f, indent=2)
print("\nSaved comparison_results.json")

print("\n=== Summary ===")
for key, res in results.items():
    print(f"  {res['label']:<45} AUROC={res['auroc']:.4f}  AUPRC={res['auprc']:.4f}  F1@best={res['at_best_f1_threshold']['f1']:.4f}")
