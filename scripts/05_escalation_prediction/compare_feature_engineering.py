from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
import xgboost as xgb

df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
feature_sets = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))

y = df["escalation_label"].values
groups = df["subject_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups))
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx):,} pairs ({df.iloc[train_idx].subject_id.nunique():,} patients)")
print(f"Test:  {len(test_idx):,} pairs ({df.iloc[test_idx].subject_id.nunique():,} patients)")

def metrics_at(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0), "sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0, "f1": f1_score(y_true, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def train_eval_default(feat_cols, label):
    X = df[feat_cols]
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
    prec_c, rec_c, th_c = precision_recall_curve(y_test, proba)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
    best_i = np.nanargmax(f1_c[:-1])
    best_thresh = float(th_c[best_i])
    pred_best = (proba >= best_thresh).astype(int)
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"[{label}] n_feat={len(feat_cols)}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1best={f1_score(y_test,pred_best):.4f}")
    return {
        "label": label, "n_features": len(feat_cols), "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "brier": round(brier, 4), "at_threshold_0.5": metrics_at(y_test, pred_05),
        "best_f1_threshold": best_thresh, "at_best_f1_threshold": metrics_at(y_test, pred_best),
        "top_features": importances.head(15).round(4).to_dict(),
    }

results = {}
labels = {
    "charlson_full_v1_baseline": "v1: Charlson full (17 cat + network, no enrichment)",
    "charlson_enriched_v2": "v2: + age/ICU/procedures/DRG (single lag)",
    "charlson_acci_multihop_v3": "v3: + ACCI + multi-hop (t-2,t-3) + admission-interval/frequency + cumulative ICU/procedures",
}
for key, cols in feature_sets.items():
    results[key] = train_eval_default(cols, labels[key])

with open(ROOT + "/work/05_escalation_prediction/feature_engineering_comparison.json", "w") as f:
    json.dump({
        "n_train_pairs": int(len(train_idx)), "n_test_pairs": int(len(test_idx)),
        "n_train_patients": int(df.iloc[train_idx].subject_id.nunique()),
        "n_test_patients": int(df.iloc[test_idx].subject_id.nunique()),
        "base_rate_test": round(float(y_test.mean()), 4),
        "models": results,
    }, f, indent=2)

print("\n=== Summary (default hyperparams, isolating feature-engineering effect) ===")
for key, res in results.items():
    print(f"  {res['label']:<70} AUROC={res['auroc']:.4f}  F1best={res['at_best_f1_threshold']['f1']:.4f}")
