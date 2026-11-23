from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
import xgboost as xgb

feature_sets = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))
FEAT = feature_sets["charlson_acci_multihop_v3"]
tr = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_train.parquet")
va = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_val.parquet")
te = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_test.parquet")
X_train_raw, y_train = tr[FEAT].copy(), tr["escalation_label"].values
X_val_raw, y_val = va[FEAT].copy(), va["escalation_label"].values
X_test_raw, y_test = te[FEAT].copy(), te["escalation_label"].values

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=FEAT, index=X_train_raw.index)
X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=FEAT, index=X_val_raw.index)
X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=FEAT, index=X_test_raw.index)

BINARY_CATEGORICAL = [
    c for c in FEAT if c in [
        "Myocardial infarction", "Congestive heart failure", "Peripheral vascular disease",
        "Cerebrovascular disease", "Dementia", "Chronic pulmonary disease", "Rheumatic disease",
        "Peptic ulcer disease", "Mild liver disease", "Diabetes without complication",
        "Diabetes with complication", "Hemiplegia or paraplegia", "Renal disease", "Malignancy",
        "Moderate or severe liver disease", "Metastatic solid tumor", "AIDS/HIV",
        "comm0_active", "comm1_active", "comm2_active", "gender_male", "had_icu_stay",
        "had_icu_prev_admission", "had_procedure", "has_apr_drg", "hospital_expire_flag",
        "dominant_community",
    ]
]
cat_idx = [FEAT.index(c) for c in BINARY_CATEGORICAL]
BEST_PARAMS = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]

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

def evaluate(name, pv, pt, extra=None):
    thr, val_f1 = best_threshold_on(y_val, pv)
    m = metrics_at(y_test, (pt >= thr).astype(int))
    out = {"label": name, "auroc": round(float(roc_auc_score(y_test, pt)), 4),
           "auprc": round(float(average_precision_score(y_test, pt)), 4),
           "brier": round(float(brier_score_loss(y_test, pt)), 4),
           "val_selected_threshold": round(thr, 4), "val_f1_at_threshold": round(val_f1, 4),
           "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
    if extra:
        out.update(extra)
    print(f"[{name}] AUROC={out['auroc']:.4f} AUPRC={out['auprc']:.4f} Brier={out['brier']:.4f} "
          f"| TEST F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f}")
    return out

results = {}

# Finish SMOTENC ratio=1.0
sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=1.0, random_state=42, k_neighbors=5)
Xr, yr = sm.fit_resample(X_train, y_train)
m = xgb.XGBClassifier(n_estimators=500, eval_metric="aucpr", scale_pos_weight=1.0,
                       random_state=42, n_jobs=4, **BEST_PARAMS)
m.fit(Xr, yr)
pv, pt = m.predict_proba(X_val)[:, 1], m.predict_proba(X_test)[:, 1]
results["smotenc_ratio100"] = evaluate(f"SMOTENC (categorical-aware), sampling_strategy=1.0 (n={len(Xr):,})", pv, pt,
                                        extra={"n_resampled": int(len(Xr)), "n_positive_after": int(yr.sum())})

# ---------------------------------------------------------------------------
# Hybrid: SMOTENC (moderate ratio) + undersample the majority class
# ---------------------------------------------------------------------------
sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=0.3, random_state=42, k_neighbors=5)
Xr, yr = sm.fit_resample(X_train, y_train)
tomek = TomekLinks()
Xr2, yr2 = tomek.fit_resample(Xr, yr)
m = xgb.XGBClassifier(n_estimators=500, eval_metric="aucpr", scale_pos_weight=1.0,
                       random_state=42, n_jobs=4, **BEST_PARAMS)
m.fit(Xr2, yr2)
pv, pt = m.predict_proba(X_val)[:, 1], m.predict_proba(X_test)[:, 1]
results["smotenc_tomek"] = evaluate(f"SMOTENC(0.3) + Tomek-links cleaning (n={len(Xr2):,})", pv, pt,
                                     extra={"n_resampled": int(len(Xr2)), "n_positive_after": int(yr2.sum())})

with open(ROOT + "/work/07_imbalance_smote/smote_results_partial2.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved smote_results_partial2.json (part 2)")
