from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix, roc_curve)
import xgboost as xgb

feature_sets = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))
FEAT = feature_sets["charlson_acci_multihop_v3"]

tr = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_train.parquet")
va = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_val.parquet")
te = pd.read_parquet(ROOT + "/work/07_imbalance_smote/split_test.parquet")

X_train, y_train = tr[FEAT], tr["escalation_label"].values
X_val, y_val = va[FEAT], va["escalation_label"].values
X_test, y_test = te[FEAT], te["escalation_label"].values
g_train = tr["subject_id"].values

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
    out = {"label": name,
           "auroc": round(float(roc_auc_score(y_test, pt)), 4),
           "auprc": round(float(average_precision_score(y_test, pt)), 4),
           "brier": round(float(brier_score_loss(y_test, pt)), 4),
           "val_selected_threshold": round(thr, 4), "val_f1_at_threshold": round(val_f1, 4),
           "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
    if extra: out.update(extra)
    print(f"[{name}] AUROC={out['auroc']:.4f} AUPRC={out['auprc']:.4f} Brier={out['brier']:.4f} "
          f"| thr={thr:.3f} -> TEST F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f}")
    return out

results = {}
BEST_PARAMS = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
print(f"Using tuned params: {BEST_PARAMS}\n")

# Inner early-stopping carve-out from TRAIN only (grouped)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
i_tr, i_es = next(gss.split(X_train, y_train, g_train))
Xi_tr, yi_tr = X_train.iloc[i_tr], y_train[i_tr]
Xi_es, yi_es = X_train.iloc[i_es], y_train[i_es]

# ---------------------------------------------------------------------------
# 1) scale_pos_weight sweep. spw=neg/pos (~6.7) maximizes recall; lower values
#    trade recall for precision and often give better F1.
# ---------------------------------------------------------------------------
pos = yi_tr.sum(); neg = len(yi_tr) - pos
spw_full = neg / pos
print(f"Full inverse-ratio scale_pos_weight = {spw_full:.2f}\n")

xgb_probas = {}
for spw_label, spw in [("1.0 (no reweighting)", 1.0), ("2.0", 2.0), ("3.0", 3.0),
                        ("4.5", 4.5), (f"{spw_full:.2f} (full inverse ratio)", spw_full)]:
    model = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=spw,
                               early_stopping_rounds=30, random_state=42, n_jobs=4, **BEST_PARAMS)
    model.fit(Xi_tr, yi_tr, eval_set=[(Xi_es, yi_es)], verbose=False)
    pv = model.predict_proba(X_val)[:, 1]
    pt = model.predict_proba(X_test)[:, 1]
    key = f"xgb_spw_{spw:.2f}".replace(".", "p")
    results[key] = evaluate(f"XGBoost, scale_pos_weight={spw_label}", pv, pt,
                             extra={"scale_pos_weight": round(float(spw), 3),
                                    "best_iteration": int(model.best_iteration)})
    xgb_probas[key] = (pv, pt)

# Pick best XGB variant by VALIDATION F1 (not test)
best_xgb_key = max(xgb_probas, key=lambda k: results[k]["val_f1_at_threshold"])
print(f"\nBest XGBoost variant by validation F1: {best_xgb_key}")
pv_xgb, pt_xgb = xgb_probas[best_xgb_key]
np.save(ROOT + "/work/07_imbalance_smote/proba_val_xgb_best.npy", pv_xgb)
np.save(ROOT + "/work/07_imbalance_smote/proba_test_xgb_best.npy", pt_xgb)

# ---------------------------------------------------------------------------
# 2) Calibration (isotonic, fit on VALIDATION) -- improves Brier and makes
#    threshold selection meaningful. Ranking (AUROC) is unchanged by a
#    monotonic transform; F1 can shift because the threshold grid changes.
# ---------------------------------------------------------------------------
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(pv_xgb, y_val)
pv_cal = iso.predict(pv_xgb)
pt_cal = iso.predict(pt_xgb)
results["xgb_isotonic_calibrated"] = evaluate("XGBoost (best) + isotonic calibration", pv_cal, pt_cal)

# ---------------------------------------------------------------------------
# 3) Ensemble: average the rank-normalized XGBoost and best linear model
# ---------------------------------------------------------------------------
lin_results = json.load(open(ROOT + "/work/07_imbalance_smote/linear_model_results.json"))
best_lin_key = max(lin_results, key=lambda k: lin_results[k]["val_f1_at_threshold"])
print(f"\nBest linear model by validation F1: {best_lin_key} ({lin_results[best_lin_key]['label']})")
pv_lin = np.load(ROOT + f"/work/07_imbalance_smote/proba_val_{best_lin_key}.npy")
pt_lin = np.load(ROOT + f"/work/07_imbalance_smote/proba_test_{best_lin_key}.npy")

def rank_norm(a):
    order = a.argsort().argsort()
    return order / (len(a) - 1)

for w in [0.5, 0.7, 0.85]:
    pv_ens = w * rank_norm(pv_xgb) + (1 - w) * rank_norm(pv_lin)
    pt_ens = w * rank_norm(pt_xgb) + (1 - w) * rank_norm(pt_lin)
    results[f"ensemble_w{int(w*100)}"] = evaluate(
        f"Ensemble: {int(w*100)}% XGBoost + {int((1-w)*100)}% {lin_results[best_lin_key]['label'].split(',')[0]} (rank-averaged)",
        pv_ens, pt_ens)

# ---------------------------------------------------------------------------
# Save everything, including ROC for the final winner
# ---------------------------------------------------------------------------
all_results = dict(lin_results)
all_results.update(results)

final_key = max(results, key=lambda k: results[k]["val_f1_at_threshold"])
print(f"\n=== WINNER by validation F1: {final_key} -> {results[final_key]['label']} ===")
print(f"    TEST: AUROC={results[final_key]['auroc']}  F1={results[final_key]['test_metrics']['f1']}")

if final_key in xgb_probas:
    pt_final = xgb_probas[final_key][1]
elif final_key == "xgb_isotonic_calibrated":
    pt_final = pt_cal
else:
    w = int(final_key.replace("ensemble_w", "")) / 100
    pt_final = w * rank_norm(pt_xgb) + (1 - w) * rank_norm(pt_lin)

fpr, tpr, _ = roc_curve(y_test, pt_final)
prec_c, rec_c, _ = precision_recall_curve(y_test, pt_final)
def ds(x, n=60):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return np.asarray(x)[idx].tolist()

with open(ROOT + "/work/07_imbalance_smote/f1_improvement_results.json", "w") as f:
    json.dump({
        "split_info": json.load(open(ROOT + "/work/07_imbalance_smote/split_info.json")),
        "tuned_params": BEST_PARAMS,
        "full_inverse_ratio_spw": round(float(spw_full), 3),
        "models": all_results,
        "winner_key": final_key,
        "roc_curve": {"fpr": ds(fpr), "tpr": ds(tpr)},
        "pr_curve": {"precision": ds(prec_c), "recall": ds(rec_c)},
    }, f, indent=2)
print("\nSaved f1_improvement_results.json")
