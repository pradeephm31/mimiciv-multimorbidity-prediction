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
                              recall_score, accuracy_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

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
            "specificity": tn/(tn+fp) if (tn+fp)>0 else 0, "f1": f1_score(y_true, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def best_threshold_on(y_true, proba):
    prec, rec, th = precision_recall_curve(y_true, proba)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    i = np.nanargmax(f1[:-1])
    return float(th[i]), float(f1[i])

def evaluate(name, pv, pt, extra=None):
    thr, val_f1 = best_threshold_on(y_val, pv)
    m = metrics_at(y_test, (pt >= thr).astype(int))
    out = {"label": name, "auroc": round(float(roc_auc_score(y_test, pt)), 4),
           "auprc": round(float(average_precision_score(y_test, pt)), 4),
           "brier": round(float(brier_score_loss(y_test, pt)), 4),
           "val_selected_threshold": round(thr, 4), "val_f1_at_threshold": round(val_f1, 4),
           "test_metrics": {k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items()}}
    if extra: out.update(extra)
    print(f"[{name}] AUROC={out['auroc']:.4f} AUPRC={out['auprc']:.4f} Brier={out['brier']:.4f} "
          f"| TEST F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f}")
    return out

results = {}
BEST_PARAMS = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
i_tr, i_es = next(gss.split(X_train, y_train, g_train))
Xi_tr, yi_tr = X_train.iloc[i_tr], y_train[i_tr]
Xi_es, yi_es = X_train.iloc[i_es], y_train[i_es]

# ---------------------------------------------------------------------------
# A) LightGBM -- different algorithm, different split-finding strategy
# ---------------------------------------------------------------------------
for nl, lr_ in [(31, 0.05), (63, 0.03)]:
    m = lgb.LGBMClassifier(n_estimators=1500, num_leaves=nl, learning_rate=lr_,
                            min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0,
                            random_state=42, n_jobs=4, verbose=-1)
    m.fit(Xi_tr, yi_tr, eval_set=[(Xi_es, yi_es)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    pv, pt = m.predict_proba(X_val)[:,1], m.predict_proba(X_test)[:,1]
    results[f"lightgbm_nl{nl}"] = evaluate(f"LightGBM (num_leaves={nl}, lr={lr_})", pv, pt,
                                            extra={"best_iteration": int(m.best_iteration_)})
    np.save(ROOT + f"/work/07_imbalance_smote/proba_val_lgb_nl{nl}.npy", pv); np.save(ROOT + f"/work/07_imbalance_smote/proba_test_lgb_nl{nl}.npy", pt)

# ---------------------------------------------------------------------------
# B) Lasso as a FEATURE SELECTOR feeding XGBoost
#    (uses L1 sparsity for its intended purpose: picking a compact feature set)
# ---------------------------------------------------------------------------
sel_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(l1_ratio=1.0, C=1e-3, solver="saga", max_iter=3000,
                                                 class_weight="balanced", random_state=42, tol=1e-3))])
sel_pipe.fit(X_train, y_train)
coefs = sel_pipe.named_steps["clf"].coef_[0]
selected = [f for f, c in zip(FEAT, coefs) if abs(c) > 1e-8]
print(f"\nLasso selected {len(selected)}/{len(FEAT)} features")

m = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=1.0,
                       early_stopping_rounds=30, random_state=42, n_jobs=4, **BEST_PARAMS)
m.fit(Xi_tr[selected], yi_tr, eval_set=[(Xi_es[selected], yi_es)], verbose=False)
pv, pt = m.predict_proba(X_val[selected])[:,1], m.predict_proba(X_test[selected])[:,1]
results["lasso_selected_xgb"] = evaluate(f"XGBoost on {len(selected)} Lasso-selected features", pv, pt,
                                          extra={"n_features_used": len(selected), "selected_features": selected})

# ---------------------------------------------------------------------------
# C) Slower learning rate + more trees (finer optimization)
# ---------------------------------------------------------------------------
slow = dict(BEST_PARAMS); slow["learning_rate"] = 0.02
m = xgb.XGBClassifier(n_estimators=3000, eval_metric="aucpr", scale_pos_weight=1.0,
                       early_stopping_rounds=60, random_state=42, n_jobs=4, **slow)
m.fit(Xi_tr, yi_tr, eval_set=[(Xi_es, yi_es)], verbose=False)
pv, pt = m.predict_proba(X_val)[:,1], m.predict_proba(X_test)[:,1]
results["xgb_slow_lr"] = evaluate("XGBoost, lr=0.02 with up to 3000 trees", pv, pt,
                                   extra={"best_iteration": int(m.best_iteration)})
np.save(ROOT + "/work/07_imbalance_smote/proba_val_xgb_slow.npy", pv); np.save(ROOT + "/work/07_imbalance_smote/proba_test_xgb_slow.npy", pt)

# ---------------------------------------------------------------------------
# D) Ensemble of the two strongest DIVERSE learners (XGBoost + LightGBM)
# ---------------------------------------------------------------------------
pv_x = np.load(ROOT + "/work/07_imbalance_smote/proba_val_xgb_best.npy"); pt_x = np.load(ROOT + "/work/07_imbalance_smote/proba_test_xgb_best.npy")
pv_l = np.load(ROOT + "/work/07_imbalance_smote/proba_val_lgb_nl31.npy"); pt_l = np.load(ROOT + "/work/07_imbalance_smote/proba_test_lgb_nl31.npy")
def rank_norm(a):
    return a.argsort().argsort() / (len(a)-1)
for w in [0.5, 0.6]:
    pv_e = w*rank_norm(pv_x) + (1-w)*rank_norm(pv_l)
    pt_e = w*rank_norm(pt_x) + (1-w)*rank_norm(pt_l)
    results[f"ens_xgb_lgb_w{int(w*100)}"] = evaluate(f"Ensemble: {int(w*100)}% XGBoost + {int((1-w)*100)}% LightGBM", pv_e, pt_e)

with open(ROOT + "/work/07_imbalance_smote/f1_round2_results.json", "w") as f:
    json.dump(results, f, indent=2)

best = max(results, key=lambda k: results[k]["val_f1_at_threshold"])
print(f"\n=== Round-2 winner by validation F1: {best} -> {results[best]['label']} ===")
print(f"    TEST AUROC={results[best]['auroc']} F1={results[best]['test_metrics']['f1']}")
print("\nSaved f1_round2_results.json")
