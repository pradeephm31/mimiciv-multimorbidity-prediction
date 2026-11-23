from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 13, script 2 (extension, no shipped counterpart): SHAP for the multi-label next-visit diagnosis model, heart-failure cohort.

Refits the 17 per-category XGBoost classifiers of Stage 09 (same cohort, split, features and parameters; the models were
never saved) and explains each with shap.TreeExplainer on up to 5,000 held-out test pairs (log-odds scale).
Outputs: for every category the top features and the importance by feature group, a category x group heatmap, and beeswarm
plots for the three most prevalent next-visit categories.
Input : work/04_enrichment_features/adm_full_enriched_multihop.parquet, work/02_network_construction/cat_to_comm.json,
        work/06_hyperparameter_tuning/final_tuned_model_results.json, scripts/11_generalisation_experiments/multilabel_lib.py
Output: work/13_shap_explainability/multilabel_hf_shap.json, figures/multilabel_hf_*.png
"""
import json, sys, time
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/11_generalisation_experiments")
sys.path.insert(0, ROOT + "/scripts/13_shap_explainability")
from charlson_map import CATEGORY_ORDER
import multilabel_lib as ml
import shap_common as sc

OUT = ROOT + "/work/13_shap_explainability/"
FIG = OUT + "figures/"
adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet")
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
params = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
GROUPS = sc.feature_groups(CATEGORY_ORDER, sorted(set(cat_to_comm.values())))

# heart-failure cohort exactly as in Stage 08: congestive heart failure on any admission, 2 or more admissions in total
n_adm = adm.groupby("subject_id")["hadm_id"].transform("count")
hf = set(adm.loc[adm["Congestive heart failure"] == 1, "subject_id"]) & set(adm.loc[n_adm >= 2, "subject_id"])
pairs, feats, hist_cols, label_cols = ml.make_pairs(adm, cat_to_comm, CATEGORY_ORDER, patient_mask=hf)
assert all(f in GROUPS for f in feats)
tr, va, te = ml.split_pairs(pairs)
print(f"HF cohort: {len(hf):,} patients; train {len(tr):,} / val {len(va):,} / test {len(te):,} pairs", flush=True)

rng = np.random.RandomState(0)
idx = np.sort(rng.choice(len(te), size=min(5000, len(te)), replace=False))
Xs = te.iloc[idx][feats].reset_index(drop=True)
Ys = te.iloc[idx][label_cols].values

results, shap_store, heat = {}, {}, []
t0 = time.time()
for i, (cat, lab) in enumerate(zip(CATEGORY_ORDER, label_cols)):
    y_tr, y_va = tr[lab].values, va[lab].values
    pos = y_tr.sum()
    if pos < 5 or pos > len(y_tr) - 5:
        print(f"  [{i + 1}/17] {cat}: near-constant label, skipped"); continue
    m = xgb.XGBClassifier(n_estimators=400, eval_metric="aucpr", scale_pos_weight=1.0, early_stopping_rounds=25,
                          random_state=42, n_jobs=4, **params)
    m.fit(tr[feats], y_tr, eval_set=[(va[feats], y_va)], verbose=False)
    sv = shap.TreeExplainer(m)(Xs)
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    gimp = sc.group_importance(sv.values, feats, GROUPS)
    auroc = float(roc_auc_score(te[lab].values, m.predict_proba(te[feats])[:, 1]))
    results[cat] = {"prevalence_next_visit": round(float(te[lab].mean()), 4), "test_auroc": round(auroc, 4),
                    "top_features": [{"feature": feats[j], "group": GROUPS[feats[j]], "mean_abs_shap": round(float(mean_abs[j]), 5)} for j in order[:8]],
                    "group_importance": gimp}
    shap_store[cat] = sv
    print(f"  [{i + 1}/17] {cat}: prevalence {results[cat]['prevalence_next_visit']:.3f}, AUROC {auroc:.3f}, top: {[feats[j] for j in order[:4]]}", flush=True)

# heatmap: category x feature group (share of mean |SHAP|)
group_names = list(next(iter(results.values()))["group_importance"].keys())
group_names = sorted(group_names)
mat = np.array([[results[c]["group_importance"].get(g, {"share": 0})["share"] * 100 for g in group_names] for c in results])
fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(results) + 2.2))
im = ax.imshow(mat, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(group_names))); ax.set_xticklabels(group_names, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(results))); ax.set_yticklabels(list(results.keys()), fontsize=8)
for r in range(mat.shape[0]):
    for c in range(mat.shape[1]):
        ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", fontsize=7, color="white" if mat[r, c] > 45 else "black")
ax.set_title("Heart-failure multi-label models: share of SHAP importance by feature group (%)", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.6)
fig.savefig(FIG + "multilabel_hf_group_heatmap.png", dpi=130, bbox_inches="tight"); plt.close("all")

# beeswarms for the three most prevalent next-visit categories
top3 = sorted(results, key=lambda c: -results[c]["prevalence_next_visit"])[:3]
for c in top3:
    sc.beeswarm(shap_store[c], FIG + f"multilabel_hf_beeswarm_{c.replace(' ', '_').replace('/', '_')}.png", max_display=15)

# how much does the category's own current flag matter (the persistence signal)?
own = {c: round(float(np.abs(shap_store[c][:, c].values).mean()), 5) for c in results}
for c in results:
    results[c]["own_current_flag_mean_abs_shap"] = own[c]
    results[c]["own_flag_rank"] = int(1 + (np.abs(shap_store[c].values).mean(axis=0) > np.abs(shap_store[c][:, c].values).mean()).sum())

json.dump({"n_patients": len(hf), "n_test_pairs": int(len(te)), "n_explained_rows": int(len(Xs)), "beeswarm_categories": top3,
           "categories": results}, open(OUT + "multilabel_hf_shap.json", "w"), indent=2)
print(f"\nSaved multilabel_hf_shap.json ({time.time() - t0:.0f}s)")
