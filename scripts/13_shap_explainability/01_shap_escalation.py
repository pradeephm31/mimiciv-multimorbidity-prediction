from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 13, script 1 (extension, no shipped counterpart): SHAP for the binary escalation model.

Explains the tuned v3 XGBoost escalation model (Stage 06 recipe) for the general population and the heart-failure
escalation model (Stage 08 recipe). The final models were never saved, so they are refit with the same splits and
parameters; the refit test AUROC is compared with the earlier stage as a check.

SHAP values come from shap.TreeExplainer on a random sample of up to 5,000 HELD-OUT test rows (log-odds scale).
Outputs per cohort: mean |SHAP| per feature and per feature group, rank stability over bootstrap resamples, agreement
with XGBoost's built-in importance, beeswarm / group-bar / dependence / waterfall PNGs and an interactive force plot fragment.
Input : work/04_enrichment_features/final_charlson_model_df.parquet, final_feature_sets.json,
        work/08_hf_cohort_liang_replication/hf_model_df.parquet, hf_feature_list.json,
        work/06_hyperparameter_tuning/final_tuned_model_results.json, work/02_network_construction/cat_to_comm.json
Output: work/13_shap_explainability/escalation_<cohort>_shap.json, figures/escalation_<cohort>_*.png, force_escalation_<cohort>.html
"""
import json, sys, time
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/13_shap_explainability")
from charlson_map import CATEGORY_ORDER
import shap_common as sc

OUT = ROOT + "/work/13_shap_explainability/"
FIG = OUT + "figures/"
params = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
GROUPS = sc.feature_groups(CATEGORY_ORDER, sorted(set(cat_to_comm.values())))
N_EXPLAIN, N_FORCE = 5000, 500

ref_general = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))
ref_hf = json.load(open(ROOT + "/work/08_hf_cohort_liang_replication/hf_model_results.json"))

cohorts = {}
df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
feats = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))["charlson_acci_multihop_v3"]
cohorts["general_population"] = (df, feats, sc.fit_escalation_general, ref_general["auroc"], "Stage 06")
df = pd.read_parquet(ROOT + "/work/08_hf_cohort_liang_replication/hf_model_df.parquet")
feats = json.load(open(ROOT + "/work/08_hf_cohort_liang_replication/hf_feature_list.json"))
cohorts["heart_failure"] = (df, feats, sc.fit_escalation_hf, ref_hf["auroc"], "Stage 08")

for cname, (df, feats, fit, ref_auroc, ref_stage) in cohorts.items():
    t0 = time.time()
    missing = [f for f in feats if f not in GROUPS]
    assert not missing, f"features without a group: {missing}"
    model, tr, te = fit(df, feats, params)
    metrics = sc.test_metrics(model, te, feats)
    print(f"\n=== {cname}: refit test AUROC {metrics['auroc']} ({ref_stage}: {ref_auroc}); AUPRC {metrics['auprc']}; best_iteration {model.best_iteration}", flush=True)

    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(te), size=min(N_EXPLAIN, len(te)), replace=False))
    rows = te.iloc[idx].reset_index(drop=True)
    X, y = rows[feats], rows["escalation_label"].values
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    vals = sv.values
    proba = model.predict_proba(X)[:, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    table = [{"feature": feats[i], "group": GROUPS[feats[i]], "mean_abs_shap": round(float(mean_abs[i]), 5),
              "mean_shap": round(float(vals[:, i].mean()), 5)} for i in order[:25]]
    gimp = sc.group_importance(vals, feats, GROUPS)
    builtin = model.feature_importances_
    rho = float(spearmanr(mean_abs, builtin)[0])
    stab = sc.stability(vals)
    print("  top 8:", [t["feature"] for t in table[:8]])
    print("  group shares:", {k: v["share"] for k, v in gimp.items()})
    print(f"  Spearman(SHAP importance, XGBoost built-in importance) = {rho:.3f}; bootstrap stability {stab}")

    # figures
    sc.beeswarm(sv, FIG + f"escalation_{cname}_beeswarm.png")
    sc.bar_groups(gimp, FIG + f"escalation_{cname}_groups.png", f"{cname.replace('_', ' ')}: SHAP importance by feature group")
    dep = [feats[i] for i in order[:6] if GROUPS[feats[i]] != "Charlson category flags"][:4]
    for f in dep:
        sc.dependence(sv, f, FIG + f"escalation_{cname}_dep_{f}.png")

    # local explanations: highest-risk true positive, lowest-risk true positive (missed case), lowest-risk true negative
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    cases = {"highest_risk_escalation": int(pos[np.argmax(proba[pos])]),
             "missed_escalation_lowest_risk": int(pos[np.argmin(proba[pos])]),
             "lowest_risk_no_escalation": int(neg[np.argmin(proba[neg])])}
    local = {}
    for name, i in cases.items():
        sc.waterfall(sv[i], FIG + f"escalation_{cname}_case_{name}.png")
        local[name] = {"subject_id": int(rows.loc[i, "subject_id"]), "hadm_id": int(rows.loc[i, "hadm_id"]),
                       "predicted_probability": round(float(proba[i]), 4), "escalation_label": int(y[i])}
    sc.force_fragment(explainer.expected_value, vals[:N_FORCE], X.iloc[:N_FORCE], OUT + f"force_escalation_{cname}.html")

    json.dump({"cohort": cname, "n_features": len(feats), "n_explained_rows": int(len(rows)),
               "expected_value_logodds": round(float(np.ravel(explainer.expected_value)[0]), 4),
               "refit_metrics": metrics, "reference_auroc": ref_auroc, "reference_stage": ref_stage,
               "best_iteration": int(model.best_iteration), "top_features": table, "group_importance": gimp,
               "spearman_with_builtin_importance": round(rho, 4), "bootstrap_stability": stab,
               "dependence_features": dep, "local_cases": local, "n_force_rows": N_FORCE},
              open(OUT + f"escalation_{cname}_shap.json", "w"), indent=2)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
print("\nSaved escalation SHAP outputs")
