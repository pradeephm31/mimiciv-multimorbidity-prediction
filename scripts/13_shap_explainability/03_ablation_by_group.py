from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 13, script 3 (extension, no shipped counterpart): leave-one-group-out ablation of the escalation model.

SHAP shows attribution, not causation, and splits credit between correlated features. As a complementary check, refit the
general-population escalation model (Stage 06 recipe) with each feature group removed in turn and report the drop in held-out
AUROC / AUPRC. The same procedure with only the Louvain community features removed answers "do the network modules add
predictive information beyond the Charlson flags?".
Note: a single XGBoost refit varies by roughly 0.0005 AUROC between runs, so differences below about 0.001 are noise.
Input : work/04_enrichment_features/final_charlson_model_df.parquet, final_feature_sets.json, Stage 06 parameters, cat_to_comm.json
Output: work/13_shap_explainability/ablation_by_group.json
"""
import json, sys, time
import pandas as pd
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
sys.path.insert(0, ROOT + "/scripts/13_shap_explainability")
from charlson_map import CATEGORY_ORDER
import shap_common as sc

OUT = ROOT + "/work/13_shap_explainability/"
params = json.load(open(ROOT + "/work/06_hyperparameter_tuning/final_tuned_model_results.json"))["best_params"]
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
GROUPS = sc.feature_groups(CATEGORY_ORDER, sorted(set(cat_to_comm.values())))
df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
feats = json.load(open(ROOT + "/work/04_enrichment_features/final_feature_sets.json"))["charlson_acci_multihop_v3"]

def run(cols):
    model, _, te = sc.fit_escalation_general(df, cols, params)
    return sc.test_metrics(model, te, cols)

t0 = time.time()
full = run(feats)
print(f"full model ({len(feats)} features): {full}", flush=True)
results = {"full": {"n_features": len(feats), **full}}
for g in sorted(set(GROUPS[f] for f in feats)):
    keep = [f for f in feats if GROUPS[f] != g]
    r = run(keep)
    results[g] = {"n_features_removed": len(feats) - len(keep), **r,
                  "delta_auroc": round(r["auroc"] - full["auroc"], 4), "delta_auprc": round(r["auprc"] - full["auprc"], 4),
                  "delta_best_f1_on_test": round(r["best_f1_on_test"] - full["best_f1_on_test"], 4)}
    print(f"without {g:36} removed {results[g]['n_features_removed']:2d}: AUROC {r['auroc']} ({results[g]['delta_auroc']:+.4f}), AUPRC {r['auprc']} ({results[g]['delta_auprc']:+.4f})", flush=True)

# only the community features (7) on top of the 17 Charlson flags, versus the flags alone
flags = [f for f in feats if GROUPS[f] == "Charlson category flags"]
comm = [f for f in feats if GROUPS[f] == "Network communities (Louvain)"]
results["flags_only"] = {"n_features": len(flags), **run(flags)}
results["flags_plus_communities"] = {"n_features": len(flags + comm), **run(flags + comm)}
print("17 flags only:", results["flags_only"], "\n17 flags + communities:", results["flags_plus_communities"])
json.dump(results, open(OUT + "ablation_by_group.json", "w"), indent=2)
print(f"\nSaved ablation_by_group.json ({time.time() - t0:.0f}s)")
