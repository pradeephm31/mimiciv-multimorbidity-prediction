from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
COMM_IDS = sorted(set(cat_to_comm.values()))
for cid in COMM_IDS:
    cats_in_comm = [c for c in CATEGORY_ORDER if cat_to_comm[c] == cid]
    adm[f"comm{cid}_count"] = adm[cats_in_comm].sum(axis=1)
    adm[f"comm{cid}_active"] = (adm[f"comm{cid}_count"] > 0).astype(int)

def dominant_comm(row):
    counts = {cid: row[f"comm{cid}_count"] for cid in COMM_IDS}
    if max(counts.values()) == 0:
        return -1
    return max(counts, key=counts.get)
adm["dominant_community"] = adm.apply(dominant_comm, axis=1)

# ---------------------------------------------------------------------------
# Build consecutive-admission-pair dataset (same escalation target as before)
# ---------------------------------------------------------------------------
adm["next_subject"] = adm.subject_id.shift(-1)
adm["next_score"] = adm.charlson_score.shift(-1)
has_next = adm.next_subject == adm.subject_id
pairs = adm[has_next].copy()
pairs["escalation_label"] = ((pairs.next_score - pairs.charlson_score) >= 2).astype(int)
pairs["score_trend"] = pairs["charlson_score"] - pairs["charlson_score_lag1"].fillna(pairs["charlson_score"])

print(f"Admission-pairs: {len(pairs):,}, escalation rate: {pairs.escalation_label.mean():.2%}")

FEATURE_SETS = {
    "charlson_full_v1_baseline": (
        CATEGORY_ORDER + [f"comm{c}_count" for c in COMM_IDS] + [f"comm{c}_active" for c in COMM_IDS]
        + ["charlson_score", "n_categories", "admit_seq", "score_trend",
           "days_since_prior_admission", "dominant_community", "hospital_expire_flag"]
    ),
    "charlson_enriched_v2": (
        CATEGORY_ORDER + [f"comm{c}_count" for c in COMM_IDS] + [f"comm{c}_active" for c in COMM_IDS]
        + ["charlson_score", "n_categories", "admit_seq", "score_trend",
           "days_since_prior_admission", "dominant_community", "hospital_expire_flag",
           "age_at_admission", "gender_male", "n_icu_stays", "icu_los_days", "had_icu_stay",
           "n_procedures", "had_procedure", "n_distinct_procedure_categories",
           "drg_severity", "drg_mortality", "has_apr_drg"]
    ),
    "charlson_acci_multihop_v3": (
        CATEGORY_ORDER + [f"comm{c}_count" for c in COMM_IDS] + [f"comm{c}_active" for c in COMM_IDS]
        + ["charlson_score", "acci_score", "age_points", "n_categories", "admit_seq", "score_trend",
           "dominant_community", "hospital_expire_flag",
           "age_at_admission", "gender_male", "n_icu_stays", "icu_los_days", "had_icu_stay",
           "had_icu_prev_admission", "cum_icu_stays", "cum_icu_los_days",
           "n_procedures", "had_procedure", "n_distinct_procedure_categories", "cum_procedures",
           "drg_severity", "drg_mortality", "has_apr_drg",
           "days_since_prior_admission", "log_days_since_prior", "admissions_per_year_so_far",
           "charlson_score_lag1", "charlson_score_lag2", "charlson_score_lag3",
           "n_categories_lag1", "n_categories_lag2", "n_categories_lag3",
           "charlson_trend_slope_3adm", "n_prior_admissions_available"]
    ),
}

for name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in pairs.columns]
    if missing:
        print(f"WARNING [{name}] missing columns: {missing}")

model_df = pairs[["subject_id", "hadm_id", "escalation_label"] + sorted(set(sum(FEATURE_SETS.values(), [])))].copy()
model_df.to_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
with open(ROOT + "/work/04_enrichment_features/final_feature_sets.json", "w") as f:
    json.dump(FEATURE_SETS, f, indent=2)
print(f"\nSaved final_charlson_model_df.parquet {model_df.shape}, final_feature_sets.json")
