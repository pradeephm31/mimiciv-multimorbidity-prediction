from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

adm_all = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
COMM_IDS = sorted(set(cat_to_comm.values()))

for cid in COMM_IDS:
    cats_in_comm = [c for c in CATEGORY_ORDER if cat_to_comm[c] == cid]
    adm_all[f"comm{cid}_count"] = adm_all[cats_in_comm].sum(axis=1)
    adm_all[f"comm{cid}_active"] = (adm_all[f"comm{cid}_count"] > 0).astype(int)

def dominant_comm(row):
    counts = {cid: row[f"comm{cid}_count"] for cid in COMM_IDS}
    if max(counts.values()) == 0:
        return -1
    return max(counts, key=counts.get)
adm_all["dominant_community"] = adm_all.apply(dominant_comm, axis=1)

adm_all["next_subject"] = adm_all.subject_id.shift(-1)
adm_all["next_score"] = adm_all.charlson_score.shift(-1)
# NEW: capture the NEXT admission's full 17-category diagnosis set as the
# multi-label target (rather than just the binary escalation flag)
for c in CATEGORY_ORDER:
    adm_all[f"next__{c}"] = adm_all[c].shift(-1)

has_next = adm_all.next_subject == adm_all.subject_id
pairs = adm_all[has_next].copy()
pairs["escalation_label"] = ((pairs.next_score - pairs.charlson_score) >= 2).astype(int)
pairs["score_trend"] = pairs["charlson_score"] - pairs["charlson_score_lag1"].fillna(pairs["charlson_score"])

LABEL_COLS = [f"next__{c}" for c in CATEGORY_ORDER]
for c in LABEL_COLS:
    pairs[c] = pairs[c].astype(int)

print(f"General-population admission-pairs: {len(pairs):,}")
print(f"Avg # categories present at NEXT visit: {pairs[LABEL_COLS].sum(axis=1).mean():.2f}")
print(f"Avg # categories persisting from current visit: "
      f"{(pairs[CATEGORY_ORDER].values.astype(bool) & pairs[LABEL_COLS].values.astype(bool)).sum(axis=1).mean():.2f}")

FEAT_V3 = (
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
)

model_df = pairs[["subject_id", "hadm_id", "escalation_label"] + FEAT_V3 + LABEL_COLS].copy()
model_df.to_parquet(ROOT + "/work/10_multilabel_general_population/general_multilabel_df.parquet")
with open(ROOT + "/work/10_multilabel_general_population/general_multilabel_meta.json", "w") as f:
    json.dump({"features": FEAT_V3, "label_cols": LABEL_COLS, "categories": CATEGORY_ORDER}, f)
print(f"\nSaved general_multilabel_df.parquet: {model_df.shape}")
