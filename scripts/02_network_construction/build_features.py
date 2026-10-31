from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
COMM_IDS = sorted(set(cat_to_comm.values()))
COMM_NAMES = {0: "Chronic-systemic/oncologic", 1: "Cardiorenal-metabolic", 2: "Neuro-frailty"}

# ---------------------------------------------------------------------------
# Per-admission community activation features (the "network" projection
# of each admission's raw 17-category profile onto the 3 Louvain communities)
# ---------------------------------------------------------------------------
for cid in COMM_IDS:
    cats_in_comm = [c for c in CATEGORY_ORDER if cat_to_comm[c] == cid]
    adm[f"comm{cid}_count"] = adm[cats_in_comm].sum(axis=1)
    adm[f"comm{cid}_active"] = (adm[f"comm{cid}_count"] > 0).astype(int)

# Dominant community per admission (for the simplified "network trajectory" view)
def dominant_comm(row):
    counts = {cid: row[f"comm{cid}_count"] for cid in COMM_IDS}
    if max(counts.values()) == 0:
        return -1  # no mapped comorbidity this admission
    return max(counts, key=counts.get)

adm["dominant_community"] = adm.apply(dominant_comm, axis=1)

# ---------------------------------------------------------------------------
# Build consecutive-admission-pair dataset: features from admission t,
# label = whether Charlson score jumps by >=2 at admission t+1 (same patient)
# ---------------------------------------------------------------------------
adm = adm.sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
adm["next_subject"] = adm.subject_id.shift(-1)
adm["next_score"] = adm.charlson_score.shift(-1)
adm["next_admittime"] = adm.admittime.shift(-1)
adm["prev_subject"] = adm.subject_id.shift(1)
adm["prev_score"] = adm.charlson_score.shift(1)
adm["prev_admittime"] = adm.admittime.shift(1)

has_next = adm.next_subject == adm.subject_id
has_prev = adm.prev_subject == adm.subject_id

adm["score_trend"] = np.where(has_prev, adm.charlson_score - adm.prev_score, 0)
adm["days_since_prior_admission"] = np.where(
    has_prev, (adm.admittime - adm.prev_admittime).dt.days, -1
)

pairs = adm[has_next].copy()
pairs["escalation_label"] = ((pairs.next_score - pairs.charlson_score) >= 2).astype(int)
pairs["days_to_next_admission"] = (pairs.next_admittime - pairs.admittime).dt.days

print(f"Admission-pairs available for modeling: {len(pairs):,}")
print(f"Escalation rate (next-admission Charlson +2 or more): {pairs.escalation_label.mean():.2%}")
print(f"Unique patients contributing pairs: {pairs.subject_id.nunique():,}")

feature_cols = (
    CATEGORY_ORDER
    + [f"comm{cid}_count" for cid in COMM_IDS]
    + [f"comm{cid}_active" for cid in COMM_IDS]
    + ["charlson_score", "n_categories", "admit_seq", "score_trend",
       "days_since_prior_admission", "dominant_community", "hospital_expire_flag"]
)
cat_feature = "admission_type"

model_df = pairs[["subject_id", "hadm_id", "escalation_label"] + feature_cols + [cat_feature]].copy()
model_df.to_parquet(ROOT + "/work/02_network_construction/model_df.parquet")
print(f"\nSaved model_df.parquet with {len(model_df):,} rows, {len(feature_cols)+1} features")
print(model_df[feature_cols[-7:] + [cat_feature]].describe(include="all").T)

adm.to_parquet(ROOT + "/work/02_network_construction/adm_features_with_comm.parquet")
print("\nSaved adm_features_with_comm.parquet")
