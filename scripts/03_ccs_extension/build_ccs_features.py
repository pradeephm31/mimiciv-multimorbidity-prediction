from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")

mat = sp.load_npz(ROOT + "/work/03_ccs_extension/ccs_matrix.npz").tocsr()
adm = pd.read_parquet(ROOT + "/work/03_ccs_extension/adm_base_for_ccs.parquet").reset_index(drop=True)
cat_list = json.load(open(ROOT + "/work/03_ccs_extension/ccs_category_list.json"))["categories"]
ccs_codes = [c["ccs"] for c in cat_list]
cat_to_comm = {int(k): v for k, v in json.load(open(ROOT + "/work/03_ccs_extension/ccs_cat_to_comm.json")).items()}

COMM_NAMES = {
    0: "Neuropsychiatric, GI & substance use",
    1: "Cardiometabolic & chronic cardiopulmonary",
    2: "Renal, infection & acute decompensation",
    3: "Trauma, injury & procedural complications",
    4: "Obstetric & reproductive",
}
# communities 5-9 are singleton residual categories -> lump into "Residual/rare"
BIG_COMMS = [0, 1, 2, 3, 4]
for c in set(cat_to_comm.values()) - set(BIG_COMMS):
    COMM_NAMES[c] = "Residual/rare"

n_adm, n_cat = mat.shape
comm_ids_per_cat = np.array([cat_to_comm[c] for c in ccs_codes])

# ---------------------------------------------------------------------------
# Community activation features per admission
# ---------------------------------------------------------------------------
comm_count_cols = {}
for cid in BIG_COMMS:
    cols_idx = np.where(comm_ids_per_cat == cid)[0]
    sub = mat[:, cols_idx]
    comm_count_cols[cid] = np.asarray(sub.sum(axis=1)).flatten()

for cid in BIG_COMMS:
    adm[f"ccs_comm{cid}_count"] = comm_count_cols[cid]
    adm[f"ccs_comm{cid}_active"] = (comm_count_cols[cid] > 0).astype(int)

def dominant_ccs_comm(row):
    counts = {cid: row[f"ccs_comm{cid}_count"] for cid in BIG_COMMS}
    if max(counts.values()) == 0:
        return -1
    return max(counts, key=counts.get)

adm["dominant_ccs_community"] = adm.apply(dominant_ccs_comm, axis=1)
adm["n_ccs_categories"] = np.asarray(mat.sum(axis=1)).flatten()

# ---------------------------------------------------------------------------
# Build consecutive-admission-pair dataset -- SAME escalation label as the
# Charlson analysis (next-admission Charlson score jumps by >=2), so the two
# feature sets are compared on an identical, apples-to-apples target.
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
adm["days_since_prior_admission"] = np.where(has_prev, (adm.admittime - adm.prev_admittime).dt.days, -1)

pairs_mask = has_next.values
pairs = adm[pairs_mask].copy()
pairs["escalation_label"] = ((pairs.next_score - pairs.charlson_score) >= 2).astype(int)

print(f"Admission-pairs available: {len(pairs):,}")
print(f"Escalation rate: {pairs.escalation_label.mean():.2%}")

# Raw CCS category flags as a DataFrame aligned to `pairs`
pair_row_idx = np.where(pairs_mask)[0]
ccs_raw = pd.DataFrame(mat[pair_row_idx].toarray(), columns=[f"ccs_{c}" for c in ccs_codes], index=pairs.index)

comm_feature_cols = [f"ccs_comm{cid}_count" for cid in BIG_COMMS] + [f"ccs_comm{cid}_active" for cid in BIG_COMMS]
base_cols = ["charlson_score", "n_categories" if "n_categories" in pairs.columns else "n_charlson_categories",
             "n_ccs_categories", "admit_seq", "score_trend", "days_since_prior_admission",
             "dominant_ccs_community", "hospital_expire_flag"]
base_cols = [c for c in base_cols if c in pairs.columns]

model_df = pd.concat([
    pairs[["subject_id", "hadm_id", "escalation_label", "admission_type"] + base_cols + comm_feature_cols].reset_index(drop=True),
    ccs_raw.reset_index(drop=True),
], axis=1)

model_df.to_parquet(ROOT + "/work/03_ccs_extension/ccs_model_df.parquet")
print(f"Saved ccs_model_df.parquet: {model_df.shape}")

with open(ROOT + "/work/03_ccs_extension/ccs_comm_meta.json", "w") as f:
    json.dump({"comm_names": COMM_NAMES, "big_comms": BIG_COMMS,
               "comm_feature_cols": comm_feature_cols, "raw_ccs_cols": [f"ccs_{c}" for c in ccs_codes],
               "base_cols": base_cols}, f, indent=2)

adm.to_parquet(ROOT + "/work/03_ccs_extension/adm_with_ccs_comm.parquet")
print("Saved ccs_comm_meta.json, adm_with_ccs_comm.parquet")
