from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")

dx = pd.read_parquet(ROOT + "/work/03_ccs_extension/dx_with_ccs.parquet")
cats = pd.read_parquet(ROOT + "/reference/ccs/ccs_categories_raw.parquet")
cat_desc = cats.drop_duplicates("category_code").set_index("category_code")["category_desc"]

adm_base = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet")[
    ["subject_id", "hadm_id", "admittime", "admit_seq", "n_admissions",
     "admission_type", "hospital_expire_flag", "charlson_score", "n_categories"]
].rename(columns={"charlson_score": "charlson_score", "n_categories": "n_charlson_categories"})
adm_base = adm_base.sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Choose CCS categories with enough support to be usable as network nodes /
# model features (>=500 admissions containing that category).
# ---------------------------------------------------------------------------
dx_ccs = dx.dropna(subset=["ccs"]).copy()
dx_ccs["ccs"] = dx_ccs["ccs"].astype(int)
dx_ccs = dx_ccs[dx_ccs.hadm_id.isin(adm_base.hadm_id)]

prevalence = dx_ccs.groupby("ccs")["hadm_id"].nunique().sort_values(ascending=False)
KEEP_CATS = prevalence[prevalence >= 500].index.tolist()
print(f"CCS categories kept (>=500 admissions): {len(KEEP_CATS)} of {len(prevalence)} observed "
      f"({dx_ccs.ccs.isin(KEEP_CATS).sum() / len(dx_ccs):.1%} of mapped rows retained)")

cat_to_idx = {c: i for i, c in enumerate(KEEP_CATS)}
n_cat = len(KEEP_CATS)

hadm_ids = adm_base.hadm_id.values
hadm_to_row = {h: i for i, h in enumerate(hadm_ids)}
n_adm = len(hadm_ids)

dx_keep = dx_ccs[dx_ccs.ccs.isin(KEEP_CATS)].drop_duplicates(["hadm_id", "ccs"])
rows = dx_keep.hadm_id.map(hadm_to_row).values
cols = dx_keep.ccs.map(cat_to_idx).values
data = np.ones(len(dx_keep), dtype=np.int8)
mat_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_adm, n_cat), dtype=np.int8)

print(f"CCS admission x category matrix: {mat_sparse.shape}, nnz={mat_sparse.nnz:,}")

sp.save_npz(ROOT + "/work/03_ccs_extension/ccs_matrix.npz", mat_sparse)
adm_base.to_parquet(ROOT + "/work/03_ccs_extension/adm_base_for_ccs.parquet")
with open(ROOT + "/work/03_ccs_extension/ccs_category_list.json", "w") as f:
    json.dump({
        "categories": [{"ccs": int(c), "desc": cat_desc.get(c, f"CCS {c}"),
                         "prevalence_admissions": int(prevalence[c])} for c in KEEP_CATS],
    }, f, indent=2)
print("Saved ccs_matrix.npz, adm_base_for_ccs.parquet, ccs_category_list.json")

# Overall mapping-coverage comparison numbers for the report
coverage = {
    "charlson_mapped_row_fraction": 0.1451,
    "ccs_mapped_row_fraction": round(float(dx["ccs"].notna().mean()), 4),
    "charlson_n_categories": 17,
    "ccs_n_categories_in_scheme": 308,
    "ccs_n_categories_observed": int(dx["ccs"].nunique()),
    "ccs_n_categories_used_500plus": n_cat,
}
with open(ROOT + "/work/03_ccs_extension/coverage_comparison.json", "w") as f:
    json.dump(coverage, f, indent=2)
print(json.dumps(coverage, indent=2))
