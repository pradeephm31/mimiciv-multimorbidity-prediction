from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import map_code_to_category, CHARLSON_WEIGHTS, CATEGORY_ORDER

pd.set_option("display.width", 140)

print("Loading admissions.csv ...")
adm = pd.read_csv(
    ROOT + "/raw/admissions.csv",
    usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
             "admission_type", "hospital_expire_flag"],
    parse_dates=["admittime", "dischtime", "deathtime"],
)
print(f"  admissions rows: {len(adm):,}, unique patients: {adm.subject_id.nunique():,}")

print("Loading diagnoses_icd.csv ...")
dx = pd.read_csv(
    ROOT + "/raw/diagnoses_icd.csv",
    dtype={"subject_id": "int64", "hadm_id": "int64", "seq_num": "float32",
           "icd_code": "string", "icd_version": "int8"},
)
print(f"  diagnosis rows: {len(dx):,}, unique admissions: {dx.hadm_id.nunique():,}, "
      f"unique patients: {dx.subject_id.nunique():,}")

# ---------------------------------------------------------------------------
# Map each diagnosis code to a Charlson category
# ---------------------------------------------------------------------------
print("Mapping ICD codes to Charlson categories ...")
unique_codes = dx[["icd_code", "icd_version"]].drop_duplicates()
print(f"  unique (code, version) pairs to map: {len(unique_codes):,}")
unique_codes["charlson_category"] = [
    map_code_to_category(c, v) for c, v in zip(unique_codes.icd_code, unique_codes.icd_version)
]
code_map = unique_codes.set_index(["icd_code", "icd_version"])["charlson_category"]

dx = dx.merge(unique_codes, on=["icd_code", "icd_version"], how="left")
mapped_frac = dx["charlson_category"].notna().mean()
print(f"  fraction of diagnosis rows mapping to a Charlson category: {mapped_frac:.1%}")

# ---------------------------------------------------------------------------
# Restrict to admissions present in both files; order chronologically
# ---------------------------------------------------------------------------
adm_sorted = adm.dropna(subset=["admittime"]).sort_values(["subject_id", "admittime"]).copy()

common_hadm = set(adm_sorted.hadm_id) & set(dx.hadm_id)
print(f"  admissions with both admittime and diagnoses: {len(common_hadm):,}")

# Filter FIRST, then compute chronological admission sequence, so admit_seq
# has no gaps from admissions that lack diagnosis codes.
adm_sorted = adm_sorted[adm_sorted.hadm_id.isin(common_hadm)].copy()
adm_sorted["admit_seq"] = adm_sorted.groupby("subject_id").cumcount() + 1
adm_sorted["n_admissions"] = adm_sorted.groupby("subject_id")["hadm_id"].transform("count")

# ---------------------------------------------------------------------------
# Per-admission Charlson category flags
# ---------------------------------------------------------------------------
print("Building per-admission Charlson category profile ...")
dx_charlson = dx.dropna(subset=["charlson_category"])
adm_cat = (
    dx_charlson.drop_duplicates(["hadm_id", "charlson_category"])
    .assign(flag=1)
    .pivot(index="hadm_id", columns="charlson_category", values="flag")
    .reindex(columns=CATEGORY_ORDER)
    .fillna(0)
    .astype("int8")
)

adm_features = adm_sorted.merge(adm_cat, on="hadm_id", how="left")
for c in CATEGORY_ORDER:
    adm_features[c] = adm_features[c].fillna(0).astype("int8")

# Apply Charlson override rules before scoring
diab_comp = adm_features["Diabetes with complication"] == 1
adm_features.loc[diab_comp, "Diabetes without complication"] = 0

mets = adm_features["Metastatic solid tumor"] == 1
adm_features.loc[mets, "Malignancy"] = 0

sev_liver = adm_features["Moderate or severe liver disease"] == 1
adm_features.loc[sev_liver, "Mild liver disease"] = 0

adm_features["charlson_score"] = sum(
    adm_features[c] * w for c, w in CHARLSON_WEIGHTS.items()
)
adm_features["n_categories"] = adm_features[CATEGORY_ORDER].sum(axis=1)

print(f"  admissions scored: {len(adm_features):,}")
print(f"  mean Charlson score: {adm_features.charlson_score.mean():.2f}")

adm_features.to_parquet(ROOT + "/work/02_network_construction/adm_features.parquet")
print("Saved adm_features.parquet")
