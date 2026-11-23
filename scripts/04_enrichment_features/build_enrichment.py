from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import pyreadr
import json

# ---------------------------------------------------------------------------
# Base admission table (chronological order already established)
# ---------------------------------------------------------------------------
adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet")[
    ["subject_id", "hadm_id", "admittime"]
].copy()
adm["admit_year"] = adm.admittime.dt.year

# ---------------------------------------------------------------------------
# 1) Age at admission (from patients.csv: anchor_age is age during anchor_year)
# ---------------------------------------------------------------------------
patients = pd.read_csv(ROOT + "/raw/patients.csv")
adm = adm.merge(patients[["subject_id", "gender", "anchor_age", "anchor_year", "dod"]], on="subject_id", how="left")
adm["age_at_admission"] = adm["anchor_age"] + (adm["admit_year"] - adm["anchor_year"])
adm["age_at_admission"] = adm["age_at_admission"].clip(lower=18, upper=100)
adm["gender_male"] = (adm["gender"] == "M").astype(int)
print(f"Age coverage: {adm.age_at_admission.notna().mean():.1%}, "
      f"mean={adm.age_at_admission.mean():.1f}, range=[{adm.age_at_admission.min():.0f},{adm.age_at_admission.max():.0f}]")

# ---------------------------------------------------------------------------
# 2) ICU utilization (from icustays.csv)
# ---------------------------------------------------------------------------
icu = pd.read_csv(ROOT + "/raw/icustays.csv")
icu_agg = icu.groupby("hadm_id").agg(n_icu_stays=("stay_id", "count"), icu_los_days=("los", "sum")).reset_index()
adm = adm.merge(icu_agg, on="hadm_id", how="left")
adm["n_icu_stays"] = adm["n_icu_stays"].fillna(0).astype(int)
adm["icu_los_days"] = adm["icu_los_days"].fillna(0.0)
adm["had_icu_stay"] = (adm["n_icu_stays"] > 0).astype(int)
print(f"ICU stay coverage: {adm.had_icu_stay.mean():.1%} of admissions had >=1 ICU stay")

# ---------------------------------------------------------------------------
# 3) Procedure burden (from procedures_icd.csv), plus distinct CCS-PCS
#    categories touched (reusing the same AHRQ CCS procedure mapping family)
# ---------------------------------------------------------------------------
proc = pd.read_csv(ROOT + "/raw/procedures_icd.csv",
                    dtype={"subject_id": "int64", "hadm_id": "int64", "seq_num": "float32",
                           "icd_code": "string", "icd_version": "int8"})
proc_count = proc.groupby("hadm_id").size().rename("n_procedures")
adm = adm.merge(proc_count, on="hadm_id", how="left")
adm["n_procedures"] = adm["n_procedures"].fillna(0).astype(int)
adm["had_procedure"] = (adm["n_procedures"] > 0).astype(int)

pcs_map = pyreadr.read_r(ROOT + "/reference/ccs/CCS_PCS_mapping.rda")["CCS_PCS_mapping"]
pcs_map["code_clean"] = pcs_map["code"].str.strip().str.upper()
pcs9 = pcs_map[pcs_map.vocabulary_id == "ICD9PCS"].drop_duplicates("code_clean").set_index("code_clean")["category_code"]
pcs10 = pcs_map[pcs_map.vocabulary_id == "ICD10PCS"].drop_duplicates("code_clean").set_index("code_clean")["category_code"]
proc["icd_clean"] = proc["icd_code"].str.strip().str.upper()
proc.loc[proc.icd_version == 9, "pcs_ccs"] = proc.loc[proc.icd_version == 9, "icd_clean"].map(pcs9)
proc.loc[proc.icd_version == 10, "pcs_ccs"] = proc.loc[proc.icd_version == 10, "icd_clean"].map(pcs10)
print(f"Procedure CCS-PCS mapping coverage: {proc['pcs_ccs'].notna().mean():.1%}")
n_distinct_pcs = proc.dropna(subset=["pcs_ccs"]).groupby("hadm_id")["pcs_ccs"].nunique().rename("n_distinct_procedure_categories")
adm = adm.merge(n_distinct_pcs, on="hadm_id", how="left")
adm["n_distinct_procedure_categories"] = adm["n_distinct_procedure_categories"].fillna(0).astype(int)

# ---------------------------------------------------------------------------
# 4) DRG severity / mortality (from drgcodes.csv, APR rows only)
# ---------------------------------------------------------------------------
drg = pd.read_csv(ROOT + "/raw/drgcodes.csv")
apr = drg[drg.drg_type == "APR"][["hadm_id", "drg_severity", "drg_mortality"]].drop_duplicates("hadm_id")
adm = adm.merge(apr, on="hadm_id", how="left")
adm["has_apr_drg"] = adm["drg_severity"].notna().astype(int)
# Impute missing severity/mortality with the median (imputation flag preserves the "missingness" signal)
adm["drg_severity"] = adm["drg_severity"].fillna(adm["drg_severity"].median())
adm["drg_mortality"] = adm["drg_mortality"].fillna(adm["drg_mortality"].median())
print(f"APR DRG coverage: {adm.has_apr_drg.mean():.1%}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
enrich_cols = ["hadm_id", "age_at_admission", "gender_male", "n_icu_stays", "icu_los_days", "had_icu_stay",
               "n_procedures", "had_procedure", "n_distinct_procedure_categories",
               "drg_severity", "drg_mortality", "has_apr_drg"]
enrich = adm[enrich_cols].copy()
enrich.to_parquet(ROOT + "/work/04_enrichment_features/enrichment_features.parquet")
print(f"\nSaved enrichment_features.parquet: {enrich.shape}")
print(enrich.describe().T)
