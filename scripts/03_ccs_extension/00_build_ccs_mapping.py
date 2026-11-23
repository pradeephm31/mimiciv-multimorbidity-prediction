from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Prerequisite for the rest of 03_ccs_extension/: maps every row of
diagnoses_icd.csv to an AHRQ CCS category and saves dx_with_ccs.parquet.

Run this BEFORE build_ccs_matrix.py.

Data source: the mapping table (ccs_mapping_raw.parquet, ccs_categories_raw.parquet)
is vendored in ../data/derived/ so this script does not depend on a live external
download. If you deliberately want the latest upstream version instead, delete those
two files and this script will fall back to cloning vcastro/CCS from GitHub (requires
network access and R's .rda format support via the `pyreadr` package) -- note the
upstream repo can change over time, so re-cloning is NOT guaranteed to reproduce the
exact category assignments used in this project. For thesis reproducibility, use the
vendored files.
"""
import pandas as pd
import numpy as np
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDORED_MAPPING = os.path.join(ROOT, "reference", "ccs", "ccs_mapping_raw.parquet")
VENDORED_CATEGORIES = os.path.join(ROOT, "reference", "ccs", "ccs_categories_raw.parquet")
RAW_DX_PATH = os.path.join(ROOT, "raw", "diagnoses_icd.csv")
OUTPUT_PATH = os.path.join(ROOT, "work", "03_ccs_extension", "dx_with_ccs.parquet")

def load_ccs_mapping():
    if os.path.exists(VENDORED_MAPPING) and os.path.exists(VENDORED_CATEGORIES):
        print("Loading vendored CCS mapping (recommended, reproducible)...")
        mapping = pd.read_parquet(VENDORED_MAPPING)
        categories = pd.read_parquet(VENDORED_CATEGORIES)
        return mapping, categories

    print("Vendored mapping not found -- falling back to cloning vcastro/CCS from GitHub.")
    print("WARNING: this pulls the CURRENT upstream repo state and is not guaranteed to")
    print("match the vendored version used to produce this project's reported results.")
    repo_dir = os.path.join(HERE, "ccs_repo")
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "--depth", "1",
                         "https://github.com/vcastro/CCS.git", repo_dir], check=True)
    import pyreadr
    mapping = pyreadr.read_r(os.path.join(repo_dir, "data", "CCS_DX_mapping.rda"))["CCS_DX_mapping"]
    categories = pyreadr.read_r(os.path.join(repo_dir, "data", "CCS_DX_categories.rda"))["CCS_DX_categories"]
    return mapping, categories

def main():
    mapping, categories = load_ccs_mapping()
    mapping["code_clean"] = mapping["code"].str.strip().str.upper()

    print(f"Loading raw diagnoses from {RAW_DX_PATH} ...")
    dx = pd.read_csv(RAW_DX_PATH,
                      dtype={"subject_id": "int64", "hadm_id": "int64", "seq_num": "float32",
                             "icd_code": "string", "icd_version": "int8"})
    dx["icd_clean"] = dx["icd_code"].str.strip().str.upper()

    m9 = mapping[mapping.vocabulary_id == "ICD9CM"].set_index("code_clean")["category_code"]
    m10 = mapping[mapping.vocabulary_id == "ICD10CM"].set_index("code_clean")["category_code"]
    m9 = m9[~m9.index.duplicated()]
    m10 = m10[~m10.index.duplicated()]

    dx9 = dx[dx.icd_version == 9]
    dx10 = dx[dx.icd_version == 10]
    dx.loc[dx9.index, "ccs"] = dx9["icd_clean"].map(m9)
    dx.loc[dx10.index, "ccs"] = dx10["icd_clean"].map(m10)

    mapped_frac = dx["ccs"].notna().mean()
    print(f"Total rows: {len(dx):,}")
    print(f"Mapped fraction (all rows): {mapped_frac:.4f}")
    print(f"Unique CCS categories observed: {dx['ccs'].nunique()}")

    dx.to_parquet(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
