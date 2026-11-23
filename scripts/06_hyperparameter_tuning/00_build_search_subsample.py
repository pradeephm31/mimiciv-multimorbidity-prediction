from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Prerequisite for tune_final_model.py. NOT part of the original package: the original
script reads search_subsample.parquet but nothing in the package creates it.

Recreates it as described in tune_final_model.py: a 35% PATIENT-level subsample of the
outer TRAINING set (same outer split as every other stage: GroupShuffleSplit, 80/20,
random_state=42). The original sampling seed is not recorded, so SEED below is an
assumption; results can differ slightly from the shipped tuning results.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 7
FRAC = 0.35

df = pd.read_parquet(ROOT + "/work/04_enrichment_features/final_charlson_model_df.parquet")
y = df["escalation_label"].values
groups = df["subject_id"].values
train_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, y, groups))
df_train = df.iloc[train_idx].reset_index(drop=True)

patients = np.sort(df_train["subject_id"].unique())
rng = np.random.RandomState(SEED)
chosen = rng.choice(patients, size=int(round(FRAC * len(patients))), replace=False)
search_df = df_train[df_train["subject_id"].isin(chosen)].reset_index(drop=True)
search_df.to_parquet(ROOT + "/work/06_hyperparameter_tuning/search_subsample.parquet")
print(f"Training patients: {len(patients):,}; sampled {len(chosen):,} ({FRAC:.0%}); "
      f"subsample pairs: {len(search_df):,}; escalation rate: {search_df.escalation_label.mean():.4f}")
