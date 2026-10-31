from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
enrich = pd.read_parquet(ROOT + "/work/04_enrichment_features/enrichment_features.parquet")
adm = adm.merge(enrich, on="hadm_id", how="left")

# ---------------------------------------------------------------------------
# Age-adjusted Charlson Comorbidity Index (ACCI)
# Charlson et al. 1994 age adjustment: +1 point per decade of age over 40
#   age 40-49: +0, 50-59: +1, 60-69: +2, 70-79: +3, 80-89: +4, 90-99: +5, ...
# ---------------------------------------------------------------------------
adm["age_points"] = np.maximum(0, ((adm["age_at_admission"] - 40) // 10).astype(int))
adm["acci_score"] = adm["charlson_score"] + adm["age_points"]
print("Age-points distribution:")
print(adm["age_points"].value_counts().sort_index())
print(f"\nRaw Charlson score:  mean={adm.charlson_score.mean():.2f}, std={adm.charlson_score.std():.2f}")
print(f"ACCI (age-adjusted): mean={adm.acci_score.mean():.2f}, std={adm.acci_score.std():.2f}")

# ---------------------------------------------------------------------------
# Multi-hop trajectory features (t-1, t-2, t-3) + interval/frequency features
# ---------------------------------------------------------------------------
g = adm.groupby("subject_id")
for lag in [1, 2, 3]:
    adm[f"charlson_score_lag{lag}"] = g["charlson_score"].shift(lag)
    adm[f"n_categories_lag{lag}"] = g["n_categories"].shift(lag)
    adm[f"admittime_lag{lag}"] = g["admittime"].shift(lag)

# Trend slope over up to the last 3 admissions (linear fit on available lags)
def trend_slope(row):
    ys, xs = [], []
    for i, lag in enumerate([3, 2, 1, 0]):  # oldest to newest: t-3, t-2, t-1, t(current)
        val = row["charlson_score"] if lag == 0 else row.get(f"charlson_score_lag{lag}")
        if pd.notna(val):
            ys.append(val)
            xs.append(i)
    if len(xs) < 2:
        return 0.0
    xs = np.array(xs); ys = np.array(ys)
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)

adm["charlson_trend_slope_3adm"] = adm.apply(trend_slope, axis=1)
adm["n_prior_admissions_available"] = adm[["charlson_score_lag1", "charlson_score_lag2", "charlson_score_lag3"]].notna().sum(axis=1)

# Interval / admission-frequency features
adm["prev_admittime"] = g["admittime"].shift(1)
has_prev = adm["prev_admittime"].notna()
adm["days_since_prior_admission"] = np.where(has_prev, (adm["admittime"] - adm["prev_admittime"]).dt.days, -1)
adm["log_days_since_prior"] = np.where(has_prev, np.log1p(adm["days_since_prior_admission"].clip(lower=0)), -1)

first_admit = g["admittime"].transform("min")
years_since_first = (adm["admittime"] - first_admit).dt.days / 365.25
adm["admissions_per_year_so_far"] = np.where(years_since_first > 0, adm["admit_seq"] / years_since_first, 0)

# Cumulative ICU / procedure history (up to and including current admission)
adm["cum_icu_stays"] = g["n_icu_stays"].cumsum()
adm["cum_icu_los_days"] = g["icu_los_days"].cumsum()
adm["cum_procedures"] = g["n_procedures"].cumsum()
adm["had_icu_prev_admission"] = g["had_icu_stay"].shift(1).fillna(0).astype(int)

print(f"\nMulti-hop feature availability: "
      f"lag1={adm.charlson_score_lag1.notna().mean():.1%}, "
      f"lag2={adm.charlson_score_lag2.notna().mean():.1%}, "
      f"lag3={adm.charlson_score_lag3.notna().mean():.1%}")

adm.to_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet")
print(f"\nSaved adm_full_enriched_multihop.parquet: {adm.shape}")
