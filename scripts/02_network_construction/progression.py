from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet")
adm = adm.sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)

multi = adm[adm.n_admissions >= 2].copy()
n_patients_total = adm.subject_id.nunique()
n_patients_multi = multi.subject_id.nunique()
print(f"Total patients: {n_patients_total:,}; with >=2 admissions: {n_patients_multi:,} "
      f"({n_patients_multi/n_patients_total:.1%})")

# ---------------------------------------------------------------------------
# 1) Charlson score trajectory by admission number
# ---------------------------------------------------------------------------
traj = (
    adm[adm.admit_seq <= 8]
    .groupby("admit_seq")
    .agg(mean_score=("charlson_score", "mean"),
         median_score=("charlson_score", "median"),
         mean_categories=("n_categories", "mean"),
         n_admissions=("hadm_id", "count"),
         n_patients=("subject_id", "nunique"),
         mortality_rate=("hospital_expire_flag", "mean"))
    .reset_index()
)
print("\nCharlson score trajectory by admission number:")
print(traj)

# ---------------------------------------------------------------------------
# 2) Consecutive-admission transitions per category (onset / persistence / resolution)
# ---------------------------------------------------------------------------
multi["prev_hadm"] = multi.groupby("subject_id")["hadm_id"].shift(1)
pairs = multi.dropna(subset=["prev_hadm"]).copy()
prev = adm.set_index("hadm_id")[CATEGORY_ORDER]
curr = adm.set_index("hadm_id")[CATEGORY_ORDER]

pairs_prev = prev.loc[pairs.prev_hadm.astype(int)].reset_index(drop=True)
pairs_curr = curr.loc[pairs.hadm_id].reset_index(drop=True)

transition_rows = []
for cat in CATEGORY_ORDER:
    p = pairs_prev[cat].values
    c = pairs_curr[cat].values
    n_pairs = len(p)
    had_before = p == 1
    not_before = p == 0
    onset = ((not_before) & (c == 1)).sum()
    persist = ((had_before) & (c == 1)).sum()
    resolved_doc = ((had_before) & (c == 0)).sum()  # dropped off active problem list
    never = ((not_before) & (c == 0)).sum()
    n_not_before = not_before.sum()
    n_had_before = had_before.sum()
    transition_rows.append({
        "category": cat,
        "weight": CHARLSON_WEIGHTS[cat],
        "prevalence_any_admission": adm[cat].mean(),
        "n_admission_pairs": n_pairs,
        "onset_rate": onset / n_not_before if n_not_before else np.nan,   # P(new at t | absent at t-1)
        "persistence_rate": persist / n_had_before if n_had_before else np.nan,  # P(still coded at t | present at t-1)
        "n_had_before": int(n_had_before),
        "n_not_before": int(n_not_before),
    })

trans_df = pd.DataFrame(transition_rows).sort_values("onset_rate", ascending=False)
print("\nCategory transition rates across consecutive admissions:")
print(trans_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3) Common "index category at first admission" -> categories gained later
# ---------------------------------------------------------------------------
first_adm = adm[adm.admit_seq == 1].set_index("subject_id")
last_adm = adm.sort_values(["subject_id", "admit_seq"]).groupby("subject_id").tail(1).set_index("subject_id")
common_subj = multi.subject_id.unique()

gained = {}
for cat in CATEGORY_ORDER:
    fa = first_adm.loc[common_subj, cat]
    la = last_adm.loc[common_subj, cat]
    newly_acquired = ((fa == 0) & (la == 1)).sum()
    had_at_first = (fa == 1).sum()
    gained[cat] = {
        "had_at_first_admission_pct": had_at_first / len(common_subj),
        "newly_acquired_by_last_admission_pct": newly_acquired / max((fa == 0).sum(), 1),
    }
gained_df = pd.DataFrame(gained).T.sort_values("newly_acquired_by_last_admission_pct", ascending=False)
print("\nCategory acquisition from first -> last admission (patients with >=2 admissions):")
print(gained_df)

# ---------------------------------------------------------------------------
# 4) Score-change archetypes (progression pathways) among multi-admission patients
# ---------------------------------------------------------------------------
score_by_patient = multi.groupby("subject_id").agg(
    n_adm=("hadm_id", "count"),
    first_score=("charlson_score", "first"),
    last_score=("charlson_score", "last"),
    max_score=("charlson_score", "max"),
    died=("hospital_expire_flag", "max"),
)
score_by_patient["delta"] = score_by_patient.last_score - score_by_patient.first_score

def archetype(row):
    if row.died == 1:
        return "Terminal admission (in-hospital death)"
    if row.delta >= 2:
        return "Escalating comorbidity burden"
    if row.delta <= -2:
        return "Declining recorded burden"  # usually documentation/coding variation, not true reversal
    return "Stable comorbidity burden"

score_by_patient["archetype"] = score_by_patient.apply(archetype, axis=1)
archetype_counts = score_by_patient.archetype.value_counts()
archetype_pct = (archetype_counts / len(score_by_patient) * 100).round(1)
print("\nProgression archetypes among patients with >=2 admissions:")
print(pd.DataFrame({"n_patients": archetype_counts, "pct": archetype_pct}))

# ---------------------------------------------------------------------------
# Save everything needed for the HTML report
# ---------------------------------------------------------------------------
out = {
    "n_patients_total": int(n_patients_total),
    "n_admissions_total": int(len(adm)),
    "n_patients_multi": int(n_patients_multi),
    "pct_patients_multi": round(n_patients_multi / n_patients_total * 100, 1),
    "mapped_category_prevalence_overall": adm[CATEGORY_ORDER].mean().sort_values(ascending=False).round(4).to_dict(),
    "trajectory": traj.round(4).to_dict(orient="records"),
    "transitions": trans_df.round(4).to_dict(orient="records"),
    "acquisition": gained_df.reset_index().rename(columns={"index": "category"}).round(4).to_dict(orient="records"),
    "archetypes": [{"archetype": k, "n_patients": int(v), "pct": float(archetype_pct[k])} for k, v in archetype_counts.items()],
    "charlson_weights": CHARLSON_WEIGHTS,
}
with open(ROOT + "/work/02_network_construction/report_data.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved report_data.json")

score_by_patient.reset_index().to_parquet(ROOT + "/work/02_network_construction/score_by_patient.parquet")
