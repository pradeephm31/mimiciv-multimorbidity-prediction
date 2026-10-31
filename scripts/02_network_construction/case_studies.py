from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet")
adm = adm.sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
score_by_patient = pd.read_parquet(ROOT + "/work/02_network_construction/score_by_patient.parquet").set_index("subject_id")

multi = adm[adm.n_admissions >= 2].copy()

def patient_timeline(subject_id):
    sub = adm[adm.subject_id == subject_id].sort_values("admit_seq")
    timeline = []
    for _, row in sub.iterrows():
        cats = [c for c in CATEGORY_ORDER if row[c] == 1]
        timeline.append({
            "admit_seq": int(row.admit_seq),
            "hadm_id": int(row.hadm_id),
            "admittime": row.admittime.strftime("%Y-%m-%d"),
            "admission_type": row.admission_type,
            "categories": cats,
            "charlson_score": int(row.charlson_score),
            "died": bool(row.hospital_expire_flag == 1),
        })
    return timeline

candidates = {}

# --- Archetype 1: Escalating comorbidity, clear stepwise category acquisition ---
esc = score_by_patient[(score_by_patient.archetype == "Escalating comorbidity burden") &
                        (score_by_patient.n_adm.between(4, 7))]
best_esc, best_esc_score = None, -1
for sid in esc.sample(min(4000, len(esc)), random_state=42).index:
    sub = adm[adm.subject_id == sid]
    cats_per_adm = [set(c for c in CATEGORY_ORDER if r[c] == 1) for _, r in sub.iterrows()]
    n_new_each_step = sum(1 for i in range(1, len(cats_per_adm)) if len(cats_per_adm[i] - cats_per_adm[i-1]) >= 1)
    monotonic = all(sub.charlson_score.values[i] <= sub.charlson_score.values[i+1] for i in range(len(sub)-1))
    span_days = (sub.admittime.max() - sub.admittime.min()).days
    score = n_new_each_step + (2 if monotonic else 0) + (1 if span_days > 200 else 0)
    if score > best_esc_score:
        best_esc_score, best_esc = score, sid
candidates["escalating"] = best_esc

# --- Archetype 2: Terminal trajectory (in-hospital death on final admission) ---
term = score_by_patient[(score_by_patient.archetype == "Terminal admission (in-hospital death)") &
                         (score_by_patient.n_adm.between(3, 6))]
best_term, best_term_score = None, -1
for sid in term.sample(min(4000, len(term)), random_state=7).index:
    sub = adm[adm.subject_id == sid]
    cats_per_adm = [set(c for c in CATEGORY_ORDER if r[c] == 1) for _, r in sub.iterrows()]
    n_new = sum(1 for i in range(1, len(cats_per_adm)) if len(cats_per_adm[i] - cats_per_adm[i-1]) >= 1)
    span_days = (sub.admittime.max() - sub.admittime.min()).days
    score = n_new + (1 if span_days > 100 else 0) + (1 if sub.charlson_score.iloc[-1] >= 5 else 0)
    if score > best_term_score:
        best_term_score, best_term = score, sid
candidates["terminal"] = best_term

# --- Archetype 3: Stable chronic burden across many admissions ---
stab = score_by_patient[(score_by_patient.archetype == "Stable comorbidity burden") &
                         (score_by_patient.n_adm.between(5, 9)) &
                         (score_by_patient.first_score >= 3)]
best_stab, best_stab_score = None, -1
for sid in stab.sample(min(4000, len(stab)), random_state=11).index:
    sub = adm[adm.subject_id == sid]
    span_days = (sub.admittime.max() - sub.admittime.min()).days
    var = sub.charlson_score.std()
    score = (1 if span_days > 300 else 0) - (var or 0)
    if score > best_stab_score:
        best_stab_score, best_stab = score, sid
candidates["stable"] = best_stab

# --- Archetype 4: Malignancy -> metastatic progression specifically ---
mal_prog = []
mal_ids = adm[adm["Malignancy"] == 1].subject_id.unique()
sub_multi = multi[multi.subject_id.isin(mal_ids)]
for sid, grp in sub_multi.groupby("subject_id"):
    grp = grp.sort_values("admit_seq")
    mal_seq = grp["Malignancy"].values
    met_seq = grp["Metastatic solid tumor"].values
    if mal_seq[0] == 1 and met_seq[0] == 0 and met_seq[-1] == 1 and len(grp) >= 3:
        mal_prog.append(sid)
    if len(mal_prog) > 500:
        break
best_mal = mal_prog[0] if mal_prog else None
candidates["malignancy"] = best_mal

print("Selected candidate subject_ids:", candidates)

case_studies = {}
for key, sid in candidates.items():
    if sid is None:
        continue
    case_studies[key] = {
        "subject_id": int(sid),
        "n_admissions": int(score_by_patient.loc[sid, "n_adm"]),
        "died": bool(score_by_patient.loc[sid, "died"] == 1),
        "timeline": patient_timeline(sid),
    }

with open(ROOT + "/work/02_network_construction/case_studies.json", "w") as f:
    json.dump(case_studies, f, indent=2)

for key, cs in case_studies.items():
    print(f"\n=== {key} : subject {cs['subject_id']} ({cs['n_admissions']} admissions) ===")
    for t in cs["timeline"]:
        print(f"  #{t['admit_seq']} {t['admittime']} [{t['admission_type']}] score={t['charlson_score']} "
              f"died={t['died']} :: {', '.join(t['categories']) if t['categories'] else '(none mapped)'}")
