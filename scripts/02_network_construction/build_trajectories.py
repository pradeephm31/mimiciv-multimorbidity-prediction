from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features_with_comm.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
cat_to_comm = json.load(open(ROOT + "/work/02_network_construction/cat_to_comm.json"))
COMM_NAMES = {0: "Chronic-systemic/oncologic", 1: "Cardiorenal-metabolic", 2: "Neuro-frailty", -1: "No mapped comorbidity"}

cases = json.load(open(ROOT + "/work/02_network_construction/case_studies.json"))

traj_out = {}
for key, cs in cases.items():
    sid = cs["subject_id"]
    sub = adm[adm.subject_id == sid].sort_values("admit_seq")
    steps = []
    for _, row in sub.iterrows():
        active_comms = [cid for cid in [0, 1, 2] if row[f"comm{cid}_count"] > 0]
        steps.append({
            "admit_seq": int(row.admit_seq),
            "admittime": row.admittime.strftime("%Y-%m-%d"),
            "dominant_community": int(row.dominant_community),
            "dominant_community_name": COMM_NAMES[int(row.dominant_community)],
            "active_communities": [COMM_NAMES[c] for c in active_comms] if active_comms else [],
            "charlson_score": int(row.charlson_score),
            "escalation_next": bool(row.get("score_trend", 0)),  # placeholder, replaced below
        })
    traj_out[key] = {"subject_id": int(sid), "steps": steps}

with open(ROOT + "/work/02_network_construction/network_trajectories.json", "w") as f:
    json.dump(traj_out, f, indent=2)

for key, t in traj_out.items():
    print(f"\n=== {key} (patient {t['subject_id']}) — path through communities ===")
    print(" -> ".join(f"[{s['admittime']}] {s['dominant_community_name']} (score {s['charlson_score']})" for s in t["steps"]))
