from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import json
import sys
import time
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

hf_adm = pd.read_parquet(ROOT + "/work/08_hf_cohort_liang_replication/hf_cohort_adm.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
tcn = json.load(open(ROOT + "/work/08_hf_cohort_liang_replication/hf_population_tcn.json"))
n_cat = len(CATEGORY_ORDER)
cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}

edge_list = [(cat_idx[e["source"]], cat_idx[e["target"]]) for e in tcn["edges"]]
edge_idx = {e: i for i, e in enumerate(edge_list)}
n_edges = len(edge_list)
print(f"Population TCN: {n_cat} nodes, {n_edges} significant directed edges")

# Adjacency (population-level): Nb(u) = out-neighbors of u in the significant TCN
pop_adj = [[] for _ in range(n_cat)]
for (u, v) in edge_list:
    pop_adj[u].append(v)

# ---------------------------------------------------------------------------
# Per-patient node/edge frequency vectors.
#   node_freq[patient, u]      = # admissions (within the visit window used)
#                                 containing category u
#   edge_freq[patient, e_idx]  = # adjacent-visit transitions u->v (within
#                                 the visit window used) matching a
#                                 significant population edge
# Computed twice: FULL history (candidate pool) and TRUNCATED history
# (first V-1 visits only, for the >=3-visit prediction targets).
# ---------------------------------------------------------------------------
def compute_freq_vectors(admdf, patient_list, truncate_last=False):
    node_freq = np.zeros((len(patient_list), n_cat), dtype=np.float32)
    edge_freq = np.zeros((len(patient_list), n_edges), dtype=np.float32)
    pid_to_row = {p: i for i, p in enumerate(patient_list)}
    grouped = admdf.groupby("subject_id")
    for pid, g in grouped:
        if pid not in pid_to_row:
            continue
        row = pid_to_row[pid]
        cats_per_visit = g.sort_values("admit_seq")[CATEGORY_ORDER].values.astype(bool)
        if truncate_last:
            cats_per_visit = cats_per_visit[:-1]
        if len(cats_per_visit) == 0:
            continue
        node_freq[row] = cats_per_visit.sum(axis=0)
        for t in range(len(cats_per_visit) - 1):
            present_t = np.where(cats_per_visit[t])[0]
            present_t1 = np.where(cats_per_visit[t + 1])[0]
            for u in present_t:
                for v in present_t1:
                    ei = edge_idx.get((u, v))
                    if ei is not None:
                        edge_freq[row, ei] += 1
    return node_freq, edge_freq

all_patients = hf_adm.subject_id.unique().tolist()
pid_row = {p: i for i, p in enumerate(all_patients)}
print(f"\nComputing FULL-history frequency vectors for {len(all_patients):,} candidate-pool patients...")
t0 = time.time()
cand_node_freq, cand_edge_freq = compute_freq_vectors(hf_adm, all_patients, truncate_last=False)
print(f"  done in {time.time()-t0:.1f}s")

vc = hf_adm.groupby("subject_id").size()
target_patients_all = vc[vc >= 3].index.tolist()
# Sample for computational tractability (still >>5x Liang's n=920 evaluation set)
rng = np.random.RandomState(42)
N_TARGET_SAMPLE = 5000
target_patients = list(rng.choice(target_patients_all, size=min(N_TARGET_SAMPLE, len(target_patients_all)), replace=False))
print(f"\nPrediction-eligible (>=3 visits): {len(target_patients_all):,}; evaluating on random sample of {len(target_patients):,}")

hf_targets = hf_adm[hf_adm.subject_id.isin(target_patients)]
print(f"Computing TRUNCATED (visits 1..V-1) frequency vectors for {len(target_patients):,} target patients...")
t0 = time.time()
targ_node_freq, targ_edge_freq = compute_freq_vectors(hf_targets, target_patients, truncate_last=True)
print(f"  done in {time.time()-t0:.1f}s")

np.save(ROOT + "/work/08_hf_cohort_liang_replication/cand_node_freq.npy", cand_node_freq)
np.save(ROOT + "/work/08_hf_cohort_liang_replication/cand_edge_freq.npy", cand_edge_freq)
np.save(ROOT + "/work/08_hf_cohort_liang_replication/targ_node_freq.npy", targ_node_freq)
np.save(ROOT + "/work/08_hf_cohort_liang_replication/targ_edge_freq.npy", targ_edge_freq)
with open(ROOT + "/work/08_hf_cohort_liang_replication/tcn_dm_patient_lists.json", "w") as f:
    json.dump({"all_patients": [int(p) for p in all_patients], "target_patients": [int(p) for p in target_patients]}, f)
print("\nSaved frequency vectors and patient lists.")
