from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import networkx as nx
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/04_enrichment_features/adm_full_enriched_multihop.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# HF cohort definition, matching Liang et al.: patients with a heart-failure
# diagnosis on record (our CHF Charlson category, which covers ICD-9 428.x
# AND ICD-10 I50.x/I09.9/etc.) and >=2 total hospital admissions. We keep
# ALL of each qualifying patient's admissions (not just the CHF-flagged
# ones), exactly as Liang's cohort construction does.
# ---------------------------------------------------------------------------
hf_patients = adm[adm["Congestive heart failure"] == 1].subject_id.unique()
hf_adm_all = adm[adm.subject_id.isin(hf_patients)].copy()
n_admissions_per_patient = hf_adm_all.groupby("subject_id")["hadm_id"].transform("count")
hf_adm = hf_adm_all[n_admissions_per_patient >= 2].copy().reset_index(drop=True)

print(f"HF cohort: {hf_adm.subject_id.nunique():,} patients, {len(hf_adm):,} admissions, "
      f"avg {len(hf_adm)/hf_adm.subject_id.nunique():.2f} visits/patient")

hf_adm.to_parquet(ROOT + "/work/08_hf_cohort_liang_replication/hf_cohort_adm.parquet")

# ---------------------------------------------------------------------------
# Rebuild the Charlson co-occurrence/transition network FROM SCRATCH on this
# cohort only -- same method as the full-population network (build_network.py)
# ---------------------------------------------------------------------------
mat = hf_adm[CATEGORY_ORDER].values.astype(np.int64)
n_cat = len(CATEGORY_ORDER)

cooc = mat.T @ mat
prevalence_counts = np.diag(cooc).copy()
np.fill_diagonal(cooc, 0)

hf_adm["next_subject"] = hf_adm.subject_id.shift(-1)
is_next_same = (hf_adm.next_subject == hf_adm.subject_id).values
idx_t = np.where(is_next_same)[0]
idx_t1 = idx_t + 1
A_t = mat[idx_t]
A_t1 = mat[idx_t1]
trans = A_t.T @ A_t1
trans_sym = trans + trans.T
np.fill_diagonal(trans_sym, 0)
print(f"Transition pairs used: {len(idx_t):,}")

cooc_norm = cooc / cooc.max()
trans_norm = trans_sym / trans_sym.max() if trans_sym.max() > 0 else trans_sym
blended = 0.5 * cooc_norm + 0.5 * trans_norm

G = nx.Graph()
for i, cat in enumerate(CATEGORY_ORDER):
    G.add_node(cat, prevalence=int(prevalence_counts[i]))
iu = np.triu_indices(n_cat, k=1)
weights = blended[iu]
nonzero = weights > 0
thresh = np.percentile(weights[nonzero], 60) if nonzero.sum() > 0 else 0
keep = weights >= thresh
rows_i, cols_j = iu[0][keep], iu[1][keep]
for i, j, w in zip(rows_i, cols_j, weights[keep]):
    G.add_edge(CATEGORY_ORDER[i], CATEGORY_ORDER[j], weight=float(w))

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
communities = sorted(communities, key=lambda c: -len(c))
cat_to_comm_hf = {}
for idx, comm in enumerate(communities):
    for cat in comm:
        cat_to_comm_hf[cat] = idx
modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
print(f"\nLouvain (HF cohort): {len(communities)} communities, modularity={modularity:.4f}")
for idx, comm in enumerate(communities):
    print(f"  Community {idx} (n={len(comm)}): {sorted(comm)}")

# Silhouette on this network (same method as full-cohort silhouette script)
from sklearn.metrics import silhouette_score
distance = 1 - blended
np.fill_diagonal(distance, 0)
labels = np.array([cat_to_comm_hf[c] for c in CATEGORY_ORDER])
if len(set(labels)) >= 2 and len(set(labels)) < n_cat:
    sil = silhouette_score(distance, labels, metric="precomputed")
else:
    sil = None
print(f"Silhouette (HF cohort network): {sil}")

with open(ROOT + "/work/08_hf_cohort_liang_replication/hf_cat_to_comm.json", "w") as f:
    json.dump({str(k): v for k, v in cat_to_comm_hf.items()}, f, indent=2)
with open(ROOT + "/work/08_hf_cohort_liang_replication/hf_network_summary.json", "w") as f:
    json.dump({
        "n_patients": int(hf_adm.subject_id.nunique()), "n_admissions": int(len(hf_adm)),
        "n_communities": len(communities), "modularity": round(modularity, 4),
        "silhouette": round(float(sil), 4) if sil is not None else None,
        "communities": [sorted(list(c)) for c in communities],
    }, f, indent=2)
print("\nSaved hf_cohort_adm.parquet, hf_cat_to_comm.json, hf_network_summary.json")
