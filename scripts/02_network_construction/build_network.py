from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import networkx as nx
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CHARLSON_WEIGHTS, CATEGORY_ORDER

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
mat = adm[CATEGORY_ORDER].values.astype(int)
n_cat = len(CATEGORY_ORDER)

# ---------------------------------------------------------------------------
# Edge weight = blend of (a) within-admission co-occurrence and
# (b) admission-to-admission transition (category A at t, category B at t+1)
# Both are symmetrized so Louvain (undirected) can run on them.
# ---------------------------------------------------------------------------
print("Building co-occurrence matrix (within-admission) ...")
cooc = mat.T @ mat  # n_cat x n_cat, diagonal = prevalence counts
np.fill_diagonal(cooc, 0)

print("Building transition matrix (admission t -> admission t+1, same patient) ...")
adm["next_subject"] = adm.subject_id.shift(-1)
adm["is_next_same_patient"] = adm.next_subject == adm.subject_id
valid = adm.is_next_same_patient.values[:-1]
A_t = mat[:-1][valid]
A_t1 = mat[1:][valid]
trans = A_t.T @ A_t1  # A_t1 counts given A_t present
trans_sym = trans + trans.T
np.fill_diagonal(trans_sym, 0)
print(f"  admission-pairs used for transitions: {valid.sum():,}")

# Normalize each matrix to comparable scale (max-normalize), then blend
cooc_norm = cooc / cooc.max()
trans_norm = trans_sym / trans_sym.max()
blended = 0.5 * cooc_norm + 0.5 * trans_norm

G = nx.Graph()
G.add_nodes_from(CATEGORY_ORDER)
for i in range(n_cat):
    for j in range(i + 1, n_cat):
        w = blended[i, j]
        if w > 0:
            G.add_edge(CATEGORY_ORDER[i], CATEGORY_ORDER[j], weight=float(w),
                       cooc_count=int(cooc[i, j]), trans_count=int(trans_sym[i, j]))

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------------------------------------------------------------------------
# Louvain community detection (networkx built-in, weight-aware)
# ---------------------------------------------------------------------------
communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
communities = sorted(communities, key=lambda c: -len(c))
cat_to_comm = {}
for idx, comm in enumerate(communities):
    for cat in comm:
        cat_to_comm[cat] = idx

modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
print(f"\nLouvain found {len(communities)} communities, modularity={modularity:.3f}")
for idx, comm in enumerate(communities):
    print(f"  Community {idx}: {sorted(comm)}")

# Try a couple of resolutions too, for sensitivity check
for res in [0.5, 1.5, 2.0]:
    c = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=res)
    print(f"  [sensitivity] resolution={res}: {len(c)} communities, sizes={sorted([len(x) for x in c], reverse=True)}")

# ---------------------------------------------------------------------------
# Save network + community assignment
# ---------------------------------------------------------------------------
nodes_out = [{"id": cat, "community": cat_to_comm[cat], "weight_charlson": CHARLSON_WEIGHTS[cat],
              "prevalence": float(mat[:, i].mean())} for i, cat in enumerate(CATEGORY_ORDER)]
edges_out = [{"source": u, "target": v, "weight": round(d["weight"], 4),
              "cooc_count": d["cooc_count"], "trans_count": d["trans_count"]}
             for u, v, d in G.edges(data=True)]

network_out = {
    "nodes": nodes_out,
    "edges": edges_out,
    "modularity": round(modularity, 4),
    "n_communities": len(communities),
    "communities": [sorted(c) for c in communities],
}
with open(ROOT + "/work/02_network_construction/network_data.json", "w") as f:
    json.dump(network_out, f, indent=2)
print("\nSaved network_data.json")

# Save category -> community mapping for downstream feature engineering
with open(ROOT + "/work/02_network_construction/cat_to_comm.json", "w") as f:
    json.dump(cat_to_comm, f, indent=2)
