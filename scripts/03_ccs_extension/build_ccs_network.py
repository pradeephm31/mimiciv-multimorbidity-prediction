from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import json

mat = sp.load_npz(ROOT + "/work/03_ccs_extension/ccs_matrix.npz").tocsr().astype(np.int32)
adm = pd.read_parquet(ROOT + "/work/03_ccs_extension/adm_base_for_ccs.parquet").reset_index(drop=True)
cat_list = json.load(open(ROOT + "/work/03_ccs_extension/ccs_category_list.json"))["categories"]
ccs_codes = [c["ccs"] for c in cat_list]
ccs_desc = {c["ccs"]: c["desc"] for c in cat_list}
n_cat = len(ccs_codes)

print(f"Matrix: {mat.shape}, categories: {n_cat}")

# ---------------------------------------------------------------------------
# Within-admission co-occurrence (sparse matmul -- fast even at this size)
# ---------------------------------------------------------------------------
print("Computing co-occurrence matrix ...")
cooc = (mat.T @ mat).toarray()
prevalence_counts = np.diag(cooc).copy()
np.fill_diagonal(cooc, 0)

# ---------------------------------------------------------------------------
# Cross-admission transition (admission t -> t+1, same patient), symmetrized
# ---------------------------------------------------------------------------
print("Computing transition matrix ...")
adm["next_subject"] = adm.subject_id.shift(-1)
is_next_same = (adm.next_subject == adm.subject_id).values
idx_t = np.where(is_next_same)[0]
idx_t1 = idx_t + 1
A_t = mat[idx_t]
A_t1 = mat[idx_t1]
trans = (A_t.T @ A_t1).toarray()
trans_sym = trans + trans.T
np.fill_diagonal(trans_sym, 0)
print(f"  admission-pairs used: {len(idx_t):,}")

# ---------------------------------------------------------------------------
# Blend + build graph (restricted to keep it computationally sane: all 251
# categories are nodes, but we only materialize edges above a floor to avoid
# an O(n^2) fully-dense edge list)
# ---------------------------------------------------------------------------
cooc_norm = cooc / cooc.max()
trans_norm = trans_sym / trans_sym.max()
blended = 0.5 * cooc_norm + 0.5 * trans_norm

G = nx.Graph()
for i, c in enumerate(ccs_codes):
    G.add_node(c, prevalence=int(prevalence_counts[i]), desc=ccs_desc[c])

iu = np.triu_indices(n_cat, k=1)
weights = blended[iu]
nonzero = weights > 0
print(f"Non-zero candidate edges: {nonzero.sum():,} of {len(weights):,} possible pairs")

# Keep edges above the 60th percentile of nonzero weights -- dense enough for
# real community structure, sparse enough to stay tractable
thresh = np.percentile(weights[nonzero], 60)
keep = weights >= thresh
rows_i, cols_j = iu[0][keep], iu[1][keep]
for i, j, w in zip(rows_i, cols_j, weights[keep]):
    G.add_edge(ccs_codes[i], ccs_codes[j], weight=float(w),
               cooc_count=int(cooc[i, j]), trans_count=int(trans_sym[i, j]))

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------
communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
communities = sorted(communities, key=lambda c: -len(c))
cat_to_comm = {}
for idx, comm in enumerate(communities):
    for cat in comm:
        cat_to_comm[cat] = idx
modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
print(f"\nLouvain found {len(communities)} communities, modularity={modularity:.4f}")
for idx, comm in enumerate(communities):
    top_members = sorted(comm, key=lambda c: -int(prevalence_counts[ccs_codes.index(c)]))[:8]
    print(f"  Community {idx} (n={len(comm)}): " + ", ".join(ccs_desc[c] for c in top_members) + (" ..." if len(comm) > 8 else ""))

for res in [0.5, 1.5, 2.0]:
    c = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=res)
    print(f"  [sensitivity] resolution={res}: {len(c)} communities, sizes={sorted([len(x) for x in c], reverse=True)[:10]}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
nodes_out = [{"id": int(c), "desc": ccs_desc[c], "community": cat_to_comm[c],
              "prevalence": float(prevalence_counts[i] / mat.shape[0])}
             for i, c in enumerate(ccs_codes)]
edges_out = [{"source": int(u), "target": int(v), "weight": round(d["weight"], 4),
              "cooc_count": d["cooc_count"], "trans_count": d["trans_count"]}
             for u, v, d in G.edges(data=True)]

network_out = {
    "nodes": nodes_out, "edges": edges_out, "modularity": round(modularity, 4),
    "n_communities": len(communities),
    "communities": [[int(c) for c in comm] for comm in communities],
}
with open(ROOT + "/work/03_ccs_extension/ccs_network_data.json", "w") as f:
    json.dump(network_out, f, indent=2)

with open(ROOT + "/work/03_ccs_extension/ccs_cat_to_comm.json", "w") as f:
    json.dump({str(k): v for k, v in cat_to_comm.items()}, f, indent=2)

print("\nSaved ccs_network_data.json, ccs_cat_to_comm.json")
