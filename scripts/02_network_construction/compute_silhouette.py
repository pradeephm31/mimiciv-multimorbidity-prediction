from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
import networkx as nx
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER
from sklearn.metrics import silhouette_score, silhouette_samples

adm = pd.read_parquet(ROOT + "/work/02_network_construction/adm_features.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
mat = adm[CATEGORY_ORDER].values.astype(int)
n_cat = len(CATEGORY_ORDER)

# ---------------------------------------------------------------------------
# Rebuild the same blended (co-occurrence + transition) similarity matrix
# used for the Louvain network, exactly as in build_network.py
# ---------------------------------------------------------------------------
cooc = mat.T @ mat
np.fill_diagonal(cooc, 0)

adm["next_subject"] = adm.subject_id.shift(-1)
adm["is_next_same_patient"] = adm.next_subject == adm.subject_id
valid = adm.is_next_same_patient.values[:-1]
A_t = mat[:-1][valid]
A_t1 = mat[1:][valid]
trans = A_t.T @ A_t1
trans_sym = trans + trans.T
np.fill_diagonal(trans_sym, 0)

cooc_norm = cooc / cooc.max()
trans_norm = trans_sym / trans_sym.max()
blended = 0.5 * cooc_norm + 0.5 * trans_norm  # 17x17 symmetric similarity matrix

# Each category's row in the blended similarity matrix is its node embedding
# (its similarity profile to all other 16 categories) -- standard practice
# for evaluating community detection quality with silhouette analysis.
embeddings = blended.copy()
distance = 1 - embeddings  # convert similarity -> distance for silhouette
np.fill_diagonal(distance, 0)

G = nx.Graph()
G.add_nodes_from(CATEGORY_ORDER)
for i in range(n_cat):
    for j in range(i + 1, n_cat):
        w = blended[i, j]
        if w > 0:
            G.add_edge(CATEGORY_ORDER[i], CATEGORY_ORDER[j], weight=float(w))

results = {}
for res in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42, resolution=res)
    communities = sorted(communities, key=lambda c: -len(c))
    labels = np.zeros(n_cat, dtype=int)
    for idx, comm in enumerate(communities):
        for cat in comm:
            labels[CATEGORY_ORDER.index(cat)] = idx
    n_comm = len(communities)
    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")

    if n_comm < 2 or n_comm >= n_cat:
        sil = None
        sil_samples = None
    else:
        sil = float(silhouette_score(distance, labels, metric="precomputed"))
        sil_samples = silhouette_samples(distance, labels, metric="precomputed")

    results[res] = {
        "resolution": res, "n_communities": n_comm,
        "community_sizes": sorted([len(c) for c in communities], reverse=True),
        "modularity": round(modularity, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
    }
    print(f"resolution={res}: n_communities={n_comm}, sizes={sorted([len(c) for c in communities], reverse=True)}, "
          f"modularity={modularity:.4f}, silhouette={sil}")

    if abs(res - 1.0) < 1e-9:
        per_node = sorted(zip(CATEGORY_ORDER, labels, sil_samples), key=lambda x: (x[1], -x[2]))
        print("\n  Per-category silhouette (resolution=1.0, the chosen partition):")
        for cat, lab, s in per_node:
            print(f"    community {lab}: {cat:<35} silhouette={s:.3f}")
        results[res]["per_category_silhouette"] = [
            {"category": cat, "community": int(lab), "silhouette": round(float(s), 4)}
            for cat, lab, s in per_node
        ]

with open(ROOT + "/work/02_network_construction/silhouette_results.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print("\nSaved silhouette_results.json")
