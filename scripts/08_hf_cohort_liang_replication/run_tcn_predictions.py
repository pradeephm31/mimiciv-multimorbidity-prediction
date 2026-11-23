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
lists = json.load(open(ROOT + "/work/08_hf_cohort_liang_replication/tcn_dm_patient_lists.json"))
n_cat = len(CATEGORY_ORDER)
cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}
edge_list = [(cat_idx[e["source"]], cat_idx[e["target"]]) for e in tcn["edges"]]
n_edges = len(edge_list)
pop_adj = [[] for _ in range(n_cat)]
for (u, v) in edge_list:
    pop_adj[u].append(v)

all_patients = lists["all_patients"]
target_patients = lists["target_patients"]
pid_row_all = {p: i for i, p in enumerate(all_patients)}
target_row_in_all = np.array([pid_row_all[p] for p in target_patients])

cand_node_freq = np.load(ROOT + "/work/08_hf_cohort_liang_replication/cand_node_freq.npy")   # (n_all, 17)
cand_edge_freq = np.load(ROOT + "/work/08_hf_cohort_liang_replication/cand_edge_freq.npy")   # (n_all, n_edges)
targ_node_freq = np.load(ROOT + "/work/08_hf_cohort_liang_replication/targ_node_freq.npy")   # (n_target, 17)
targ_edge_freq = np.load(ROOT + "/work/08_hf_cohort_liang_replication/targ_edge_freq.npy")   # (n_target, n_edges)

n_all = len(all_patients)
n_target = len(target_patients)
print(f"Candidate pool: {n_all:,} patients. Targets: {n_target:,} patients.")

# Precompute, per edge, the (u,k) node indices for fast lookups
edge_u = np.array([e[0] for e in edge_list])
edge_v = np.array([e[1] for e in edge_list])

# ---------------------------------------------------------------------------
# Ground truth: last (held-out) visit's diagnosis set for each target patient
# ---------------------------------------------------------------------------
hf_targets_full = hf_adm[hf_adm.subject_id.isin(target_patients)].sort_values(["subject_id", "admit_seq"])
last_visit_dx = {}
prelast_visit_dx = {}  # visit V-1 (source/head nodes for retrieval)
for pid, g in hf_targets_full.groupby("subject_id"):
    cats = g[CATEGORY_ORDER].values.astype(bool)
    last_visit_dx[pid] = set(np.where(cats[-1])[0])
    prelast_visit_dx[pid] = set(np.where(cats[-2])[0])

# ---------------------------------------------------------------------------
# TCN-DM: vectorized distance computation per target against ALL candidates
# dist(Pi,Pj) = MS_node + MS_edge  (skipping demographics term for simplicity)
# ---------------------------------------------------------------------------
def nearest_neighbor(t_idx):
    a_i = targ_node_freq[t_idx]                      # (17,)
    active_nodes = np.where(a_i > 0)[0]
    if len(active_nodes) == 0:
        return None
    # MS_node: sum over u in G_i of (A_i(u) - A_j(u))^2, sqrt
    diff_node = a_i[active_nodes][None, :] - cand_node_freq[:, active_nodes]   # (n_all, |active|)
    ms_node_sq = (diff_node ** 2).sum(axis=1)

    # MS_edge: sum over ordered pairs (u,k) both in active_nodes, u!=k
    w_i = targ_edge_freq[t_idx]                        # (n_edges,)
    pair_mask = np.isin(edge_u, active_nodes) & np.isin(edge_v, active_nodes) & (edge_u != edge_v)
    if pair_mask.sum() > 0:
        diff_edge = w_i[pair_mask][None, :] - cand_edge_freq[:, pair_mask]     # (n_all, n_pairs)
        ms_edge_sq = (diff_edge ** 2).sum(axis=1)
    else:
        ms_edge_sq = np.zeros(n_all)

    dist = np.sqrt(ms_node_sq) + np.sqrt(ms_edge_sq)
    dist[target_row_in_all[t_idx]] = np.inf  # exclude self
    return int(np.argmin(dist))

# ---------------------------------------------------------------------------
# Degree centrality (Liang eq 1-5), computed on an arbitrary directed edge set
# ---------------------------------------------------------------------------
def degree_centrality(active_edges, l=n_cat):
    """active_edges: set of (u,v) tuples present in this local graph."""
    nb = {u: set() for u in range(l)}
    for (u, v) in active_edges:
        nb[u].add(v)
    k = {u: len(nb[u]) for u in range(l)}
    rho = {u: k[u] / (l - 1) for u in range(l)}

    def jaccard_dist(u, v):
        su, sv = nb[u], nb[v]
        union = su | sv
        if not union:
            return 1.0
        return 1 - len(su & sv) / len(union)

    gamma = {}
    for u in range(l):
        higher = [v for v in range(l) if rho[v] > rho[u]]
        if higher:
            delta = min(jaccard_dist(u, v) for v in higher)
        else:
            delta = max((jaccard_dist(u, v) for v in range(l) if v != u), default=0)
        gamma[u] = rho[u] * delta
    return gamma

def hop_neighbors(source_nodes, adj, max_hops=2):
    frontier = set(source_nodes)
    visited = set(source_nodes)
    for _ in range(max_hops):
        nxt = set()
        for u in frontier:
            nxt |= set(adj.get(u, []))
        nxt -= visited
        visited |= nxt
        frontier = nxt
    return visited

pop_adj_dict = {u: pop_adj[u] for u in range(n_cat)}

# ---------------------------------------------------------------------------
# Run predictions for each method, at L = 3, 5, 10 (adapted from Liang's
# 5/10/20 because our category space is 17, not dozens of raw ICD codes --
# L=20 would exceed the total possible categories)
# ---------------------------------------------------------------------------
L_VALUES = [3, 5, 10]
methods = ["TCN-ALL", "TCN-CN", "TCN-JI", "TCN-DM"]
predictions = {m: {L: {} for L in L_VALUES} for m in methods}

print("\nRunning predictions...")
t0 = time.time()
nn_cache = {}
for idx, pid in enumerate(target_patients):
    source = prelast_visit_dx[pid]
    if not source:
        continue

    # TCN-ALL: 1-hop neighbors of source in population TCN, ranked by population-level gamma
    pop_gamma = degree_centrality(set(edge_list))
    cand_all = hop_neighbors(source, pop_adj_dict, max_hops=1)
    ranked_all = sorted(cand_all, key=lambda u: -pop_gamma.get(u, 0))

    # TCN-CN / TCN-JI: score candidate categories by common-neighbor / Jaccard
    # overlap with the source diagnosis set, using population Nb()
    nb_pop = {u: set(pop_adj[u]) for u in range(n_cat)}
    candidates_2hop = hop_neighbors(source, pop_adj_dict, max_hops=2)
    cn_scores, ji_scores = {}, {}
    for v in candidates_2hop:
        cn = sum(len(nb_pop.get(vi, set()) & nb_pop.get(v, set())) for vi in source)
        union_size = sum(len(nb_pop.get(vi, set()) | nb_pop.get(v, set())) or 1 for vi in source)
        ji = sum(len(nb_pop.get(vi, set()) & nb_pop.get(v, set())) / (len(nb_pop.get(vi, set()) | nb_pop.get(v, set())) or 1) for vi in source)
        cn_scores[v] = cn
        ji_scores[v] = ji
    ranked_cn = sorted(candidates_2hop, key=lambda u: -cn_scores.get(u, 0))
    ranked_ji = sorted(candidates_2hop, key=lambda u: -ji_scores.get(u, 0))

    # TCN-DM: nearest neighbor + local dyad graph
    nn = nearest_neighbor(idx)
    if nn is not None:
        nn_pid = all_patients[nn]
        nn_cache[pid] = nn_pid
        # Build combined dyad edge set: target's own truncated edges + neighbor's full edges
        dyad_edges = set()
        w_i = targ_edge_freq[idx]
        w_k = cand_edge_freq[nn]
        for ei, (u, v) in enumerate(edge_list):
            if w_i[ei] > 0 or w_k[ei] > 0:
                dyad_edges.add((u, v))
        dyad_gamma = degree_centrality(dyad_edges) if dyad_edges else {}
        cand_dm = hop_neighbors(source, {u: [v for (uu, v) in dyad_edges if uu == u] for u in range(n_cat)}, max_hops=2)
        ranked_dm = sorted(cand_dm, key=lambda u: -dyad_gamma.get(u, 0))
    else:
        ranked_dm = []

    for L in L_VALUES:
        predictions["TCN-ALL"][L][pid] = set(ranked_all[:L])
        predictions["TCN-CN"][L][pid] = set(ranked_cn[:L])
        predictions["TCN-JI"][L][pid] = set(ranked_ji[:L])
        predictions["TCN-DM"][L][pid] = set(ranked_dm[:L])

    if (idx + 1) % 1000 == 0:
        print(f"  {idx+1}/{n_target} done ({time.time()-t0:.1f}s elapsed)")

print(f"Total prediction time: {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation: Liang's exact micro-averaged Precision/Recall/F1 (eq 16-18)
# ---------------------------------------------------------------------------
results = {}
for m in methods:
    results[m] = {}
    for L in L_VALUES:
        num_inter, num_pred, num_true = 0, 0, 0
        for pid in target_patients:
            R = predictions[m][L].get(pid, set())
            T = last_visit_dx.get(pid, set())
            num_inter += len(R & T)
            num_pred += len(R)
            num_true += len(T)
        precision = num_inter / num_pred if num_pred > 0 else 0
        recall = num_inter / num_true if num_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[m][L] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        print(f"[{m}] L={L}: Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

with open(ROOT + "/work/08_hf_cohort_liang_replication/tcn_prediction_results.json", "w") as f:
    json.dump({"results": results, "n_targets": n_target, "L_values": L_VALUES}, f, indent=2)
print("\nSaved tcn_prediction_results.json")
