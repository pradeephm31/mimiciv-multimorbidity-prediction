from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
import pandas as pd
import numpy as np
from scipy import stats
import json
import sys
sys.path.insert(0, ROOT + "/scripts/01_charlson_mapping")
from charlson_map import CATEGORY_ORDER

hf_adm = pd.read_parquet(ROOT + "/work/08_hf_cohort_liang_replication/hf_cohort_adm.parquet").sort_values(["subject_id", "admit_seq"]).reset_index(drop=True)
n_cat = len(CATEGORY_ORDER)
cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}

mat = hf_adm[CATEGORY_ORDER].values.astype(bool)
hf_adm["next_subject"] = hf_adm.subject_id.shift(-1)
valid = (hf_adm.next_subject == hf_adm.subject_id).values[:-1]
A_t = mat[:-1][valid]       # visit t diagnosis presence
A_t1 = mat[1:][valid]       # visit t+1 diagnosis presence
n_transitions = A_t.shape[0]
print(f"Total adjacent-visit transition instances: {n_transitions:,}")

# ---------------------------------------------------------------------------
# For every ordered pair (i, j), build the 2x2 contingency table for
#   present(i at t) x present(j at t+1)
# and run a chi-square test of independence. This mirrors Liang's "pairwise
# statistical analysis... using Chi-square test" to retain only TCPs with a
# statistically significant relationship.
# ---------------------------------------------------------------------------
freq_i_t = A_t.sum(axis=0)          # marginal: how often i present at t
freq_j_t1 = A_t1.sum(axis=0)        # marginal: how often j present at t+1
joint = A_t.T.astype(np.int64) @ A_t1.astype(np.int64)   # joint count(i at t AND j at t+1)

edges = []
for i in range(n_cat):
    for j in range(n_cat):
        a = joint[i, j]                                   # i@t, j@t+1
        b = freq_i_t[i] - a                                # i@t, not-j@t+1
        c = freq_j_t1[j] - a                               # not-i@t, j@t+1
        d = n_transitions - a - b - c                      # neither
        table = np.array([[a, b], [c, d]])
        if a < 5:   # too sparse to test meaningfully
            continue
        try:
            chi2, p, dof, exp = stats.chi2_contingency(table, correction=True)
        except ValueError:
            continue
        if p < 0.05:
            edges.append({"i": i, "j": j, "cat_i": CATEGORY_ORDER[i], "cat_j": CATEGORY_ORDER[j],
                           "count": int(a), "chi2": float(chi2), "p_chi2": float(p)})

print(f"Significant directed TCPs (chi-square p<0.05): {len(edges)}")

# ---------------------------------------------------------------------------
# Binomial directionality test: for each unordered pair {i,j} that passed
# chi-square in at least one direction, compare count(i->j) vs count(j->i)
# to establish a preferred direction (p<0.05) or mark bi-directional.
# ---------------------------------------------------------------------------
edge_lookup = {(e["i"], e["j"]): e for e in edges}
final_edges = []
seen_pairs = set()
for e in edges:
    i, j = e["i"], e["j"]
    if i == j:
        # self-transition (recurrence) -- always keep, no direction to test
        final_edges.append({**e, "direction": "self"})
        continue
    pair = tuple(sorted([i, j]))
    if pair in seen_pairs:
        continue
    seen_pairs.add(pair)
    n_ij = edge_lookup.get((i, j), {}).get("count", 0)
    n_ji = edge_lookup.get((j, i), {}).get("count", 0)
    total = n_ij + n_ji
    if total == 0:
        continue
    p_binom = stats.binomtest(n_ij, total, 0.5).pvalue
    if p_binom < 0.05 and n_ij > n_ji:
        final_edges.append({**edge_lookup[(i, j)], "direction": "forward"})
    elif p_binom < 0.05 and n_ji > n_ij:
        final_edges.append({**edge_lookup[(j, i)], "direction": "forward"})
    else:
        # bi-directional: keep both directions
        if (i, j) in edge_lookup:
            final_edges.append({**edge_lookup[(i, j)], "direction": "bidirectional"})
        if (j, i) in edge_lookup:
            final_edges.append({**edge_lookup[(j, i)], "direction": "bidirectional"})

print(f"Final TCN edges after directionality filtering: {len(final_edges)}")

# Node attribute: overall prevalence (frequency of category across all admissions)
node_freq = {CATEGORY_ORDER[i]: int(mat[:, i].sum()) for i in range(n_cat)}

tcn = {
    "nodes": [{"category": c, "frequency": node_freq[c]} for c in CATEGORY_ORDER],
    "edges": [{"source": e["cat_i"], "target": e["cat_j"], "weight": e["count"],
               "direction": e["direction"]} for e in final_edges],
    "n_transitions_analyzed": int(n_transitions),
}
with open(ROOT + "/work/08_hf_cohort_liang_replication/hf_population_tcn.json", "w") as f:
    json.dump(tcn, f, indent=2)
print("\nSaved hf_population_tcn.json")
print("\nSample edges:")
for e in final_edges[:15]:
    print(f"  {e['cat_i']} -> {e['cat_j']}  (n={e['count']}, {e['direction']})")
