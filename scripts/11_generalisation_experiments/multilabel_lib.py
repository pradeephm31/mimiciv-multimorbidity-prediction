"""
Shared helpers for the extension stages 11 and 12 (added in this project; no shipped counterpart).

Reproduces the data preparation, split and model of Stages 09/10 so that results are comparable:
  * pairs = (admission t, admission t+1) of the same patient; labels = the 17 Charlson flags at t+1
  * outer split: GroupShuffleSplit 80/20 (seed 42); inner split of the 80%: 75/25 (seed 7); grouped by patient
  * one XGBoost classifier per category with the Stage 06 hyperparameters (400 trees max, early stopping 25)
  * evaluation: last transition of each test patient, Liang's micro-averaged precision / recall / F1 at top-L
Optional "history" features add, per category, how many admissions up to and including the current one had it
(cum_<category>), which uses only information available at the current admission.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

L_VALUES = [3, 5, 10]


def base_features(cats, comm_ids):
    return (
        list(cats) + [f"comm{c}_count" for c in comm_ids] + [f"comm{c}_active" for c in comm_ids]
        + ["charlson_score", "acci_score", "age_points", "n_categories", "admit_seq", "score_trend",
           "dominant_community", "hospital_expire_flag",
           "age_at_admission", "gender_male", "n_icu_stays", "icu_los_days", "had_icu_stay",
           "had_icu_prev_admission", "cum_icu_stays", "cum_icu_los_days",
           "n_procedures", "had_procedure", "n_distinct_procedure_categories", "cum_procedures",
           "drg_severity", "drg_mortality", "has_apr_drg",
           "days_since_prior_admission", "log_days_since_prior", "admissions_per_year_so_far",
           "charlson_score_lag1", "charlson_score_lag2", "charlson_score_lag3",
           "n_categories_lag1", "n_categories_lag2", "n_categories_lag3",
           "charlson_trend_slope_3adm", "n_prior_admissions_available"]
    )


def make_pairs(adm, cat_to_comm, cats, patient_mask=None):
    """Admission-pair table. patient_mask: optional set of subject_ids to keep (all their admissions are kept)."""
    adm = adm.sort_values(["subject_id", "admit_seq"]).reset_index(drop=True).copy()
    if patient_mask is not None:
        adm = adm[adm.subject_id.isin(patient_mask)].reset_index(drop=True)
    comm_ids = sorted(set(cat_to_comm.values()))
    for cid in comm_ids:
        in_comm = [c for c in cats if cat_to_comm[c] == cid]
        adm[f"comm{cid}_count"] = adm[in_comm].sum(axis=1)
        adm[f"comm{cid}_active"] = (adm[f"comm{cid}_count"] > 0).astype(int)
    counts = adm[[f"comm{cid}_count" for cid in comm_ids]].values
    adm["dominant_community"] = np.where(counts.max(axis=1) == 0, -1, np.array(comm_ids)[counts.argmax(axis=1)])

    g = adm.groupby("subject_id")
    for c in cats:
        adm[f"cum_{c}"] = g[c].cumsum()          # admissions up to and including the current one with this category
        adm[f"next__{c}"] = g[c].shift(-1)
    adm["next_score"] = g["charlson_score"].shift(-1)
    has_next = g["admit_seq"].shift(-1).notna()
    pairs = adm[has_next].copy()
    pairs["escalation_label"] = ((pairs.next_score - pairs.charlson_score) >= 2).astype(int)
    pairs["score_trend"] = pairs["charlson_score"] - pairs["charlson_score_lag1"].fillna(pairs["charlson_score"])
    label_cols = [f"next__{c}" for c in cats]
    for c in label_cols:
        pairs[c] = pairs[c].astype(int)
    feats = base_features(cats, comm_ids)
    hist_cols = [f"cum_{c}" for c in cats]
    return pairs.reset_index(drop=True), feats, hist_cols, label_cols


def split_pairs(pairs):
    y = pairs["escalation_label"].values
    groups = pairs["subject_id"].values
    trainval_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(pairs, y, groups))
    tv = pairs.iloc[trainval_idx].reset_index(drop=True)
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7).split(tv, tv["escalation_label"].values, tv["subject_id"].values))
    return tv.iloc[tr_i].reset_index(drop=True), tv.iloc[va_i].reset_index(drop=True), pairs.iloc[test_idx].reset_index(drop=True)


def train_multilabel(tr, va, te, feats, label_cols, params, n_estimators=400, log=print):
    """One binary XGBoost per label; returns probabilities for te, shape (len(te), n_labels)."""
    proba = np.zeros((len(te), len(label_cols)))
    for i, lab in enumerate(label_cols):
        y_tr, y_va = tr[lab].values, va[lab].values
        pos = y_tr.sum()
        if pos < 5 or pos > len(y_tr) - 5:
            proba[:, i] = y_tr.mean()
            continue
        m = xgb.XGBClassifier(n_estimators=n_estimators, eval_metric="aucpr", scale_pos_weight=1.0,
                              early_stopping_rounds=25, random_state=42, n_jobs=4, **params)
        m.fit(tr[feats], y_tr, eval_set=[(va[feats], y_va)], verbose=False)
        proba[:, i] = m.predict_proba(te[feats])[:, 1]
        log(f"  [{i + 1}/{len(label_cols)}] {lab}: pos_rate_train={pos / len(y_tr):.3f}, best_iter={m.best_iteration}")
    return proba


def last_transition_rows(te):
    return np.where(te.groupby("subject_id")["admit_seq"].transform("max").values == te["admit_seq"].values)[0]


def score_sets(Y, rows, rank_fn):
    """Liang's micro-averaged precision / recall / F1. rank_fn(row) -> set of predicted category indices."""
    ni = npred = ntrue = 0
    for r in rows:
        R = rank_fn(r)
        T = set(np.where(Y[r])[0].tolist())
        ni += len(R & T); npred += len(R); ntrue += len(T)
    p = ni / npred if npred else 0.0
    rc = ni / ntrue if ntrue else 0.0
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4), "n_predicted": int(npred), "n_true": int(ntrue)}


def top_by(scores_row, k):
    return set(np.argsort(-scores_row)[:k].tolist()) if k > 0 else set()


class Rankers:
    """Prediction rules. cur: bool (n, n_cat) current flags; cum: counts (n, n_cat) up to the current admission."""
    def __init__(self, cur, cum, proba=None):
        self.cur, self.cum, self.P = cur, cum, proba
        self.n_cur = cur.sum(axis=1).astype(int)
        self.n_ever = (cum > 0).sum(axis=1).astype(int)

    def persistence_current(self, L):
        return lambda r: set(np.where(self.cur[r])[0][:L].tolist())

    def persistence_history(self, L):
        def f(r):
            ev = np.where(self.cum[r] > 0)[0]
            order = ev[np.argsort(-self.cum[r][ev], kind="stable")]
            return set(order[:L].tolist())
        return f

    def model_rule_a(self, L, P=None):      # at least one prediction (rule used in Stages 09 and 10)
        P = self.P if P is None else P
        return lambda r: top_by(P[r], min(L, max(self.n_cur[r], 1)))

    def model_rule_b(self, L, P=None):      # as many predictions as current diagnoses (none if none)
        P = self.P if P is None else P
        return lambda r: top_by(P[r], min(L, self.n_cur[r]))

    def model_rule_ever(self, L, P=None):   # as many predictions as distinct categories ever diagnosed so far
        P = self.P if P is None else P
        return lambda r: top_by(P[r], min(L, self.n_ever[r]))
