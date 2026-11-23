"""
Shared helpers for Stage 13 (SHAP explainability, an extension added in this project; no shipped counterpart).

* FEATURE_GROUPS: every v3 feature is assigned to one interpretable group, so SHAP importance can be summed per group
  (single features are unreliable here because the score, ACCI, lag and trend features are strongly correlated).
* Model fitting reproduces the earlier stages (same splits, same recipe) because no final model was ever saved.
* Plot helpers save matplotlib figures as PNG; the report script embeds them in one HTML file.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def feature_groups(cats, comm_ids):
    g = {}
    for c in cats:
        g[c] = "Charlson category flags"
    for cid in comm_ids:
        g[f"comm{cid}_count"] = "Network communities (Louvain)"
        g[f"comm{cid}_active"] = "Network communities (Louvain)"
    g["dominant_community"] = "Network communities (Louvain)"
    for f in ["charlson_score", "acci_score", "n_categories", "score_trend", "charlson_trend_slope_3adm",
              "charlson_score_lag1", "charlson_score_lag2", "charlson_score_lag3",
              "n_categories_lag1", "n_categories_lag2", "n_categories_lag3", "n_prior_admissions_available"]:
        g[f] = "Burden score and trajectory"
    for f in ["age_at_admission", "age_points", "gender_male"]:
        g[f] = "Age and sex"
    for f in ["n_icu_stays", "icu_los_days", "had_icu_stay", "had_icu_prev_admission", "cum_icu_stays", "cum_icu_los_days",
              "n_procedures", "had_procedure", "n_distinct_procedure_categories", "cum_procedures",
              "drg_severity", "drg_mortality", "has_apr_drg"]:
        g[f] = "ICU, procedures and DRG"
    for f in ["admit_seq", "days_since_prior_admission", "log_days_since_prior", "admissions_per_year_so_far",
              "hospital_expire_flag"]:
        g[f] = "Admission history and timing"
    for c in cats:
        g[f"cum_{c}"] = "Category history (cum_*)"
    return g


def group_importance(sv_values, feature_names, groups):
    """Mean |SHAP| summed by group, as share of the total."""
    imp = np.abs(sv_values).mean(axis=0)
    out = {}
    for f, v in zip(feature_names, imp):
        out[groups[f]] = out.get(groups[f], 0.0) + float(v)
    tot = sum(out.values())
    return {k: {"mean_abs_shap": round(v, 5), "share": round(v / tot, 4)} for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def fit_escalation_general(df, feats, params):
    """Stage 06 recipe: outer 80/20 grouped split (seed 42); refit on the training part with a 15% grouped early-stopping carve-out."""
    y = df["escalation_label"].values
    groups = df["subject_id"].values
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, y, groups))
    tr = df.iloc[train_idx].reset_index(drop=True)
    te = df.iloc[test_idx].reset_index(drop=True)
    i_tr, i_es = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(tr, tr["escalation_label"].values, tr["subject_id"].values))
    X_tr, y_tr = tr.iloc[i_tr][feats], tr["escalation_label"].values[i_tr]
    X_es, y_es = tr.iloc[i_es][feats], tr["escalation_label"].values[i_es]
    pos = y_tr.sum(); neg = len(y_tr) - pos
    m = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=neg / pos, early_stopping_rounds=30,
                          random_state=42, n_jobs=4, **params)
    m.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return m, tr, te


def fit_escalation_hf(df, feats, params):
    """Stage 08 recipe (train_hf_model.py): outer 80/20 (seed 42), inner 75/25 (seed 7), scale_pos_weight 1."""
    y = df["escalation_label"].values
    groups = df["subject_id"].values
    tv_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, y, groups))
    tv = df.iloc[tv_idx].reset_index(drop=True)
    te = df.iloc[test_idx].reset_index(drop=True)
    i_tr, i_va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7).split(tv, tv["escalation_label"].values, tv["subject_id"].values))
    tr, va = tv.iloc[i_tr], tv.iloc[i_va]
    m = xgb.XGBClassifier(n_estimators=600, eval_metric="aucpr", scale_pos_weight=1.0, early_stopping_rounds=30,
                          random_state=42, n_jobs=4, **params)
    m.fit(tr[feats], tr["escalation_label"].values, eval_set=[(va[feats], va["escalation_label"].values)], verbose=False)
    return m, tr.reset_index(drop=True), te


def best_f1(y, p):
    prec, rec, th = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(np.nanmax(f1[:-1]))


def test_metrics(model, te, feats):
    p = model.predict_proba(te[feats])[:, 1]
    y = te["escalation_label"].values
    return {"auroc": round(float(roc_auc_score(y, p)), 4), "auprc": round(float(average_precision_score(y, p)), 4),
            "best_f1_on_test": round(best_f1(y, p), 4), "base_rate": round(float(y.mean()), 4)}


def save_fig(path, tight=True):
    fig = plt.gcf()
    fig.savefig(path, dpi=130, bbox_inches="tight" if tight else None)
    plt.close("all")


def beeswarm(sv, path, max_display=20):
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    save_fig(path)


def bar_groups(group_imp, path, title):
    names = list(group_imp.keys())[::-1]
    vals = [group_imp[n]["share"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(names) + 1.2))
    ax.barh(names, vals, color="#4C78A8")
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def dependence(sv, feature, path):
    plt.figure(figsize=(5.6, 4))
    shap.plots.scatter(sv[:, feature], show=False)
    save_fig(path)


def waterfall(sv_row, path, max_display=12):
    plt.figure()
    shap.plots.waterfall(sv_row, max_display=max_display, show=False)
    save_fig(path)


def force_fragment(base_value, values, X, path):
    """Interactive force plot fragment (HTML + JS call, without the shap library JS)."""
    plot = shap.force_plot(base_value, values, X, matplotlib=False)
    open(path, "w", encoding="utf-8").write(plot.html())


def stability(values, n_boot=5, seed=0, top_k=10):
    """Rank stability of mean |SHAP| across bootstrap resamples of the explained rows."""
    rng = np.random.RandomState(seed)
    base = np.abs(values).mean(axis=0)
    base_top = set(np.argsort(-base)[:top_k])
    rho, overlap = [], []
    from scipy.stats import spearmanr
    for _ in range(n_boot):
        idx = rng.choice(len(values), size=len(values), replace=True)
        b = np.abs(values[idx]).mean(axis=0)
        rho.append(float(spearmanr(base, b)[0]))
        overlap.append(len(base_top & set(np.argsort(-b)[:top_k])) / top_k)
    return {"n_bootstrap": n_boot, "mean_spearman": round(float(np.mean(rho)), 4), "mean_top10_overlap": round(float(np.mean(overlap)), 3)}
