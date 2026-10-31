from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Stage 13, script 4 (extension, no shipped counterpart): assemble the SHAP outputs into ONE self-contained HTML report.

Static figures are embedded as base64 PNG, the interactive force plots are embedded with the SHAP JavaScript, so the file can
be opened in any browser, moved or emailed with nothing else.
Input : everything written by scripts 1 to 3 in work/13_shap_explainability/
Output: work/13_shap_explainability/shap_report.html
"""
import base64, html, json
import shap

OUT = ROOT + "/work/13_shap_explainability/"
FIG = OUT + "figures/"
esc = html.escape


def img(name, alt, width="100%"):
    b = base64.b64encode(open(FIG + name, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{esc(alt)}" style="max-width:{width}">'


def table(rows, headers, fmt=None):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def pct(x):
    return f"{100 * x:.1f}%"


gen = json.load(open(OUT + "escalation_general_population_shap.json"))
hfe = json.load(open(OUT + "escalation_heart_failure_shap.json"))
mlh = json.load(open(OUT + "multilabel_hf_shap.json"))
abl = json.load(open(OUT + "ablation_by_group.json"))

CASE_TEXT = {
    "highest_risk_escalation": "Highest predicted risk among patients who did escalate",
    "missed_escalation_lowest_risk": "A missed escalation: the escalating patient with the lowest predicted risk",
    "lowest_risk_no_escalation": "Lowest predicted risk among patients who did not escalate",
}


def cohort_section(key, title, d):
    s = f"<h2 id='{key}'>{esc(title)}</h2>"
    m = d["refit_metrics"]
    s += (f"<p>Model refit with the {esc(d['reference_stage'])} recipe: test AUROC <b>{m['auroc']}</b> "
          f"(earlier stage: {d['reference_auroc']}), AUPRC {m['auprc']}, base rate {m['base_rate']}. "
          f"SHAP values are for {d['n_explained_rows']:,} held-out test rows, in log-odds units "
          f"(a positive value pushes the prediction toward escalation).</p>")
    g = d["group_importance"]
    comm = g.get("Network communities (Louvain)", {"share": 0})["share"]
    top_group = next(iter(g))
    s += (f"<p class='takeaway'>The <b>{esc(top_group)}</b> features carry the largest share of the model's attributions "
          f"({pct(g[top_group]['share'])}). The Louvain community features carry <b>{pct(comm)}</b>.</p>")
    s += "<h3>Importance by feature group</h3>" + img(f"escalation_{key}_groups.png", "SHAP importance by feature group", "720px")
    s += "<h3>Top 15 features</h3>" + table(
        [(i + 1, t["feature"], t["group"], f"{t['mean_abs_shap']:.4f}", f"{t['mean_shap']:+.4f}") for i, t in enumerate(d["top_features"][:15])],
        ["#", "Feature", "Group", "Mean |SHAP|", "Mean SHAP"])
    s += "<h3>How feature values push the prediction (beeswarm)</h3><p class='note'>Each dot is one patient admission. Red is a high feature value, blue a low one; the horizontal position is the contribution to the prediction.</p>"
    "<p class='note'><b>Reading the direction.</b> A high current Charlson score or a high score trend pushes the predicted risk <i>down</i>. The outcome is a further rise of 2 or more at the next admission, so patients who have already risen steeply have less room to rise again (a likely regression-to-the-mean effect; this is an interpretation, not something SHAP tests). Older age pushes the risk up.</p>"
    s += img(f"escalation_{key}_beeswarm.png", "SHAP beeswarm", "860px")
    s += "<h3>Dependence plots for the leading non-flag features</h3><div class='grid'>"
    for f in d["dependence_features"]:
        s += f"<div>{img(f'escalation_{key}_dep_{f}.png', f'dependence {f}', '100%')}</div>"
    s += "</div><h3>Individual patient explanations</h3>"
    for name, info in d["local_cases"].items():
        s += (f"<h4>{esc(CASE_TEXT[name])}</h4><p class='note'>subject {info['subject_id']}, admission {info['hadm_id']}: predicted probability "
              f"{info['predicted_probability']}, actual escalation = {info['escalation_label']}</p>"
              + img(f"escalation_{key}_case_{name}.png", name, "760px"))
    s += (f"<h3>Interactive force plot ({d['n_force_rows']} patients)</h3><p class='note'>Hover for values. Use the two drop-downs to change the "
          "ordering and the y-axis. Each bar column is one patient.</p><div class='force'>"
          + open(OUT + f"force_escalation_{key}.html", encoding="utf-8").read() + "</div>")
    st = d["bootstrap_stability"]
    s += (f"<h3>Checks</h3><ul><li>Rank agreement with XGBoost's built-in feature importance: Spearman {d['spearman_with_builtin_importance']}.</li>"
          f"<li>Stability of the importance ranking over {st['n_bootstrap']} bootstrap resamples of the explained rows: Spearman {st['mean_spearman']}, "
          f"top-10 overlap {st['mean_top10_overlap']}. This measures sampling noise in the explained rows only, not model-refit variability.</li></ul>")
    return s


def multilabel_section():
    cats = mlh["categories"]
    s = "<h2 id='multilabel'>Next-visit diagnoses model (heart-failure cohort)</h2>"
    s += (f"<p>The 17 per-category classifiers of Stage 09, refit on the heart-failure cohort ({mlh['n_patients']:,} patients) and explained on "
          f"{mlh['n_explained_rows']:,} held-out test pairs. Each model predicts whether one Charlson category is present at the next admission.</p>")
    s += "<h3>Importance by feature group, per category</h3>" + img("multilabel_hf_group_heatmap.png", "category by feature group heatmap", "880px")
    own = [(c, v["own_flag_rank"]) for c, v in cats.items()]
    first = sum(1 for _, r in own if r == 1)
    s += (f"<p class='takeaway'>For {first} of {len(own)} categories the single most important feature is the category's own flag at the current visit, "
          "that is, the persistence signal. The remaining features add the extra information the model uses.</p>")
    rows = []
    for c, v in cats.items():
        rows.append((c, f"{v['prevalence_next_visit']:.3f}", f"{v['test_auroc']:.3f}", v["own_flag_rank"],
                     ", ".join(t["feature"] for t in v["top_features"][:4])))
    s += "<h3>Per category</h3>" + table(rows, ["Category", "Next-visit prevalence", "Test AUROC", "Rank of own current flag", "Top 4 features"])
    s += "<h3>Beeswarm plots for the three most prevalent categories</h3>"
    for c in mlh["beeswarm_categories"]:
        s += f"<h4>{esc(c)}</h4>" + img(f"multilabel_hf_beeswarm_{c.replace(' ', '_').replace('/', '_')}.png", c, "820px")
    return s


def ablation_section():
    s = "<h2 id='ablation'>Do the network modules add predictive information? Leave-one-group-out ablation</h2>"
    s += ("<p>SHAP shows attribution, not causation, and splits credit between correlated features. As a complementary check the "
          "general-population escalation model was refit with each feature group removed in turn.</p>")
    f = abl["full"]
    rows = [("Full model (57 features)", f["n_features"], f["auroc"], f["auprc"], f["best_f1_on_test"], "-", "-")]
    for g, v in abl.items():
        if g in ("full", "flags_only", "flags_plus_communities"):
            continue
        rows.append((f"Without: {g}", f["n_features"] - v["n_features_removed"], v["auroc"], v["auprc"], v["best_f1_on_test"],
                     f"{v['delta_auroc']:+.4f}", f"{v['delta_auprc']:+.4f}"))
    s += table(rows, ["Model", "Features", "AUROC", "AUPRC", "Best F1 (test-tuned)", "Change in AUROC", "Change in AUPRC"])
    a, b = abl["flags_only"], abl["flags_plus_communities"]
    s += (f"<p class='takeaway'>Using only the 17 Charlson flags gives AUROC {a['auroc']}; adding the 7 Louvain community features gives "
          f"{b['auroc']} (change {b['auroc'] - a['auroc']:+.4f}).</p>")
    s += "<p class='note'>A single XGBoost refit varies by about 0.0005 AUROC between runs, so changes below about 0.001 are not meaningful.</p>"
    return s


CSS = """
:root{--bg:#fbfaf8;--fg:#1f2328;--muted:#5f6670;--card:#ffffff;--line:#e3e0da;--accent:#2f6f9f}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--fg:#e8e6e1;--muted:#a3a8b0;--card:#202328;--line:#34383e;--accent:#6aaed6}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1020px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:1.9rem;margin:0 0 4px} h2{margin-top:56px;padding-top:10px;border-top:1px solid var(--line);font-size:1.45rem}
h3{margin-top:30px;font-size:1.1rem} h4{margin:22px 0 4px;font-size:1rem}
.sub{color:var(--muted);margin:0 0 24px}
nav a{margin-right:14px;color:var(--accent);text-decoration:none;font-size:.95rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:10px 0;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top} th{background:var(--bg)}
img{background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px;height:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.note{color:var(--muted);font-size:.9rem;margin:4px 0 8px}
.takeaway{background:var(--card);border-left:4px solid var(--accent);padding:10px 14px;margin:14px 0}
.force{background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px;overflow-x:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card b{font-size:1.3rem;display:block}
@media(max-width:600px){main{padding:16px}}
"""


def card(title, d):
    g = d["group_importance"]
    top = list(g.items())[:3]
    comm = g.get("Network communities (Louvain)", {"share": 0})["share"]
    tops = ", ".join(t["feature"] for t in d["top_features"][:3])
    return (f"<div class='card'><span class='note'>{esc(title)}</span><b>AUROC {d['refit_metrics']['auroc']}</b>"
            f"<div>Top features: {esc(tops)}</div><div class='note'>Top group: {esc(top[0][0])} ({pct(top[0][1]['share'])}); "
            f"Louvain communities {pct(comm)}</div></div>")


page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHAP explainability report</title><style>{CSS}</style>{shap.getjs()}</head><body><main>
<h1>SHAP explainability report</h1>
<p class="sub">MIMIC-IV multimorbidity pipeline, Stage 13 (extension). Models are XGBoost classifiers explained with TreeExplainer on held-out test rows.</p>
<nav><a href="#general_population">General population</a><a href="#heart_failure">Heart failure</a><a href="#multilabel">Next-visit diagnoses</a><a href="#ablation">Ablation</a><a href="#method">Method and caveats</a></nav>
<div class="cards">{card('Escalation model, general population', gen)}{card('Escalation model, heart-failure cohort', hfe)}</div>
<p><b>What SHAP is.</b> For every prediction, SHAP splits the model's output into one contribution per feature, so you can see which inputs pushed a
patient's predicted risk up or down. Averaging the absolute contributions over many patients ranks the features. Contributions are in log-odds
units. Because related features (Charlson score, age-adjusted score, lagged scores, trend) share credit, results are also summed by feature group.</p>
{cohort_section('general_population', 'Escalation model, general population (223,291 patients)', gen)}
{cohort_section('heart_failure', 'Escalation model, heart-failure cohort (21,975 patients)', hfe)}
{multilabel_section()}
{ablation_section()}
<h2 id="method">Method and caveats</h2>
<ul>
<li>The final models were not saved in earlier stages, so they were refit with the same splits and parameters. The refit AUROCs equal the earlier stages' (see each section).</li>
<li>SHAP values were computed on a random sample of up to 5,000 held-out test rows, never on training data.</li>
<li>SHAP describes what the model uses, not what causes escalation. Strongly correlated features share credit, so read feature groups first.</li>
<li>Feature groups: Charlson category flags; Louvain community features; burden score and trajectory (score, age-adjusted score, lags, trend);
age and sex; ICU, procedures and DRG; admission history and timing.</li>
<li>The bootstrap stability check resamples the explained rows only. Variability across model refits was not measured except in the ablation note.</li>
<li>The next-visit model section covers the heart-failure cohort only. Other cohorts and the general-population multi-label model were not explained.</li>
</ul></main></body></html>"""
open(OUT + "shap_report.html", "w", encoding="utf-8").write(page)
print(f"Saved shap_report.html ({len(page) / 1e6:.1f} MB)")
