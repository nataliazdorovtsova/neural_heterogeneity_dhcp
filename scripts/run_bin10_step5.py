"""Step 5 of RESEARCH_PLAN_BIN10.md (section 6): polygenic scores and the
position in the spread (associative, cross-sectional).

    NH_RUN=bin10_no7 python scripts/run_bin10_step5.py

Sample: infants with age-adjusted positions (D21, without the 7 `low7`, D20)
and polygenic scores (EUR ancestry only, D9): n = 411. The scores and
ancestry PCs are read in place (paths.PRS, paths.ANCESTRY_PCS), merged in
memory and never written out; every output holds summary statistics only.

Model, one score at a time:

    y ~ PRS (z) + PC1-10 + male + GA at birth + rcs(PMA at scan, 4 knots)

y z-scored; OLS with standard errors clustered by family (twins share
their genotype, so they are not independent; CR1).

Families (D22):
- primary: 10 scores x axes 1-3 = 30 tests, BH FDR;
- positive controls: volume, ICVF, FA (beside the family);
- exploratory: all 48 scores x axes 1-3, BH FDR over 144;
- lobes (secondary): the 10 primary scores x 12 lobe z-scores, BH over 120.
Sensitivity sets (primary scores, axes): term-born only; PC1-5; training
infants only. Exploratory, flagged: PRS x PMA (BH over the 30).

Writes results/<run>/step5/prs.csv, step5_summary.json; figure b11_prs.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import paths
import figstyle as fs
from scripts.run_bin10_step4 import AXES, LOBES, AXIS_LABELS, rcs, ols_cluster, bh, load_data

OUT = paths.RESULTS / "step5"
PRIMARY = ["adhd2022", "asd2019", "asd_early_2025", "scz2022", "bip2024_noUKBB",
           "mdd2025_noUKBB", "anx2024", "ocd2024", "an2019", "sud2023"]
CONTROLS = ["volume", "ICVF", "FA"]
NEURO = ["epi2023", "alz2022", "pd2019", "als2021", "vasdem2024", "glioma2023"]
BRAIN = ["volume", "ICVF", "FA", "MD", "ISOVF", "OD", "CT", "SA", "SC_GMV", "GC", "MC", "FI", "ICI",
         "cortical_expansion", "scgmv_adj_expansion"]
NAMES = {"adhd2022": "ADHD", "asd2019": "autism", "asd_early_2025": "autism, early diagnosis",
         "scz2022": "schizophrenia", "bip2024_noUKBB": "bipolar disorder", "mdd2025_noUKBB": "depression",
         "anx2024": "anxiety", "ocd2024": "OCD", "an2019": "anorexia nervosa", "sud2023": "substance use",
         "volume": "brain volume", "ICVF": "neurite density (ICVF)", "FA": "FA"}


def load(positions="positions_scaled.csv"):
    d = load_data(positions)
    prs = pd.read_csv(paths.PRS)
    prs["dhcp_id"] = prs.participant_id.astype(str).str.replace("sub-", "", regex=False)
    scores = [c for c in prs.columns if c not in ("participant_id", "dhcp_id")]
    pcs = pd.read_csv(paths.ANCESTRY_PCS, sep=r"\s+", header=None)
    pcs = pcs.iloc[:, [1] + list(range(2, 12))]
    pcs.columns = ["dhcp_id"] + [f"PC{i}" for i in range(1, 11)]
    d = d.merge(prs.drop(columns="participant_id"), on="dhcp_id").merge(pcs, on="dhcp_id")
    return d.reset_index(drop=True), scores


def fit(d, y_col, s, n_pc=10, interaction=False):
    cols = [y_col, s, "male", "ga_birth", "scan_age", "family"] + [f"PC{i}" for i in range(1, n_pc + 1)]
    d = d[cols].dropna()
    y = d[y_col].values
    y = (y - y.mean()) / y.std(ddof=1)
    x = d[s].values
    x = (x - x.mean()) / x.std(ddof=1)
    Z = [np.ones(len(d)), x, d.male.values, d.ga_birth.values, rcs(d.scan_age.values),
         d[[f"PC{i}" for i in range(1, n_pc + 1)]].values]
    j = 1
    if interaction:
        u = (d.scan_age.values - 40) / 5
        Z.append(x * u)
        j = -1
    X = np.column_stack(Z)
    b, se, df = ols_cluster(y, X, d.family.values)
    tq = stats.t.ppf(0.975, df)
    t = b[j] / se[j]
    return dict(beta=b[j], lo=b[j] - tq * se[j], hi=b[j] + tq * se[j], t=t,
                p=2 * stats.t.sf(abs(t), df), n=len(d))


def run(d, scores):
    rows = []
    for s in scores:
        for y in AXES:
            rows.append(dict(set="main", term="prs", outcome=y, score=s, **fit(d, y, s)))
        if s in PRIMARY or s in CONTROLS:
            for y in LOBES:
                rows.append(dict(set="main", term="prs", outcome=y, score=s, **fit(d, y, s)))
    for s in PRIMARY + CONTROLS:
        for y in AXES:
            rows.append(dict(set="term_born", term="prs", outcome=y, score=s, **fit(d[d.ga_birth >= 37], y, s)))
            rows.append(dict(set="pc1_5", term="prs", outcome=y, score=s, **fit(d, y, s, n_pc=5)))
            rows.append(dict(set="train_only", term="prs", outcome=y, score=s, **fit(d[d.split == "train"], y, s)))
            rows.append(dict(set="main", term="prs_x_pma", outcome=y, score=s, **fit(d, y, s, interaction=True)))
    r = pd.DataFrame(rows)
    r["family"] = ""
    r["q"] = np.nan
    def fam(mask, name):
        r.loc[mask, "family"] = name
        r.loc[mask, "q"] = bh(r.loc[mask, "p"])
    main_ax = (r.set == "main") & (r.term == "prs") & r.outcome.isin(AXES)
    fam(main_ax & r.score.isin(PRIMARY), "primary")
    r["q_exploratory"] = np.nan
    r.loc[main_ax, "q_exploratory"] = bh(r.loc[main_ax, "p"])
    fam((r.set == "main") & (r.term == "prs") & r.outcome.isin(LOBES) & r.score.isin(PRIMARY), "lobes")
    fam((r.set == "main") & (r.term == "prs_x_pma") & r.score.isin(PRIMARY), "interaction")
    return r


def figure(r, scores, n):
    fs.apply_style()
    m = r[(r.set == "main") & (r.term == "prs") & r.outcome.isin(AXES)]
    other_psych = [s for s in scores if s not in PRIMARY + NEURO + BRAIN]
    groups = [("primary family (D22)", PRIMARY), ("other psychiatric and subtypes", other_psych),
              ("brain imaging", [s for s in BRAIN if s in scores]), ("neurological", NEURO)]
    order = [s for _, g in groups for s in g]
    T = np.array([[m[(m.score == s) & (m.outcome == a)].t.iloc[0] for a in AXES] for s in order])
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], wspace=0.55, left=0.19, right=0.98, top=0.91, bottom=0.08)

    ax = fig.add_subplot(gs[0])
    vmax = max(3.0, float(np.ceil(np.abs(T).max())))
    im = ax.imshow(T, cmap=fs.DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, s in enumerate(order):
        for j, a in enumerate(AXES):
            row = m[(m.score == s) & (m.outcome == a)].iloc[0]
            mark = "*" if (row.family == "primary" and row.q < 0.05) else ("†" if row.q_exploratory < 0.05 else "")
            ax.text(j, i, f"{row.t:+.1f}{mark}", ha="center", va="center", fontsize=8,
                    color="white" if abs(row.t) > 0.6 * vmax else fs.TEXT_PRIMARY)
    ax.set_yticks(range(len(order)), [NAMES.get(s, s) if s in PRIMARY else s for s in order], fontsize=8.5)
    for lab, s in zip(ax.get_yticklabels(), order):
        if s in PRIMARY:
            lab.set_fontweight("bold")
    ax.set_xticks(range(3), ["axis 1", "axis 2", "axis 3"])
    ax.xaxis.tick_top()
    k = 0
    for name, g in groups:
        if k:
            ax.axhline(k - 0.5, color="white", lw=4)
        ax.text(-0.82, k - 0.5 + len(g) / 2, name, transform=ax.get_yaxis_transform(), ha="center",
                va="center", rotation=90, fontsize=9, color=fs.TEXT_MUTED, style="italic")
        k += len(g)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("A   t-values, all 48 scores", pad=28)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02)
    cb.set_label("t (score → axis)", fontsize=10)

    # B: forest, primary family + positive controls, one column per axis
    sub = gs[1].subgridspec(1, 3, wspace=0.08)
    rows = PRIMARY + CONTROLS
    lim = max(0.3, float(r[(r.set == "main") & (r.term == "prs") & r.outcome.isin(AXES)
                           & r.score.isin(rows)][["lo", "hi"]].abs().max().max()) + 0.02)
    for j, a in enumerate(AXES):
        ax = fig.add_subplot(sub[j])
        for i, s in enumerate(rows[::-1]):
            row = m[(m.score == s) & (m.outcome == a)].iloc[0]
            col = fs.COLOR_CYCLE[0] if s in PRIMARY else fs.NEUTRAL_GREY
            filled = row.family == "primary" and row.q < 0.05
            ax.plot([row.lo, row.hi], [i, i], color=col, lw=2)
            ax.plot(row.beta, i, "o", color=col, ms=7, mfc=col if filled else "white", mew=1.8)
        fs.reference_line(ax, 0, axis="x")
        ax.axhline(len(CONTROLS) - 0.5, color=fs.NEUTRAL_GREY, lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(range(len(rows)), [NAMES[s] for s in rows[::-1]] if j == 0 else [])
        ax.set_title(AXIS_LABELS[a].replace(" vs ", " vs\n"), fontsize=9.5)
        if j == 1:
            ax.set_xlabel("standardised β with 95% CI")
    fig.text(0.60, 0.975, "B   Primary family (orange) and positive controls (grey)", ha="center",
             fontsize=16, fontweight="bold")
    fig.text(0.02, 0.012,
             f"Run {paths.RUN}, n = {n} infants with polygenic scores (EUR ancestry). Each cell: y ~ score + PC1–10 + "
             "sex + GA at birth + PMA spline, SEs clustered by family.\n*: passes BH FDR in the primary family "
             "(30 tests); †: passes BH FDR over all 48 scores × 3 axes (144 tests). Filled points: pass in the "
             "primary family. Scores computed with PRS-CS; GWAS sources to confirm (D22).",
             fontsize=fs.NOTE_SIZE - 1, color=fs.TEXT_SECONDARY, va="bottom")
    fs.save_figure(fig, paths.FIGURES / "b11_prs")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, scores = load()
    assert all(s in scores for s in PRIMARY + CONTROLS)
    r = run(d, scores)
    r.to_csv(OUT / "prs.csv", index=False)
    m = r[(r.set == "main") & (r.term == "prs")]
    prim = m[m.family == "primary"].sort_values("p")
    sens = r[r.score.isin(PRIMARY + CONTROLS) & r.outcome.isin(AXES) & (r.term == "prs")]
    summ = dict(
        n=int(len(d)), n_term=int((d.ga_birth >= 37).sum()), n_train=int((d.split == "train").sum()),
        n_families_multi=int((d.family.value_counts() > 1).sum()),
        power_note="80% power at alpha 0.05 (one test) for |r| = 0.14 at n = 400",
        primary=prim[["score", "outcome", "beta", "lo", "hi", "p", "q"]].round(4).to_dict("records"),
        primary_passing=prim[prim.q < 0.05][["score", "outcome", "beta", "p", "q"]].round(4).to_dict("records"),
        controls=m[m.score.isin(CONTROLS) & m.outcome.isin(AXES)][["score", "outcome", "beta", "lo", "hi", "p"]]
        .round(4).to_dict("records"),
        exploratory_passing=m[m.q_exploratory < 0.05][["score", "outcome", "beta", "p", "q_exploratory"]]
        .round(4).to_dict("records"),
        exploratory_smallest_p=m[m.outcome.isin(AXES)].nsmallest(8, "p")[["score", "outcome", "beta", "p",
                                                                         "q_exploratory"]].round(4).to_dict("records"),
        lobes_passing=m[(m.family == "lobes") & (m.q < 0.05)][["score", "outcome", "beta", "p", "q"]]
        .round(4).to_dict("records"),
        lobes_smallest_q=float(m[m.family == "lobes"].q.min()),
        interaction_passing=r[(r.family == "interaction") & (r.q < 0.05)][["score", "outcome", "beta", "p", "q"]]
        .round(4).to_dict("records"),
        interaction_smallest_p=float(r[r.family == "interaction"].p.min()),
        sensitivity_primary_top=sens[sens.score.isin(prim.head(3).score) & sens.outcome.isin(prim.head(3).outcome)]
        .pivot_table(index=["score", "outcome"], columns="set", values="beta").round(3).reset_index()
        .to_dict("records"),
    )
    (OUT / "step5_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    figure(r, scores, len(d))
    with pd.option_context("display.width", 200):
        print(prim[["score", "outcome", "n", "beta", "lo", "hi", "p", "q"]].round(3).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k != "primary"}, indent=1, default=float))


if __name__ == "__main__":
    main()
