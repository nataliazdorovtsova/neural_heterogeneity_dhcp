"""Step 5b (DECISIONS D23): five genetic factors and the position in the spread.

    NH_RUN=bin10_no7 python scripts/run_bin10_step5b.py

Defined after the step 5 null (S39) and committed before this run. Factors
follow Grotzinger et al. 2022: each factor score is the mean of its z-scored
polygenic scores (z within the analysis sample), z-scored again. Model as in
step 5 (scripts/run_bin10_step5.py): y ~ factor + PC1-10 + male + GA at birth
+ PMA spline, SEs clustered by family.

Primary: 5 factors x axes 1-3 = 15 tests, BH FDR. Secondary: 5 x 12 lobes,
BH. Sensitivity: term-born, PC1-5, training only.
Summary statistics only. Writes results/<run>/step5b/; figure b11b_prs_factors.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import paths
import figstyle as fs
from scripts.build_bin10 import GROUP_NAMES
from scripts.run_bin10_step4 import AXES, LOBES, AXIS_LABELS, bh
from scripts.run_bin10_step5 import load, fit

OUT = paths.RESULTS / "step5b"
FACTORS = {"neurodevelopmental": ["adhd2022", "asd2019"],
           "psychotic": ["scz2022", "bip2024_noUKBB"],
           "internalising": ["mdd2025_noUKBB", "anx2024"],
           "compulsive": ["ocd2024", "an2019"],
           "substance use": ["sud2023"]}


def add_factors(d):
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    for f, cols in FACTORS.items():
        d[f] = z(np.mean([z(d[c]) for c in cols], axis=0))
    return d


def figure(r, n):
    fs.apply_style()
    m = r[(r.set == "main")]
    fig = plt.figure(figsize=(16, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.35, left=0.12, right=0.97, top=0.86, bottom=0.2)
    sub = gs[0].subgridspec(1, 3, wspace=0.08)
    names = list(FACTORS)
    lim = max(0.3, float(m[m.outcome.isin(AXES)][["lo", "hi"]].abs().max().max()) + 0.02)
    for j, a in enumerate(AXES):
        ax = fig.add_subplot(sub[j])
        for i, f in enumerate(names[::-1]):
            row = m[(m.score == f) & (m.outcome == a)].iloc[0]
            filled = row.q < 0.05
            ax.plot([row.lo, row.hi], [i, i], color=fs.COLOR_CYCLE[0], lw=2.2)
            ax.plot(row.beta, i, "o", color=fs.COLOR_CYCLE[0], ms=8,
                    mfc=fs.COLOR_CYCLE[0] if filled else "white", mew=2)
            ax.text(lim * 0.97, i + 0.22, f"p = {row.p:.2f}", ha="right", fontsize=8, color=fs.TEXT_MUTED)
        fs.reference_line(ax, 0, axis="x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_yticks(range(len(names)), names[::-1] if j == 0 else [])
        ax.set_title(AXIS_LABELS[a].replace(" vs ", " vs\n"), fontsize=9.5)
        if j == 1:
            ax.set_xlabel("standardised β with 95% CI")
    fig.text(0.33, 0.955, "A   Genetic factors and axes 1–3 (15 tests, BH FDR)", ha="center",
             fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    lob = m[m.outcome.isin(LOBES)]
    T = np.array([[lob[(lob.score == f) & (lob.outcome == l)].t.iloc[0] for l in LOBES] for f in names])
    vmax = max(3.0, float(np.ceil(np.abs(T).max())))
    ax.imshow(T, cmap=fs.DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, f in enumerate(names):
        for j, l in enumerate(LOBES):
            row = lob[(lob.score == f) & (lob.outcome == l)].iloc[0]
            ax.text(j, i, f"{row.t:+.1f}" + ("*" if row.q < 0.05 else ""), ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(LOBES)), [g.replace("_", " ") for g in GROUP_NAMES], rotation=45, ha="right",
                  fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("B   t-values, lobe z-scores (60 tests, BH)", pad=12)
    fig.text(0.02, 0.01,
             f"Run {paths.RUN}, n = {n} infants with polygenic scores (EUR ancestry). Factor = mean of z-scored scores "
             "(Grotzinger et al. 2022 groups; D23). y ~ factor + PC1–10 + sex + GA at birth + PMA spline, SEs clustered "
             "by family.\nFilled points / *: pass BH FDR. Defined after the step 5 null; a follow-up, "
             "not a confirmatory test.", fontsize=fs.NOTE_SIZE - 1, color=fs.TEXT_SECONDARY, va="bottom")
    fs.save_figure(fig, paths.FIGURES / "b11b_prs_factors")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, _ = load()
    d = add_factors(d)
    rows = []
    for f in FACTORS:
        for y in AXES + LOBES:
            rows.append(dict(set="main", outcome=y, score=f, **fit(d, y, f)))
        for y in AXES:
            for name, sub, npc in (("term_born", d[d.ga_birth >= 37], 10), ("pc1_5", d, 5),
                                   ("train_only", d[d.split == "train"], 10)):
                rows.append(dict(set=name, outcome=y, score=f, **fit(sub, y, f, n_pc=npc)))
    r = pd.DataFrame(rows)
    r["family"], r["q"] = "", np.nan
    for mask, name in (((r.set == "main") & r.outcome.isin(AXES), "primary"),
                       ((r.set == "main") & r.outcome.isin(LOBES), "lobes")):
        r.loc[mask, "family"] = name
        r.loc[mask, "q"] = bh(r.loc[mask, "p"])
    r.to_csv(OUT / "prs_factors.csv", index=False)
    within_r = {f: float(np.corrcoef(d[c[0]], d[c[1]])[0, 1]) for f, c in FACTORS.items() if len(c) == 2}
    prim = r[r.family == "primary"].sort_values("p")
    lob = r[r.family == "lobes"]
    summ = dict(n=int(len(d)), within_factor_r=within_r,
                factor_r=pd.DataFrame({f: d[f] for f in FACTORS}).corr().round(2).to_dict(),
                primary=prim[["score", "outcome", "beta", "lo", "hi", "p", "q"]].round(4).to_dict("records"),
                n_primary_p05=int((prim.p < 0.05).sum()),
                lobes_passing=lob[lob.q < 0.05][["score", "outcome", "beta", "p", "q"]].round(4).to_dict("records"),
                n_lobes_p05=int((lob.p < 0.05).sum()), lobes_smallest_q=float(lob.q.min()),
                sensitivity=r[r.outcome.isin(AXES)].pivot_table(index=["score", "outcome"], columns="set",
                                                               values="beta").round(3).reset_index()
                .to_dict("records"))
    (OUT / "step5b_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    figure(r, len(d))
    with pd.option_context("display.width", 200):
        print(prim[["score", "outcome", "n", "beta", "lo", "hi", "p", "q"]].round(3).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k not in ("primary", "sensitivity")}, indent=1, default=float))
    print(pd.DataFrame(summ["sensitivity"]).to_string())


if __name__ == "__main__":
    main()
