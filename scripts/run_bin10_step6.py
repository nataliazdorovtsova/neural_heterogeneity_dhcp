"""Step 6 of RESEARCH_PLAN_BIN10.md (section 7): positions in the spread at
birth and behaviour / temperament at 18 months.

    NH_RUN=bin10_no7 python scripts/run_bin10_step6.py

Sample: age-adjusted positions (D21, without the 7, D20) with CBCL (n = 518)
or ECBQ (n = 516). Outcomes: the 7 CBCL syndrome t-scores in cbcl_clean.csv
(no internalising / externalising totals in the file) and the 3 ECBQ scales.

Model, one outcome x one position at a time:

    outcome (z) ~ position (z) + male + GA at birth + PMA at scan + age at assessment

OLS, SEs clustered by family. Primary: 10 outcomes x axes 1-3 = 30 tests,
BH FDR. Secondary: 10 outcomes x 12 lobes, BH. CBCL t-scores are floored at
50 (33-68% of infants at the floor), so every test is repeated as a partial
Spearman correlation (ranks of outcome and position, residualised on the
covariates). Sensitivity: training only; + mother's education. No
PRS-outcome analysis (no PRS-axis link passed in step 5, plan section 7).
Writes results/<run>/step6/; figure b12_outcomes.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import paths
import figstyle as fs
from scripts.build_bin10 import GROUP_NAMES
from scripts.run_bin10_step4 import AXES, LOBES, AXIS_LABELS, ols_cluster, bh, load_data

OUT = paths.RESULTS / "step6"
CBCL = ["emotional_reactive_t", "anxious_depressed_t", "somatic_complaints_t", "withdrawn_t",
        "sleep_problems_t", "attention_problems_t", "aggressive_behavior_t"]
ECBQ = ["surgency", "negative_affect", "effortful_control"]
LABEL = {c: c.replace("_t", "").replace("_", " ") for c in CBCL + ECBQ}
LABEL["aggressive_behavior_t"] = "aggressive behaviour"
LABEL["emotional_reactive_t"] = "emotionally reactive"


def load(positions="positions_scaled.csv"):
    d = load_data(positions)
    out = []
    for f, cols in ((paths.CBCL, CBCL), (paths.ECBQ, ECBQ)):
        o = pd.read_csv(f)
        o["dhcp_id"] = o.src_id.astype(str).str.replace("sub-", "", regex=False)
        o = o[["dhcp_id", "age_months"] + cols].rename(columns={"age_months": f"age_{f.stem}"})
        out.append(o)
    d = d.merge(out[0], on="dhcp_id", how="left").merge(out[1], on="dhcp_id", how="left")
    d["age_assess"] = d.age_cbcl_clean.fillna(d.age_ecbq_clean)
    for c in CBCL:
        d.loc[d.age_cbcl_clean.notna(), "age_assess_" + c] = d.age_cbcl_clean
    for c in ECBQ:
        d.loc[d.age_ecbq_clean.notna(), "age_assess_" + c] = d.age_ecbq_clean
    return d


def fit(d, y_col, x_col, extra=()):
    age = "age_assess_" + y_col
    cols = [y_col, x_col, "male", "ga_birth", "scan_age", age, "family", *extra]
    d = d[cols].dropna()
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    y, x = z(d[y_col].values), z(d[x_col].values)
    C = np.column_stack([np.ones(len(d)), d.male, d.ga_birth, d.scan_age, d[age], *[d[e] for e in extra]])
    b, se, df = ols_cluster(y, np.column_stack([C[:, :1], x, C[:, 1:]]), d.family.values)
    tq = stats.t.ppf(0.975, df)
    # partial Spearman: ranks of y and x, residualised on the covariates
    ry, rx = stats.rankdata(d[y_col]), stats.rankdata(d[x_col])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rho = np.corrcoef(res(ry), res(rx))[0, 1]
    dof = len(d) - C.shape[1] - 1
    t_rho = rho * np.sqrt(dof / (1 - rho ** 2))
    return dict(beta=b[1], lo=b[1] - tq * se[1], hi=b[1] + tq * se[1], t=b[1] / se[1],
                p=2 * stats.t.sf(abs(b[1] / se[1]), df), rho=rho, p_rank=2 * stats.t.sf(abs(t_rho), dof),
                n=len(d))


def figure(r, n_c, n_e):
    fs.apply_style()
    m = r[r.set == "main"]
    outs = CBCL + ECBQ
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], wspace=0.35, left=0.13, right=0.98, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    a = m[m.outcome.isin(outs) & m.position.isin(AXES)]
    B = np.array([[a[(a.outcome == o) & (a.position == x)].beta.iloc[0] for x in AXES] for o in outs])
    vmax = max(0.2, float(np.ceil(np.abs(m.beta).max() * 10) / 10))
    im = ax.imshow(B, cmap=fs.DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, o in enumerate(outs):
        for j, x in enumerate(AXES):
            row = a[(a.outcome == o) & (a.position == x)].iloc[0]
            ax.text(j, i, f"{row.beta:+.2f}" + ("*" if row.q < 0.05 else ""), ha="center", va="center",
                    fontsize=9, fontweight="bold" if row.q < 0.05 else "normal")
    ax.axhline(len(CBCL) - 0.5, color="white", lw=5)
    ax.set_yticks(range(len(outs)), [LABEL[o] for o in outs])
    ax.set_xticks(range(3), ["axis 1", "axis 2", "axis 3"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.95, (len(CBCL) - 1) / 2, f"CBCL (n = {n_c})", transform=ax.get_yaxis_transform(), rotation=90,
            ha="center", va="center", fontsize=10, color=fs.TEXT_MUTED, style="italic")
    ax.text(-0.95, len(CBCL) + (len(ECBQ) - 1) / 2, f"ECBQ (n = {n_e})", transform=ax.get_yaxis_transform(),
            rotation=90, ha="center", va="center", fontsize=10, color=fs.TEXT_MUTED, style="italic")
    ax.set_title("A   Axes (30 tests, BH FDR)", pad=30)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03).set_label("standardised β")

    ax = fig.add_subplot(gs[1])
    l = m[m.outcome.isin(outs) & m.position.isin(LOBES)]
    T = np.array([[l[(l.outcome == o) & (l.position == x)].t.iloc[0] for x in LOBES] for o in outs])
    tv = max(3.0, float(np.ceil(np.abs(T).max())))
    ax.imshow(T, cmap=fs.DIVERGING, norm=TwoSlopeNorm(0, -tv, tv), aspect="auto")
    for i, o in enumerate(outs):
        for j, x in enumerate(LOBES):
            row = l[(l.outcome == o) & (l.position == x)].iloc[0]
            ax.text(j, i, f"{row.t:+.1f}" + ("*" if row.q < 0.05 else ""), ha="center", va="center", fontsize=8)
    ax.axhline(len(CBCL) - 0.5, color="white", lw=5)
    ax.set_yticks(range(len(outs)), [])
    ax.set_xticks(range(len(LOBES)), [g.replace("_", " ") for g in GROUP_NAMES], rotation=45, ha="right")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("B   t-values, lobe z-scores (120 tests, BH)", pad=12)
    fig.text(0.02, 0.01,
             f"Run {paths.RUN}, age-adjusted positions at birth (D21); outcomes at about 18 months. outcome ~ position + "
             "sex + GA at birth + PMA at scan + age at assessment, SEs clustered by family.\n*: passes BH FDR. "
             "CBCL t-scores are floored at 50; every test is repeated as a partial Spearman correlation "
             "(results/<run>/step6).", fontsize=fs.NOTE_SIZE - 1, color=fs.TEXT_SECONDARY, va="bottom")
    fs.save_figure(fig, paths.FIGURES / "b12_outcomes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    rows = []
    for o in CBCL + ECBQ:
        for x in AXES + LOBES:
            rows.append(dict(set="main", outcome=o, position=x, **fit(d, o, x)))
        for x in AXES:
            rows.append(dict(set="train_only", outcome=o, position=x, **fit(d[d.split == "train"], o, x)))
            rows.append(dict(set="mother_education", outcome=o, position=x, **fit(d, o, x, ["mother_education"])))
    r = pd.DataFrame(rows)
    r["family"], r["q"], r["q_rank"] = "", np.nan, np.nan
    for mask, name in (((r.set == "main") & r.position.isin(AXES), "axes"),
                       ((r.set == "main") & r.position.isin(LOBES), "lobes")):
        r.loc[mask, "family"] = name
        r.loc[mask, "q"] = bh(r.loc[mask, "p"])
        r.loc[mask, "q_rank"] = bh(r.loc[mask, "p_rank"])
    r.to_csv(OUT / "outcomes.csv", index=False)
    ax_ = r[r.family == "axes"].sort_values("p")
    lob = r[r.family == "lobes"]
    n_c = int(d[CBCL[0]].notna().sum())
    n_e = int(d[ECBQ[0]].notna().sum())
    cols = ["outcome", "position", "n", "beta", "lo", "hi", "p", "q", "rho", "p_rank", "q_rank"]
    summ = dict(n_cbcl=n_c, n_ecbq=n_e, floor_share={c: float((d[c].dropna() == 50).mean()) for c in CBCL},
                axes=ax_[cols].round(4).to_dict("records"),
                axes_passing=ax_[(ax_.q < 0.05) | (ax_.q_rank < 0.05)][cols].round(4).to_dict("records"),
                n_axes_p05=int((ax_.p < 0.05).sum()), n_axes_prank05=int((ax_.p_rank < 0.05).sum()),
                lobes_passing=lob[(lob.q < 0.05) | (lob.q_rank < 0.05)][cols].round(4).to_dict("records"),
                n_lobes_p05=int((lob.p < 0.05).sum()), lobes_smallest_q=float(lob.q.min()),
                lobes_smallest_q_rank=float(lob.q_rank.min()),
                max_abs_ci_axes=float(ax_[["lo", "hi"]].abs().max().max()),
                sensitivity_top=r[r.position.isin(AXES)].pivot_table(index=["outcome", "position"], columns="set",
                                                                    values="beta").loc[
                    list(zip(ax_.head(4).outcome, ax_.head(4).position))].round(3).reset_index().to_dict("records"))
    (OUT / "step6_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    figure(r, n_c, n_e)
    with pd.option_context("display.width", 220):
        print(ax_[cols].round(3).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k != "axes"}, indent=1, default=float))


if __name__ == "__main__":
    main()
