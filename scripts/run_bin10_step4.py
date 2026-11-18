"""Step 4 of RESEARCH_PLAN_BIN10.md (section 5): what goes with an infant's
position in the spread?

    NH_RUN=bin10_no7 python scripts/run_bin10_step4.py

Positions: the age-adjusted positions of D21 (results/<run>/positions_scaled.csv),
for every infant in the primary set except the 7 `low7` infants (D20).
- primary: axes 1-3 (the axes above noise in step 3);
- secondary: the 12 lobe z-scores.

Model, one outcome y (z-scored) and one predictor x at a time:

    y ~ x + rcs(PMA at scan, 4 knots) [+ GA at birth, unless x is GA at birth or preterm]

OLS with standard errors clustered by family (twins and triplets share the
first 7 characters of the dHCP ID; CR1, t with G - 1 df). Continuous
predictors are z-scored (beta = SD of y per SD of x); binary ones (male,
preterm) are 0/1 (beta = difference in SD of y).

Predictors:
- biological, the tested family: GA at birth; preterm (< 37 weeks); male;
  birth weight z for GA (within this cohort, see `birth_weight_z`); head
  circumference at scan; radiology score (1-5, as a number).
  Axes: Bonferroni over 6 predictors x 3 axes = 18. Lobes: BH FDR over
  6 x 12 = 72.
- artefact checks (not in the family): log motion (eddy relative); density
  before thresholding.
- exploratory, flagged (not in the family): the mother's age when she left
  full-time education (dHCP enrolment form, cpenr01). The release has no
  deprivation index; education is the nearest socioeconomic measure.

Sensitivity sets: + log motion as a covariate (motion goes with axis 2 and
with GA at birth); training infants only; without the infants scanned before
32 weeks (D21: their age-adjusted positions are about 20% too small); the
constant-spread positions of step 3 (positions.csv; axes 1-2 there are
rotated by 22 degrees in their plane, so this is approximate for them).

Writes results/<run>/step4/covariates.csv and step4_summary.json, and
figures b10_covariates (and b10b_ga_spread when GA at birth passes).
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
from scripts.run_spec import clean_covariates

OUT = paths.RESULTS / "step4"
ENROLMENT = paths.DHCP2026_DATA / "1_Participant_enrolment" / "cpenr01.txt"
AXES = ["axis1", "axis2", "axis3"]
LOBES = [f"z_{n}" for n in GROUP_NAMES]
# short names from the loadings of C(40) (results/<run>/scale/summary.json)
AXIS_LABELS = {"axis1": "axis 1\nfrontal vs limbic R, subcortical",
               "axis2": "axis 2\nsubcortical, occipital vs limbic L",
               "axis3": "axis 3\nparietal vs occipital, limbic"}
BIO = ["ga_birth", "preterm", "male", "birth_weight_z", "head_circumference", "radiology"]
ARTEFACT = ["log_motion", "density_unthresh"]
EXPLORATORY = ["mother_education"]
LABELS = {"ga_birth": "GA at\nbirth", "preterm": "preterm\n(< 37 wk)", "male": "male",
          "birth_weight_z": "birth weight\nz for GA", "head_circumference": "head\ncircumference",
          "radiology": "radiology\nscore", "log_motion": "motion (log)",
          "density_unthresh": "density before\nthreshold", "mother_education": "mother's\neducation (age)"}
BINARY = {"preterm", "male"}
NO_GA_ADJ = {"ga_birth", "preterm"}


# ---------------------------------------------------------------- helpers ---

def rcs(x, n_knots=4):
    """Restricted cubic spline basis (Harrell), knots at the 5/35/65/95th percentiles."""
    x = np.asarray(x, float)
    k = np.percentile(x, {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[n_knots])
    s = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(n_knots - 2):
        cols.append((np.maximum(x - k[j], 0) ** 3
                     - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / s)
    return np.column_stack(cols)


def ols_cluster(y, X, groups):
    """OLS coefficients and CR1 cluster-robust covariance; returns beta, se, df."""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g_codes, g = np.unique(groups, return_inverse=True)
    G = len(g_codes)
    S = np.zeros((G, p))
    np.add.at(S, g, X * u[:, None])
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - p)
    V = c * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V)), G - 1


def fit_one(d, outcome, pred, extra=()):
    """beta, 95% CI, p and n for one outcome x predictor (+ extra covariates)."""
    extra = [e for e in extra if e != pred]
    cols = [outcome, pred, "scan_age", "ga_birth", "family", *extra]
    d = d[cols].dropna()
    y = d[outcome].values
    y = (y - y.mean()) / y.std(ddof=1)
    x = d[pred].values.astype(float)
    if pred not in BINARY:
        x = (x - x.mean()) / x.std(ddof=1)
    Z = [np.ones(len(d)), x, rcs(d.scan_age.values)]
    if pred not in NO_GA_ADJ:
        Z.append(d.ga_birth.values)
    Z += [d[e].values for e in extra]
    X = np.column_stack(Z)
    b, se, df = ols_cluster(y, X, d.family.values)
    tq = stats.t.ppf(0.975, df)
    t = b[1] / se[1]
    return dict(beta=b[1], lo=b[1] - tq * se[1], hi=b[1] + tq * se[1],
                p=2 * stats.t.sf(abs(t), df), n=len(d))


def birth_weight_z(d):
    """Residual of log birth weight on rcs(GA at birth) + sex, divided by its SD."""
    m = d.birth_weight.notna()
    X = np.column_stack([np.ones(m.sum()), rcs(d.ga_birth[m]), d.male[m]])
    y = np.log(d.birth_weight[m].values)
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    z = pd.Series(np.nan, index=d.index)
    z[m] = r / r.std(ddof=1)
    return z


def mother_education(ids):
    e = pd.read_csv(ENROLMENT, sep="\t", skiprows=[1], low_memory=False)
    e["dhcp_id"] = e.src_subject_id.astype(str).str.replace("sub-", "", regex=False)
    v = pd.to_numeric(e.groupby("dhcp_id").mother_education1.first(), errors="coerce")
    v[v <= 0] = np.nan                           # -999 / -998: missing codes
    return v.reindex(ids).values


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------- data ---

def load_data(positions_file):
    coh = clean_covariates(pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv"))
    coh = coh[coh.in_primary.astype(bool)].copy()
    coh["dhcp_id"] = coh.dhcp_id.astype(str)
    pos = pd.read_csv(paths.RESULTS / positions_file)
    pos["dhcp_id"] = pos.dhcp_id.astype(str)
    d = pos.drop(columns=["scan_age"], errors="ignore").merge(coh, on=["dhcp_id"], suffixes=("", "_coh"))
    d = d[~d.low7.astype(bool)].reset_index(drop=True)
    d["family"] = d.dhcp_id.str[:7]
    d["ga_birth"] = d.birth_age
    d["preterm"] = (d.birth_age < 37).astype(float)
    d["male"] = (d.sex == "male").astype(float)
    d["birth_weight_z"] = birth_weight_z(d)
    d["head_circumference"] = d.head_circumference_scan
    d["radiology"] = d.radiology_score
    d["log_motion"] = np.log(d.qc_dmri_eddy_rel_motion)
    d["mother_education"] = mother_education(d.dhcp_id)
    return d


def run_set(d, name, extra=()):
    rows = []
    for outcome in AXES + LOBES:
        for pred in BIO + ARTEFACT + EXPLORATORY:
            r = fit_one(d, outcome, pred, extra)
            block = "bio" if pred in BIO else "artefact" if pred in ARTEFACT else "exploratory"
            rows.append(dict(set=name, outcome=outcome, family="axes" if outcome in AXES else "lobes",
                             predictor=pred, block=block, **r))
    df = pd.DataFrame(rows)
    df["p_bonf"] = np.nan
    df["q_fdr"] = np.nan
    a = (df.family == "axes") & (df.block == "bio")
    df.loc[a, "p_bonf"] = np.minimum(df.loc[a, "p"] * a.sum(), 1)
    l = (df.family == "lobes") & (df.block == "bio")
    df.loc[l, "q_fdr"] = bh(df.loc[l, "p"])
    df["passes"] = (df.p_bonf < 0.05) | (df.q_fdr < 0.05)
    return df


# ----------------------------------------------------------------- figure ---

def heat(ax, df, rows, cols, row_labels, vmax, marks):
    M = np.array([[df[(df.outcome == r) & (df.predictor == c)].beta.iloc[0] for c in cols] for r in rows])
    im = ax.imshow(M, cmap=fs.DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            row = df[(df.outcome == r) & (df.predictor == c)].iloc[0]
            txt = f"{row.beta:+.2f}"
            strong = marks(row)
            ax.text(j, i, txt + ("*" if strong else ""), ha="center", va="center",
                    fontsize=fs.VALUE_SIZE, fontweight="bold" if strong else "normal",
                    color="white" if abs(row.beta) > 0.6 * vmax else fs.TEXT_PRIMARY)
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], rotation=0, fontsize=8.5)
    ax.set_yticks(range(len(rows)), row_labels, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def figure(main):
    fs.apply_style()
    ax_df = main[main.family == "axes"]
    lobe_df = main[main.family == "lobes"]
    vmax = max(0.3, float(np.ceil(main.beta.abs().max() * 10) / 10))
    cols = BIO + ARTEFACT + EXPLORATORY
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 1.45], width_ratios=[1.75, 1],
                          hspace=0.32, wspace=0.28, left=0.11, right=0.98, top=0.93, bottom=0.12)

    # A: axes x predictors
    ax = fig.add_subplot(gs[0, :])
    im = heat(ax, ax_df, AXES, cols, [AXIS_LABELS[a] for a in AXES], vmax,
              lambda r: (r.block == "bio" and r.p_bonf < 0.05) or (r.block != "bio" and r.p < 0.05))
    for x in (len(BIO) - 0.5, len(BIO) + len(ARTEFACT) - 0.5):
        ax.axvline(x, color="white", lw=6)
    for x0, x1, lab in ((0, len(BIO) - 1, "biological (tested family)"),
                        (len(BIO), len(BIO) + len(ARTEFACT) - 1, "artefact checks"),
                        (len(cols) - 1, len(cols) - 1, "exploratory")):
        ax.text((x0 + x1) / 2, -0.75, lab, ha="center", va="bottom", fontsize=10, color=fs.TEXT_SECONDARY)
    ax.set_title("A   Which infant characteristics go with the position on each axis?", pad=30)

    # B: forest plot, the axis GA at birth relates to most
    k = ax_df[ax_df.predictor == "ga_birth"].assign(a=lambda d: d.beta.abs()).sort_values("a").outcome.iloc[-1]
    ax = fig.add_subplot(gs[1, 0])
    f = ax_df[ax_df.outcome == k].set_index("predictor").loc[cols[::-1]]
    for i, (pred, r) in enumerate(f.iterrows()):
        col = fs.COLOR_CYCLE[0] if r.block == "bio" else fs.NEUTRAL_GREY
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.2)
        ax.plot(r.beta, i, "o", color=col, ms=8, mfc=col if r.passes else "white", mew=2)
        ax.text(max(vmax, f.hi.max()) * 1.04, i, f"n = {r.n}", va="center", fontsize=8.5, color=fs.TEXT_MUTED)
    ax.set_yticks(range(len(f)), [LABELS[c].replace("\n", " ") for c in f.index], fontsize=9.5)
    fs.reference_line(ax, 0, axis="x")
    ax.set_xlim(min(-vmax, f.lo.min() - 0.03), max(vmax, f.hi.max()) * 1.25)
    ax.set_xlabel("standardised β with 95% CI (filled: passes Bonferroni)")
    fs.panel_title(ax, "B", f"{AXIS_LABELS[k].splitlines()[0].capitalize()}: all predictors")

    # C: lobes x biological predictors
    ax = fig.add_subplot(gs[1, 1])
    heat(ax, lobe_df, LOBES, BIO, [n.replace("_", " ") for n in GROUP_NAMES], vmax,
         lambda r: r.q_fdr < 0.05)
    ax.tick_params(axis="x", labelsize=7.5)
    fs.panel_title(ax, "C", "Lobe z-scores (FDR)")
    cax = fig.add_axes([0.40, 0.045, 0.2, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("standardised β", fontsize=10)
    cax.xaxis.set_label_position("top")
    n = int(ax_df[ax_df.predictor == "ga_birth"].n.iloc[0])
    fig.text(0.02, 0.012,
             f"Run {paths.RUN}, n = {n} infants (age-adjusted positions, D21). Each cell: one regression, y ~ x + PMA "
             "spline (+ GA at birth), SEs clustered by family. *: passes Bonferroni (axes, 18 tests) or BH FDR "
             "(lobes, 72 tests);\nfor artefact checks and the exploratory column, *: p < 0.05 without correction. "
             "Continuous x per SD; male and preterm: difference in SD.",
             fontsize=fs.NOTE_SIZE - 1, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "b10_covariates")
    return k


def figure_ga(d, k):
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(d.scan_age, d[k], c=d.ga_birth, cmap="viridis_r", s=14, alpha=0.8, lw=0)
    fig.colorbar(sc, ax=ax).set_label("GA at birth (weeks)")
    fs.reference_line(ax, 0)
    ax.set_xlabel("PMA at scan (weeks)"); ax.set_ylabel(f"{k.replace('axis', 'axis ')} score (age-adjusted)")
    ax.set_title(f"{AXIS_LABELS[k].splitlines()[0].capitalize()} by age at scan and at birth")
    fs.save_figure(fig, paths.FIGURES / "b10b_ga_spread")


# ------------------------------------------------------------------- main ---

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_data("positions_scaled.csv")
    sets = {"main": d, "train_only": d[d.split == "train"], "no_lt32": d[d.scan_age >= 32]}
    res = pd.concat([run_set(v, k) for k, v in sets.items()]
                    + [run_set(d, "motion_adjusted", extra=["log_motion"])]
                    + [run_set(load_data("positions.csv"), "constant_spread")], ignore_index=True)
    res.to_csv(OUT / "covariates.csv", index=False)

    main_ = res[res.set == "main"]
    pas = main_[main_.passes]
    sens = {}
    for _, r in pas.iterrows():
        s = res[(res.outcome == r.outcome) & (res.predictor == r.predictor)].set_index("set")
        sens[f"{r.outcome}~{r.predictor}"] = {k: [round(s.loc[k, "beta"], 3), round(s.loc[k, "p"], 4)]
                                             for k in s.index}
    nominal = main_[(main_.block != "bio") & (main_.p < 0.05)]
    summ = dict(
        n=int(len(d)), n_train=int((d.split == "train").sum()), n_lt32=int((d.scan_age < 32).sum()),
        n_families_multi=int((d.family.value_counts() > 1).sum()),
        n_by_predictor={p: int(main_[(main_.outcome == "axis1") & (main_.predictor == p)].n.iloc[0])
                        for p in BIO + ARTEFACT + EXPLORATORY},
        deprivation="no deprivation index in the dHCP 2026 release (enrolment, clinical and follow-up "
                    "folders searched); mother's age at leaving full-time education used as an exploratory proxy",
        passing=pas[["outcome", "predictor", "beta", "lo", "hi", "p", "p_bonf", "q_fdr", "n"]]
        .round(4).to_dict("records"),
        sensitivity=sens,
        nominal_artefact_or_exploratory=nominal[["outcome", "predictor", "beta", "p"]].round(4).to_dict("records"),
        axes_table=main_[main_.family == "axes"][["outcome", "predictor", "beta", "lo", "hi", "p", "p_bonf"]]
        .round(4).to_dict("records"),
    )
    k = figure(main_)
    if (main_[(main_.family == "axes") & (main_.predictor == "ga_birth")].p_bonf < 0.05).any():
        figure_ga(d, k)
        summ["ga_figure_axis"] = k
    (OUT / "step4_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    with pd.option_context("display.width", 200):
        print(main_[main_.family == "axes"][["outcome", "predictor", "n", "beta", "lo", "hi", "p", "p_bonf"]]
              .round(3).to_string())
    print(json.dumps({k: summ[k] for k in ("n", "n_lt32", "n_families_multi", "n_by_predictor")}, indent=1))
    print(json.dumps(summ["passing"], indent=1))
    print(json.dumps(summ["sensitivity"], indent=1))
    print(json.dumps(summ["nominal_artefact_or_exploratory"], indent=1))


if __name__ == "__main__":
    main()
