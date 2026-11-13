"""Step 1 pilot: are the 10% networks sound, and which measure to follow?

    python scripts/pilot_bin10.py        # after scripts/build_bin10.py

Training infants only (569); no polygenic scores or outcomes are read.
For each candidate measure (communicability, between-lobe share) and each of
the 12 lobe features, a penalised spline on PMA (the model's basis, smoothing
chosen by GCV) gives the age curve and the residual spread. The criteria were
fixed before looking (RESEARCH_PLAN_BIN10.md, step 1):
  a  developmental change that differs between lobes
  b  robustness to the density choice (8% and 12% vs 10%)
  c  low age-adjusted links with motion and pre-threshold density
  d  structure in the spread (share of the first principal component)
  e  distribution (skew)

Writes results/bin10/pilot/pilot_summary.json and figures/bin10/b1-b4.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skew, spearmanr

import paths
import atlas
import figstyle as fs
from model import splines
from scripts.build_bin10 import lobe_means, GROUP_NAMES, DENSITIES

FEAT = paths.PROJECT / "results" / "features"
OUT = paths.PROJECT / "results" / "bin10" / "pilot"
FIG = paths.PROJECT / "figures" / "bin10"
T = np.linspace(27, 45, 181)
N_BOOT = 200
MEASURES = {"communicability": "communicability", "between_lobe_share": "between-lobe share"}
M_COLOURS = {"communicability": fs.MEASURE_COLOURS["communicability"],
             "between_lobe_share": fs.COLORS["blue"]}
LOBE_COLOURS = dict(zip(atlas.LOBES, fs.COLOR_CYCLE[:6]))
SHORT = [f"{n[:4]} {n[-1]}" for n in GROUP_NAMES]


class Curve:
    """Penalised spline y ~ B(t) beta, smoothing chosen by GCV."""

    def __init__(self, t, y, lam=None):
        self.B = splines.basis(t)
        self.P = splines.difference_penalty(splines.M_DEFAULT)
        BtB, Bty = self.B.T @ self.B, self.B.T @ y
        if lam is None:
            best = np.inf
            for l in np.logspace(-2, 4, 25):
                A = np.linalg.solve(BtB + l * self.P, np.c_[Bty, BtB])
                rss = ((y - self.B @ A[:, 0]) ** 2).sum()
                gcv = len(y) * rss / (len(y) - np.trace(A[:, 1:])) ** 2
                if gcv < best:
                    best, lam = gcv, l
        self.lam = lam
        self.beta = np.linalg.solve(BtB + lam * self.P, Bty)
        self.resid = y - self.B @ self.beta

    def __call__(self, t):
        return splines.basis(np.atleast_1d(t)) @ self.beta


def residualise(t, y):
    ok = np.isfinite(y)
    r = np.full_like(y, np.nan, dtype=float)
    r[ok] = Curve(t[ok], y[ok]).resid
    return r


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    coh = pd.read_csv(paths.PROJECT / "results" / "cohort" / "cohort.csv")
    tr = (coh.in_primary.values.astype(bool) & (coh.split == "train").values)
    c = coh[tr].reset_index(drop=True)
    t = c.scan_age.values
    nodes = np.load(FEAT / "bin10_mst_nodes.npz")
    dens = list(nodes["densities"]); j10 = dens.index(10)
    rng = np.random.default_rng(11)
    OUT.mkdir(parents=True, exist_ok=True)
    fs.apply_style()

    # --- network checks ---------------------------------------------------------
    weakest = nodes["weakest_weight"][tr, j10]
    edges = nodes["edges_10"][tr]
    freq = edges.mean(0)
    consensus = np.argsort(-freq, kind="stable")[:DENSITIES[10]]
    overlap = edges[:, consensus].mean(1)
    net = dict(n_infants=int(tr.sum()), n_edges=DENSITIES[10],
               weakest_weight_median=float(np.median(weakest)),
               weakest_vs_pma_spearman=float(spearmanr(weakest, t)[0]),
               edges_ever_retained=int((freq > 0).sum()),
               edges_in_all_infants=int((freq == 1).sum()),
               edges_in_half_or_more=int((freq >= 0.5).sum()),
               consensus_overlap_median=float(np.median(overlap)),
               consensus_overlap_vs_pma_r=corr(overlap, t))
    fig_network(c, t, weakest, freq, overlap, net)

    # --- the two measures ---------------------------------------------------------
    summary = dict(network=net, measures={})
    motion = residualise(t, c.qc_dmri_eddy_rel_motion.values.astype(float))
    density = residualise(t, c.density_unthresh.values.astype(float))
    curves, resid10 = {}, {}
    for m in MEASURES:
        X = {d: lobe_means(nodes[m][tr, k]) for k, d in enumerate(dens)}
        fits = [Curve(t, X[10][:, g]) for g in range(12)]
        R = np.column_stack([f.resid for f in fits])
        sd = R.std(0, ddof=1)
        change = np.array([(f(42) - f(30))[0] for f in fits]) / sd
        bands = []
        for g, f in enumerate(fits):
            boot = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(t), len(t))
                boot.append(Curve(t[i], X[10][i, g], lam=f.lam)(T))
            bands.append(np.percentile(boot, [2.5, 97.5], axis=0))
        curves[m] = (fits, bands, X[10]); resid10[m] = R
        robust = {d: [corr(R[:, g], Curve(t, X[d][:, g]).resid) for g in range(12)]
                  for d in dens if d != 10}
        change_d = {}
        for d in dens:
            if d == 10: continue
            fd = [Curve(t, X[d][:, g]) for g in range(12)]
            change_d[f"{d}%"] = [round(float((f(42) - f(30))[0] / f.resid.std(ddof=1)), 3) for f in fd]
        Z = (R - R.mean(0)) / sd
        ev = np.linalg.eigvalsh(np.corrcoef(Z.T))[::-1]
        summary["measures"][m] = dict(
            change_30_42_sd=change.round(3).tolist(),
            change_30_42_sd_other_density=change_d,
            change_mean=float(change.mean()), change_sd_across_lobes=float(change.std(ddof=1)),
            spread_sd=sd.tolist(),
            robust_r={f"{d}%": np.round(v, 3).tolist() for d, v in robust.items()},
            robust_r_min=float(min(min(v) for v in robust.values())),
            r_motion=[round(corr(R[:, g], motion), 3) for g in range(12)],
            r_density=[round(corr(R[:, g], density), 3) for g in range(12)],
            pc_share=(ev / ev.sum()).round(3)[:4].tolist(),
            skew_raw=np.round(skew(X[10], axis=0), 2).tolist(),
            skew_resid=np.round(skew(R, axis=0), 2).tolist(),
            gcv_lambda=[float(f.lam) for f in fits])
    summary["measures_r_by_lobe"] = [round(corr(resid10["communicability"][:, g],
                                                resid10["between_lobe_share"][:, g]), 3)
                                     for g in range(12)]
    summary["lobes"] = GROUP_NAMES.tolist()
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))

    fig_lobes("communicability", *curves["communicability"], t, "b2_communicability_by_lobe",
              "communicability")
    fig_lobes("between_lobe_share", *curves["between_lobe_share"], t,
              "b3_between_share_by_lobe", "share of edges leaving the lobe")
    fig_comparison(summary)


def fig_network(c, t, weakest, freq, overlap, net):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.42, bottom=0.28)
    pre = c.preterm.values.astype(bool)
    ax = axes[0]
    for m, col, lab in ((~pre, fs.TERM_COLOUR, "term-born"), (pre, fs.PRETERM_COLOUR, "preterm-born")):
        ax.scatter(t[m], weakest[m], s=8, color=col, alpha=0.6, label=lab, linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("PMA (weeks)"); ax.set_ylabel("weakest kept edge weight")
    fs.panel_title(ax, "A", "Weakest kept edge")
    ax.text(0.97, 0.05, f"Spearman r = {net['weakest_vs_pma_spearman']:+.2f}",
            transform=ax.transAxes, ha="right", fontsize=fs.NOTE_SIZE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    ax.hist(freq[freq > 0], bins=40, color=fs.COLORS["blue"], edgecolor=fs.TEXT_PRIMARY, linewidth=0.5)
    ax.set_xlabel("share of infants keeping the edge"); ax.set_ylabel("edges")
    fs.panel_title(ax, "B", "Edges kept, across infants")
    ax.text(0.97, 0.95, f"{net['edges_ever_retained']:,} of 4,005 edges\nkept in any infant",
            transform=ax.transAxes, ha="right", va="top", fontsize=fs.NOTE_SIZE)
    ax = axes[2]
    for m, col in ((~pre, fs.TERM_COLOUR), (pre, fs.PRETERM_COLOUR)):
        ax.scatter(t[m], overlap[m], s=8, color=col, alpha=0.6, linewidths=0)
    ax.set_xlabel("PMA (weeks)"); ax.set_ylabel("share in the consensus")
    fs.panel_title(ax, "C", "Overlap with consensus")
    ax.text(0.03, 0.05, f"r = {net['consensus_overlap_vs_pma_r']:+.2f}",
            transform=ax.transAxes, fontsize=fs.NOTE_SIZE)
    fig.text(0.07, -0.04, f"{net['n_infants']} training infants. Each network: maximum spanning "
             f"tree + strongest edges, {net['n_edges']} edges (10%). C: consensus = the "
             f"{net['n_edges']} edges kept most often across infants.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, FIG / "b1_network_check")


def fig_lobes(m, fits, bands, X, t, name, ylabel):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.3, bottom=0.14)
    for k, (lobe, ax) in enumerate(zip(atlas.LOBES, axes.flat)):
        for h, ls in (("L", "-"), ("R", "--")):
            g = list(GROUP_NAMES).index(f"{lobe}_{h}")
            ax.scatter(t, X[:, g], s=4, color=fs.NEUTRAL_GREY, alpha=0.15, linewidths=0, zorder=1)
            ax.fill_between(T, *bands[g], color=LOBE_COLOURS[lobe], alpha=0.25, linewidth=0, zorder=2)
            ax.plot(T, fits[g](T), color=LOBE_COLOURS[lobe], ls=ls, lw=2, zorder=3)
        fs.panel_title(ax, "ABCDEF"[k], lobe)
        if k >= 3: ax.set_xlabel("PMA (weeks)")
        if k % 3 == 0: ax.set_ylabel(ylabel)
    fig.legend(handles=[Line2D([], [], color=fs.TEXT_PRIMARY, ls="-", lw=2, label="left"),
                        Line2D([], [], color=fs.TEXT_PRIMARY, ls="--", lw=2, label="right")],
               loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2)
    fig.text(0.07, 0.0, f"{len(t)} training infants, binarised at 10% density. Lines: penalised "
             f"spline on PMA; shaded: 95% band from {N_BOOT} bootstrap resamples of infants;\n"
             "grey: each infant's lobe value (mean over the lobe's regions, computed on the "
             "90-region network). Each panel has its own y-scale.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, FIG / name)


def fig_comparison(s):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    x = np.arange(12)
    off = {"communicability": -0.15, "between_lobe_share": 0.15}
    handles = [Line2D([], [], color=M_COLOURS[m], marker="o", ls="", ms=8, label=lab)
               for m, lab in MEASURES.items()]

    ax = axes[0, 0]
    for m in MEASURES:
        v = s["measures"][m]
        ax.scatter(x + off[m], v["change_30_42_sd"], color=M_COLOURS[m], s=40, zorder=3,
                   edgecolors=fs.TEXT_PRIMARY, linewidths=0.5)
    fs.reference_line(ax, 0)
    ax.set_xticks(x, SHORT, rotation=60, ha="right")
    ax.set_ylabel("change, in SDs of the spread")
    fs.panel_title(ax, "A", "Development, 30 to 42 weeks")
    ax.text(0.02, 0.97, "\n".join(f"{lab}: SD across lobes "
                                  f"{s['measures'][m]['change_sd_across_lobes']:.2f}"
                                  for m, lab in MEASURES.items()),
            transform=ax.transAxes, va="top", fontsize=fs.NOTE_SIZE)

    ax = axes[0, 1]
    for m in MEASURES:
        for d, mk in (("8%", "o"), ("12%", "s")):
            ax.scatter(x + off[m], s["measures"][m]["robust_r"][d], color=M_COLOURS[m], marker=mk,
                       s=36, zorder=3, edgecolors=fs.TEXT_PRIMARY, linewidths=0.5)
    fs.reference_line(ax, 0.8)
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
    ax.set_xticks(x, SHORT, rotation=60, ha="right")
    ax.set_ylabel("r with the 10% values")
    fs.panel_title(ax, "B", "Same infants at 8% and 12%?")
    ax.legend(handles=[Line2D([], [], color=fs.TEXT_SECONDARY, marker="o", ls="", label="8%"),
                       Line2D([], [], color=fs.TEXT_SECONDARY, marker="s", ls="", label="12%")],
              loc="lower right")

    ax = axes[1, 0]
    cats = [(m, k) for m in MEASURES for k in ("r_motion", "r_density")]
    for i, (m, k) in enumerate(cats):
        v = np.abs(s["measures"][m][k])
        ax.scatter(np.full(12, i) + rng_jitter(12, i), v, color=M_COLOURS[m], s=30,
                   edgecolors=fs.TEXT_PRIMARY, linewidths=0.5, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=fs.TEXT_PRIMARY, lw=2)
    ax.set_xticks(range(4), ["motion", "density", "motion", "density"])
    ax.set_ylabel("|r|, age-adjusted"); ax.set_ylim(0, None)
    fs.panel_title(ax, "C", "Links with artefacts")

    ax = axes[1, 1]
    for i, m in enumerate(MEASURES):
        pc = s["measures"][m]["pc_share"]
        ax.bar(i, pc[0], color=M_COLOURS[m], edgecolor=fs.TEXT_PRIMARY, linewidth=0.6, width=0.6)
        ax.bar(i, 1 - pc[0], bottom=pc[0], color=fs.NEUTRAL_GREY, alpha=0.5,
               edgecolor=fs.TEXT_PRIMARY, linewidth=0.6, width=0.6)
        ax.text(i, pc[0] / 2, f"{pc[0]:.0%}", ha="center", va="center", fontweight="bold",
                fontsize=fs.NOTE_SIZE, color="white")
    ax.set_xticks(range(2), list(MEASURES.values()))
    ax.set_xlim(-0.7, 1.7); ax.set_ylim(0, 1)
    ax.set_ylabel("share of the spread")
    fs.panel_title(ax, "D", "Spread on one global component")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.text(0.07, -0.05, "Training infants, 10% density. Spread = residuals around each "
             "lobe's age curve. D: colour = first principal component of the 12 residuals\n"
             "(all lobes moving together); grey = the rest.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, FIG / "b4_measure_comparison")


def rng_jitter(n, seed):
    return np.random.default_rng(seed).uniform(-0.12, 0.12, n)


if __name__ == "__main__":
    main()
